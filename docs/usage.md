# Using driftlock

How to run the library, the native agent, and the pinned LHTB experiment harness.
For what driftlock is and why it is built this way, see [README](../README.md) and
[architecture](architecture.md).

## Core library quick start

The runner wraps an async function that performs one agent step. Each result carries
JSON-serializable agent state plus the observations used by the zero-token heuristics.
The filesystem checkpoint store is deliberately separate from the agent so other
backends (Docker, Harbor, cloud sandboxes) can implement the same interface.
Local snapshots include Git metadata, tracked files, and untracked files so a restore
returns both the worktree and repository state to the same point. Linked Git
worktrees and submodules are rejected because their mutable Git state lives outside
the workspace; use a self-contained clone for now. Snapshots are exact by default.

```python
from pathlib import Path

from driftlock import (
    DirectoryCheckpointStore,
    DriftlockRunner,
    HeuristicJudge,
    StepOutcome,
)

workspace = Path("/path/to/agent/workspace")
snapshots = Path("/path/to/snapshots")  # must be outside workspace


async def next_step(context):
    # Ask your agent for one action, execute it, and return its new state.
    # Reserve request prefill, then cap output by the remaining token allowance.
    return StepOutcome(
        action="pytest -q",
        state={"messages": []},
        changed_paths=("src/parser.py",),
        diff="...",
        tokens=1200,
        completed=False,
    )


runner = DriftlockRunner(
    DirectoryCheckpointStore(workspace, snapshots),
    HeuristicJudge(),
)
result = await runner.run(
    goal="Fix the parser without changing its public API",
    plan="Reproduce, patch, test",
    step=next_step,
    initial_state={"messages": []},
)
```

With no fine judge, coarse signals trigger rollback directly (the heuristics-only
ablation). Pass `CallableLLMJudge(async_completion_function)` to enable the two-tier
mode. The callable owns its provider SDK and credentials; driftlock sends it the
original goal, plan, recent trajectory, heuristic signals, and latest diff, and
expects a structured JSON verdict.

Periodic snapshots are retained across detector windows. When drift is confirmed,
the runner selects the newest checkpoint from before the earliest triggered signal
window, avoiding a superficially recent snapshot that already contains the loop,
stall, or error spike.

### Parallel workspace reads

Set `parallel_tool_calls=True` on `ToolCallingAgent` to overlap contiguous
`read_file` and `search_files` calls in one provider response:

```python
agent = ToolCallingAgent(
    environment,
    observer,
    async_completion_function,
    parallel_tool_calls=True,
)
```

All other tools are serial barriers, including shell commands, writes, completion,
planning, memory, retrieval, delegation, and MCP. Results, errors, history, and
audits retain the provider's call order. The existing `max_tool_calls_per_step`
limits both total calls and concurrency; the defaults remain 4 calls and 96,000
history characters. Truncated responses and responses above the call limit execute
no tools. Enabled read failure details are truncated to `max_tool_output_chars`.
Cancellation cancels and joins launched reads before propagating, without starting
later barriers.

The default is `False`, preserving serial execution and existing provider requests.
Opted-in environments must tolerate concurrent read `exec` requests;
`LocalEnvironment` supports this. Delegated children do not inherit this option.

### MCP tools over stdio

The optional MCP client connects to explicitly configured local servers. It supports
the [MCP stdio lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle)
and [tool discovery/calls](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
for protocol versions `2025-11-25` and `2025-06-18`. No additional dependencies are
required. HTTP, resources, prompts, sampling, and authorization flows are not
implemented in this version.

```python
import sys
from driftlock import MCPClient, MCPServerConfig, ToolCallingAgent

config = MCPServerConfig(
    name="project",
    command=(sys.executable, "/path/to/mcp_server.py"),
    allowed_tools=frozenset({"lookup"}),
)
async with MCPClient(config) as client:
    agent = ToolCallingAgent(
        environment,
        observer,
        async_completion_function,
        mcp_clients=(client,),
    )
    result = await runner.run(
        goal="Look up the project settings",
        step=agent,
        initial_state=agent.initial_state(),
    )
```

The host owns server lifecycle and authorizes native tool names through the required
allowlist; an empty allowlist exposes no tools. Names advertised to the model are
namespaced per server. Tools and schemas are snapshotted at connection time; create
a new client/agent to adopt a changed catalog. Up to 8 servers and 64 total external
tools can be attached to one agent. `MCPLimits` bounds requests, responses, discovery,
results and shutdown. MCP results use the existing per-step call and conversation
limits. Oversized results become explicit tool errors, not truncated successes.

Client failures and server `isError` results are auditable tool errors. Failed or
timed-out transport sessions are closed; potentially mutating calls are never retried
automatically. The parent model's token usage remains separate from server work.
External tool effects and connections are not checkpoint resources: restoring a
workspace or conversation does not undo an external action. Choose read-only tools
or tools with suitable idempotency when using rollback. Server text is untrusted
data, and configured server programs run with the host's local permissions. Explicit
environment overrides belong in `MCPServerConfig.env`; the client does not copy the
host's whole environment. With `mcp_clients=()` the legacy request and checkpoint
schema are unchanged. Delegated children do not inherit MCP capabilities implicitly.

### Remote and Harbor environments

`RemoteArchiveCheckpointStore` implements the same interface over the three methods
POSIX Harbor environments already expose: `exec`, `upload_file`, and
`download_file`. It requires Linux-style `sh`, `tar`, `find`, `rm`, `cp -a`,
`realpath`, `sha256sum`, `mkfifo`, and `tee`; Windows containers are not supported.
Archives and agent state are persisted on the host. Remote cleanup failures emit a
warning instead of being silently treated as success.

```python
from driftlock import RemoteArchiveCheckpointStore

store = RemoteArchiveCheckpointStore(
    harbor_environment,
    remote_workspace="/app",
    store_dir="./runs/checkpoints",  # keep outside agent-visible mounts
    user="root",
)
```

Restore validates canonical paths remotely, rejects staging directories that resolve
or mount inside the workspace, and downloads a pre-restore recovery archive to the
host—and verifies it against the remote SHA-256—before changing live files. It
preserves the workspace-root inode, but child directories are recreated: a Harbor
adapter must use `before_restore` to move tmux panes parked in a child directory back
to the workspace root before applying the snapshot. On an ordinary copy failure, an
exact pre-restore tree is rebuilt from the untouched remote backup (or a separately
named, checksum-verified host fallback). Recovery hashes and extracts the same
archive byte stream before mutating the live tree, so a changed archive is rejected.
Recovery archives are retained on failure, timeout, or cancellation; other staging
artifacts are cleaned after ordinary failures. The configured workspace cannot be
`/`.

### Terminus-2 checkpoint boundaries

> This adapter wraps Harbor's stock Terminus-2 agent in driftlock's checkpoint and
> rollback machinery. It is **the path the published results came from** — all 287
> archived job configs used `driftlock.harbor_agent:LHTBDriftlockAgent` — and it is
> retained so that run stays replayable. driftlock's own agent loop lives beside it;
> see *Native LHTB agent* below and [architecture](architecture.md).

`TerminusStepAdapter` connects the runner to a small, dependency-free runtime
protocol that yields after exactly one billed Terminus episode. Its versioned
codec checkpoints the message history, the terminal observation waiting to become
the next prompt, the two-step completion-confirmation flag, and the logical episode
number.

```python
from driftlock import TerminusStepAdapter

step = TerminusStepAdapter(checkpointable_terminus_runtime)
result = await runner.run(
    goal=instruction,
    plan="inspect, implement, verify",
    step=step,
    initial_state=step.initial_state(),
)
```

Harbor's stock `Terminus2.run()` owns the whole loop and resets per-run state, so it
must not be called once per driftlock step. The fork implements a two-phase
`prepare_start()` / `start()` plus `resume()`, and yields after every LLM response.
`prepare_start()` performs no model call: it resets semantic state, reads the initial
terminal screen, and returns the exact rendered Terminus user prompt. The adapter
passes that string unchanged to `start()` and verifies it is the first chat message,
so an unrelated or stale initial conversation cannot be checkpointed.
For a normal response, the boundary is after commands execute and the next terminal
observation is ready. A parser-error response is also a billed episode: it must yield
before Harbor's early `continue`, with the parser correction as `next_prompt` and the
parse failure in `TerminusBoundary.error`. The runtime can use
`Terminus2StateBridge` to capture and restore the existing `Chat` object. Restoring
clears the provider response-chain id so the next call sends the restored full
history.

The same rule applies below `Chat`: Harbor currently turns an output-length response
into an exception and recursively retries without adding its usage to `Chat`. The
fork must intercept that response, return it as an error boundary with its actual
token usage and shorter-response correction prompt, and let driftlock decide whether
to continue. Multiple provider responses may never be hidden inside one boundary.

Terminus must be constructed with context summarization disabled, and the fork must
disable `_query_llm`'s internal retry decorator. Summarization can make three
subagent calls before the main call; it also derives copied audit steps from a
trajectory prefix that no longer matches restored chat after rollback. The runtime
therefore exposes `summarization_enabled`, `internal_retries_enabled`, and a monotonic
`provider_call_count` incremented around the lowest-level provider request. The
adapter refuses either hidden-call feature and verifies that the physical counter
advances by exactly one on every driftlock step. It also verifies the captured chat
is the restored history as an exact prefix followed by the submitted user prompt and
one assistant response, preventing an early or wrong-branch capture from silently
discarding context.

The adapter also enforces Terminus's completion handshake: a boundary may report
`completed=True` only when the restored previous boundary was already awaiting
completion confirmation and the current boundary still carries that flag. A single
premature completion claim cannot end the driftlock run.

Only semantic state rewinds. Token/cost accumulators, rollout details, trajectory
files, session ids, and Harbor's physical turn counter remain monotonic so rolled-back
work is still billed and auditable. The adapter rejects runtimes that skip or combine
episode boundaries. A rollback reason is appended to the restored pending observation
without contaminating stored checkpoint state; when rollback reaches the initial
checkpoint, the same reason is passed explicitly to `prepare_start()`.

Filesystem rollback is not enough for Terminus's persistent tmux shell: rejected
branches can leave a different cwd, exported variables, aliases, foreground jobs, or
background servers behind. Pass the adapter hook to the remote store:

```python
store = RemoteArchiveCheckpointStore(
    harbor_environment,
    remote_workspace="/app",
    store_dir="./runs/checkpoints",
    before_restore=step.before_workspace_restore,
)
```

The runtime implementation must quiesce every process from the rejected branch,
replace the tmux shell, start the new shell at the canonical workspace root, and
reset incremental terminal-output tracking. If cleanup fails, it must raise; the
remote store then aborts before mutating the workspace.

The concrete `LHTBTerminusRuntime` targets LHTB commit
`0d9918f6b66eda0752f8c7d17c9a73a18ee32f98`. Its companion patch preserves the
otherwise discarded LiteLLM usage on output truncation, counts the lowest-level
provider attempt, disables both retry layers, and installs a revision marker that the
runtime checks before making a provider call. The runtime also verifies Harbor's
frozen LiteLLM version, reserves input tokens before capping output, and retains pane
and cast audit history across shell replacement. The patch also makes terminal
commands reach a shell completion marker before a workspace boundary is observed.
Installation and construction
instructions are in [`integrations/lhtb/README.md`](../integrations/lhtb/README.md).
`HarborWorkspaceDeltaObserver` hashes content and POSIX metadata for the full remote
workspace around each episode and records a before/after Git view, so the heuristics
receive metadata-only edits and changes made to files that were already dirty as well
as newly changed paths.

`RunnerConfig.max_tokens` is shared by agent and fine-judge calls. The step adapter
receives the total remaining budget in `context.tokens_remaining`. Before issuing a
provider call, it must reserve the request prefill plus a usable output allowance;
when that cannot fit, it must raise `StepTokenBudgetExhausted`. Otherwise it caps output
at the smaller of its configured maximum and the budget left after prefill, then reports
actual billed tokens in `StepOutcome.tokens`, including failed model calls.
Unexpected adapter exceptions propagate because treating them as zero-token agent
steps would corrupt compute-matched experiments. For fine judges, return
`JudgeCompletion(text=..., tokens=...)` from the completion callback to include judge
usage; returning a bare string is supported when usage is genuinely unavailable.

### One-command LHTB runs

Install driftlock into the pinned Harbor virtual environment and apply the companion
patch as described in
[`integrations/lhtb/README.md`](../integrations/lhtb/README.md). On a native amd64 host,
the experiment CLI checks the exact LHTB revision, patch, LiteLLM version, Docker
architecture, and credential presence before a paid request. It never accepts or
writes a credential value.

```bash
: "${OPENROUTER_API_KEY:?inject OPENROUTER_API_KEY with your secret manager}"
driftlock-lhtb run \
  --lhtb-dir /srv/LHTB \
  --arm driftlock \
  --job-name driftlock-smoke \
  --tasks 2048 chess-mate \
  --max-total-tokens 2000000
```

The `retry`, `driftlock-heuristic`, and `driftlock` arms share one total-token budget
across all Harbor `continue_until_timeout` phases. `retry` discards verifier text and
blindly restores the original workspace and fresh conversation after a binary
rejection. `driftlock-heuristic` is the zero-judge-token ablation; `driftlock` adds a
single-attempt DeepSeek V4-Flash fine judge and folds its input, cache, output, and
dollar usage into Harbor's trial accounting. The harness pins controlled arms to
Harbor's `same_conversation` mode and starts Harbor with the same Python environment
that passed preflight. It also rejects task-tree changes and any Harbor bytes beyond
the packaged patch. A stock Terminus run has no comparable total-token ceiling; the CLI
therefore requires an explicit `--ack-unbounded-stock-tokens` after a provider-side
spend cap is configured. After screening, `driftlock-lhtb select JOB_DIR` ranks tasks
by measured mean partial credit and records the trial result files behind the choice.

Completed arms can be aggregated into one strict, auditable report. By default the
analyzer requires identical task/attempt matrices, task checksums, and model identity;
it rejects missing rewards or usage instead of silently turning infrastructure errors
into model failures. It also requires each Harbor job summary to be finished and
error-free, and verifies each recorded task checksum against the selected LHTB
checkout. Arm labels are checked against the pinned agent configuration, controlled
arms must share one total-token budget, and every trial's job ID and name must match
its job summary and directory. Canonical namespaced Harbor task names are resolved
through each `task.toml`, while agent versions and all common non-treatment settings
(model API, temperatures, request limits, timeouts, environment, and verifier) are
validated and summarized by a configuration SHA-256. Harbor retries must be zero,
and official job-level token/cache/cost totals must reconcile with the trial files.
The canonical Harbor `lock.json` binds zero configured retries, concurrency, task
matrix, pinned Harbor revision, and a build fingerprint over all installed driftlock
Python sources plus the companion patch. Trial UUIDs must be globally unique. Every
input `result.json` path and SHA-256 is retained.

```bash
driftlock-lhtb analyze --lhtb-dir /srv/LHTB \
  --arm-dir stock=/srv/LHTB/jobs/stock \
  --arm-dir retry=/srv/LHTB/jobs/retry \
  --arm-dir driftlock-heuristic=/srv/LHTB/jobs/driftlock-heuristic \
  --arm-dir driftlock=/srv/LHTB/jobs/driftlock \
  --arm-dir oracle=/srv/LHTB/jobs/oracle \
  --output analysis.json
```

The report includes reward and solved-rate summaries, token/cache/cost accounting,
paired task deltas versus stock, and an ordinary least-squares failure-rate slope
against `log2(expert_time_estimate_min)`. Missing planned arms are explicit.

The planned hindsight oracle is not exposed as an online agent arm. A valid oracle
must replay retained candidate checkpoints in isolated copies against the hidden
verifier, then choose with hindsight; the CLI rejects attempts to label an ordinary
agent config as that upper bound.

Retained checkpoints can also be measured as a scored timeline without creating an
oracle analysis arm or calling a model:

```bash
driftlock-lhtb score-checkpoints \
  --lhtb-dir /srv/LHTB \
  --source-job-dir /srv/LHTB/jobs/round-five-driftlock \
  --output-dir /srv/LHTB/jobs/round-five-checkpoint-scores
```

Each checkpoint is restored into a fresh task environment and graded by the task's
ordinary hidden verifier. The resulting `checkpoint-scores.json` records phase,
step, checkpoint reward, the source trial's job-level final reward, and computable
best-versus-final headroom. It is written after every score; reruns reuse completed
entries. The replay agent reports zero provider tokens and needs no provider
credential. Use `--dry-run` to enumerate retained and missing checkpoint timelines
without Docker.

Development uses Python 3.11+ and `uv`:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

