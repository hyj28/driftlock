# driftlock

**A checkpoint-and-rollback layer for long-horizon agents.**

> 🚧 **Status: early. No benchmark results yet.** The first library slice implements
> the checkpoint/rollback control loop and local filesystem snapshots. The LHTB
> integration and experiments are still in progress. Nothing here is a result claim.

---

## The problem

Agents fail differently on long tasks than on short ones. Frontier models solve
near-100% of tasks a human expert finishes in under four minutes, and **under 10%**
of tasks that take a human more than four hours. A recurring rule of thumb across
recent work: **double a task's length and its failure rate roughly quadruples.**

Two failure modes dominate the long-horizon regime:

1. **Context rot** — as history grows, relevant information gets harder to retrieve,
   and performance falls off a cliff past a critical context-utilization threshold.
   This affects frontier and small models alike.
2. **Compounding error and goal drift** — small early mistakes snowball along the
   trajectory, gradually steering the agent away from what it was asked to do.

`driftlock` attacks **the second one**.

## The approach

Snapshot the filesystem and agent state together, then let an independent judge
periodically ask: *is the current state still a sound basis for continuing?*
If not, roll back to the last healthy checkpoint and retry from there.

The judge is two-tier by design:

| Tier | Mechanism | Cost |
| --- | --- | --- |
| **Coarse** | Heuristics — no file changes for N steps, action loops, error-rate spikes, reward stalls | Zero tokens |
| **Fine** | An LLM reads (original goal + current plan + recent trajectory + file diff) and judges semantic drift | Cheap model, negligible |

The coarse tier keeps the cost near zero; the fine tier catches what rules can't
express — *"the agent is now working on the wrong thing."*

## The question this has to answer

> *"How is checkpoint-and-rollback different from just retrying on failure?
> Aren't you buying success rate with extra compute?"*

The claim only holds if the judge detects a broken trajectory **before** the task
fails — early stop plus precise rollback, not blind restart. So the experiment
includes a **compute-matched retry** arm that gets the same token budget to retry
blindly. If `driftlock` can't beat that, the idea doesn't work, and the writeup
will say so.

## Experiment design

Four arms, evaluated on a subset of
**[LHTB (Long-Horizon Terminal-Bench)](https://github.com/zli12321/LHTB)** — 46
reproducible terminal tasks designed to resist memorization, shortcutting, and
reward hacking. The official harness and hidden verifiers are kept intact so
numbers stay comparable to the public leaderboard.

| Arm | Purpose |
| --- | --- |
| No intervention | Baseline |
| **Compute-matched retry** | Rules out "you just spent more compute" |
| **driftlock** | The claim |
| **Oracle upper bound** | A hindsight-perfect judge — shows how much headroom the real judge leaves |

**Metrics**

- Success rate (LHTB continuous reward, partial credit included)
- `GD_actions` / `GD_inaction` — goal-drift metrics adopted from
  *Asymmetric Goal Drift in Coding Agents* (ICLR 2026) rather than invented here
- Token cost per task
- **Slope of the task-length vs. failure-rate curve** — flattening this is the
  strongest result available

## Why LHTB is hard (and why a subset)

Of 46 tasks, **29 remain unsolved by every model evaluated**, and only 7% of 782
recorded runs reached the solve threshold. The best score to date is a mean reward
of 0.505. Tasks average 69–93 minutes and roughly 231 agent steps.

Running the full suite costs 53–71 hours of wall-clock per model, so this project
uses a **screened 8–12 task subset**, chosen by *measured* partial credit — tasks
nobody can solve provide no headroom to measure against.

## Planned deliverables

1. This library — a rollback middle layer you can wrap around your own agent loop
2. A technical writeup: the four curves, failure-case analysis, and the judge design tradeoffs

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
    # Cap the provider request at context.tokens_remaining when it is not None.
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
instructions are in [`integrations/lhtb/README.md`](integrations/lhtb/README.md).
`HarborWorkspaceDeltaObserver` hashes content and POSIX metadata for the full remote
workspace around each episode and records a before/after Git view, so the heuristics
receive metadata-only edits and changes made to files that were already dirty as well
as newly changed paths.

`RunnerConfig.max_tokens` is shared by agent and fine-judge calls. The step adapter
receives `context.tokens_remaining` and must use it to cap the provider request, then
report actual billed tokens in `StepOutcome.tokens`, including failed model calls.
Unexpected adapter exceptions propagate because treating them as zero-token agent
steps would corrupt compute-matched experiments. For fine judges, return
`JudgeCompletion(text=..., tokens=...)` from the completion callback to include judge
usage; returning a bare string is supported when usage is genuinely unavailable.

Development uses Python 3.11+ and `uv`:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Repo layout

```
src/driftlock/   # local/remote checkpoint stores, judges, heuristics, runner
tests/           # unit and integration-style local tests
PLAN.md          # full working plan, risk register, phase gates
README.md        # this file
```

## License

MIT
