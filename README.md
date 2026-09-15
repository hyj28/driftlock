# driftlock

<p align="center">
  <img src="docs/assets/driftlock-hero.svg" alt="driftlock — Checkpoint, detect drift, roll back, learn" width="100%">
</p>

<p align="center">
  <a href="https://github.com/hyj28/driftlock/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/hyj28/driftlock/ci.yml?branch=main&amp;style=flat-square&amp;label=CI"></a>
  <a href="https://github.com/hyj28/driftlock/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/hyj28/driftlock?display_name=tag&amp;sort=semver&amp;style=flat-square&amp;color=10b981"></a>
  <a href="https://www.python.org/"><img alt="Python 3.13" src="https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white"></a>
  <img alt="No runtime dependencies" src="https://img.shields.io/badge/runtime%20dependencies-none-0f766e?style=flat-square">
  <a href="RESULTS.md"><img alt="170 archived trials" src="https://img.shields.io/badge/archived%20trials-170-a855f7?style=flat-square"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/github/license/hyj28/driftlock?style=flat-square&amp;color=22c55e"></a>
  <a href="https://github.com/hyj28/driftlock/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/hyj28/driftlock?style=flat-square&amp;logo=github&amp;color=f59e0b"></a>
</p>

<p align="center">
  <strong>A long-horizon coding agent that checkpoints its work, rolls back when it drifts,<br>and distils what it learns into skills that must earn their place.</strong>
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="RESULTS.md">Results</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/usage.md">Usage</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

| Checkpoint | Detect | Roll back | Learn |
|:--|:--|:--|:--|
| Snapshot filesystem and agent state together | Zero-token heuristics escalate to a trajectory-aware judge | Return to the last healthy state with bounded retries | Distil localized failures into skills, then admit only validated gains |

> [!IMPORTANT]
> The published 170-trial result measures **Terminus-2 wrapped in driftlock's checkpoint,
> rollback, distillation, and validation machinery**. driftlock's native tool-calling agent
> ships, but does not yet have a published measurement. The distinction is retained in every
> run record and explained in [RESULTS.md](RESULTS.md).

## Quick start

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required. The library itself is
stdlib-only; development tools are optional extras.

```bash
git clone https://github.com/hyj28/driftlock
cd driftlock
uv sync --extra dev
uv run pytest
```

Start with the [usage guide](docs/usage.md), inspect the measured evidence in
[RESULTS.md](RESULTS.md), or jump directly to the runner API below.

Agents fail differently on long tasks than on short ones. Frontier models solve near-100% of tasks a
human expert finishes in under four minutes and **under 10%** of tasks that take a human more than
four hours; a recurring rule of thumb is that doubling a task's length roughly quadruples its
failure rate.

Two failure modes dominate that regime. **Context rot** — relevant information gets harder to
retrieve as history grows. **Compounding error and goal drift** — small early mistakes snowball
until the agent is working on the wrong thing. driftlock attacks the second, and turns what it
learns there into skills it carries into the next task.

---

## Undo is the missing lever

The standard vocabulary for context engineering has four levers, and every one of them assumes the
agent keeps moving forward:

| Lever | Mechanism in driftlock |
|---|---|
| write | Rollback-grounded skill distillation into a persistent library |
| select | Agent-initiated retrieval over skills, workspace and memory in one corpus |
| compress | Context compaction at checkpoint boundaries |
| isolate | Fresh bounded sub-agents with their own conversation and no recursion |
| **undo** | **Checkpoint + progress-aware rollback** |

Snapshot the filesystem and agent state together, then let a two-tier judge periodically ask whether
the current state is still a sound basis for continuing. The coarse tier is zero-token heuristics —
no file changes for N steps, action loops, error spikes, reward stalls. The fine tier is a cheap
model reading goal, plan, recent trajectory and diff. If the answer is no, roll back to the last
healthy checkpoint and retry from there.

## Skills have to earn their place

Recent work found that self-evolving agents improve through *validation-filtered search*, not
accumulation: only **55 of 388** candidate skills produced a real gain, and **every** selected
improvement was grounded in a *failed* trajectory.

driftlock narrows the grounding further. Because every checkpoint can be scored by the task's own
verifier for free, a trajectory becomes a timeline, and a flat segment is a stretch of steps that
provably bought nothing. That localized evidence — a bounded region with a diff attached — becomes
the supervision signal for distillation, in place of a whole failed trajectory.

A candidate enters the library only after a **paired validation run**: ten replicates of its own
source task, with and without the skill, differenced per replicate, against a control shared by
every candidate from that task. When retrieval selects nothing, the treatment prompt is
byte-identical to its control — which yields a **measured noise floor at no extra cost**.

## What was measured, and what was not

A 170-trial validation run cost **$15.06** and is reported in full, including the parts that did not
work, in **[RESULTS.md](RESULTS.md)**. One candidate of fourteen was admitted against a chance
expectation of 0.150; the two distillation arms were indistinguishable at this sample size; and six
candidates were never retrieved at all, which is the finding that drove the agentic-RAG work.

**That run measured Terminus-2 wrapped in driftlock's checkpoint, rollback, distillation and
validation machinery.** All 287 archived job configs used
`driftlock.harbor_agent:LHTBDriftlockAgent`. driftlock's own tool-calling agent, and the components
built on it, have no published measurement yet. Both paths ship, and which one a run used is
recoverable from its record.

---

## What is built

| Component | Status |
|---|---|
| Checkpointing, progress-aware rollback, two-tier judge | done |
| Free checkpoint scoring via the task's own verifier | done |
| Failure localization to a checkpoint segment | done |
| Skill distillation, retrieval, once-per-task injection | done |
| Paired validation and admission with a measured noise floor | done |
| Resumable runs, bounded retries, degraded-observation reporting | done |
| Tool-calling agent (`run_shell`, `read_file`, `write_file`, `search_files`, `complete`) | done |
| Agentic RAG — retrieval as a tool, over code *and* skills | done |
| Context compaction | done |
| Planning / task decomposition | done |
| Persistent memory across tasks | done |
| Bounded sequential sub-agent delegation | done |
| MCP client — stdio and Streamable HTTP, host-supplied authorization | done |
| Bounded opt-in parallel workspace reads | done |
| Prompt-cache management | done |
| Output self-verification | done |
| Bounded exact-string file edit | done |

Everything past the tool-calling loop is **opt-in**. An agent built without them offers exactly the
five historical tools and sends a byte-identical request, which is what keeps the archived
experiment replayable. Each carries a `driftlock_*` flag into the experiment harness and appears in
the run record's active-component set, so a trial can always be attributed to the configuration that
produced it.

## Development setup

```bash
git clone https://github.com/hyj28/driftlock
cd driftlock
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

Python 3.13. **No runtime dependencies** — the library is stdlib-only. `pytest` and `ruff` are dev
extras; `sentence-transformers` is optional and needed only for the pinned-embedder integration
test.

## Runner API

```python
from driftlock.runner import DriftlockRunner, RunnerConfig
from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.heuristics import HeuristicJudge, HeuristicConfig

result = await DriftlockRunner(
    DirectoryCheckpointStore(workspace, store_dir),
    HeuristicJudge(HeuristicConfig()),
    config=RunnerConfig(max_steps=50, max_rollbacks=3, checkpoint_interval=5),
).run(
    goal="repair the parser",
    plan="inspect, patch, verify",
    step=agent,
    initial_state=agent.initial_state(),
)
```

Full usage — the native agent, every optional component, remote and Harbor environments, and the
one-command LHTB harness — is in **[docs/usage.md](docs/usage.md)**.

## Documentation

| | |
|---|---|
| **[RESULTS.md](RESULTS.md)** | The 170-trial run, its numbers, and its limits |
| **[docs/architecture.md](docs/architecture.md)** | How the pieces fit, and the invariant each one holds |
| **[docs/usage.md](docs/usage.md)** | Running the library, the agent, and the experiment harness |
| **[docs/design-journal.md](docs/design-journal.md)** | The dated working plan, kept as a record of how the design moved |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | The engineering discipline this repository holds itself to |
| **[CHANGELOG.md](CHANGELOG.md)** | What landed, in order |

## Repo layout

```
src/driftlock/    # the library: runner, agent, checkpoints, judges, skills, components
tests/            # no network: real subprocesses and real loopback servers
docs/             # architecture, usage, design journal
RESULTS.md        # the measured run
```

## What this repository optimizes for

Every component states what it does **not** guarantee as plainly as what it does, and that statement
lives in the code rather than only in the docs. Self-verification says it is not adversarially sound,
and why. The MCP client says a credential reaches it only from an injected supplier. The local
environment says it is not a process sandbox. The file edit says hard links break.

The same standard applies to measurement. A status that cannot distinguish *we could not observe
this* from *we observed nothing* is treated as a defect, because a blind channel reading as a
negative result has destroyed real measurements here — five separate times, catalogued in
[RESULTS.md §6](RESULTS.md).

## License

MIT — see [LICENSE](LICENSE).
