# Architecture

How driftlock fits together, and — more usefully — the single invariant each part exists to hold.

For what was measured see [RESULTS](../RESULTS.md); for how to run any of this see [usage](usage.md).

---

## 1. The control loop

`runner.py` drives one agent step at a time and owns three decisions: when to checkpoint, when to
ask the judge, and when to roll back.

```
          ┌─────────── step ───────────┐
goal ──►  │  agent produces a step     │  ──►  workspace delta + state
          └────────────┬───────────────┘
                       │  every N steps
                       ▼
            checkpoint (filesystem + conversation + component state)
                       │
                       ▼
        coarse judge (zero tokens: no-change, loops, error spikes, reward stalls)
                       │  triggered
                       ▼
        fine judge (cheap model: goal, plan, recent trajectory, diff)
                       │  drifted
                       ▼
        roll back to the last healthy checkpoint, with feedback
```

`checkpoints.py` defines the store interface and a local directory implementation; `remote.py`
implements the same interface for a workspace reachable only through `exec`, `upload_file` and
`download_file`. `heuristics.py` is the coarse tier, `judges.py` the fine one.

**The invariant:** a checkpoint is restorable only when everything it captured is accounted for. A
degraded observation — a file that vanished mid-hash, a `tar` warning, a truncated manifest — makes
a checkpoint non-restorable but never ends the run. Those two decisions used to be one, and merging
them cost five paid measurements; see [RESULTS §6](../RESULTS.md).

## 2. Self-evolution

```
scored timeline          flat segment            candidate skill         admitted skill
     │                        │                        │                      │
checkpoint_scoring ──► checkpoint_localization ──► skill_distillation ──► skill_validation
     │                                                                  ──► skill_admission
     └── the task's own hidden verifier, replayed on retained checkpoints, at zero token cost
```

- **`checkpoint_scoring.py`** replays retained checkpoints through fresh hidden-verifier jobs. The
  agent never sees the verifier; letting it would mean optimizing against the oracle.
- **`checkpoint_localization.py`** turns the score curve into segments. A flat segment is a stretch
  of steps that provably bought nothing, and it bounds the failure to a region with a diff attached.
- **`skill_distillation.py`** takes that localized evidence rather than a whole failed trajectory.
  Skills use the ProcMEM `activation` / `execution` / `termination` schema and carry preventative
  content — *when X appears, do not do Y; do Z instead*.
- **`skill_retrieval.py`** / **`skill_injection.py`** index activation conditions and inject at most
  once per task, with provenance.
- **`skill_validation.py`** runs ten paired replicates on the candidate's own source task, sharing
  one control across every candidate from that task.
- **`skill_admission.py`** applies the rule: ≥9 of 10 positive deltas, which bounds the all-null
  admission probability at 11/1024 ≈ 1.07% per candidate.

**The invariant:** when retrieval selects nothing, the treatment prompt is byte-identical to its
control. That is not a nicety — it is what makes those trials a *measured noise floor* rather than
wasted spend, and it is why the injection path returns the original object unchanged rather than a
copy.

## 3. The agent

`agent.py` is a provider-neutral, checkpointable tool-calling loop. Five tools are historical and
always present:

```
run_shell   read_file   write_file   search_files   complete
```

Everything else is opt-in, and that word carries weight: **an agent constructed without a component
offers exactly those five tools and sends a byte-identical request**, constructing the historical
request type itself rather than an equivalent one. The archived 170-trial run depends on this and
cost $15.06 on a server that no longer exists.

| Module | What it adds | The invariant it holds |
|---|---|---|
| `agentic_retrieval.py` | `retrieve_context` over skills, workspace and memory in one corpus | A relevance floor derived from vector dimension, not a fixed cosine cut. Memory loses an equal-similarity tie to workspace evidence *structurally*, not by prompt instruction |
| `agent.py` (compaction) | History rewriting at checkpoint boundaries | A rewrite is recorded; tool-call/result pairing survives it |
| `planning.py` | `manage_plan`, a durable plan that outlives compaction | The plan lives beside history, so it cannot be compacted away or starve recent turns |
| `memory.py` | `manage_memory`, unvalidated cross-task claims | Memory has no admission gate, so a retrieved memory is a hint about where to look and can never outrank a fresh observation |
| `delegation.py` | `delegate_task`, one fresh bounded sub-agent | Sequential, no recursion, checkpointed quota; an interrupted child reports its accounting as *unknown* rather than zero |
| `mcp.py` | External tools over stdio or Streamable HTTP | A credential arrives only from an injected supplier; sensitive headers are rebuilt per redirect hop, so cross-origin replay cannot happen |
| `prompt_cache.py` | Cache-prefix intent and effectiveness reporting | Mutable state renders *after* history, so a plan mutation cannot rewrite the cached prefix |
| `verification.py` | A completion claim opens a bounded evidence check | The model proposes a falsifiable command; it does not decide whether that command passed |
| `agent.py` (edit) | `edit_file`, exact-string replacement | Non-unique, absent, and no-op matches are all refused loudly; a concurrent writer can never produce a mixed file |

## 4. The discipline these share

Four rules recur, and each was learned by paying for its absence.

**A status must distinguish *could not observe* from *observed nothing*.** Absent cache telemetry is
not a cache miss. A provider that reports no cached tokens is not a provider reporting zero. A
completion that could not be checked is neither verified nor refuted. An interrupted child's call
count is unknown, not zero. Collapsing either pair makes a blind channel read as a negative result —
which is what `no_reward` did when it charged instrumentation failures to the agent under test.

**Bounded in every dimension, and hitting a cap is a recorded outcome.** Entry counts, character
budgets, store sizes, attempt limits, report sizes. A cap that silently truncates is worse than no
cap, because the truncation is invisible in the artifact you analyze later.

**A rule enforced by prompt text is not enforced.** Memory's subordination to observation is a sort
key and a capacity rule, not a sentence in the system prompt. The verification barrier is an exit
code, not the model's opinion of its own work. Where a guarantee cannot be made mechanical, the
docstring says so.

**Say what you do not guarantee.** `LocalEnvironment` is not a process sandbox. Self-verification is
not adversarially sound against a model that sets out to fake a verdict. `edit_file` breaks hard
links. The MCP client performs no interactive OAuth. Each of these is in the code, next to the thing
it qualifies.

## 5. Two agent paths

driftlock ships two Harbor entry points, and they are not interchangeable:

| | `harbor_agent.py` | `harbor_native_agent.py` |
|---|---|---|
| Class | `LHTBDriftlockAgent(Terminus2)` | `LHTBNativeDriftlockAgent(BaseAgent)` |
| Agent loop | Harbor's stock Terminus-2 | driftlock's own (`agent.py`) |
| Components | none — checkpoint and rollback only | all of them, each behind a `driftlock_*` flag |
| Published results | **all 287 archived job configs** | none yet |

The Terminus path is retained because the measured run came from it. The native path is where the
component work lives. Every enabled component appears in the run record's active set, so a trial is
always attributable to the configuration that produced it — and `lhtb_analysis.py` rejects a
configuration that changes mid-trial.

## 6. Module map

| | |
|---|---|
| **Loop** | `runner.py` `models.py` `checkpoints.py` `remote.py` `local.py` `heuristics.py` `judges.py` `terminal_quiescence.py` |
| **Self-evolution** | `checkpoint_scoring.py` `checkpoint_localization.py` `skill_distillation.py` `skill_distillation_driver.py` `skill_retrieval.py` `skill_injection.py` `skill_validation.py` `skill_admission.py` `oracle.py` |
| **Agent** | `agent.py` `agentic_retrieval.py` `planning.py` `memory.py` `delegation.py` `mcp.py` `prompt_cache.py` `verification.py` `st_embedder.py` |
| **Harbor / LHTB** | `harbor_agent.py` `harbor_native_agent.py` `terminus.py` `lhtb.py` `native_lhtb.py` `lhtb_experiment.py` `lhtb_analysis.py` `usage.py` |
