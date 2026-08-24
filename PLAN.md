# driftlock — Project Plan

> A self-evolving long-horizon coding agent whose learning signal is its own rollbacks.
> This is the working plan and gets updated as the project moves. For the
> public-facing introduction, see `README.md`.

**Last updated:** 2026-08-20

### Implementation status

The project started as a checkpoint-and-rollback middleware wrapped around Harbor's
stock Terminus-2 agent. As of 2026-08-20 the scope changed: **driftlock is now the
agent itself**, and rollback events became the supervision signal for skill
distillation. The rewrite keeps roughly three quarters of the existing code.

- **Built and unit-tested (survives the rewrite, ~7,700 lines).** Provider-neutral
  async runner; atomic workspace + agent-state checkpoints with integrity
  verification; no-change, action-loop, error-spike, and reward-stall heuristics;
  structured LLM judge; shared agent/judge token accounting and step/rollback
  budgets; signal-window-aware rollback selection; sync/async checkpoint backends;
  host-side archive snapshots for POSIX remote environments with canonical-path
  validation, SHA-256-verified fallback, and exact failed-restore recovery;
  workspace snapshotting with xattr and content digests; tmux lifecycle cleanup;
  native-amd64/credential/Docker preflight; reproducible job generation; measured
  partial-credit task selector; strict multi-arm result ingestion with provenance,
  comparability checks, paired deltas, cost/token summaries, and failure-rate slope
  analysis; 136 passing unit tests.
- **Discarded by the rewrite (~2,750 lines).** `terminus.py` and its tests (Terminus-2
  semantic conversation codec), the `Terminus2` subclass and config plumbing inside
  `harbor_agent.py`, and `LHTBTerminusRuntime` plus the Harbor-internals coupling in
  `lhtb.py`. All of it exists only to drive someone else's agent loop.
- **Not yet built.** The agent loop, the subagent layer, and the entire skill layer.
- **Not yet measured.** No experiment number exists. The week-1 end-to-end gate has
  never been run — unit tests pass, but nothing has touched a real container, a real
  model, or a real task. Validating the surviving code is a job for that smoke run,
  not for code review.

---

## 1. The claims, in two sentences

**Drift.** Same model, same compute budget — add checkpointing and progress-aware
rollback and long-horizon terminal task success goes from X% to Y%, with a clear
account of where the ceiling is.

**Transfer.** Skills distilled from *rollback events* — a checkpoint-localized failure
region plus the judge's verdict on it — beat skills distilled from *whole failed
trajectories* by Z points on held-out tasks, closing W% of the gap between automatic
methods and human-curated skills that EvoAgentBench reports as the field's standing
deficit.

(X / Y / Z / W are placeholders. They stay unfilled until measured. This rule is not
decoration: an unfilled placeholder in a public repo is a stronger signal than a
number nobody can reproduce.)

---

## 2. Why this project

### 2.1 Choosing the track

An analysis of ~170,000 AI papers from H1 2026 found that **long-horizon planning is
the single fastest-growing topic**: mentions went from 264 to 1,611 (+510%). The
largest by volume is agentic workflows (4,585 → 10,496). The more useful finding:
**specialized agent components are growing two to five times faster** than the broader
agentic-workflows category — the field has moved past "can we build agents?" to "how
do we make agents plan, use tools, and judge their own output?"

### 2.2 Why goal drift is the right target

- **Industry treats it as a real pain point; the academic side is thin.** Search
  turns up mostly engineering blogs and playbooks rather than the arXiv density seen
  in the skill-synthesis space.
- **Published metrics already exist.** [Arike et al. (2025)](https://arxiv.org/abs/2505.02709)
  formalized `GD_actions` (drift through commission) and `GD_inaction` (drift through
  omission). They require aligned-action budget and residual-state annotations. LHTB
  does not provide those, so they remain unavailable for the generic benchmark
  analysis unless a task-specific annotation protocol is added. Inventing reward- or
  token-based proxies would not be credible.
- **There's a quantitative law to attack.** Doubling a task's duration roughly
  quadruples its failure rate. Flattening that curve even slightly is a clean result.
- **The capability gap is stark.** Frontier models solve close to 100% of tasks a
  human expert finishes in under four minutes, and under 10% of tasks taking a human
  more than four hours.

### 2.3 Why self-evolution, and why it isn't the crowded version

Generic skill synthesis got very crowded in H1 2026 — SkillFoundry, SkillOpt,
SkillComposer, SkillGen, SkillDAG, SkillX, SkillRL, EvoSkills, Trace2Skill,
SkillWeaver, AutoSkill, plus a dedicated benchmark for skill-retrieval ambiguity.
Walking that happy path invites one fatal question: *"how is this different from
SkillX?"*

The answer here is a specific, defensible one, and it comes from the field's own
recent results:

- [*Rethinking Self-Evolving Agent Skills*](https://arxiv.org/html/2608.02636) (2026-08)
  finds that skill evolution is **not** cumulative but is "validation-filtered
  search": only **55 of 388** candidate skills (14.2%) produced a distinct validation
  improvement. Crucially, **all 11 selected improvements included failed
  trajectories** — success-only feedback never produced a selected improvement.
- [EvoAgentBench](https://arxiv.org/html/2607.05202v1) finds automatic evolution
  methods brittle against human curation: Memento −2.4 to +1.5, ReasoningBank +0.4 to
  +3.6, GEPA +1.2 to +5.7, versus a curated Anchor Skill at **+7.5 to +10.5**.

So the field knows failure trajectories are the necessary ingredient, but feeds the
model the *whole* trajectory and lets it guess where things went wrong. **driftlock
already localizes the failure**: the checkpoint delta bounds it to the region between
checkpoint *k* and *k+n*, with a diff and a judge verdict attached. Nobody else has
that, because nobody else built the rollback infrastructure first.

That is the contribution: **rollback-grounded skill distillation.** It is not "another
skill synthesis method"; it is a claim about the *supervision signal*, and it is
testable against the exact baseline the field already uses.

### 2.4 Rollback is the missing fifth context-engineering lever

The 2026 consensus vocabulary for context engineering is four levers — **write,
select, compress, isolate**. The supporting numbers are strong: information that
scores 98.1 in a clean prompt drops to **64.1** when distributed across a multi-turn
agent run; context editing alone is worth **+29%**, and +39% combined with a memory
tool; a 100-turn eval cut token consumption by **84%**.

Every one of those four levers assumes the agent keeps moving forward. **None of them
is undo.** Rollback is the fifth lever, and it is the one this project owns.

### 2.5 The three recognized long-horizon failure modes

1. **Context rot** — as history grows, relevant information becomes harder to
   retrieve, and performance falls off sharply once context utilization crosses a
   critical threshold. This hits frontier and small models alike (Claude Sonnet 4,
   GPT-4.1, Qwen3-32B, and Gemini 2.5 Flash all degrade as input tokens grow).
2. **Compounding error and goal drift** ← **what this project attacks**
3. The interaction between the two

---

## 3. Technical approach

### 3.1 The agent

driftlock is a terminal coding agent, written here, running against LHTB and
SWE-bench Verified containers. It implements all five levers:

| Lever | Mechanism | Status |
| --- | --- | --- |
| **write** | Rollback-grounded skill distillation into a persistent library | To build |
| **select** | Embedding retrieval over `activation` conditions + a router that injects the top-k skills | To build |
| **compress** | Context editing / summarization at checkpoint boundaries | To build |
| **isolate** | Read-only subagents (read, grep, run tests; never write) returning 1–2k token condensed summaries | To build |
| **undo** | Checkpoint + progress-aware rollback | **Built** |

**Subagents are deliberately side-effect free.** They locate, read, and analyze;
they never touch the filesystem. This matches the production pattern behind the
+90.2% multi-agent research result, and it means rollback semantics need no change at
all: with no filesystem side effects, there is nothing in flight to undo. The
checkpoint layer is untouched by the subagent addition.

Writing the loop ourselves also **eliminates the original plan's second risk**. Harbor
and Terminus-2 expose no agent plugin interface, so the old design had to wedge a
rollback layer into someone else's loop. Owning both sides removes that problem
entirely, and it is what makes skill injection into prompt construction tractable.

The cost is losing direct comparability with the LHTB public leaderboard. This is
acceptable: every claim above is a *within-experiment* comparison between arms that
share an agent, and the LHTB tasks, hidden verifiers, and scoring stay intact.

### 3.2 The judge: two tiers

| Tier | What it does | Cost |
| --- | --- | --- |
| **Coarse — heuristics** | Action loops, error-rate spikes, sustained command failure, reward stalls | Zero tokens |
| **Fine — LLM** | Once the coarse tier fires, hand (original goal + current plan + recent trajectory + file diff) to a cheap model to judge semantic drift | DeepSeek V4-Pro 0813; see §5.2 |

A useful side effect: **heuristics-only / LLM-only / two-tier are three ready-made
ablation arms.** The experiment design falls out of the architecture.

**Not every signal may open a review.** A detector kind can be marked
*corroborating*: it is passed to the fine judge as evidence when something else
fires, but on its own it is recorded and dropped. `no_file_change` is
corroborating, and it is the only one — measured, not assumed. The 2026-08-23
diagnostic run (3 LHTB tasks, 462 steps, `driftlock` arm) raised it alone **109
times and the fine judge rejected all 109**, while both rollbacks that did happen
came from `action_loop` + `error_spike` + `no_file_change` firing together. 54% of
the solo firings landed in the first fifth of a phase, in runs of up to 28
consecutive steps — the detector was reporting exploration, which is reading
rather than writing, as drift.

Two consequences follow. The fine judge was **52% of that run's spend**
($0.63 of $1.23) and roughly 80% of its calls came from a signal that never
survived, so gating is a budget decision as much as a correctness one. And the
`driftlock-heuristic` arm, which has no fine judge to veto anything, previously
exhausted its rollback budget within 16–28 steps on 8/8 trials; gating is what
makes it a comparable arm rather than a degenerate one.

Suppressed triggers are still recorded, in their own `suppressed` bucket
alongside `upheld` and `vetoed`, so a gated detector's firing rate stays
measurable and the decision above stays re-checkable against later data.

### 3.2.1 What the boundary constraint costs

Checkpointing requires the shell to be quiescent at a step boundary, so the
driftlock arms append one instruction stock does not get: *each response's
terminal-command batch must return to the login shell before it ends*. That
constraint is load-bearing — a checkpoint taken mid-command archives a workspace
the agent is still writing to — but it is not free, and the first four-arm round
measured the price.

| Arm | Median output tokens | Responses cut off at the 8,192-token ceiling |
| --- | --- | --- |
| `stock` | 912 | **0 / 1,242** |
| `retry` | 1,533 | 53 / 1,052 (5.0%) |
| `driftlock` | 2,034 | 34 / 570 (6.0%) |

The gap is present in the first 25k tokens of context, before the trajectories
could have diverged, and truncation does not correlate with context length —
`retry`'s truncated calls had a *lower* median input than its untruncated ones.
The constraint changes how the model writes: it favours large self-contained
batches that set everything up and return, which are exactly the batches that run
out of output budget.

Two consequences.

**It was killing trials, and that was a defect.** A response cut off at the
ceiling ends mid-token, so its last keystrokes never execute. The companion patch
refused to run such a batch — correctly — but did so by raising, which Harbor
turns into an errored trial, and `lhtb_analysis` rejects any job containing one.
Three of eight `retry` trials and three of eight `driftlock` trials died this way
while `stock` lost none. From companion patch v11 the batch is refused and handed
back to the model as feedback, the same path a parse error already took.

**Part of any `stock` vs `driftlock` gap is prompt, not rollback.** This is why
the `retry` arm exists: it carries the identical constraint and the identical
loop, and differs from `driftlock` only in that it never rolls back. `retry` vs
`driftlock` isolates rollback; `stock` vs `driftlock` does not, and is reported
as loop-plus-constraint-plus-rollback rather than as a rollback effect.

### 3.2.1a Why the agent provider moved off `baidu/fp8`

`baidu/fp8` was the original pin: fp8 rather than fp4, a full 1M context, and the
cheapest fp8 endpoint at the time. It is no longer usable. On 2026-08-24 it
returned `tpm_rate_limit_exceeded` continuously from roughly 00:35 to at least
03:30 — a live one-token probe at 03:40 was still refused — and a four-arm round
launched into that window produced nothing at all across three hours and 32
trials.

The agent is now pinned to `deepinfra/fp8`: same fp8 quantisation, same 1M
context, cheaper ($0.08/$0.18 against Baidu's current $0.14/$0.28), and answering
when Baidu was not. The judge stays on `alibaba`, which answered throughout.

None of this is a claim that DeepInfra has more headroom — every routable
endpoint here is a shared pool and the API exposes no per-pool capacity. It is a
claim that the previous pin was measurably dead and this one was measurably
alive. What actually protects the round is the check below, not the choice of
slug.

**`run` now probes before it spends.** Every paid launch first asks each pinned
provider — the agent's, and the judge's for the arms that pay for one — for a
single token, and refuses to start if either does not answer. It costs a
fraction of a cent and it is the only paid call the launcher makes itself. Had it
existed, the 2026-08-24 round would have refused to start at 00:34 instead of
discovering the same fact three hours and four dead arms later. `preflight`
itself is unchanged and still spends nothing.

### 3.2.1b What the build fingerprint guards, and what it does not

Every trial records a hash of the driftlock source that produced it. The guard
that matters is **agreement**: two trials being compared must carry the same
hash, or the comparison is between different programs. That is now checked
across arms, which it previously was not — each lock was only matched against
whatever build happened to be installed at analysis time, so the cross-arm
question was never actually asked.

Matching the *installed* build was the wrong test, and it failed in the obvious
way. The hash covered the whole package including `lhtb_analysis.py`, which
never executes inside a trial, so fixing a reporting bug changed the hash and
locked the analyzer out of the very run that exposed the bug. The producing
build was the only build permitted to read a run, and that build could not read
it. Analysis was also non-portable: a second machine could never read the first
machine's results.

So: runs now record `lhtb_runtime_fingerprint()`, which covers everything that
can execute inside a trial and excludes `lhtb_analysis.py` alone. Agreement
across arms is a hard error. Whether the reader is the producing build is a
recorded fact — `analyzed_by_producing_build` in the report — not a refusal.

The exclusion is only sound while nothing on the runtime path depends on the
excluded module, so `task_directory_sha256` moved to `lhtb.py` (it verifies a
checkout during `oracle-prepare`, so it was never analysis), and a test walks
the package asserting no runtime module imports the analyzer.

**This change is also what made the round-4 data readable at all**, which is a
reason to be suspicious of it. The check it replaces is strictly weaker at
detecting mixed builds than the one now in place; what was given up is the
requirement that the reader be the writer, which protected nothing the
agreement check does not.

### 3.2.2 Why the arms now run concurrently

Round 1 and round 2 ran the four arms one after another, so `stock` executed
between 21:52 and 23:34 and the other three started only after it finished. That
is two problems, not one.

It is a **confound**: provider latency, queue depth and rate-limit headroom are
not constant across a twelve-hour window, and an arm that ran at a quiet hour is
not compute-matched to one that ran at a busy one in any sense the token counter
can see.

It is also **fragile**: when the shared pool saturated at 23:28 it caught three
arms that had not started, and all 24 of their trials died within 90 seconds of
each other. Sequencing turned one provider incident into a total loss instead of
a partial one.

The arms therefore run as four concurrent Harbor jobs, one trial in flight each,
rather than one job at a time with four trials in flight. Total concurrency and
total wall clock are unchanged; what changes is that the four arms meet the same
provider conditions at the same moment, and an incident degrades all of them
equally rather than annihilating the ones that had not started yet.

### 3.2.3 A fine judge that cannot answer is not a veto

The 2026-08-24 four-arm round exposed a second degeneracy: 33 of 34 escalated
triggers produced no parseable JSON. The pinned reasoning model had only 512
output tokens for both hidden reasoning and its answer, while the eight-step
judge trajectory reached 33,079 characters. Those failures became ordinary
`uncertain` verdicts and therefore looked like 33 considered decisions not to
roll back.

Judge transport/parse failures, preflight output-budget exhaustion, and real
model verdicts are now separate states in every trigger record and in
`signal_counts`. One failed call followed by an answer remains non-fatal. At four
or more escalations, a failure rate of at least 75% records judge reliability as
`failed`; this catches both observed broken rounds (117/137 and 33/34) while
allowing a single transient among four calls. If every call fails in a smaller
sample, reliability is `inconclusive` rather than guessed clean or dead. Judge
reliability is separate from the terminal run status, so a token-limited phase
records both `status=token_limit` and `judge_reliability=failed` instead of blaming
the judge for the budget stop. Trial metadata accumulates the evidence across
phases, and neither failed nor inconclusive reliability is admitted as a valid
measurement. The output cap is the pinned model's declared 8,192-token maximum.
The judge is a reasoning model, so its reasoning and compact JSON must share that
full allowance rather than the former 512-token slice.

### 3.3 Skill representation

One schema, shared by **both** distillation arms. This is a correctness requirement,
not a convenience: if the two arms used different formats, the comparison would be
confounded by format rather than by grounding. The arms differ in exactly one thing —
the evidence handed to the distiller.

Structured markdown following the [ProcMEM](https://arxiv.org/pdf/2606.23127) /
[Memento-Skills](https://arxiv.org/pdf/2603.18743) convention:

- `activation` — when this skill applies (also the retrieval index and router key)
- `execution` — what to do
- `termination` — when it is finished, and when it no longer applies

Content is **preventative** ("when X appears, do not do Y; do Z instead").
[ReasoningBank](https://research.google/blog/reasoningbank-enabling-agents-to-learn-from-experience/)
observed that curated checklists spontaneously evolve toward compositional,
preventative logic; rollback events produce that form natively, so we start where they
ended up.

Retrieval uses a local sentence-transformers model on the cloud box: **zero API cost**,
outside the token budget.

### 3.4 Skill admission: validation-filtered, never automatic

An unfiltered skill library **makes the agent worse**. This is measured, not
hypothetical: in the 2026-08 study, GPT-5.5 on LiveMath gained +5.7 validation points
and lost **−6.6** on the released test.

So every candidate skill must earn its place by measured improvement on a held-out
validation split before entering the library. This is budget-neutral — it only
requires splitting the training side three ways (see §4.3).

The field's own pass rate, **14.2%**, becomes a second independent piece of evidence:
if checkpoint localization genuinely makes distillation sharper, our candidate pass
rate should sit visibly above it.

### 3.5 The two questions this has to survive

> *"How is checkpoint-and-rollback different from just retrying on failure? Aren't you
> buying success rate with extra compute?"*

**The judge detects a broken trajectory before the task fails** — early stop plus
precise rollback to the last healthy point, not a blind restart. Which means the
judge's quality is the project's quality, and which is why the compute-matched control
arm is non-negotiable.

> *"How is rollback-grounded distillation different from just showing the model the
> failed trajectory?"*

**Localization.** The whole-trajectory arm exists precisely to answer this, and if it
wins, that gets published as the result.

---

## 4. Experiment design

Two benchmarks, because they measure different claims and only one of them can support
each.

### 4.1 LHTB — the drift claim

[repo](https://github.com/zli12321/LHTB) ·
[dataset](https://huggingface.co/datasets/IntelligenceLab/Long-Horizon-Terminal-Bench)

| Item | Value |
| --- | --- |
| Full suite | 46 tasks across 9 categories |
| Tokens per task | ~9.9M (231 steps average; 120–320 range) |
| Time per task | 69–93 min average, 90 min budget cap |
| Scoring | Continuous reward 0–1 with partial credit; ≥0.95 counts as solved |
| Verification | Hidden verifiers, deterministic replay, seeded environments |
| Best result to date | Grok 4.5, mean reward 0.505, 13 of 46 solved |
| Unsolved | 29/46; only 7% of 782 runs reached solve status |

**Use an 8-task subset**, selected by *measured* partial credit in week 1. The 29
tasks nobody has solved must be excluded — they offer no headroom, only a floor
effect.

**Why LHTB cannot carry the transfer claim.** Excluding the 29 unsolved tasks leaves
a usable pool of 17, spread across 9 categories — under 2 tasks per category. A
train/test split would give roughly 9/8. Detecting an effect the field measures at a
14.2% candidate pass rate, with n=8 held out, has essentially no statistical power.
LHTB is also the most expensive benchmark available (~9.9M tokens, ~90 min per task)
paired with the method that most needs repeated runs. Forcing it would produce a
number nobody should believe.

### 4.2 SWE-bench Verified — the transfer claim

Short, cheap, plentiful tasks; the agent benchmark with the widest recognition; and
the benchmark EvoAgentBench itself uses (87/56 split), so Memento / ReasoningBank /
GEPA / Anchor Skill numbers are directly citable as reference points.

### 4.3 Splits and arms

**LHTB — 4 arms × 8 tasks = 32 runs**

| Arm | Purpose |
| --- | --- |
| 1. No intervention | Reference point |
| 2. **Compute-matched retry** | Same token budget, spent on blind retries — **kills the "you just spent more compute" objection** |
| 3. **driftlock** (checkpoint + rollback) | The drift claim |
| 4. **Oracle upper bound** | A hindsight-perfect judge — shows how much headroom the real judge leaves on the table |

**SWE-bench Verified — 50 tasks, split three ways**

| Split | Tasks | Use |
| --- | --- | --- |
| Train | 20 | Run, collect rollback events, distill candidate skills |
| Validation | 10 | Measure each candidate; only measured improvements enter the library |
| Held-out test | 20 | Final four-arm comparison; never seen during evolution |

**SWE-bench — 4 arms**

| Arm | Purpose |
| --- | --- |
| 1. No skills | Reference point |
| 2. **Whole-trajectory distillation** | The field's standard grounding (ReasoningBank-style) — **the arm that decides whether localization matters** |
| 3. **Rollback-grounded distillation** | The transfer claim |
| 4. **Human-curated skills** | Upper bound, mirroring EvoAgentBench's Anchor Skill (+7.5 to +10.5) — turns "better than nothing" into "closes N% of the field's standing gap" |

Evolution: 2 distillation arms × 3 rounds × 30 runs (20 train + 10 validation) = 180
runs. Final eval: 4 arms × 20 held-out = 80 runs.

### 4.4 Metrics

- Success rate (LHTB continuous reward with partial credit; SWE-bench resolve rate)
- **Candidate skill pass rate**, against the 14.2% reference
- `GD_actions` / `GD_inaction` (Arike et al. definitions, only for tasks carrying the
  required aligned-action and residual-state annotations)
- Token cost per task
- **Slope of the task-length vs. failure-rate curve** — flattening this is the
  strongest single result available

---

## 5. Environment and cost

### 5.1 Compute host: a cheap x86 cloud box

LHTB images are essentially amd64-only — the docs instruct Apple Silicon users to set
`DOCKER_DEFAULT_PLATFORM=linux/amd64`, i.e. run under emulation. The local M4 Air
(24 GB, fanless) throttles hard under an overnight full-load run, and some images may
not run at all.

**Split:** write code and analyze data locally; run experiments on the cloud box.

### 5.2 Models

DeepSeek ended its 75% launch promotion on 2026-08-16, roughly doubling V4-Pro and
making it about ten times the price of V4-Flash. The allocation below follows from
that, and from a constraint the original plan got backwards.

| Role | Model and pinned provider | Price per 1M tokens |
| --- | --- | --- |
| Agent — high call count | DeepSeek **V4-Flash 0731**, `deepinfra/fp8` | $0.08 in / $0.18 out |
| Fine judge — fires only when the coarse tier trips | DeepSeek **V4-Pro 0813**, `alibaba` | $1.162 in / $0.1162 cache / $3.485 out |
| Skill retrieval | Local sentence-transformers | $0 |

**The expensive model belongs on the low-call-count side.** Cost is dominated by the
agent loop, so putting Flash there and Pro on the judge is cheaper than the reverse
*and* keeps the judge independent: a judge sharing the agent's weights shares its
blind spots, and §3.3 already commits to the judge's quality being the project's
quality.

**Cache-hit pricing is 5–10× cheaper than a miss**, which matters because an agent
loop re-sends a nearly identical history every step.

**Both identifiers carry a dated build.** An unversioned alias such as
`deepseek-v4-pro` follows whatever OpenRouter currently points it at, so arms run on
different days would silently use different models while the recorded model string
stayed identical.

**Provider routing is part of experiment identity.** Every paid request uses one
provider in `only` with `allow_fallbacks: false`. The canonical Harbor lock and trial
result must record the same agent provider, every arm must share that provider, and
fine-judge arms must share their separately pinned provider. Judge pricing is keyed
by provider so an unpriced provider cannot run with stale rates.

### 5.3 Budget (hard $100 constraint)

| Item | Runs | Cost |
| --- | --- | --- |
| LHTB, 4 arms × 8 tasks — agent | 32 | ~$7 |
| LHTB — fine judge | — | ~$8 |
| SWE-bench evolution, 2 arms × 3 rounds × 30 | 180 | ~$7 |
| SWE-bench held-out eval, 4 arms × 20 | 80 | ~$4 |
| **API total** | **292** | **~$26** |
| Cloud box, hourly, destroyed between sessions | ~250 h | ~$33 |
| **Total** | | **~$59** |

Assumes ~90% cache hits. The cloud box bills hourly and is billed for as long as it
*exists*, not while it runs, so the figure above depends on destroying it between
sessions rather than powering it off; leaving it up for the full ten weeks would cost
~$206 on its own.

--- | --- | --- |
| LHTB, 4 arms × 8 tasks | 32 | ~$21 |
| SWE-bench evolution, 2 arms × 3 rounds × 30 | 180 | ~$13 |
| SWE-bench held-out eval, 4 arms × 20 | 80 | ~$7 |
| **API total** | **292** | **~$41** |
| Cloud box, 2–3 months | — | $30–45 |
| **Total** | | **~$71–86** |

Leaves $14–29 for one iteration after a bad round. Assumes ~90% cache hit rate.

---

## 6. Phases and gates

Ten weeks from 2026-08-20, targeting a complete release at the end of October.

| Phase | Work | Gate (fail → change the plan) |
| --- | --- | --- |
| **Week 1** | Cloud box; end-to-end smoke on 2 stock LHTB tasks; screen ~20 tasks and pick the 8-task subset by **measured partial credit**. This validates the surviving code against reality for the first time. | V4-Pro scores near zero on everything → floor effect; switch to a stronger model or easier tasks |
| **Weeks 2–3** | Write the agent: tool-calling loop, context compression, read-only subagent layer. Wire it into Harbor as a plugin. Delete the ~2,750 lines of Terminus-2 coupling. | Our agent scores far below stock Terminus-2 on the screened subset → fix the loop before anything else |
| **Week 4** | Skill layer: ProcMEM schema, both distillation prompts, embedding retrieval + activation router, validation-set admission | — |
| **Week 5** | LHTB four-arm round | driftlock doesn't beat compute-matched retry → fix the judge immediately, don't keep running |
| **Week 6** | SWE-bench Verified environment; commit the 20/10/20 split | — |
| **Weeks 7–8** | Evolution rounds: 2 distillation arms × 3 rounds | Candidate pass rate not visibly above 14.2% → localization isn't helping; report that honestly rather than tuning until it does |
| **Week 9** | Held-out four-arm eval; hand-author the curated-skill arm; ablations (heuristics-only / LLM-only / two-tier) | — |
| **Week 10** | Failure-case analysis, packaging, blog post | — |

**Note on timing.** New-grad applications for 2027 starts run roughly August through
October, so this schedule finishes at the tail of that window. During September and
October the repo is public with the full design and the agent implementation in place;
the results section stays empty, per §1.

---

## 7. Risk register

| # | Risk | Impact | Mitigation |
| --- | --- | --- | --- |
| 1 | **Floor effect** — V4-Pro too weak, no headroom to improve | No result | Week-1 screening by measured partial credit; fall back to a stronger model |
| 2 | **Our agent underperforms stock Terminus-2** — writing the loop ourselves costs raw capability | Weak baseline, results uninteresting | Week-3 gate compares against the week-1 stock numbers on the same tasks |
| 3 | **Judge isn't accurate enough** — decisions no better than random, all curves overlap | No result | The oracle arm surfaces this early |
| 4 | **Skill library degrades performance** — the measured −6.6 failure mode | Transfer claim inverts | Validation-set admission (§3.4); the no-skills arm makes degradation visible |
| 5 | **Two benchmarks, two environments** — SWE-bench setup slips into the evolution weeks | 1–2 week slip | Week 6 is dedicated to it; LHTB results are already banked by then |
| 6 | **amd64 emulation** — images fail or run too slowly | Can't iterate | Already avoided via the x86 cloud box; verify in week 1 |
| 7 | **Wall-clock** — 292 runs, of which 32 are 90-minute LHTB tasks | Schedule pressure | Parallelize on the cloud box; box must be sized for concurrent containers |
| 8 | **Budget overrun** — ~$41 of API leaves room for roughly one redo | Can't iterate | Use V4-Flash during development; hold the 8-task LHTB subset |
| 9 | **Attribution against external baselines** — with subagents in the agent, comparisons to Memento / ReasoningBank / GEPA can't isolate which component won | Weaker external claim | Accepted. All four arms per benchmark share the same agent, so every *internal* comparison is clean; external numbers are cited as reference points, not as controlled comparisons |
| 10 | **`sustained_command_failure` misses an alternating wedged toolchain** — the detector requires all commands to fail on every step of an 8-step window, which is exactly what excludes ordinary red-green TDD; an agent whose toolchain is broken but which alternates edit-only steps with command steps evades it entirely (measured: fires on 8/8 all-fail, silent on alternating) | Missed intervention | Accepted for now. The miss is in the safe direction: a missed detection makes the driftlock arm behave like the no-intervention arm, so it can only understate the effect, never inflate it. Loosening the threshold enough to catch it also re-catches TDD, because every command-running step in a TDD loop is also all-fail. Retune from week-1 measurements rather than from speculation |
| 11 | **OpenRouter provider drift** — one model slug is served by numerically different provider builds | Arm comparison void | Mitigated: strict no-fallback routing on every paid call; provider recorded in lock/trials, checked within each arm and across arms; judge rates keyed to its pin. **Resilience is bought back inside the step, not by relaxing the pin.** The pinned provider is served from a *shared* upstream pool: on 2026-08-23 it returned `tpm_rate_limit_exceeded` continuously for at least 11 minutes and took 28 of that round's 32 trials with it, including three whole arms that never ran a single step. A 429 is refused before the model runs, so it generates and bills nothing — which is what lets a step retry the *same* provider without weakening the one-billable-call-per-step rule. Refused attempts are counted, excluded from that rule, and reported per phase as `rate_limited_calls`, so a run that only survived by hammering a saturated provider does not look clean. `max_retries` stays 0 at the Harbor level and the analyzer keeps rejecting trial-level retries, which would re-spend tokens the final `result.json` never records |
| 12 | **The corroborating gate is fitted on 3 tasks** — `no_file_change` was demoted on one diagnostic run: 109 solo firings, 109 rejected, 2 rollbacks total (§3.2). The tasks were chosen for their spread on the week-1 screen, not sampled, and a task where the agent genuinely stalls *without* looping would now be missed | Missed intervention; a knob fitted on the data it is judged by | The miss is in the safe direction — a gated detector makes the driftlock arm behave more like the no-intervention arm, which can only understate the effect. Suppressed triggers keep being recorded, so the held-out week-9 run re-measures the gate on tasks it was not fitted on: if solo `no_file_change` ever precedes a genuine stall there, the demotion is wrong and the record shows it |
| 13 | **Output-ceiling truncation** — the checkpoint boundary constraint roughly doubles median response length, so the driftlock arms hit the 8,192-token ceiling on 5–6% of calls where stock never does (§3.2.1) | Wasted steps; a confound in `stock` vs `driftlock` | A truncated batch is no longer fatal (patch v11) — it returns to the model as feedback and costs one step. The confound is structural, not fixable: `retry` is the arm that isolates rollback from the constraint |

---

## 8. Deliverables

1. **Open-source agent + library** — a terminal coding agent with checkpoint,
   rollback, and rollback-grounded skill distillation, pip-installable, documented,
   with a one-command reproduction script. The rollback layer stays usable
   standalone around someone else's loop.
2. **Technical blog post** — the drift curves, the transfer results, the candidate
   pass rate against 14.2%, failure-case analysis, and the design tradeoffs behind
   the judge.

---

## 9. Key references

**Benchmarks and harnesses**
- [LHTB — Long-Horizon Terminal-Bench](https://zli12321.github.io/LHTB/) ·
  [GitHub](https://github.com/zli12321/LHTB) ·
  [HF dataset](https://huggingface.co/datasets/IntelligenceLab/Long-Horizon-Terminal-Bench)
- [EvoAgentBench: Benchmarking Agent Self-Evolution via Ability Transfer](https://arxiv.org/html/2607.05202v1) ·
  [dataset](https://huggingface.co/datasets/EverMind-AI/EvoAgentBench)
- [AgentCE-Bench](https://arxiv.org/pdf/2604.06111) — fallback: tunable horizon, lightweight environments
- [LOCA-bench](https://arxiv.org/pdf/2602.07962) — controllable, extreme context growth

**Self-evolution and skills**
- [Rethinking Self-Evolving Agent Skills: Feedback Dynamics over Multiple Rounds](https://arxiv.org/html/2608.02636)
  (2026-08) — source of the 14.2% candidate pass rate and the validation-filtered-search framing
- [Managing Procedural Memory in LLM Agents (ProcMEM)](https://arxiv.org/pdf/2606.23127) —
  source of the activation / execution / termination schema
- [Memento-Skills: Let Agents Design Agents](https://arxiv.org/pdf/2603.18743)
- [ReasoningBank](https://research.google/blog/reasoningbank-enabling-agents-to-learn-from-experience/)
- [A Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046)

**Context engineering**
- [Context Engineering: A Practical Guide for AI Agents](https://sourcegraph.com/blog/context-engineering) —
  the write / select / compress / isolate framing
- [Context Engineering: Agent Reliability Playbook 2026](https://www.digitalapplied.com/blog/context-engineering-agent-reliability-playbook-2026)
- [Self-Compacting Language Model Agents](https://arxiv.org/pdf/2606.23525)

**Failure modes and metrics**
- [Arike et al., *Evaluating Goal Drift in Language Model Agents* (2025)](https://arxiv.org/abs/2505.02709) —
  source of the `GD_actions` / `GD_inaction` definitions
- [Beyond the Leaderboard: tool-use, planning, and reasoning failures](https://arxiv.org/pdf/2607.05775)
- [The Long-Horizon Task Mirage?](https://arxiv.org/html/2604.11978v1)
- [jhammant/agent-drift](https://github.com/jhammant/agent-drift)

**Engineering references**
- [Agent Rollback and Checkpoint Patterns](https://www.digitalapplied.com/blog/agent-rollback-checkpoint-patterns-2026-engineering-reference)
- [Agent drift: why long-running AI agents lose the plot](https://usewire.io/blog/agent-drift-why-long-running-ai-agents-lose-the-plot/)

**Pricing**
- [DeepSeek official pricing](https://deepseek.ai/pricing) · [summary](https://benchlm.ai/deepseek/api-pricing)
