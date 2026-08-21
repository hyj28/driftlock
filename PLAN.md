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
| **Coarse — heuristics** | No file changes for N steps, action loops, error-rate spikes, reward stalls | Zero tokens |
| **Fine — LLM** | Once the coarse tier fires, hand (original goal + current plan + recent trajectory + file diff) to a cheap model to judge semantic drift | DeepSeek V4-Flash; negligible |

A useful side effect: **heuristics-only / LLM-only / two-tier are three ready-made
ablation arms.** The experiment design falls out of the architecture.

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

| Role | Model | Price per 1M tokens |
| --- | --- | --- |
| Agent | DeepSeek **V4-Pro** | $0.435 in / $0.003625 cache hit / $0.87 out |
| LLM judge + skill distiller | DeepSeek **V4-Flash** | $0.14 in / $0.0028 cache hit / $0.28 out |
| Skill retrieval | Local sentence-transformers | $0 |

**Cache-hit pricing is 50–120× cheaper, which is decisive here** — an agent loop
re-sends a nearly identical history every step, so the cache hit rate is naturally
very high.

### 5.3 Budget (hard $100 constraint)

| Item | Runs | Cost |
| --- | --- | --- |
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
