# driftlock — Project Plan

> A checkpoint-and-rollback layer for long-horizon agents.
> This is the working plan and gets updated as the project moves. For the
> public-facing introduction, see `README.md`.

**Last updated:** 2026-08-09

### Implementation status

- **Implemented:** provider-neutral async runner; atomic local workspace + agent-state
  checkpoints with integrity verification; no-change, action-loop, error-spike, and
  reward-stall heuristics; optional structured LLM judge; shared agent/judge token
  accounting and step/rollback budgets; signal-window-aware rollback selection;
  sync/async checkpoint backends; host-side archive snapshots for Harbor-compatible
  POSIX remote environments with canonical-path validation and retained recovery
  archives; SHA-256-verified host fallback and exact failed-restore recovery; unit
  tests.
- **Next:** add the Terminus-2 conversation-state codec and checkpoint-boundary agent
  adapter, then run the week-1 LHTB smoke tests before selecting experiment tasks.

---

## 1. The claim, in one sentence

Same model, same compute budget — add a layer of checkpointing and progress-aware
rollback and long-horizon terminal task success goes from X% to Y%, with a clear
account of where the ceiling is.

(X / Y are placeholders. They stay unfilled until measured.)

---

## 2. Why this project

### 2.1 Choosing the track

An analysis of ~170,000 AI papers from H1 2026 found that **long-horizon planning is
the single fastest-growing topic**: mentions went from 264 to 1,611 (+510%). The
largest by volume is agentic workflows (4,585 → 10,496). The more useful finding:
**specialized agent components are growing two to five times faster** than the broader
agentic-workflows category — the field has moved past "can we build agents?" to "how
do we make agents plan, use tools, and judge their own output?"

Conclusion: **the component layer is what's valuable, not another general-purpose agent.**

### 2.2 Why not self-evolving agents / skill synthesis

That line got extremely crowded in H1 2026: SkillFoundry, SkillOpt, SkillComposer,
SkillGen, SkillDAG, SkillX, SkillRL, EvoSkills, Trace2Skill, SkillWeaver, AutoSkill,
and more — a dozen-plus papers in six months, plus a dedicated benchmark
(SkillResolve-Bench) for the second-order problem of skill-retrieval ambiguity.
Walking the happy path there invites one fatal question: *"how is this different from
SkillX?"*

### 2.3 Why goal drift is the right target

- **Industry treats it as a real pain point; the academic side is thin.** Search
  turns up mostly engineering blogs and playbooks (Wire, NxCode, Redis,
  digitalapplied's rollback-patterns reference) rather than the arXiv density seen in
  the skill-synthesis space.
- **Published metrics already exist.** *Asymmetric Goal Drift in Coding Agents*
  (ICLR 2026) formalized goal drift with `GD_actions` (drift through commission) and
  `GD_inaction` (drift through omission), with a companion stress-test repo,
  [jhammant/agent-drift](https://github.com/jhammant/agent-drift). **Adopting published
  metrics is what makes the numbers credible** — inventing our own would not.
- **There's a quantitative law to attack.** Doubling a task's duration roughly
  quadruples its failure rate. Flattening that curve even slightly is a clean result.
- **The capability gap is stark.** Frontier models solve close to 100% of tasks a
  human expert finishes in under four minutes, and under 10% of tasks taking a human
  more than four hours.

### 2.4 The three recognized long-horizon failure modes

1. **Context rot** — as history grows, relevant information becomes harder to
   retrieve, and performance falls off sharply once context utilization crosses a
   critical threshold. This hits frontier and small models alike (Claude Sonnet 4,
   GPT-4.1, Qwen3-32B, and Gemini 2.5 Flash all degrade as input tokens grow).
2. **Compounding error and goal drift** ← **what this project attacks**
3. The interaction between the two

---

## 3. Technical approach

### 3.1 Core mechanism: checkpoints + progress-aware rollback

Snapshot filesystem state (Docker layers + git) together with agent state. An
independent judge periodically asks whether the current state is still a sound basis
for continuing. If not, roll back to the most recent healthy checkpoint and retry
from there.

### 3.2 The judge: two tiers

| Tier | What it does | Cost |
| --- | --- | --- |
| **Coarse — heuristics** | No file changes for N steps, action loops, error-rate spikes, reward stalls | Zero tokens |
| **Fine — LLM** | Once the coarse tier fires, hand (original goal + current plan + recent trajectory + file diff) to a cheap model to judge semantic drift | DeepSeek V4-Flash; negligible |

A useful side effect of this structure: **heuristics-only / LLM-only / two-tier are
three ready-made ablation arms.** The experiment design falls out of the architecture.

### 3.3 The question this has to survive

> *"How is checkpoint-and-rollback different from just retrying on failure? Aren't you
> buying success rate with extra compute?"*

The answer has to be: **the judge detects a broken trajectory before the task fails** —
early stop plus precise rollback to the last healthy point, not a blind restart.
Which means **the judge's quality is the project's quality**, and which is why the
compute-matched control arm below is non-negotiable.

---

## 4. Experiment design

### 4.1 Benchmark

**A subset of LHTB (Long-Horizon Terminal-Bench)** —
[repo](https://github.com/zli12321/LHTB) ·
[dataset](https://huggingface.co/datasets/IntelligenceLab/Long-Horizon-Terminal-Bench).

Key facts driving cost and schedule estimates:

| Item | Value |
| --- | --- |
| Full suite | 46 tasks across 9 categories |
| Tokens per task | ~9.9M (231 steps average; 120–320 range) |
| Time per task | 69–93 min average, 90 min budget cap |
| Wall-clock for a full single-model run | 53–71 hours |
| Scoring | Continuous reward 0–1 with partial credit; ≥0.95 counts as solved |
| Verification | Hidden verifiers, deterministic replay, seeded environments |
| Best result to date | Grok 4.5, mean reward 0.505, 13 of 46 solved |
| Unsolved | 29/46; only 7% of 782 runs reached solve status |
| Cost reference | MiniMax M3 $6.13/task; Claude models $38–73/task |

**Use an 8–12 task subset**, selected by *measured* partial credit in week 1. The 29
tasks nobody has solved must be excluded — they offer no headroom, only a floor effect.
The official harness and hidden verifiers stay intact so numbers remain comparable to
the public leaderboard.

Task structure is standardized as five files:
`task.toml` / `instruction.md` / `environment/` / `tests/` / `solution/`.

### 4.2 Four arms

| Arm | Purpose |
| --- | --- |
| 1. No intervention | Reference point |
| 2. **Compute-matched retry** | Same token budget, spent on blind retries — **kills the "you just spent more compute" objection** |
| 3. **driftlock** (checkpoint + rollback) | The claim |
| 4. **Oracle upper bound** | A hindsight-perfect judge — shows how much headroom the real judge leaves on the table |

### 4.3 Metrics

- Success rate (LHTB continuous reward, partial credit included)
- `GD_actions` / `GD_inaction` (adopted from the ICLR 2026 definitions)
- Token cost per task
- **Slope of the task-length vs. failure-rate curve** — flattening this is the
  strongest result available

---

## 5. Environment and cost

### 5.1 Compute host: a cheap x86 cloud box

**Why:** LHTB images are essentially amd64-only — the docs instruct Apple Silicon
users to set `DOCKER_DEFAULT_PLATFORM=linux/amd64`, i.e. run under emulation. The
local M4 Air (24 GB, fanless) throttles hard under an overnight full-load run, and
some images may not run at all.

**Split:** write code and analyze data locally; run experiments on the cloud box.

### 5.2 Models

| Role | Model | Price per 1M tokens |
| --- | --- | --- |
| Agent | DeepSeek **V4-Pro** | $0.435 in / $0.003625 cache hit / $0.87 out |
| LLM judge | DeepSeek **V4-Flash** | $0.14 in / $0.0028 cache hit / $0.28 out |

**Cache-hit pricing is 50–120× cheaper, which is decisive here** — an agent loop
re-sends a nearly identical history every step, so the cache hit rate is naturally
very high.

### 5.3 Budget (hard $100 constraint)

Estimating 12 tasks × 4 arms at a ~90% cache hit rate:

- ~476M input tokens: 10% miss ≈ $20.7 + 90% hit ≈ $1.6
- ~12M output tokens ≈ $10.4
- **≈ $33 for one complete four-arm round**

Plus $30–45 for two to three months of the cloud box, **$100 covers 1–2 complete
rounds plus scattered development runs.**

**Mitigation: start with 8 tasks**, bringing the first round to just over $20 and
leaving room to revise the judge and run a second round. Expand to 12 once the
result is stable.

---

## 6. Phases and gates

| Phase | Work | Gate (fail → change the plan) |
| --- | --- | --- |
| **Week 1** | Stand up the cloud box, run 2 stock LHTB tasks end to end; screen ~20 tasks with V4-Pro and pick the subset by **measured partial credit** | If V4-Pro scores near zero on everything → floor effect; switch to a stronger model or easier tasks |
| **Weeks 2–3** | Fork Harbor, insert the checkpoint / snapshot / rollback skeleton; heuristics-only judge first | If the fork proves intractable → fall back to AgentCE-Bench |
| **Weeks 4–5** | Add the LLM judge tier; run the first complete four-arm round | If driftlock doesn't beat compute-matched retry → fix the judge immediately, don't keep running |
| **Weeks 6–7** | Ablations, failure-case analysis, oracle upper bound | — |
| **Week 8** | Package the library (pip-installable, one-command repro script) + technical blog post | — |

---

## 7. Risk register

| # | Risk | Impact | Mitigation |
| --- | --- | --- | --- |
| 1 | **Floor effect** — V4-Pro too weak, no headroom to improve | No result | Week-1 screening by measured partial credit; fall back to a stronger model |
| 2 | **Harness fork underestimated** — Harbor / Terminus-2 expose no agent plugin interface, so the rollback layer must be wedged into someone else's loop | 2+ week slip | Gate at end of week 2; fall back to AgentCE-Bench |
| 3 | **Judge isn't accurate enough** — decisions no better than random, all four curves overlap | No result | The oracle arm surfaces this early |
| 4 | **amd64 emulation** — images fail or run too slowly | Can't iterate | Already avoided via the x86 cloud box; verify in week 1 |
| 5 | **Wall-clock** — 12 tasks × 4 arms = 48 runs × 90 min cap | One experiment round per day | Parallelize on the cloud box; drop to 8 tasks if needed |
| 6 | **Budget overrun** — $33/round only covers 1–2 rounds | Can't iterate | Start at 8 tasks; use V4-Flash during development |

---

## 8. Deliverables

1. **Open-source library** — a rollback middle layer others can wrap around their own
   agent loop: pip-installable, documented, with a one-command reproduction script
2. **Technical blog post** — the four curves, failure-case analysis, and the design
   tradeoffs behind the judge

---

## 9. Key references

**Benchmarks and harnesses**
- [LHTB — Long-Horizon Terminal-Bench](https://zli12321.github.io/LHTB/) ·
  [GitHub](https://github.com/zli12321/LHTB) ·
  [HF dataset](https://huggingface.co/datasets/IntelligenceLab/Long-Horizon-Terminal-Bench)
- [AgentCE-Bench](https://arxiv.org/pdf/2604.06111) — fallback: tunable horizon, lightweight environments
- [LOCA-bench](https://arxiv.org/pdf/2602.07962) — controllable, extreme context growth
- [LongHorizon-Harness](https://arxiv.org/html/2608.01964)

**Failure modes and metrics**
- [Beyond the Leaderboard: tool-use, planning, and reasoning failures](https://arxiv.org/pdf/2607.05775)
- [The Long-Horizon Task Mirage?](https://arxiv.org/html/2604.11978v1)
- *Asymmetric Goal Drift in Coding Agents* (ICLR 2026) — source of `GD_actions` / `GD_inaction`
- [jhammant/agent-drift](https://github.com/jhammant/agent-drift) — drift stress-testing

**Engineering references**
- [Agent Rollback and Checkpoint Patterns](https://www.digitalapplied.com/blog/agent-rollback-checkpoint-patterns-2026-engineering-reference)
- [Agent drift: why long-running AI agents lose the plot](https://usewire.io/blog/agent-drift-why-long-running-ai-agents-lose-the-plot/)
- [Long-Horizon Agent Trajectory Governance Playbook](https://www.nxcode.io/resources/news/long-horizon-agent-trajectory-governance-playbook-2026)

**Adjacent work (worth keeping the boundary clear)**
- [Self-Compacting Language Model Agents](https://arxiv.org/pdf/2606.23525) — the context-compaction direction
- [A Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046)

**Pricing**
- [DeepSeek official pricing](https://deepseek.ai/pricing) · [summary](https://benchlm.ai/deepseek/api-pricing)
