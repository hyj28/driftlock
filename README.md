# driftlock

**A checkpoint-and-rollback layer for long-horizon agents.**

> 🚧 **Status: early. No results yet.** This repo currently contains the research plan
> and will accumulate the implementation, experiments, and write-up over the next ~8 weeks.
> Nothing here is a claim — the numbers below are placeholders until measured.

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

## Repo layout

```
PLAN.md     # full working plan, risk register, phase gates
README.md   # this file
```

## License

MIT
