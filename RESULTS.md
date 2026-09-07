# Results

A 170-trial paired experiment on three LHTB tasks, run to completion for **$15.06**. Every
number below is regenerable offline from the archived run records; see *Reproducing* at the end.

**Summary.** Checkpoint-localized skill distillation shows **no measurable advantage** over the
whole-trajectory baseline, and the experiment as designed **cannot resolve** an effect of the
size either arm produces. The defensible contribution is not the method — it is the measurement
apparatus, which caught four distinct ways this result could have been overstated, including two
I had already written down and believed.

---

## 1. What was measured

14 skill candidates were distilled from failed agent trajectories on three tasks, in two arms:
`localized` (the checkpoint-localized method) and `baseline` (whole-trajectory). Each candidate
was validated on **its own source task** with 10 paired replicates: the same task run with the
skill available and without it, differenced per replicate.

| | |
|---|---|
| Trials | 170 planned, **170 measured**, 0 pending |
| Attempts | 287 (includes retries of infrastructure failures) |
| Cost | **$15.06** against a $25.67 plan |
| Model | `deepseek-v4-flash-0731` via OpenRouter, `deepinfra/fp8` |
| Judge | `deepseek-v4-pro-0813` via `alibaba` |

Validating each candidate on its own source task is a deliberate weakening. An earlier design
validated across tasks and was abandoned after measurement showed it would produce 140 zero
deltas: a skill distilled from task T is only retrieved on task T, so every cross-task
observation would have been byte-identical to its control. Own-task validation answers a
narrower question — *does this skill help on the task it came from?* — but it is a question the
data can actually address.

## 2. The headline

```
tested 14 complete candidates;  admitted 1;  rejected 13
retrieval split: never retrieved 6, retrieved and unhelpful 7, retrieved and admitted 1
pass rate 1/8 (12.5%) among retrieved candidates
field reference 55/388 (14.2%) under a different validation filter — not like-for-like
all-null chance expectation: at most 0.150 admissions across 14 tests
```

One admission. Under the stated null model, **0.150 admissions are expected by chance** across
this cohort. One observed against 0.150 expected is not distinguishable from luck, and the
admission rule is a directional screen, not a significance test.

The naive framing would be `1/14 = 7.1%` against the field's 14.2%. That denominator is wrong:
six candidates were never retrieved at all, so their deltas measure run-to-run noise rather than
skill quality. Against candidates that actually entered the test it is `1/8 = 12.5%`. Both
numbers are true; only the second is meaningful, and the report now prints only that one.

## 3. The core hypothesis: localized vs baseline

This is what the project exists to test. Restricted to candidates whose skill was actually
retrieved:

| arm | n | mean delta | sd | signs (+/0/−) |
|---|---|---|---|---|
| `baseline` | 50 | **+0.0285** | 0.0905 | 25 / 14 / 11 |
| `localized` | 30 | **+0.0246** | 0.1637 | 19 / 6 / 5 |

The localized arm does not beat the baseline. The gap (−0.004) is a fortieth of the localized
arm's own standard deviation.

But those two groups have different task mixes, and this report has already been burned once by
exactly that (§5). Within task:

| task | localized − baseline | n |
|---|---|---|
| `riscv-core-debug` | **−0.0388** | 20 vs 30 |
| `spice-ephemeris-regression` | **+0.0455** | 10 vs 20 |

Opposite signs, from one to three candidates per cell. The single admission came from the
**baseline** arm.

**Conclusion: no advantage demonstrated, and the design cannot resolve one at this scale.**

## 4. Retrieval fails on the task a skill was distilled from

Six of fourteen candidates were never retrieved on their own source task. Measured with the
pinned embedder (`all-MiniLM-L6-v2`, revision `c9745ed1…`), cosine similarity between each
skill's activation text and its own task instruction:

```
alp-paper-reproduction        0.049  0.103  0.184  0.231              0 of 4 retrieved
riscv-core-debug              0.256  0.422  0.425  0.491  0.500  0.541  4 of 6
spice-ephemeris-regression    0.265  0.356  0.603  0.613              2 of 4
                                            threshold 0.35
```

All four `alp` candidates fall below threshold, so `alp` contributes **no injected observations
at all** — a third of the split cannot inform the question.

The threshold was calibrated at 0.35 on provenance-labelled pairs, down from an initial 0.75
that retrieved nothing. Calibration did not fix the problem because the problem is not the
threshold: own-task similarities span **0.049 to 0.613**, and no single cut separates them.
A skill's activation text describes a *situation* ("when you are in a placeholder repository and
need to match externally specified output"), while a task instruction describes a *goal*. These
are different genres of text, and embedding similarity between them is weak even when the skill
is exactly the right one.

**This is a retrieval-calibration result, not a skill-quality result**, and the two must not be
reported as one number. Lowering the threshold to catch 0.049 would retrieve everything and
destroy specificity; the fix is a different retrieval key, not a different cut point. Not
attempted here.

## 5. The null channel, and what it caught

When retrieval selects nothing, the treatment prompt is byte-identical to its control — the
injection code returns the original object unchanged. Those deltas are therefore a **measured
noise floor**, obtained at no extra cost.

Pooled across the cohort, the comparison reads:

```
no skill injected (noise floor):  n=60  mean +0.089  sd 0.205
skill injected:                   n=80  mean +0.027  sd 0.122
```

Read alone, the injected group's 44-positive/16-negative split looks like something. Against a
noise floor three times larger, it does not. That is the mechanism working.

**But the pooled comparison is itself confounded, and I reported it before noticing.** `alp`
supplies 40 of the 60 noise-floor observations and zero injected ones, and `alp` is the
highest-mean, highest-variance task in the split. The pooled figure is a task-mix artifact.
Within task:

| task | no-skill | injected | difference |
|---|---|---|---|
| `alp-paper-reproduction` | n=40, +0.1300 | n=0 | **no contrast possible** |
| `riscv-core-debug` | n=10, +0.0622 | n=50, +0.0541 | **−0.0081** |
| `spice-ephemeris-regression` | n=10, −0.0455 | n=30, −0.0182 | **+0.0273** |

Opposite signs, both tiny, null channels of n=10.

The instability is the finding. Across three successive data cuts as the matrix filled in, the
`riscv` contrast read `+0.021`, then `−0.020`, then `−0.008` — **it changed sign twice**. A
quantity that flips sign as data arrives is not measuring an effect. Reporting either sign as a
result would have been an artifact of when I looked.

## 6. What the instrumentation cost

68 of the first 170 trials failed. Diagnosis found **five distinct sites** where driftlock's own
instrumentation destroyed a paid measurement, each surfaced by fixing the previous one:

| site | trigger | fixed in |
|---|---|---|
| checkpoint quiesce handshake | missed a 10-second deadline | #41 |
| `tar` archive | an unrecognised exit-1 warning | #42 |
| workspace hashing | `find`/sha256 walk hit a vanished file | #42 |
| manifest parsing | exit 0 with truncated stdout | #42 |
| remote path safety | `find` could not stat an entry | #42 |

All five are one sentence: **every point that observed a workspace a background process was
concurrently mutating treated a degraded observation as fatal to a paid run.** Agents start
builds and long test runs; those keep writing while we archive and hash.

The cost was concrete. One `riscv` control trial died four times — three times on quiesce, once
on tar — and because controls are shared, that single trial blocked **all six `riscv` candidates
from ever being decided**. Worse, the failures were recorded as `no_reward`, which reads as *the
agent failed to produce a reward*, charging our instrumentation's failure to the agent under
test.

The fix separates two decisions the code had merged: whether the run continues (always, for a
degraded observation) and whether a checkpoint may later be restored (not when anything is
unaccounted for). That trades lost measurement for possible mismeasurement, so the exposure is
reported — counts of uncheckpointable boundaries and non-restorable checkpoints reach the
validation summary, and analysis can exclude affected trials. In the final run both were zero.

## 7. Limits

- **n.** 10 replicates per candidate, 10 observations per within-task null channel. Sign flips
  across data cuts are direct evidence this is too few.
- **Three tasks**, one of which contributes no injected observations. Effectively two.
- **Own-task validation** answers a narrower question than transfer, which is the question a
  skill library ultimately has to answer.
- **The screen is directional.** Admission requires ≥9 of 10 positive deltas; it is not a
  significance test, and the multiplicity bound (0.150 expected admissions) is a cohort-level
  expectation, not a per-candidate p-value.
- **One model, one provider.** No claim generalises across model families.
- **Instrumentation changed mid-experiment.** The first 158 trials ran under Harbor patch v11,
  the remainder under v14. The change affects only what happens when an observation degrades —
  identical behaviour on trials that never hit it — but it is a difference, and it is recorded
  per trial.

## 8. Reproducing

Run records are archived outside this repo (18 MB, 1552 JSON files: per-trial Harbor results,
driftlock phase records, the 14 skill documents, all 287 job configs, and the pinned embedder).

```bash
driftlock-lhtb admit-skills <archive>/jobs-summaries/r5-validation.json \
  --library-dir <fresh dir> --output /tmp/admission.json
```

That regenerates the admission report, the retrieval split, and the per-task null channel
offline at no cost. The archive's own `README.md` documents its layout.

---

## What I would do differently

Fix retrieval before spending on validation. Six candidates never entered the test, and that was
knowable for free — the similarity measurement in §4 costs nothing and would have shown that a
third of the split could not contribute. Roughly $6 of the $15 bought observations that could
only ever measure noise.

And state a defect's *pattern* on first contact, not its instance. Each of the five
instrumentation sites was found by fixing the one before it. The sentence in §6 was writable
after the first, and would have turned five sequential rounds into one.
