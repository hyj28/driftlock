# Changelog

All notable changes to driftlock. The format follows [Keep a Changelog](https://keepachangelog.com);
this project has not cut a tagged release yet, so everything below is the road to `0.1.0`.

Dates are the merge dates of the pull requests listed; numbers in brackets are PR numbers.

## [0.1.0] — first release

**2026-09-13 → 2026-09-14.** Documentation, CI, and the platform gap CI immediately found.

### Added

- Documentation for a project someone else could pick up: a README that orients rather than
  argues, `docs/architecture.md` stating the one invariant each component holds, `docs/usage.md`,
  `CONTRIBUTING.md`, this file, and a CI workflow running the gates on Linux and macOS.

### Fixed

- **The suite had never run on Linux.** Its first CI run reported 22 failures against a green
  macOS baseline. The process-identity snapshot emitted PID 1 while the validator rejected it — two
  halves of one contract disagreeing — and nobody had noticed because macOS has no `/proc`, so an
  unobservable process table produced exactly what an observed-empty one would. An absent table is
  now its own status, a kernel where it is structurally unavailable is distinguished from a reading
  that failed, and the exposure reaches the validation summary.
- GNU tar and BSD tar disagree about backslashes in paths; the implementation is detected rather
  than assumed.
- CI passed `-q` on top of the `-q` already in `pyproject.toml`, making it `-qq` and suppressing the
  summary line — a log that cannot show a test count cannot distinguish a green suite from one that
  collected nothing.

### Corrected

- `RESULTS.md` never named the agent that produced its numbers. All 287 archived job configs used
  the Terminus-2 path, so the report is about the checkpoint, scoring, distillation and validation
  machinery — not driftlock's own tool-calling agent, which has no published measurement.
- The Terminus-2 adapter was documented as legacy awaiting deletion. It is the path the published
  results came from.

---

## [Unreleased] — the agent and its components

**2026-09-05 → 2026-09-13.** driftlock stops being middleware around someone else's agent and
becomes one. Every component here is opt-in: an agent built without it offers exactly the five
historical tools and sends a byte-identical request, which is what keeps the measured run
replayable.

### Added

- **Agentic RAG** [#45] — `retrieve_context` as a tool the agent invokes from live context, over
  skills, workspace and memory in one corpus, behind a relevance floor derived from vector dimension
  rather than a fixed cosine cut.
- **Context compaction** [#46] — bounded history rewriting at checkpoint boundaries, with the
  rewrite recorded and tool-call pairing preserved.
- **Planning** [#47] — `manage_plan`, a durable plan that lives beside history so compaction cannot
  remove it, and that survives rollback.
- **Persistent memory, delegation, MCP over stdio, parallel workspace reads** [#48] — memory that
  cannot outrank a fresh observation; one fresh bounded sub-agent with no recursion; external tools
  behind an explicit allowlist; opt-in parallel reads for contiguous read-only calls.
- **Prompt-cache management** [#49] — a provider-neutral cache breakpoint, mutable state rendered
  after history so a plan mutation cannot rewrite the cached prefix, and four-valued telemetry in
  which a provider reporting nothing is distinct from one reporting zero.
- **Output self-verification** [#50] — a completion claim opens a bounded, isolated evidence check.
  The model proposes a falsifiable command; the host decides from exit codes, and a command that
  would pass equally well before the work was done is rejected as non-discriminating.
- **MCP Streamable HTTP with host-supplied authorization** [#51] — a token reaches the client only
  from an injected supplier, sensitive headers are rebuilt per redirect hop, and a `401` yields a
  typed result naming where to authorize.
- **Component composition into the experiment harness** [#52] — every component reachable behind a
  `driftlock_*` flag, defaulting off, recorded in the run record's active set.
- **Bounded exact-string file edit** [#53] — `edit_file`, with non-unique, absent and no-op matches
  all refused loudly, and an atomic exchange that verifies the bytes it displaced.

### Fixed

- Verification-budget exhaustion and delegation timeouts no longer abort a trial through the
  provider-call reconciler [#52]. These would have been the sixth and seventh instances of the
  failure catalogued in [RESULTS §6](RESULTS.md).
- The credential no longer escapes through the metadata-discovery redirect chain [#51] — one
  parameter was answering both *may I transmit here* and *does this URL contain the credential*, so
  discovery switched off the containment check by switching off transmission.

---

## [0.0.3] — self-evolution closes the loop

**2026-08-22 → 2026-09-07.** Scored timelines, distillation from localized evidence, paired
validation, and the 170-trial run that measured all of it.

### Added

- **Checkpoint-scored timelines** [#22] and **stalled-segment detection** [#24] — the task's own
  hidden verifier, replayed on retained checkpoints at zero token cost, turning a trajectory into a
  score curve and a flat segment into bounded evidence.
- **Skill schema, evidence bounds, admission, retrieval, injection** [#25–#30] — the ProcMEM
  activation/execution/termination schema, a persistent library, deterministic retrieval, and
  once-per-task injection with provenance.
- **Paired validation** [#33, #34, #36, #38–#40] — ten replicates on a candidate's own source task
  against a control shared by every candidate from that task, run concurrently and resumably.
- **Threshold calibration** [#35] and **per-task null channel** [#43] — the measured noise floor,
  and the per-task decomposition that showed the pooled figure was a task-mix artifact.
- **[RESULTS.md](RESULTS.md)** [#44] — the full report, including the parts that did not work.

### Fixed

Five sites where driftlock's own instrumentation destroyed a paid measurement and recorded it as
`no_reward` — *the agent failed to produce a reward* — while the real cause was our own observation
of a workspace a background process was still writing to:

- the checkpoint quiesce handshake missing a ten-second deadline [#41]
- a `tar` archive treating an unrecognised exit-1 warning as fatal [#42]
- workspace hashing hitting a file that vanished mid-walk [#42]
- manifest parsing accepting exit 0 with truncated stdout [#42]
- remote path safety failing when `find` could not stat an entry [#42]

The fix separates two decisions the code had merged: whether the run continues (always, for a
degraded observation) and whether a checkpoint may later be restored (not when anything is
unaccounted for).

---

## [0.0.2] — making a long experiment survivable

**2026-08-14 → 2026-08-21.** The harness stops losing paid work.

### Fixed

- A truncated provider batch no longer kills a trial [#9].
- Shared-pool rate limits are survived rather than fatal [#11].
- An absent cache field is no longer read as zero [#14] — the first instance of the rule that
  *could not observe* and *observed nothing* are different facts.
- A build fingerprint guards agreement between config and run [#15].
- A fine-judge failure is visible instead of silently degrading the arm [#17].
- A dead trial is charged to its own task rather than the pool [#18].
- Usage accounting reads real inputs instead of reconstructing them [#21].

---

## [0.0.1] — checkpoint and rollback

**2026-08-09 → 2026-08-14.** The original thesis: undo as a context-engineering lever.

### Added

- The progress-aware checkpoint and rollback control loop, with a local directory store [#1].
- A Harbor environment adapter and remote checkpoint store [#2].
- Terminus-2 checkpoint-boundary adapters [#3, #4].
- The pinned LHTB experiment harness, its arms, and dependency-free result analysis [#5–#7].
- Exposed detector thresholds [#8].
