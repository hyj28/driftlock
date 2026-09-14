# Contributing

This repository holds itself to a specific discipline. It is written down because it is the reason
the code looks the way it does, and because every rule below was learned by paying for its absence.

## The gates

Every change passes four, and exit codes decide — not an assertion that tests pass.

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

1. **Exit code.** Any gate returning non-zero is a failure.
2. **Test count.** Below baseline, zero, or a suite that collected nothing all fail. A runner that
   matches no files still exits 0.
3. **Test integrity.** The diff must not weaken the suite — see below.
4. **Run it.** A green suite proves you did not break what worked, not that the new behaviour is
   exercised. Something starts the program, or the reason it cannot is recorded.

## What counts as weakening the suite

Each of these is a red gate, not a style note:

- a deleted test, or a deleted or loosened assertion in one that remains
- an expected value recomputed the way the code computes it — `assert add(a, b) == a + b` passes by
  construction and can never disagree
- a newly mocked internal collaborator standing between the test and the thing under test
- `skip` / `xfail` / commented-out cases added to the suite
- new behaviour with no new test

**A guard that no test can falsify is not covered.** Revert it alone and confirm its test fails.
This caught a real gap here: five guards in the file-edit path all survived deletion with the suite
green, because a single concurrency test proved only that *one of four redundant guards* fired and
each covered for the others.

> When mutating a copy of the tree to check falsifiability, force `PYTHONPATH` — the venv installs
> `driftlock` editable, so a scratch copy is shadowed and the mutation silently no-ops. A batch that
> does nothing looks exactly like a batch that found nothing.

## What a new component must prove

Every optional component follows the same contract, and the tests assert each point:

1. **Byte-identical when absent.** An agent built without it offers exactly the five historical
   tools and sends a byte-identical request, constructing the historical request type itself.
   Assert against a literal, never a recomputation. The archived 170-trial run depends on this and
   cannot be re-bought.
2. **Bounded in every dimension**, with hitting a cap recorded rather than silently truncating.
3. **Three-valued where two will not do.** *Could not observe* is a different fact from *observed
   nothing*, and collapsing them is treated as a defect.
4. **Mechanism, not prompt text.** A rule the system relies on is enforced structurally. Where it
   cannot be, the docstring says so.
5. **Survives checkpoint and rollback**, with tool-call/result pairing intact.
6. **Reachable from the experiment harness** behind a `driftlock_*` flag, defaulting off, and
   present in the run record's active-component set. A component the harness cannot reach is a
   component nothing can attribute a result to.
7. **States what it does not guarantee**, in the code, next to the thing it qualifies.

## Commits

Messages say what changed and why, in prose. A reader should be able to reconstruct the defect from
the message without the diff — several of the commit messages here are the only place a subtle
failure mode is written down.

Commit at coherent checkpoints: one for an implementation once its gates are green, one per review
round. Never commit while a gate is red. Never stage build artifacts.

## How changes are reviewed here

Work lands through a deliberately adversarial loop: an implementation, then a runtime driver that
**is not shown the diff** and reports evidence rather than a verdict, then a fresh reviewer that
reads the code. The split matters — each finds what the other structurally cannot.

The driver walks scenarios it can imagine and proves behaviour with verbatim output. The reviewer
finds the place where a guard was *disabled by construction* — such as a containment check made
inert because one parameter answered two different questions, which no amount of scenario driving
would have reached.

Two rules keep this honest:

- **Report evidence, not verdicts.** "Ran it, works fine" is refused. One row per scenario: the
  exact input, the observed output verbatim, the expected result, and match or mismatch. A scenario
  that could not be run is *unavailable*, never *passing*.
- **Prove the instrument can move before trusting that it did not.** A leak check that shows "no
  change after" proves nothing unless it also shows the numbers moving *during*. A measurement that
  cannot fail is not a measurement.

## Environment

Python 3.13, `uv`, project-local only.

```bash
uv venv && uv pip install -e ".[dev]"
```

The library has **no runtime dependencies** and new ones are not accepted lightly. The test suite
reaches no external host: it starts real subprocesses and real loopback HTTP servers instead of
mocking them.
