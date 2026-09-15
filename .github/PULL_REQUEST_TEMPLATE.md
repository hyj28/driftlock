## What changed

<!-- Describe the failure mode or research need and the externally observable change. -->

## Invariant

<!-- What must remain true? What does this change explicitly not guarantee? -->

## Evidence

- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run pytest` with the test-count summary retained
- [ ] New behavior has a falsifiable test at its public seam
- [ ] The test diff does not delete, loosen, skip, or mock away existing coverage
- [ ] A runtime scenario was exercised, or its unavailability is stated precisely
- [ ] Optional components remain byte-identical when disabled and appear in run provenance when enabled

<!-- Add exact commands and concise verbatim output. Do not include credentials or private task data. -->
