# Code of conduct

This project adopts the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)
as its code of conduct. Harassment, personal attacks, and demeaning or exclusionary behaviour are
not welcome here, in issues, pull requests, reviews, or anywhere else this project is discussed.

To report a concern, contact the maintainer through GitHub at
[@hyj28](https://github.com/hyj28). Reports are handled privately. Security vulnerabilities go
through [SECURITY.md](SECURITY.md) instead.

## What this means for review here

driftlock's review is adversarial by design — [CONTRIBUTING.md](CONTRIBUTING.md) asks reviewers to
try to break a change, and the commit history is full of defects that survived one review and were
caught by the next. That is a standard applied to work, never to people, and the distinction is not
a formality:

- **Attack the claim, not the author.** "This status cannot distinguish an unobservable channel from
  an empty one, and here is the input that proves it" is the review this project wants. "This is
  sloppy" is not a finding.
- **A finding carries a failure scenario.** Inputs or state, and the wrong behaviour they produce.
  Without one, it is an opinion, and opinions do not block a change.
- **Being wrong is ordinary.** Every component here has a round where the fix was itself defective.
  Saying so plainly, and saying what you did not verify, is the expected behaviour — not something
  to be embarrassed about or to hold against anyone.
- **"It works" is not evidence, and neither is seniority.** Exit codes, counts, and verbatim output
  decide. Nobody's judgement overrides a measurement, including the maintainer's.

A person who cannot separate criticism of their code from criticism of themselves will have a bad
time here, and so will a person who uses a technical review as cover for making someone else feel
small. Both are in scope for this document.
