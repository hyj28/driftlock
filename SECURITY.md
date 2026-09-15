# Security policy

driftlock executes model-proposed commands, snapshots workspaces and agent state, connects to MCP
servers, and can receive host-supplied authorization. A quiet failure at any of those boundaries
can be more dangerous than an obvious crash.

## Reporting a vulnerability

Use [GitHub's private vulnerability reporting](https://github.com/hyj28/driftlock/security/advisories/new)
for command or path injection, workspace escape, unsafe checkpoint restoration, credential
disclosure, cross-origin authorization replay, or a result that silently turns an unavailable
observation into a successful one.

Include the affected driftlock version, platform, smallest safe reproduction, expected boundary,
and observed behavior. Remove credentials, provider prompts, proprietary task data, and archived
experiment records before sending the report.

You should receive an acknowledgement within seven days. Please allow time for a fix and a
coordinated release before publishing details.

## Supported versions

Security fixes target the latest tagged release and `main`. Older pre-1.0 releases are not
guaranteed to receive backports.

## Threat-model note

`LocalEnvironment` is not a hostile-process sandbox, self-verification is not adversarially sound,
and the MCP client does not perform interactive OAuth. These are documented product boundaries,
not vulnerabilities by themselves. A mismatch between code and a documented boundary is in scope.
