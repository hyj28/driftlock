# Pinned LHTB Harbor integration

`driftlock` targets the LHTB repository at commit
`0d9918f6b66eda0752f8c7d17c9a73a18ee32f98`. The small companion patch does two
things the boundary runtime cannot safely infer from stock Harbor:

1. it adds a revision marker checked when `LHTBTerminusRuntime` starts;
2. it retains the full `LLMResponse`, including provider usage, when LiteLLM reports
   an output-length truncation; and
3. it counts calls immediately around `acompletion` / `aresponses` and disables
   LiteLLM's parameter-fallback retry while driftlock owns a one-response boundary.

The patch also factors Harbor's normal chat accounting into `Chat.record_response()`
so a billed truncated response can be recorded exactly once without a retry.

## Install

Run these commands from a directory outside the driftlock checkout:

```bash
git clone https://github.com/zli12321/LHTB.git
cd LHTB
git checkout 0d9918f6b66eda0752f8c7d17c9a73a18ee32f98
git apply /absolute/path/to/driftlock/integrations/lhtb/driftlock-harbor.patch
uv pip install -e ./harbor
```

Verify the exact integration before spending provider tokens:

```bash
python - <<'PY'
from harbor._driftlock_pin import (
    DRIFTLOCK_HARBOR_PATCH_VERSION,
    LHTB_REPOSITORY_REVISION,
)

assert LHTB_REPOSITORY_REVISION == "0d9918f6b66eda0752f8c7d17c9a73a18ee32f98"
assert DRIFTLOCK_HARBOR_PATCH_VERSION == 2
print("pinned LHTB Harbor integration ready")
PY
```

Run the no-network integration smoke test against Harbor's real Terminus loop:

```bash
uv run --with pytest --with pytest-asyncio \
  pytest -q /absolute/path/to/driftlock/integrations/lhtb/test_runtime_smoke.py
```

The runtime supports LHTB's LiteLLM backend. Construct Terminus-2 with
`enable_summarize=False`; do not install LHTB's process-reward tracker because
driftlock owns the one-response checkpoints. Call `agent.setup(environment)` before
constructing the runtime:

```python
from driftlock import (
    HarborWorkspaceDeltaObserver,
    LHTBTerminusRuntime,
    TerminusStepAdapter,
)

observer = HarborWorkspaceDeltaObserver(
    environment,
    remote_workspace="/app",
    user=environment.default_user,
)
runtime = LHTBTerminusRuntime(
    terminus_agent,
    environment,
    agent_context,
    remote_workspace="/app",
    observer=observer,
)
step = TerminusStepAdapter(runtime)
```

`LHTBTerminusRuntime` replaces the Tenacity-decorated agent query method and calls
LiteLLM's unwrapped single-attempt method with `num_retries=0`. Each call temporarily
sets Harbor's episode limit to the next physical episode, executes one provider
response plus its terminal commands, and restores the configured limit. The patched
counter proves exactly one lowest-level provider attempt occurred. Rollback restores
semantic chat state while physical calls, usage, trajectory steps, and session
identifiers remain monotonic.

Use `step.before_workspace_restore` as the remote checkpoint store's `before_restore`
hook. Before the first episode the runtime records PID/start-time identities for
pre-existing task services. On rollback it freezes and kills the rejected tmux tree,
then removes post-baseline processes that escaped via double-fork, replaces the login
shell, verifies the new pane is at the canonical workspace root, and resets
incremental screen tracking before the archive store mutates the workspace.
