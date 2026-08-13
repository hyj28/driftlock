# Pinned LHTB Harbor integration

`driftlock` targets the LHTB repository at commit
`0d9918f6b66eda0752f8c7d17c9a73a18ee32f98`. The companion patch does five
things the boundary runtime cannot safely infer from stock Harbor:

1. it adds a revision marker checked when `LHTBTerminusRuntime` starts;
2. it retains the full `LLMResponse`, including provider usage, when LiteLLM reports
   an output-length truncation; and
3. it counts calls immediately around `acompletion` / `aresponses` and disables
   LiteLLM's parameter-fallback retry while driftlock owns a one-response boundary;
4. it appends to the tmux pane log so replacing a rejected shell cannot erase the
   earlier physical trajectory; and
5. it waits for a shell completion marker before exposing a command boundary, and
   interrupts plus verifies quiescence after a command timeout.

The patch also factors Harbor's normal chat accounting into `Chat.record_response()`
so a billed truncated response can be recorded exactly once without a retry.

## Install

Run these commands from a directory outside the driftlock checkout:

```bash
git clone https://github.com/zli12321/LHTB.git
cd LHTB
git checkout 0d9918f6b66eda0752f8c7d17c9a73a18ee32f98
uv sync --project harbor --frozen --no-dev
uv pip install --python harbor/.venv/bin/python /absolute/path/to/driftlock
PATCH=$(harbor/.venv/bin/python -c \
  'from driftlock import lhtb_harbor_patch_path; print(lhtb_harbor_patch_path())')
git apply "$PATCH"
```

Verify the exact integration before spending provider tokens:

```bash
harbor/.venv/bin/python - <<'PY'
from importlib.metadata import version

from harbor._driftlock_pin import (
    DRIFTLOCK_HARBOR_PATCH_VERSION,
    LHTB_REPOSITORY_REVISION,
)

assert LHTB_REPOSITORY_REVISION == "0d9918f6b66eda0752f8c7d17c9a73a18ee32f98"
assert DRIFTLOCK_HARBOR_PATCH_VERSION == 8
assert version("litellm") == "1.83.14"
print("pinned LHTB Harbor integration ready")
PY
```

Run the no-network integration smoke test against Harbor's real Terminus loop:

```bash
uv pip install --python harbor/.venv/bin/python pytest pytest-asyncio
harbor/.venv/bin/python -m pytest -q \
  /absolute/path/to/driftlock/integrations/lhtb/test_runtime_smoke.py
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
LiteLLM's unwrapped single-attempt method with both retry ceilings set to zero. The
runtime rejects router, fallback, or retry configuration and validates LiteLLM
`1.83.14` from Harbor's frozen lockfile. Each call temporarily
sets Harbor's episode limit to the next physical episode, executes one provider
response plus its terminal commands, and restores the configured limit. It reserves
a conservative bound for input tokens before setting the chat or Responses API
output ceiling. Harbor waits for a shell completion marker before the runtime takes
its post-episode workspace snapshot, preventing a foreground command from racing an
accepted checkpoint. Each blocking send uses a fresh, unguessable tmux completion
channel so task commands and concurrent sessions cannot release another boundary's
waiter. The marker is queued on its own input line, so multiline and heredoc commands
remain byte-for-byte intact, and it restores the original command's exit status. A
batch whose final non-empty keystroke does not execute a shell command is rejected
before any part of that batch is sent; Harbor's documented `C-c` and `C-d` control
keys are accepted as explicit ways to return from an interactive program before the
marker is queued. Intermediate commands keep Harbor's requested duration, allowing
an interactive program to be opened, used, and exited within one response.
Persistent interactive state across provider responses is intentionally unsupported
because it cannot be represented by workspace and conversation checkpoints; the
runtime adds this constraint to the agent prompt. Rollback restores semantic chat
state while physical calls, usage, trajectory steps, pane logs, cast segments, and
session identifiers remain monotonic.

Use `step.before_workspace_restore` as the remote checkpoint store's `before_restore`
hook. Before the first episode the runtime records PID/start-time identities for
pre-existing task services. On rollback it freezes and kills the rejected tmux tree,
then removes post-baseline processes that escaped via double-fork, replaces the login
shell, verifies the new pane is at the canonical workspace root, and resets
incremental screen tracking before the archive store mutates the workspace.

The workspace observer requires `python3` inside the task environment. It records
content, file type, mode, ownership, size, nanosecond mtime, symlink targets, device
identities, and extended attributes for tracked, untracked, and ignored entries.
