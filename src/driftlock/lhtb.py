"""Pinned LHTB Harbor integration for one-response Terminus-2 boundaries.

The implementation intentionally imports Harbor lazily.  ``driftlock`` remains a
small provider-neutral package, while an experiment environment can install the
pinned LHTB checkout and apply the companion patch under ``integrations/lhtb``.
"""

from __future__ import annotations

import difflib
import inspect
import shlex
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MethodType
from typing import Any, Protocol

from driftlock.terminus import (
    Terminus2StateBridge,
    TerminusBoundary,
    TerminusConversationState,
)

LHTB_REPOSITORY_REVISION = "0d9918f6b66eda0752f8c7d17c9a73a18ee32f98"
DRIFTLOCK_HARBOR_PATCH_VERSION = 2


class LHTBRuntimeCompatibilityError(RuntimeError):
    """Raised when the installed Harbor fork is not the pinned integration."""


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """A content manifest and Git view captured around one agent episode."""

    files: Mapping[str, str]
    git_view: str = ""


@dataclass(frozen=True, slots=True)
class WorkspaceDelta:
    """Filesystem evidence attributable to one agent episode."""

    changed_paths: tuple[str, ...] = ()
    diff: str = ""


class WorkspaceDeltaObserver(Protocol):
    """Capture remote workspace state without mutating the task."""

    async def snapshot(self) -> WorkspaceSnapshot: ...

    def compare(
        self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> WorkspaceDelta: ...


class HarborWorkspaceDeltaObserver:
    """Hash remote files and retain a Git diff view for per-step evidence.

    The manifest excludes ``.git`` internals but includes tracked, untracked, and
    ignored workspace files.  The textual evidence is a diff between the Git views
    observed before and after the episode; it therefore remains meaningful when a
    file was already dirty before the step.
    """

    _MANIFEST_COMMAND = r"""
find . -xdev ! -path './.git' ! -path './.git/*' -type d \
  -printf 'd\0%p\0\0'
find . -xdev ! -path './.git' ! -path './.git/*' -type l \
  -printf 'l\0%p\0%l\0'
find . -xdev ! -path './.git' ! -path './.git/*' -type f \
  -exec sh -c '
    for path do
      line=$(sha256sum -- "$path") || exit
      case "$line" in \\*) line=${line#\\};; esac
      digest=${line%% *}
      printf "f\0%s\0%s\0" "$path" "$digest"
    done
  ' sh {} +
""".strip()
    _GIT_VIEW_COMMAND = (
        "if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
        "git diff --no-ext-diff --binary -- . && "
        "git diff --no-ext-diff --binary --cached -- . && "
        "git status --porcelain=v1 --untracked-files=all; "
        "fi"
    )

    def __init__(
        self,
        environment: Any,
        *,
        remote_workspace: str,
        user: str | int | None = None,
    ) -> None:
        if not isinstance(remote_workspace, str):
            raise TypeError("remote_workspace must be a string")
        workspace_path = PurePosixPath(remote_workspace)
        if not workspace_path.is_absolute() or workspace_path == PurePosixPath("/"):
            raise ValueError("remote_workspace must be an absolute non-root POSIX path")
        self.environment = environment
        self.remote_workspace = remote_workspace
        self.user = user

    async def snapshot(self) -> WorkspaceSnapshot:
        manifest_result = await self.environment.exec(
            self._MANIFEST_COMMAND,
            cwd=self.remote_workspace,
            user=self.user,
        )
        _require_remote_success(manifest_result, "hash remote workspace")
        git_result = await self.environment.exec(
            self._GIT_VIEW_COMMAND,
            cwd=self.remote_workspace,
            user=self.user,
        )
        _require_remote_success(git_result, "capture remote Git view")
        return WorkspaceSnapshot(
            files=_parse_sha256_manifest(manifest_result.stdout or ""),
            git_view=git_result.stdout or "",
        )

    def compare(
        self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> WorkspaceDelta:
        paths = sorted(set(before.files) | set(after.files))
        changed_paths = tuple(
            path for path in paths if before.files.get(path) != after.files.get(path)
        )
        diff = "\n".join(
            difflib.unified_diff(
                before.git_view.splitlines(),
                after.git_view.splitlines(),
                fromfile="workspace-before",
                tofile="workspace-after",
                lineterm="",
            )
        )
        if not diff and changed_paths:
            diff = "\n".join(f"content changed: {path}" for path in changed_paths)
        return WorkspaceDelta(changed_paths=changed_paths, diff=diff)


class _CountingLLM:
    """Count calls at the lowest Harbor provider abstraction."""

    def __init__(self, delegate: Any, *, bypass_retry_wrapper: bool) -> None:
        self.delegate = delegate
        self.call_count = 0
        self.bypass_retry_wrapper = bypass_retry_wrapper

    async def call(self, prompt: str, **kwargs: Any) -> Any:
        self.call_count += 1
        call = self.delegate.call
        if self.bypass_retry_wrapper:
            unwrapped = getattr(call, "__wrapped__", None)
            if unwrapped is None:
                raise LHTBRuntimeCompatibilityError(
                    "pinned LiteLLM.call must expose its unwrapped single attempt"
                )
            return await unwrapped(self.delegate, prompt=prompt, **kwargs)
        return await call(prompt=prompt, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class LHTBTerminusRuntime:
    """Expose the pinned LHTB Terminus-2 loop one provider response at a time.

    ``agent.setup(environment)`` must have completed before construction.  The
    runtime owns semantic resets after rollback but deliberately preserves Harbor's
    physical token counters, episode counter, session id, and trajectory steps.
    """

    summarization_enabled = False
    internal_retries_enabled = False

    def __init__(
        self,
        agent: Any,
        environment: Any,
        context: Any,
        *,
        remote_workspace: str,
        observer: WorkspaceDeltaObserver,
        require_pinned_harbor: bool = True,
    ) -> None:
        self.agent = agent
        self.environment = environment
        self.context = context
        workspace_path = PurePosixPath(remote_workspace)
        if not workspace_path.is_absolute() or workspace_path == PurePosixPath("/"):
            raise ValueError("remote_workspace must be an absolute non-root POSIX path")
        self.remote_workspace = remote_workspace
        if not callable(getattr(observer, "snapshot", None)) or not callable(
            getattr(observer, "compare", None)
        ):
            raise TypeError("observer must implement snapshot() and compare()")
        self.observer = observer
        self.bridge = Terminus2StateBridge()
        self._prepared_prompt: str | None = None
        self._initialized = False
        self._process_baseline: tuple[str, ...] | None = None
        self._require_pinned_harbor = require_pinned_harbor
        self._validate_agent()
        if require_pinned_harbor:
            _validate_pinned_harbor()

        _validate_loop_signature(agent)
        llm = agent._llm
        if require_pinned_harbor:
            lite_llm_type = _harbor_symbol("harbor.llms.lite_llm", "LiteLLM")
            if not isinstance(llm, lite_llm_type):
                raise LHTBRuntimeCompatibilityError(
                    "the pinned runtime supports Harbor's LiteLLM backend only"
                )
        if isinstance(llm, _CountingLLM):
            raise LHTBRuntimeCompatibilityError(
                "Terminus agent is already owned by an LHTBTerminusRuntime"
            )
        self._counting_llm = _CountingLLM(
            llm, bypass_retry_wrapper=require_pinned_harbor
        )
        if require_pinned_harbor:
            llm._driftlock_single_attempt = True
        agent._llm = self._counting_llm
        agent._enable_summarize = False
        agent._query_llm = MethodType(_single_query_llm, agent)

    @property
    def provider_call_count(self) -> int:
        physical_count = getattr(
            self._counting_llm.delegate, "_driftlock_provider_call_count", None
        )
        if isinstance(physical_count, int) and not isinstance(physical_count, bool):
            return physical_count
        return self._counting_llm.call_count

    async def prepare_start(
        self,
        instruction: str,
        *,
        plan: str,
        rollback_feedback: str | None,
    ) -> str:
        """Reset semantic state and render Harbor's exact first user prompt."""
        if not isinstance(instruction, str) or not isinstance(plan, str):
            raise TypeError("instruction and plan must be strings")
        if rollback_feedback is not None and not isinstance(rollback_feedback, str):
            raise TypeError("rollback_feedback must be a string or None")

        calls_before = self.provider_call_count
        agent = self.agent
        if self._process_baseline is None:
            self._process_baseline = await self._capture_process_baseline()
        if not self._initialized:
            agent._reset_per_run_state()
            chat_type = _harbor_symbol("harbor.llms.chat", "Chat")
            agent._chat = chat_type(
                self._counting_llm,
                interleaved_thinking=agent._interleaved_thinking,
            )
            self._initialized = True
        else:
            chat = _agent_chat(agent)
            chat.messages[:] = []
            chat.reset_response_chain()
            agent._pending_completion = False
            agent._pending_subagent_refs = None
            agent._pending_handoff_prompt = None
            agent._termination_reason = None

        agent._context = self.context
        agent._original_instruction = instruction
        if agent._run_started_monotonic is None:
            agent._run_started_monotonic = time.monotonic()

        augmented_instruction = instruction
        if plan:
            augmented_instruction += f"\n\nExecution plan:\n{plan}"
        if rollback_feedback:
            augmented_instruction += (
                "\n\nThe prior trajectory was rolled back. Choose a different "
                f"approach using this feedback:\n{rollback_feedback}"
            )
        augmented_instruction += _mcp_section(agent)
        skills_section = await agent._build_skills_section(self.environment)
        if skills_section:
            augmented_instruction += skills_section

        terminal_state = agent._limit_output_length(
            await agent._session.get_incremental_output()
        )
        prompt = agent._prompt_template.format(
            instruction=augmented_instruction,
            terminal_state=terminal_state,
        )
        _append_user_audit_step(agent, prompt)
        if self.provider_call_count != calls_before:
            raise RuntimeError("prepare_start must not make a provider call")
        self._prepared_prompt = prompt
        return prompt

    async def start(
        self,
        *,
        prompt: str,
        tokens_remaining: int | None,
    ) -> TerminusBoundary:
        if prompt != self._prepared_prompt:
            raise RuntimeError("start prompt does not match the prepared Harbor prompt")
        self._prepared_prompt = None
        return await self._run_boundary(
            prompt=prompt,
            semantic_episode=1,
            tokens_remaining=tokens_remaining,
        )

    async def resume(
        self,
        state: TerminusConversationState,
        *,
        prompt: str,
        tokens_remaining: int | None,
    ) -> TerminusBoundary:
        if not self._initialized:
            raise RuntimeError("cannot resume before prepare_start")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        self.bridge.restore(self.agent, state)
        self.agent._context = self.context
        self.agent._termination_reason = None
        return await self._run_boundary(
            prompt=prompt,
            semantic_episode=state.episode + 1,
            tokens_remaining=tokens_remaining,
        )

    async def _run_boundary(
        self,
        *,
        prompt: str,
        semantic_episode: int,
        tokens_remaining: int | None,
    ) -> TerminusBoundary:
        _validate_token_ceiling(tokens_remaining)
        before = await self.observer.snapshot()
        calls_before = self.provider_call_count
        steps_before = len(self.agent._trajectory_steps)
        old_max_episodes = self.agent._max_episodes
        old_call_kwargs = dict(self.agent._llm_call_kwargs)
        if tokens_remaining is not None:
            configured = old_call_kwargs.get("max_tokens")
            ceiling = tokens_remaining
            if isinstance(configured, int) and not isinstance(configured, bool):
                ceiling = min(ceiling, configured)
            self.agent._llm_call_kwargs["max_tokens"] = ceiling
        self.agent._llm_call_kwargs["num_retries"] = 0
        self.agent._max_episodes = self.agent._n_episodes + 1

        truncation: BaseException | None = None
        try:
            await self.agent._run_agent_loop(
                initial_prompt=prompt,
                chat=_agent_chat(self.agent),
                logging_dir=self.agent.logs_dir,
                original_instruction=self.agent._original_instruction,
                reset_context=False,
            )
        except BaseException as error:
            if _is_output_length_error(error):
                truncation = error
            else:
                raise
        finally:
            self.agent._max_episodes = old_max_episodes
            self.agent._llm_call_kwargs.clear()
            self.agent._llm_call_kwargs.update(old_call_kwargs)

        if self.provider_call_count != calls_before + 1:
            raise RuntimeError(
                "pinned Terminus boundary did not make exactly one provider call"
            )

        if truncation is not None:
            step = _record_truncation(self.agent, prompt, truncation)
        else:
            new_steps = [
                step
                for step in self.agent._trajectory_steps[steps_before:]
                if getattr(step, "source", None) == "agent"
            ]
            if len(new_steps) != 1:
                raise LHTBRuntimeCompatibilityError(
                    "one provider response must append exactly one Harbor agent step"
                )
            step = new_steps[0]

        self.agent._update_context_from_state(self.context)
        self.agent._dump_trajectory()
        after = await self.observer.snapshot()
        delta = self.observer.compare(before, after)
        next_prompt = _step_observation(step)
        conversation = self.bridge.capture(
            self.agent,
            next_prompt=next_prompt,
            episode=semantic_episode,
        )
        error = _step_error(step, truncation)
        return TerminusBoundary(
            conversation=conversation,
            action=_step_action(step),
            changed_paths=delta.changed_paths,
            diff=delta.diff,
            error=error,
            tokens=_step_tokens(step),
            completed=self.agent._termination_reason == "confirmed_task_complete",
            summary=_step_message(step),
        )

    async def before_workspace_restore(self, remote_workspace: str) -> None:
        """Kill the rejected tmux tree and recreate a clean shell at the root."""
        if remote_workspace != self.remote_workspace:
            raise ValueError("restore workspace does not match runtime workspace")
        session = self.agent._session
        session_name = getattr(session, "_session_name", None)
        if not isinstance(session_name, str) or not session_name:
            raise LHTBRuntimeCompatibilityError("TmuxSession._session_name is required")
        if self._process_baseline is None:
            raise RuntimeError("prepare_start must capture the process baseline first")
        user = getattr(session, "_user", None)
        result = await self.environment.exec(
            _kill_tmux_tree_command(
                session_name,
                remote_workspace,
                process_baseline=self._process_baseline,
            ),
            user=user,
            timeout_sec=30,
        )
        _require_remote_success(result, "quiesce rejected tmux process tree")
        await session.start()
        await session.send_keys(
            [f"cd -- {shlex.quote(remote_workspace)}", "Enter"],
            min_timeout_sec=0.1,
        )
        cwd_result = await self.environment.exec(
            "tmux display-message -p -t "
            f"{shlex.quote(session_name)} '#{{pane_current_path}}'",
            user=user,
            timeout_sec=10,
        )
        _require_remote_success(cwd_result, "verify replacement tmux cwd")
        actual_cwd = (cwd_result.stdout or "").strip()
        expected_result = await self.environment.exec(
            f"realpath -e -- {shlex.quote(remote_workspace)}",
            user=user,
            timeout_sec=10,
        )
        _require_remote_success(expected_result, "canonicalize replacement tmux cwd")
        if actual_cwd != (expected_result.stdout or "").strip():
            raise RuntimeError(
                f"replacement tmux cwd is {actual_cwd!r}, expected remote workspace"
            )
        session._previous_buffer = None

    async def _capture_process_baseline(self) -> tuple[str, ...]:
        session = self.agent._session
        session_name = getattr(session, "_session_name", None)
        if not isinstance(session_name, str) or not session_name:
            raise LHTBRuntimeCompatibilityError("TmuxSession._session_name is required")
        result = await self.environment.exec(
            _process_baseline_command(session_name),
            user=getattr(session, "_user", None),
            timeout_sec=30,
        )
        _require_remote_success(result, "capture pre-agent process baseline")
        identities: list[str] = []
        for line in (result.stdout or "").splitlines():
            identity = line.strip()
            if not _valid_process_identity(identity) or identity in identities:
                raise RuntimeError("remote process baseline is malformed")
            identities.append(identity)
        return tuple(identities)

    def _validate_agent(self) -> None:
        agent = self.agent
        required = (
            "_llm",
            "_session",
            "_run_agent_loop",
            "_reset_per_run_state",
            "_build_skills_section",
            "_update_context_from_state",
            "_dump_trajectory",
            "_prompt_template",
            "_llm_call_kwargs",
            "_max_episodes",
            "_n_episodes",
            "_trajectory_steps",
            "_interleaved_thinking",
        )
        missing = [name for name in required if not hasattr(agent, name)]
        if missing:
            raise LHTBRuntimeCompatibilityError(
                "incompatible LHTB Terminus-2 agent; missing " + ", ".join(missing)
            )
        if agent._session is None:
            raise LHTBRuntimeCompatibilityError(
                "agent.setup(environment) must run before runtime construction"
            )
        if getattr(agent, "_process_reward_tracker", None) is not None:
            raise LHTBRuntimeCompatibilityError(
                "LHTB process-reward checkpoints must be disabled; driftlock owns "
                "episode boundaries"
            )
        if getattr(agent, "_save_raw_content_in_trajectory", False):
            raise LHTBRuntimeCompatibilityError(
                "raw_content trajectories omit parsed actions and are unsupported"
            )
        if not isinstance(agent._llm_call_kwargs, dict):
            raise LHTBRuntimeCompatibilityError("_llm_call_kwargs must be a dict")


async def _single_query_llm(
    agent: Any,
    chat: Any,
    prompt: str,
    logging_paths: tuple[Path | None, Path | None, Path | None],
    original_instruction: str = "",
    session: Any = None,
) -> Any:
    """Replacement for the Tenacity-decorated Harbor query method."""
    del original_instruction, session
    logging_path, prompt_path, response_path = logging_paths
    if prompt_path is not None:
        prompt_path.write_text(prompt)
    started = time.time()
    try:
        response = await chat.chat(
            prompt,
            logging_path=logging_path,
            **agent._llm_call_kwargs,
        )
    finally:
        agent._api_request_times.append((time.time() - started) * 1000)
    if response_path is not None:
        response_path.write_text(response.content)
    return response


def _validate_pinned_harbor() -> None:
    try:
        marker = __import__("harbor._driftlock_pin", fromlist=["*"])
    except ImportError as error:
        raise LHTBRuntimeCompatibilityError(
            "install LHTB Harbor at the pinned revision and apply "
            "integrations/lhtb/driftlock-harbor.patch"
        ) from error
    revision = getattr(marker, "LHTB_REPOSITORY_REVISION", None)
    patch_version = getattr(marker, "DRIFTLOCK_HARBOR_PATCH_VERSION", None)
    if revision != LHTB_REPOSITORY_REVISION or (
        patch_version != DRIFTLOCK_HARBOR_PATCH_VERSION
    ):
        raise LHTBRuntimeCompatibilityError(
            "installed Harbor does not match driftlock's pinned LHTB integration"
        )


def _harbor_symbol(module: str, name: str) -> Any:
    try:
        imported = __import__(module, fromlist=[name])
        return getattr(imported, name)
    except (AttributeError, ImportError) as error:
        raise LHTBRuntimeCompatibilityError(
            f"pinned Harbor symbol {module}.{name} is unavailable"
        ) from error


def _agent_chat(agent: Any) -> Any:
    chat = getattr(agent, "_chat", None)
    if chat is None or not hasattr(chat, "messages"):
        raise LHTBRuntimeCompatibilityError("initialized Harbor Chat is required")
    return chat


def _mcp_section(agent: Any) -> str:
    servers = getattr(agent, "mcp_servers", ())
    if not servers:
        return ""
    value = "\n\nMCP Servers:\nThe following MCP servers are available for this task.\n"
    for server in servers:
        if server.transport == "stdio":
            value += (
                f"- {server.name}: stdio transport, command: {server.command} "
                f"{' '.join(server.args)}\n"
            )
        else:
            value += (
                f"- {server.name}: {server.transport} transport, "
                f"url: {server.url}\n"
            )
    return value


def _append_user_audit_step(agent: Any, prompt: str) -> None:
    step_type = _harbor_symbol("harbor.models.trajectories", "Step")
    agent._trajectory_steps.append(
        step_type(
            step_id=len(agent._trajectory_steps) + 1,
            timestamp=datetime.now(UTC).isoformat(),
            source="user",
            message=prompt,
        )
    )


def _record_truncation(agent: Any, prompt: str, error: BaseException) -> Any:
    response = getattr(error, "response", None)
    usage = getattr(response, "usage", None)
    if response is None or usage is None:
        raise LHTBRuntimeCompatibilityError(
            "patched Harbor truncation must retain its LLMResponse and usage"
        )
    chat = _agent_chat(agent)
    record_response = getattr(chat, "record_response", None)
    if not callable(record_response):
        raise LHTBRuntimeCompatibilityError(
            "patched Harbor Chat.record_response() is required"
        )
    record_response(prompt, response)
    agent._pending_completion = False
    correction = (
        "Your previous response reached the provider output limit. Continue with a "
        "shorter response containing one focused action."
    )
    step_type = _harbor_symbol("harbor.models.trajectories", "Step")
    observation_type = _harbor_symbol("harbor.models.trajectories", "Observation")
    result_type = _harbor_symbol(
        "harbor.models.trajectories", "ObservationResult"
    )
    metrics_type = _harbor_symbol("harbor.models.trajectories", "Metrics")
    step = step_type(
        step_id=len(agent._trajectory_steps) + 1,
        timestamp=datetime.now(UTC).isoformat(),
        source="agent",
        model_name=response.model_name or agent._model_name,
        message=response.content,
        reasoning_content=response.reasoning_content,
        observation=observation_type(results=[result_type(content=correction)]),
        metrics=metrics_type(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cache_tokens or None,
            cost_usd=usage.cost_usd or None,
            prompt_token_ids=response.prompt_token_ids,
            completion_token_ids=response.completion_token_ids,
            logprobs=response.logprobs,
        ),
    )
    agent._trajectory_steps.append(step)
    agent._termination_reason = "driftlock_output_length_boundary"
    return step


def _step_observation(step: Any) -> str:
    observation = getattr(step, "observation", None)
    results = getattr(observation, "results", None)
    if not isinstance(results, list) or not results:
        raise LHTBRuntimeCompatibilityError("Harbor step has no next observation")
    contents = [result.content for result in results if isinstance(result.content, str)]
    if len(contents) != 1:
        raise LHTBRuntimeCompatibilityError(
            "Harbor step must expose exactly one textual next observation"
        )
    return contents[0]


def _step_action(step: Any) -> str:
    actions: list[str] = []
    for call in getattr(step, "tool_calls", None) or ():
        if call.function_name == "bash_command":
            value = call.arguments.get("keystrokes")
            actions.append(value if isinstance(value, str) else "bash_command")
        elif call.function_name == "mark_task_complete":
            actions.append("mark_task_complete")
        else:
            actions.append(str(call.function_name))
    return "\n".join(actions) or "model response without executable command"


def _step_message(step: Any) -> str:
    message = getattr(step, "message", "")
    return message if isinstance(message, str) else str(message)


def _step_tokens(step: Any) -> int:
    metrics = getattr(step, "metrics", None)
    prompt_tokens = getattr(metrics, "prompt_tokens", None)
    completion_tokens = getattr(metrics, "completion_tokens", None)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (prompt_tokens, completion_tokens)
    ):
        raise LHTBRuntimeCompatibilityError(
            "Harbor must report prompt and completion tokens for every response"
        )
    return prompt_tokens + completion_tokens


def _step_error(step: Any, truncation: BaseException | None) -> str | None:
    if truncation is not None:
        return str(truncation)
    observation = _step_observation(step)
    if observation.startswith("Previous response had parsing errors:"):
        return observation
    return None


def _is_output_length_error(error: BaseException) -> bool:
    return error.__class__.__name__ == "OutputLengthExceededError"


def _validate_token_ceiling(tokens_remaining: int | None) -> None:
    if tokens_remaining is None:
        return
    if (
        not isinstance(tokens_remaining, int)
        or isinstance(tokens_remaining, bool)
        or tokens_remaining <= 0
    ):
        raise ValueError("tokens_remaining must be a positive integer or None")


def _parse_sha256_manifest(output: str) -> dict[str, str]:
    files: dict[str, str] = {}
    fields = output.split("\0")
    if fields[-1:] != [""] or (len(fields) - 1) % 3:
        raise RuntimeError("remote workspace returned an invalid manifest")
    for offset in range(0, len(fields) - 1, 3):
        kind, name, value = fields[offset : offset + 3]
        normalized = name.removeprefix("./")
        if kind not in {"d", "f", "l"} or not normalized or normalized in files:
            raise RuntimeError("remote workspace returned duplicate or empty paths")
        if kind == "f" and (
            len(value) != 64
            or any(
                character not in "0123456789abcdefABCDEF" for character in value
            )
        ):
            raise RuntimeError("remote workspace returned an invalid SHA-256 digest")
        if kind == "d" and value:
            raise RuntimeError("remote directory manifest value must be empty")
        files[normalized] = f"{kind}:{value.lower() if kind == 'f' else value}"
    return files


def _require_remote_success(result: Any, operation: str) -> None:
    return_code = getattr(result, "return_code", None)
    if return_code != 0:
        stderr = (getattr(result, "stderr", None) or "").strip()
        raise RuntimeError(f"failed to {operation}: {stderr or f'exit {return_code}'}")


def _process_baseline_command(session_name: str) -> str:
    session = shlex.quote(session_name)
    return f"""
set -eu
pane=$(tmux display-message -p -t {session} '#{{pane_pid}}')
case "$pane" in ''|*[!0-9]*) exit 31;; esac
excluded=$pane
changed=1
while [ "$changed" -eq 1 ]; do
  changed=0
  for status in /proc/[0-9]*/status; do
    [ -r "$status" ] || continue
    pid=${{status#/proc/}}
    pid=${{pid%/status}}
    case " $excluded " in *" $pid "*) continue;; esac
    ppid=
    while IFS= read -r line; do
      case "$line" in PPid:*) set -- $line; ppid=$2; break;; esac
    done < "$status"
    case " $excluded " in
      *" $ppid "*) excluded="$excluded $pid"; changed=1;;
    esac
  done
done
for stat in /proc/[0-9]*/stat; do
  [ -r "$stat" ] || continue
  pid=${{stat#/proc/}}
  pid=${{pid%/stat}}
  case " $excluded " in *" $pid "*) continue;; esac
  IFS= read -r value < "$stat" || continue
  rest=${{value##*) }}
  set -- $rest
  start=${{20:-}}
  case "$start" in ''|*[!0-9]*) continue;; esac
  printf '%s:%s\n' "$pid" "$start"
done
""".strip()


def _kill_tmux_tree_command(
    session_name: str,
    workspace: str,
    *,
    process_baseline: tuple[str, ...],
) -> str:
    session = shlex.quote(session_name)
    root = shlex.quote(workspace)
    if any(not _valid_process_identity(value) for value in process_baseline):
        raise ValueError("process_baseline contains an invalid identity")
    baseline = shlex.quote(" ".join(process_baseline))
    return f"""
set -eu
workspace=$(realpath -e -- {root})
[ "$workspace" != / ]
[ -d "$workspace" ]
pane=$(tmux display-message -p -t {session} '#{{pane_pid}}')
case "$pane" in ''|*[!0-9]*) exit 31;; esac
marked=$pane
changed=1
while [ "$changed" -eq 1 ]; do
  changed=0
  for status in /proc/[0-9]*/status; do
    [ -r "$status" ] || continue
    pid=${{status#/proc/}}
    pid=${{pid%/status}}
    case " $marked " in *" $pid "*) continue;; esac
    ppid=$(sed -n 's/^PPid:[[:space:]]*//p' "$status" 2>/dev/null)
    case " $marked " in
      *" $ppid "*) marked="$marked $pid"; changed=1;;
    esac
  done
done
tmux send-keys -t {session} C-c 2>/dev/null || true
# Freeze the shell and its known descendants before killing them. This closes the
# fork-during-cleanup race that would otherwise let a rejected background job escape.
for pid in $marked; do kill -STOP "$pid" 2>/dev/null || true; done
changed=1
while [ "$changed" -eq 1 ]; do
  changed=0
  for status in /proc/[0-9]*/status; do
    [ -r "$status" ] || continue
    pid=${{status#/proc/}}
    pid=${{pid%/status}}
    case " $marked " in *" $pid "*) continue;; esac
    ppid=$(sed -n 's/^PPid:[[:space:]]*//p' "$status" 2>/dev/null)
    case " $marked " in
      *" $ppid "*)
        marked="$marked $pid"
        kill -STOP "$pid" 2>/dev/null || true
        changed=1
        ;;
    esac
  done
done
for pid in $marked; do
  kill -KILL "$pid" 2>/dev/null || true
done
tmux kill-session -t {session} 2>/dev/null || true
! tmux has-session -t {session} 2>/dev/null

# The tmux ancestry misses daemonized jobs that double-forked and were reparented.
# Preserve only processes that predated the agent plus this cleanup command's own
# ancestry; freeze and then kill every other rejected-branch process by PID/starttime.
baseline={baseline}
protected=$$
cursor=$$
while [ "$cursor" -gt 1 ]; do
  status=/proc/$cursor/status
  [ -r "$status" ] || break
  parent=
  while IFS= read -r line; do
    case "$line" in PPid:*) set -- $line; parent=$2; break;; esac
  done < "$status"
  case "$parent" in ''|*[!0-9]*) break;; esac
  protected="$protected $parent"
  cursor=$parent
done

candidates=
changed=1
while [ "$changed" -eq 1 ]; do
  changed=0
  for stat in /proc/[0-9]*/stat; do
    [ -r "$stat" ] || continue
    pid=${{stat#/proc/}}
    pid=${{pid%/stat}}
    case " $protected $candidates " in *" $pid "*) continue;; esac
    IFS= read -r value < "$stat" || continue
    rest=${{value##*) }}
    set -- $rest
    start=${{20:-}}
    case "$start" in ''|*[!0-9]*) continue;; esac
    case " $baseline " in *" $pid:$start "*) continue;; esac
    candidates="$candidates $pid"
    kill -STOP "$pid" 2>/dev/null || true
    changed=1
  done
done
sleep 1
for pid in $candidates; do kill -KILL "$pid" 2>/dev/null || true; done
""".strip()


def _valid_process_identity(value: str) -> bool:
    pid, separator, start = value.partition(":")
    return separator == ":" and pid.isascii() and start.isascii() and (
        pid.isdigit() and start.isdigit() and int(pid) > 0 and int(start) > 0
    )


def _validate_loop_signature(agent: Any) -> None:
    signature = inspect.signature(agent._run_agent_loop)
    if "reset_context" not in signature.parameters:
        raise LHTBRuntimeCompatibilityError(
            "pinned _run_agent_loop(reset_context=...) signature is required"
        )
