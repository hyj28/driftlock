"""Pinned LHTB Harbor integration for one-response Terminus-2 boundaries.

The implementation intentionally imports Harbor lazily.  ``driftlock`` remains a
small provider-neutral package, while an experiment environment can install the
pinned LHTB checkout and apply the packaged companion patch.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import importlib.metadata
import importlib.resources
import inspect
import json
import shlex
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MethodType
from typing import Any, Protocol

from driftlock.models import StepTokenBudgetExhausted
from driftlock.terminus import (
    Terminus2StateBridge,
    TerminusBoundary,
    TerminusConversationState,
)

LHTB_REPOSITORY_REVISION = "0d9918f6b66eda0752f8c7d17c9a73a18ee32f98"
LHTB_LITELLM_VERSION = "1.83.14"
DRIFTLOCK_HARBOR_PATCH_VERSION = 11

# On 2026-08-23 the pinned agent provider's *shared* upstream pool was saturated
# for at least 11 minutes and every trial then in flight died. These defaults cover
# roughly 12 minutes of continuous 429s per step: 15 + 30 + 60 + 120 * 5.
DEFAULT_RATE_LIMIT_RETRIES = 8
DEFAULT_RATE_LIMIT_BACKOFF_SEC = 15.0
DEFAULT_RATE_LIMIT_BACKOFF_CAP_SEC = 120.0

_RETRY_OR_FALLBACK_KEYS = frozenset(
    {
        "allowed_fails",
        "content_policy_fallbacks",
        "context_window_fallbacks",
        "cooldown_time",
        "fallbacks",
        "max_fallbacks",
        "max_retries",
        "model_list",
        "num_retries",
        "retry_policy",
        "routing_strategy",
    }
)
_OUTPUT_TOKEN_KEYS = frozenset(
    {"max_completion_tokens", "max_output_tokens", "max_tokens"}
)


def openrouter_provider_call_kwargs(provider: str) -> dict[str, Any]:
    """Return the strict, no-fallback OpenRouter routing request body."""
    if not isinstance(provider, str) or not provider:
        raise ValueError("OpenRouter provider must be a non-empty string")
    return {"extra_body": {"provider": {"only": [provider], "allow_fallbacks": False}}}


def openrouter_provider_from_call_kwargs(
    call_kwargs: Mapping[str, Any], *, source: str
) -> str:
    """Validate strict OpenRouter routing and return its sole provider slug."""
    if "extra_body" not in call_kwargs:
        raise ValueError(f"{source} must contain extra_body")
    extra_body = call_kwargs["extra_body"]
    if not isinstance(extra_body, Mapping):
        raise ValueError(f"{source}.extra_body must be a mapping")
    if set(extra_body) != {"provider"}:
        raise ValueError(f"{source}.extra_body must contain exactly provider")
    routing = extra_body["provider"]
    if not isinstance(routing, Mapping):
        raise ValueError(f"{source}.extra_body.provider must be a mapping")
    if set(routing) != {"only", "allow_fallbacks"}:
        raise ValueError(
            f"{source}.extra_body.provider must contain exactly only and "
            "allow_fallbacks"
        )
    only = routing["only"]
    if not isinstance(only, list):
        raise ValueError(f"{source}.extra_body.provider.only must be a list")
    if len(only) != 1:
        raise ValueError(
            f"{source}.extra_body.provider.only must contain exactly one provider slug"
        )
    if not isinstance(only[0], str) or not only[0]:
        raise ValueError(
            f"{source}.extra_body.provider.only provider slug must be a "
            "non-empty string"
        )
    if routing["allow_fallbacks"] is not False:
        raise ValueError(f"{source}.extra_body.provider.allow_fallbacks must be false")
    return only[0]


class LHTBRuntimeCompatibilityError(RuntimeError):
    """Raised when the installed Harbor fork is not the pinned integration."""


def lhtb_experiment_fingerprint() -> str:
    """Hash the installed driftlock source and packaged Harbor companion patch."""
    package_root = Path(__file__).resolve().parent
    sources = sorted(package_root.rglob("*.py"))
    patch = lhtb_harbor_patch_path().resolve()
    sources.append(patch)
    digest = hashlib.sha256()
    for source in sources:
        relative = source.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


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

    async def canonical_workspace(self) -> str: ...

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
python3 - <<'PY'
import hashlib
import os
import stat
import sys

root = os.lstat(b".")
out = sys.stdout.buffer
if not hasattr(os, "listxattr") or not hasattr(os, "getxattr"):
    raise RuntimeError("remote Python must expose POSIX extended-attribute APIs")


def digest_xattrs(path):
    digest = hashlib.sha256()
    for name in sorted(os.listxattr(path, follow_symlinks=False)):
        encoded_name = os.fsencode(name)
        value = os.getxattr(path, name, follow_symlinks=False)
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def content_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


root_metadata = ":".join(
    str(item)
    for item in (
        root.st_mode,
        root.st_uid,
        root.st_gid,
        root.st_size,
        root.st_mtime_ns,
    )
).encode()
root_value = b":".join((root_metadata, digest_xattrs(b".").encode(), b""))
out.write(b"d\0.\0" + root_value + b"\0")


def visit(directory, relative=b"."):
    entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    for entry in entries:
        if relative == b"." and entry.name == b".git":
            continue
        path = os.path.join(directory, entry.name)
        shown = relative + b"/" + entry.name
        value = os.lstat(path)
        mode = value.st_mode
        if stat.S_ISREG(mode):
            kind = b"f"
            payload = content_digest(path).encode()
        elif stat.S_ISDIR(mode):
            kind = b"d"
            payload = b""
        elif stat.S_ISLNK(mode):
            kind = b"l"
            payload = os.fsencode(os.readlink(path)).hex().encode()
        elif stat.S_ISFIFO(mode):
            kind = b"p"
            payload = b""
        elif stat.S_ISSOCK(mode):
            kind = b"s"
            payload = b""
        elif stat.S_ISBLK(mode):
            kind = b"b"
            payload = str(value.st_rdev).encode()
        elif stat.S_ISCHR(mode):
            kind = b"c"
            payload = str(value.st_rdev).encode()
        else:
            raise RuntimeError(f"unsupported workspace entry: {os.fsdecode(shown)}")
        metadata = ":".join(
            str(item)
            for item in (
                mode,
                value.st_uid,
                value.st_gid,
                value.st_size,
                value.st_mtime_ns,
            )
        ).encode()
        manifest_value = b":".join(
            (metadata, digest_xattrs(path).encode(), payload)
        )
        out.write(kind + b"\0" + shown + b"\0" + manifest_value + b"\0")
        if kind == b"d" and value.st_dev == root.st_dev:
            visit(path, shown)


visit(b".")
PY
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
        self._canonical_workspace: str | None = None

    async def canonical_workspace(self) -> str:
        canonical = await _canonical_remote_workspace(
            self.environment,
            self.remote_workspace,
            user=self.user,
        )
        if (
            self._canonical_workspace is not None
            and canonical != self._canonical_workspace
        ):
            raise RuntimeError("remote workspace canonical path changed during the run")
        self._canonical_workspace = canonical
        return canonical

    async def snapshot(self) -> WorkspaceSnapshot:
        workspace = await self.canonical_workspace()
        manifest_result = await self.environment.exec(
            self._MANIFEST_COMMAND,
            cwd=workspace,
            user=self.user,
        )
        _require_remote_success(manifest_result, "hash remote workspace")
        git_result = await self.environment.exec(
            self._GIT_VIEW_COMMAND,
            cwd=workspace,
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


def is_rate_limit_rejection(error: BaseException) -> bool:
    """Whether *error* is a provider rejection that generated and billed nothing.

    Only HTTP 429 qualifies. A 429 is refused before the model runs, so no
    completion exists to be billed and retrying the *same* provider cannot make
    a billed call look free -- which is the property that lets a retry coexist
    with the one-billable-call-per-step invariant in ``terminus.py``.

    LiteLLM is not importable where this is tested, so the check is
    duck-typed: an explicit 429 status, or the exception's own class name.
    Anything else is left to propagate.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status == 429
    return type(error).__name__ == "RateLimitError"


class _CountingLLM:
    """Count calls at the lowest Harbor provider abstraction."""

    def __init__(
        self,
        delegate: Any,
        *,
        bypass_retry_wrapper: bool,
        rate_limit_retries: int = 0,
        rate_limit_backoff_sec: float = 0.0,
        rate_limit_backoff_cap_sec: float = 0.0,
        sleep: Any = None,
    ) -> None:
        if rate_limit_retries < 0:
            raise ValueError("rate_limit_retries cannot be negative")
        if rate_limit_backoff_sec < 0 or rate_limit_backoff_cap_sec < 0:
            raise ValueError("rate limit backoff cannot be negative")
        self.delegate = delegate
        self.call_count = 0
        # Physical calls that ended in a 429. They generated nothing, so they are
        # excluded from the one-call-per-step check -- but they are counted, not
        # discarded, so a run that only survived by hammering a saturated
        # provider is visible in the record instead of looking clean.
        self.rate_limited_call_count = 0
        self.bypass_retry_wrapper = bypass_retry_wrapper
        self.rate_limit_retries = rate_limit_retries
        self.rate_limit_backoff_sec = rate_limit_backoff_sec
        self.rate_limit_backoff_cap_sec = rate_limit_backoff_cap_sec
        self._sleep = sleep or asyncio.sleep

    async def _attempt(self, prompt: str, **kwargs: Any) -> Any:
        call = self.delegate.call
        if self.bypass_retry_wrapper:
            unwrapped = getattr(call, "__wrapped__", None)
            if unwrapped is None:
                raise LHTBRuntimeCompatibilityError(
                    "pinned LiteLLM.call must expose its unwrapped single attempt"
                )
            return await unwrapped(self.delegate, prompt=prompt, **kwargs)
        return await call(prompt=prompt, **kwargs)

    async def call(self, prompt: str, **kwargs: Any) -> Any:
        self.call_count += 1
        delay = self.rate_limit_backoff_sec
        for remaining in range(self.rate_limit_retries, -1, -1):
            try:
                return await self._attempt(prompt, **kwargs)
            except LHTBRuntimeCompatibilityError:
                raise
            except Exception as error:
                if not remaining or not is_rate_limit_rejection(error):
                    raise
                self.rate_limited_call_count += 1
                if delay:
                    await self._sleep(delay)
                    if self.rate_limit_backoff_cap_sec:
                        delay = min(delay * 2, self.rate_limit_backoff_cap_sec)
                    else:
                        delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

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
        rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
        rate_limit_backoff_sec: float = DEFAULT_RATE_LIMIT_BACKOFF_SEC,
        rate_limit_backoff_cap_sec: float = DEFAULT_RATE_LIMIT_BACKOFF_CAP_SEC,
    ) -> None:
        self.agent = agent
        self.environment = environment
        self.context = context
        workspace_path = PurePosixPath(remote_workspace)
        if not workspace_path.is_absolute() or workspace_path == PurePosixPath("/"):
            raise ValueError("remote_workspace must be an absolute non-root POSIX path")
        self.remote_workspace = remote_workspace
        observer_methods = ("canonical_workspace", "snapshot", "compare")
        if any(
            not callable(getattr(observer, name, None)) for name in observer_methods
        ):
            raise TypeError(
                "observer must implement canonical_workspace(), snapshot(), "
                "and compare()"
            )
        self.observer = observer
        self.bridge = Terminus2StateBridge()
        self._prepared_prompt: str | None = None
        self._initialized = False
        self._process_baseline: tuple[str, ...] | None = None
        self._canonical_workspace: str | None = None
        self._recording_generation = 0
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
            _validate_single_attempt_configuration(agent, llm)
        if isinstance(llm, _CountingLLM):
            raise LHTBRuntimeCompatibilityError(
                "Terminus agent is already owned by an LHTBTerminusRuntime"
            )
        self._counting_llm = _CountingLLM(
            llm,
            bypass_retry_wrapper=require_pinned_harbor,
            rate_limit_retries=rate_limit_retries,
            rate_limit_backoff_sec=rate_limit_backoff_sec,
            rate_limit_backoff_cap_sec=rate_limit_backoff_cap_sec,
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

    @property
    def rate_limited_call_count(self) -> int:
        """Physical calls refused with 429, which generated and billed nothing."""
        return self._counting_llm.rate_limited_call_count

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
        runtime_workspace = await _canonical_remote_workspace(
            self.environment,
            self.remote_workspace,
            user=getattr(agent._session, "_user", None),
        )
        observer_workspace = await self.observer.canonical_workspace()
        if observer_workspace != runtime_workspace:
            raise LHTBRuntimeCompatibilityError(
                "runtime and workspace observer resolve different canonical paths"
            )
        if (
            self._canonical_workspace is not None
            and runtime_workspace != self._canonical_workspace
        ):
            raise RuntimeError("remote workspace canonical path changed during the run")
        self._canonical_workspace = runtime_workspace
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
        augmented_instruction += (
            "\n\nCheckpoint boundary constraint:\n"
            "Each response's terminal-command batch must return to the login shell "
            "before it ends. You may use an interactive program only when the same "
            "response also exits it. Do not leave a REPL, pager, foreground job, "
            "partial command, or pending stdin read across responses."
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
        if self._require_pinned_harbor:
            _validate_single_attempt_configuration(
                self.agent, self._counting_llm.delegate
            )
        if tokens_remaining is not None:
            input_tokens = _conservative_input_token_bound(
                self.agent,
                _agent_chat(self.agent),
                prompt,
            )
            ceiling = tokens_remaining - input_tokens
            if ceiling <= 0:
                raise StepTokenBudgetExhausted(
                    "remaining token budget cannot cover the next provider input"
                )
            output_key = (
                "max_output_tokens"
                if getattr(self._counting_llm.delegate, "_use_responses_api", False)
                else "max_tokens"
            )
            configured_values = (old_call_kwargs.get(key) for key in _OUTPUT_TOKEN_KEYS)
            for configured in configured_values:
                if isinstance(configured, int) and not isinstance(configured, bool):
                    ceiling = min(ceiling, configured)
            for key in _OUTPUT_TOKEN_KEYS:
                self.agent._llm_call_kwargs.pop(key, None)
            self.agent._llm_call_kwargs[output_key] = ceiling
        self.agent._llm_call_kwargs["num_retries"] = 0
        self.agent._llm_call_kwargs["max_retries"] = 0
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
        if self._canonical_workspace is None:
            raise RuntimeError("prepare_start must canonicalize the workspace first")
        user = getattr(session, "_user", None)
        canonical = await _canonical_remote_workspace(
            self.environment,
            remote_workspace,
            user=user,
        )
        if canonical != self._canonical_workspace:
            raise ValueError("restore workspace canonical path does not match runtime")
        await session.stop()
        result = await self.environment.exec(
            _kill_tmux_tree_command(
                session_name,
                canonical,
                process_baseline=self._process_baseline,
            ),
            user=user,
            timeout_sec=30,
        )
        _require_remote_success(result, "quiesce rejected tmux process tree")
        self._recording_generation += 1
        _rotate_session_recording(session, self._recording_generation)
        await session.start()
        await session.send_keys(
            [f"cd -- {shlex.quote(canonical)}", "Enter"],
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
        if actual_cwd != canonical:
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
            "install LHTB Harbor at the pinned revision and apply the patch from "
            "driftlock.lhtb_harbor_patch_path()"
        ) from error
    revision = getattr(marker, "LHTB_REPOSITORY_REVISION", None)
    patch_version = getattr(marker, "DRIFTLOCK_HARBOR_PATCH_VERSION", None)
    if revision != LHTB_REPOSITORY_REVISION or (
        patch_version != DRIFTLOCK_HARBOR_PATCH_VERSION
    ):
        raise LHTBRuntimeCompatibilityError(
            "installed Harbor does not match driftlock's pinned LHTB integration"
        )
    try:
        installed_litellm = importlib.metadata.version("litellm")
    except importlib.metadata.PackageNotFoundError as error:
        raise LHTBRuntimeCompatibilityError(
            "the pinned Harbor environment must install LiteLLM from its lockfile"
        ) from error
    if installed_litellm != LHTB_LITELLM_VERSION:
        raise LHTBRuntimeCompatibilityError(
            "installed LiteLLM does not match the pinned Harbor lockfile: "
            f"expected {LHTB_LITELLM_VERSION}, found {installed_litellm}"
        )


def lhtb_harbor_patch_path() -> Path:
    """Return the installed companion patch for the pinned Harbor checkout."""
    resource = importlib.resources.files("driftlock").joinpath(
        "integrations/lhtb/driftlock-harbor.patch"
    )
    return Path(str(resource))


def _validate_single_attempt_configuration(agent: Any, llm: Any) -> None:
    configured: set[str] = set()
    for value in (
        getattr(agent, "_llm_kwargs", None),
        getattr(agent, "_llm_call_kwargs", None),
        getattr(llm, "_llm_kwargs", None),
    ):
        if isinstance(value, Mapping):
            configured.update(_RETRY_OR_FALLBACK_KEYS.intersection(value))
    if configured:
        names = ", ".join(sorted(configured))
        raise LHTBRuntimeCompatibilityError(
            "retry, router, and fallback configuration is unsupported: " + names
        )
    llm_kwargs = getattr(llm, "_llm_kwargs", None)
    if isinstance(llm_kwargs, Mapping):
        base_output_keys = sorted(_OUTPUT_TOKEN_KEYS.intersection(llm_kwargs))
        if base_output_keys:
            raise LHTBRuntimeCompatibilityError(
                "output token limits must be supplied through Terminus "
                "llm_call_kwargs, not LiteLLM model kwargs: "
                + ", ".join(base_output_keys)
            )
    try:
        litellm = __import__("litellm")
    except ImportError:
        return
    if getattr(litellm, "model_fallbacks", None):
        raise LHTBRuntimeCompatibilityError(
            "global litellm.model_fallbacks must be disabled"
        )


def _conservative_input_token_bound(agent: Any, chat: Any, prompt: str) -> int:
    messages = [*chat.messages, {"role": "user", "content": prompt}]
    encoded = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    byte_bound = len(encoded) + 32 * len(messages) + 256
    try:
        token_counter = __import__("litellm", fromlist=["token_counter"]).token_counter
        estimated = token_counter(model=agent._model_name, messages=messages)
    except Exception:
        estimated = 0
    if not isinstance(estimated, int) or isinstance(estimated, bool) or estimated < 0:
        estimated = 0
    return max(byte_bound, estimated)


def _rotate_session_recording(session: Any, generation: int) -> None:
    """Use a fresh cast file after each stopped rejected trajectory."""
    for name in (
        "_local_asciinema_recording_path",
        "_remote_asciinema_recording_path",
    ):
        original_name = f"_driftlock_original{name}"
        if not hasattr(session, original_name):
            setattr(session, original_name, getattr(session, name, None))
        value = getattr(session, original_name)
        if value is None:
            continue
        path = Path(value) if name.startswith("_local") else PurePosixPath(value)
        stem = path.name.removesuffix(path.suffix)
        rotated = path.with_name(f"{stem}.rollback-{generation}{path.suffix}")
        setattr(session, name, rotated)
    markers = getattr(session, "_markers", None)
    if isinstance(markers, list):
        markers.clear()


async def _canonical_remote_workspace(
    environment: Any,
    remote_workspace: str,
    *,
    user: str | int | None,
) -> str:
    result = await environment.exec(
        f"realpath -e -- {shlex.quote(remote_workspace)}",
        user=user,
        timeout_sec=10,
    )
    _require_remote_success(result, "canonicalize remote workspace")
    lines = (getattr(result, "stdout", None) or "").splitlines()
    if len(lines) != 1:
        raise RuntimeError("remote workspace canonicalization returned invalid output")
    canonical = lines[0]
    path = PurePosixPath(canonical)
    if not path.is_absolute() or path == PurePosixPath("/") or canonical != str(path):
        raise ValueError("remote workspace resolves to an invalid or root path")
    return canonical


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
                f"- {server.name}: {server.transport} transport, url: {server.url}\n"
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
    result_type = _harbor_symbol("harbor.models.trajectories", "ObservationResult")
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
        if kind not in {"b", "c", "d", "f", "l", "p", "s"} or (
            not normalized or normalized in files
        ):
            raise RuntimeError("remote workspace returned duplicate or empty paths")
        if not value:
            raise RuntimeError("remote workspace returned empty metadata")
        payload = value.rsplit(":", 1)[-1]
        if kind == "f" and (
            len(payload) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in payload)
        ):
            raise RuntimeError("remote workspace returned an invalid SHA-256 digest")
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
targets=
for pid in $marked; do
  stat=/proc/$pid/stat
  [ -r "$stat" ] || continue
  IFS= read -r value < "$stat" || continue
  rest=${{value##*) }}
  set -- $rest
  start=${{20:-}}
  case "$start" in ''|*[!0-9]*) continue;; esac
  targets="$targets $pid:$start"
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
    case " $protected " in *" $pid "*) continue;; esac
    IFS= read -r value < "$stat" || continue
    rest=${{value##*) }}
    set -- $rest
    start=${{20:-}}
    case "$start" in ''|*[!0-9]*) continue;; esac
    case " $baseline " in *" $pid:$start "*) continue;; esac
    case " $candidates " in *" $pid:$start "*) continue;; esac
    candidates="$candidates $pid:$start"
    kill -STOP "$pid" 2>/dev/null || true
    changed=1
  done
done
for identity in $candidates; do
  pid=${{identity%%:*}}
  kill -KILL "$pid" 2>/dev/null || true
done
targets="$targets$candidates"

# A failed signal is harmless only if the exact PID/start-time identity has exited.
# Poll briefly so the restore hook cannot report success while rejected code survives.
attempt=0
while [ "$attempt" -lt 50 ]; do
  survivors=
  for identity in $targets; do
    pid=${{identity%%:*}}
    expected=${{identity#*:}}
    stat=/proc/$pid/stat
    [ -r "$stat" ] || continue
    IFS= read -r value < "$stat" || continue
    rest=${{value##*) }}
    set -- $rest
    state=${{1:-}}
    actual=${{20:-}}
    if [ "$actual" = "$expected" ] && [ "$state" != Z ]; then
      survivors="$survivors $identity"
    fi
  done
  [ -z "$survivors" ] && exit 0
  attempt=$((attempt + 1))
  sleep 0.1
done
printf 'rejected processes survived cleanup:%s\n' "$survivors" >&2
exit 42
""".strip()


def _valid_process_identity(value: str) -> bool:
    pid, separator, start = value.partition(":")
    return (
        separator == ":"
        and pid.isascii()
        and start.isascii()
        and (pid.isdigit() and start.isdigit() and int(pid) > 0 and int(start) > 0)
    )


def _validate_loop_signature(agent: Any) -> None:
    signature = inspect.signature(agent._run_agent_loop)
    if "reset_context" not in signature.parameters:
        raise LHTBRuntimeCompatibilityError(
            "pinned _run_agent_loop(reset_context=...) signature is required"
        )
