"""Checkpoint-boundary adapters for Harbor's Terminus-2 agent loop.

The module deliberately has no Harbor dependency. A Harbor fork only needs to
expose one billed Terminus episode at a time through ``TerminusBoundaryRuntime``.
Conversation state is JSON-compatible so the regular checkpoint stores can persist
it atomically with the workspace.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, MutableSequence
from dataclasses import dataclass
from typing import Any, Protocol

from driftlock.models import StepContext, StepOutcome


class TerminusStateError(ValueError):
    """Raised when persisted Terminus state does not match the codec contract."""


class TerminusChat(Protocol):
    """The stable subset of Harbor's ``Chat`` used by the state bridge."""

    @property
    def messages(self) -> MutableSequence[Any]: ...

    def reset_response_chain(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TerminusConversationState:
    """Semantic Terminus state captured after one billed agent episode."""

    messages: tuple[Mapping[str, Any], ...]
    next_prompt: str
    pending_completion: bool
    episode: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode, int)
            or isinstance(self.episode, bool)
            or self.episode <= 0
        ):
            raise TerminusStateError("Terminus episode must be positive")
        if not isinstance(self.next_prompt, str):
            raise TerminusStateError("Terminus next_prompt must be a string")
        if not isinstance(self.pending_completion, bool):
            raise TerminusStateError("Terminus pending_completion must be a boolean")
        _validate_messages(self.messages)


@dataclass(frozen=True, slots=True)
class TerminusBoundary:
    """One billed Terminus episode and its driftlock observations.

    Parser-error responses are episodes too.  A runtime must return them with
    ``error`` populated even though no terminal command was executed.
    """

    conversation: TerminusConversationState
    action: str
    changed_paths: tuple[str, ...] = ()
    diff: str = ""
    error: str | None = None
    reward: float | None = None
    tokens: int = 0
    completed: bool = False
    summary: str = ""

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens cannot be negative")


class TerminusBoundaryRuntime(Protocol):
    """A Terminus fork that yields after exactly one billed episode.

    ``start`` may be called again after rollback to the initial checkpoint.  It
    must reset semantic conversation state while keeping physical usage counters
    and trajectory audit data monotonic.  Both methods must enforce the supplied
    provider token ceiling and report actual usage in the returned boundary.
    Output-length responses must not be retried inside one call: they return an
    error boundary with their billed usage and correction prompt instead.
    """

    @property
    def provider_call_count(self) -> int:
        """Monotonic count incremented at the lowest provider-call boundary."""
        ...

    @property
    def summarization_enabled(self) -> bool:
        """Whether Terminus may make internal summarization model calls."""
        ...

    @property
    def internal_retries_enabled(self) -> bool:
        """Whether one Terminus query may make multiple provider attempts."""
        ...

    async def start(
        self,
        instruction: str,
        *,
        plan: str,
        rollback_feedback: str | None,
        tokens_remaining: int | None,
    ) -> TerminusBoundary: ...

    async def resume(
        self,
        state: TerminusConversationState,
        *,
        prompt: str,
        tokens_remaining: int | None,
    ) -> TerminusBoundary: ...

    async def before_workspace_restore(self, remote_workspace: str) -> None:
        """Quiesce branch processes and recreate a clean shell at the workspace."""
        ...


class TerminusConversationCodec:
    """Encode, decode, capture, and restore Terminus-2 conversation state."""

    schema_version = 1
    state_key = "terminus_2"

    def initial_state(self) -> dict[str, Any]:
        """Return the JSON state used before the first Terminus episode."""
        return {
            self.state_key: {
                "schema_version": self.schema_version,
                "started": False,
            }
        }

    def encode(self, state: TerminusConversationState) -> dict[str, Any]:
        """Return a detached JSON-compatible checkpoint payload."""
        payload = {
            "schema_version": self.schema_version,
            "started": True,
            "messages": copy.deepcopy(list(state.messages)),
            "next_prompt": state.next_prompt,
            "pending_completion": state.pending_completion,
            "episode": state.episode,
        }
        _validate_json(payload)
        return {self.state_key: payload}

    def decode(self, value: Mapping[str, Any]) -> TerminusConversationState | None:
        """Decode checkpoint state, returning ``None`` before the first episode."""
        payload = value.get(self.state_key)
        if not isinstance(payload, Mapping):
            raise TerminusStateError(
                f"checkpoint state is missing the {self.state_key!r} object"
            )
        schema_version = payload.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version != self.schema_version
        ):
            raise TerminusStateError("unsupported Terminus state schema version")
        started = payload.get("started")
        if started is False:
            return None
        if started is not True:
            raise TerminusStateError("Terminus started flag must be a boolean")

        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise TerminusStateError("Terminus messages must be a list")
        next_prompt = payload.get("next_prompt")
        pending_completion = payload.get("pending_completion")
        episode = payload.get("episode")
        if not isinstance(next_prompt, str):
            raise TerminusStateError("Terminus next_prompt must be a string")
        if not isinstance(pending_completion, bool):
            raise TerminusStateError("Terminus pending_completion must be a boolean")
        if not isinstance(episode, int) or isinstance(episode, bool):
            raise TerminusStateError("Terminus episode must be an integer")
        return TerminusConversationState(
            messages=tuple(copy.deepcopy(messages)),
            next_prompt=next_prompt,
            pending_completion=pending_completion,
            episode=episode,
        )

    def capture(
        self,
        chat: TerminusChat,
        *,
        next_prompt: str,
        pending_completion: bool,
        episode: int,
    ) -> TerminusConversationState:
        """Capture the semantic fields from Harbor after an episode boundary."""
        messages = copy.deepcopy(list(chat.messages))
        return TerminusConversationState(
            messages=tuple(messages),
            next_prompt=next_prompt,
            pending_completion=pending_completion,
            episode=episode,
        )

    def restore(self, chat: TerminusChat, state: TerminusConversationState) -> None:
        """Restore chat history and force the provider to consume the full history.

        Token counters and rollout details intentionally remain untouched.  They
        account for physical compute and must not rewind with semantic state.
        """
        chat.messages[:] = copy.deepcopy(list(state.messages))
        chat.reset_response_chain()


class Terminus2StateBridge:
    """Narrow bridge for the relevant private fields of Harbor Terminus-2.

    Harbor currently does not expose checkpointable state publicly.  The forked
    episode loop supplies ``next_prompt`` and uses this bridge only after a billed
    response is recorded. Compatibility failures are explicit instead of silently
    producing a partial restore.
    """

    def __init__(self, codec: TerminusConversationCodec | None = None) -> None:
        self.codec = codec or TerminusConversationCodec()

    def capture(
        self,
        agent: Any,
        *,
        next_prompt: str,
        episode: int,
    ) -> TerminusConversationState:
        chat = _agent_chat(agent)
        pending_completion = _agent_field(agent, "_pending_completion", bool)
        return self.codec.capture(
            chat,
            next_prompt=next_prompt,
            pending_completion=pending_completion,
            episode=episode,
        )

    def restore(self, agent: Any, state: TerminusConversationState) -> None:
        chat = _agent_chat(agent)
        self.codec.restore(chat, state)
        agent._pending_completion = state.pending_completion


class TerminusStepAdapter:
    """Translate checkpoint-boundary Terminus episodes into driftlock steps."""

    def __init__(
        self,
        runtime: TerminusBoundaryRuntime,
        *,
        codec: TerminusConversationCodec | None = None,
        rollback_prefix: str = "Trajectory rollback reason:",
    ) -> None:
        self.runtime = runtime
        self.codec = codec or TerminusConversationCodec()
        self.rollback_prefix = rollback_prefix
        if runtime.summarization_enabled:
            raise ValueError(
                "Terminus internal summarization must be disabled for "
                "checkpoint boundaries"
            )
        if runtime.internal_retries_enabled:
            raise ValueError(
                "Terminus internal provider retries must be disabled for "
                "checkpoint boundaries"
            )
        _provider_call_count(runtime)

    def initial_state(self) -> dict[str, Any]:
        return self.codec.initial_state()

    async def before_workspace_restore(self, remote_workspace: str) -> None:
        """Delegate the remote store's pre-restore lifecycle to Terminus.

        The runtime must terminate processes from the rejected branch, replace the
        persistent tmux shell with a clean one whose cwd is ``remote_workspace``,
        and reset its terminal-output cursor. Any failure must propagate so the
        checkpoint store leaves the live workspace untouched.
        """
        await self.runtime.before_workspace_restore(remote_workspace)

    async def __call__(self, context: StepContext) -> StepOutcome:
        provider_calls_before = _provider_call_count(self.runtime)
        previous = self.codec.decode(context.state)
        if previous is None:
            boundary = await self.runtime.start(
                context.goal,
                plan=context.plan,
                rollback_feedback=context.rollback_feedback,
                tokens_remaining=context.tokens_remaining,
            )
            expected_episode = 1
        else:
            prompt = previous.next_prompt
            if context.rollback_feedback:
                prompt = (
                    f"{prompt}\n\n{self.rollback_prefix} {context.rollback_feedback}"
                )
            boundary = await self.runtime.resume(
                previous,
                prompt=prompt,
                tokens_remaining=context.tokens_remaining,
            )
            expected_episode = previous.episode + 1

        provider_calls_after = _provider_call_count(self.runtime)
        if provider_calls_after != provider_calls_before + 1:
            raise RuntimeError(
                "Terminus runtime must make exactly one provider call per "
                "driftlock step: "
                f"expected counter {provider_calls_before + 1}, got "
                f"{provider_calls_after}"
            )

        if boundary.conversation.episode != expected_episode:
            raise RuntimeError(
                "Terminus runtime must advance exactly one episode per driftlock step: "
                f"expected {expected_episode}, got {boundary.conversation.episode}"
            )

        return StepOutcome(
            action=boundary.action,
            state=self.codec.encode(boundary.conversation),
            changed_paths=boundary.changed_paths,
            diff=boundary.diff,
            error=boundary.error,
            reward=boundary.reward,
            tokens=boundary.tokens,
            completed=boundary.completed,
            summary=boundary.summary,
        )


def _agent_chat(agent: Any) -> TerminusChat:
    chat = getattr(agent, "_chat", None)
    if (
        chat is None
        or not hasattr(chat, "messages")
        or not callable(getattr(chat, "reset_response_chain", None))
    ):
        raise RuntimeError(
            "incompatible Terminus-2 agent: initialized _chat with messages and "
            "reset_response_chain() is required"
        )
    return chat


def _provider_call_count(runtime: TerminusBoundaryRuntime) -> int:
    value = runtime.provider_call_count
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(
            "Terminus runtime provider_call_count must be a non-negative integer"
        )
    return value


def _agent_field(agent: Any, name: str, expected_type: type[Any]) -> Any:
    value = getattr(agent, name, None)
    if not isinstance(value, expected_type):
        raise RuntimeError(
            f"incompatible Terminus-2 agent: {name} must be {expected_type.__name__}"
        )
    return value


def _validate_messages(messages: tuple[Mapping[str, Any], ...]) -> None:
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TerminusStateError(f"Terminus message {index} must be an object")
        if not isinstance(message.get("role"), str):
            raise TerminusStateError(
                f"Terminus message {index} must contain a string role"
            )
        if "content" not in message:
            raise TerminusStateError(f"Terminus message {index} must contain content")
    _validate_json(list(messages))


def _validate_json(value: Any) -> None:
    _validate_json_tree(value)
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise TerminusStateError(
            "Terminus checkpoint state must be strict JSON"
        ) from error


def _validate_json_tree(value: Any) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TerminusStateError("Terminus checkpoint state must be strict JSON")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_tree(item)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TerminusStateError(
                "Terminus checkpoint state must use string object keys"
            )
        for item in value.values():
            _validate_json_tree(item)
        return
    raise TerminusStateError("Terminus checkpoint state must be strict JSON")
