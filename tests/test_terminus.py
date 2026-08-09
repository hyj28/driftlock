from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pytest

from driftlock.models import StepContext
from driftlock.terminus import (
    Terminus2StateBridge,
    TerminusBoundary,
    TerminusConversationCodec,
    TerminusConversationState,
    TerminusStateError,
    TerminusStepAdapter,
)


class FakeChat:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages
        self.response_chain_resets = 0
        self.total_input_tokens = 100

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    def reset_response_chain(self) -> None:
        self.response_chain_resets += 1


class FakeAgent:
    def __init__(self, chat: FakeChat) -> None:
        self._chat = chat
        self._pending_completion = True
        self._n_episodes = 3


@dataclass
class RuntimeCall:
    operation: str
    state: TerminusConversationState | None
    prompt: str
    tokens_remaining: int | None


class FakeBoundaryRuntime:
    def __init__(self) -> None:
        self.calls: list[RuntimeCall] = []

    async def start(
        self,
        instruction: str,
        *,
        plan: str,
        tokens_remaining: int | None,
    ) -> TerminusBoundary:
        self.calls.append(
            RuntimeCall("start", None, f"{instruction}|{plan}", tokens_remaining)
        )
        return _boundary(episode=1, next_prompt="terminal one", tokens=11)

    async def resume(
        self,
        state: TerminusConversationState,
        *,
        prompt: str,
        tokens_remaining: int | None,
    ) -> TerminusBoundary:
        self.calls.append(RuntimeCall("resume", state, prompt, tokens_remaining))
        return _boundary(
            episode=state.episode + 1,
            next_prompt="terminal two",
            tokens=13,
            completed=True,
        )


def _conversation(
    *, episode: int = 1, next_prompt: str = "terminal output"
) -> TerminusConversationState:
    return TerminusConversationState(
        messages=(
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"commands": []}'},
        ),
        next_prompt=next_prompt,
        pending_completion=False,
        episode=episode,
    )


def _boundary(
    *,
    episode: int,
    next_prompt: str,
    tokens: int,
    completed: bool = False,
) -> TerminusBoundary:
    return TerminusBoundary(
        conversation=_conversation(episode=episode, next_prompt=next_prompt),
        action="run tests",
        changed_paths=("src/app.py",),
        diff="+healthy",
        tokens=tokens,
        completed=completed,
        summary="one Terminus episode",
    )


def _context(
    state: dict[str, Any],
    *,
    rollback_feedback: str | None = None,
    logical_step: int = 1,
) -> StepContext:
    return StepContext(
        goal="finish task",
        plan="inspect then test",
        state=state,
        sequence=logical_step,
        logical_step=logical_step,
        attempt=1,
        rollback_feedback=rollback_feedback,
        tokens_remaining=500,
    )


def test_conversation_codec_round_trip_is_detached() -> None:
    codec = TerminusConversationCodec()
    original = _conversation()

    encoded = codec.encode(original)
    decoded = codec.decode(encoded)

    assert decoded == original
    assert decoded is not None
    encoded["terminus_2"]["messages"][0]["content"] = "mutated"
    assert decoded.messages[0]["content"] == "task"


def test_conversation_codec_rejects_non_json_and_unknown_schema() -> None:
    with pytest.raises(TerminusStateError, match="strict JSON"):
        TerminusConversationState(
            messages=({"role": "user", "content": {1, 2}},),
            next_prompt="",
            pending_completion=False,
            episode=1,
        )

    with pytest.raises(TerminusStateError, match="schema version"):
        TerminusConversationCodec().decode(
            {"terminus_2": {"schema_version": 2, "started": False}}
        )


def test_state_bridge_restores_only_semantic_state() -> None:
    codec = TerminusConversationCodec()
    chat = FakeChat(
        [
            {"role": "user", "content": "checkpoint"},
            {"role": "assistant", "content": "answer"},
        ]
    )
    agent = FakeAgent(chat)
    bridge = Terminus2StateBridge(codec)
    state = bridge.capture(agent, next_prompt="checkpoint observation", episode=3)
    saved_messages = copy.deepcopy(chat.messages)

    chat.messages.append({"role": "user", "content": "drift"})
    agent._pending_completion = False
    agent._n_episodes = 8
    bridge.restore(agent, state)

    assert chat.messages == saved_messages
    assert chat.response_chain_resets == 1
    assert chat.total_input_tokens == 100
    assert agent._pending_completion is True
    assert agent._n_episodes == 8


async def test_step_adapter_starts_then_resumes_at_exact_boundaries() -> None:
    runtime = FakeBoundaryRuntime()
    adapter = TerminusStepAdapter(runtime)

    first = await adapter(_context(adapter.initial_state()))
    second = await adapter(_context(dict(first.state), logical_step=2))

    assert first.tokens == 11
    assert first.changed_paths == ("src/app.py",)
    assert second.tokens == 13
    assert second.completed
    assert [call.operation for call in runtime.calls] == ["start", "resume"]
    assert runtime.calls[1].prompt == "terminal one"
    assert runtime.calls[1].tokens_remaining == 500


async def test_step_adapter_adds_feedback_only_after_rollback() -> None:
    runtime = FakeBoundaryRuntime()
    adapter = TerminusStepAdapter(runtime)
    state = adapter.codec.encode(_conversation())

    await adapter(
        _context(
            state,
            rollback_feedback="the previous branch changed the wrong files",
            logical_step=2,
        )
    )

    assert runtime.calls[0].prompt == (
        "terminal output\n\nTrajectory rollback reason: "
        "the previous branch changed the wrong files"
    )


async def test_step_adapter_rejects_runtime_that_skips_episodes() -> None:
    class SkippingRuntime(FakeBoundaryRuntime):
        async def start(
            self,
            instruction: str,
            *,
            plan: str,
            tokens_remaining: int | None,
        ) -> TerminusBoundary:
            return _boundary(episode=2, next_prompt="late", tokens=1)

    adapter = TerminusStepAdapter(SkippingRuntime())

    with pytest.raises(RuntimeError, match="exactly one episode"):
        await adapter(_context(adapter.initial_state()))
