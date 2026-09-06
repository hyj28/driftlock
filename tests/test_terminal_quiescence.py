from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from driftlock.terminal_quiescence import (
    PANE_LIVENESS_TIMEOUT_SECONDS,
    QUIESCE_HANDSHAKE_TIMEOUT_SECONDS,
    CheckpointQuiesceStatus,
    DriftlockTerminalUnusableError,
    quiesce_terminal_after_timeout,
)


@dataclass(frozen=True, slots=True)
class SendKeysCall:
    keys: tuple[str, ...]
    options: dict[str, Any]


class FakeSendKeys:
    def __init__(
        self,
        timeout_calls: Sequence[int] = (),
        errors: dict[int, BaseException] | None = None,
    ) -> None:
        self.timeout_calls = frozenset(timeout_calls)
        self.errors = errors or {}
        self.calls: list[SendKeysCall] = []

    async def __call__(self, keys: list[str], **options: Any) -> None:
        self.calls.append(SendKeysCall(tuple(keys), options))
        call_number = len(self.calls)
        if call_number in self.errors:
            raise self.errors[call_number]
        if call_number in self.timeout_calls:
            raise TimeoutError("simulated send timeout")


def _unexpected_probe() -> Callable[[], Awaitable[str]]:
    async def probe() -> str:
        raise AssertionError("pane capture was not expected")

    return probe


@pytest.mark.asyncio
async def test_first_quiesce_handshake_recovers_checkpoint_boundary() -> None:
    send_keys = FakeSendKeys()

    outcome = await quiesce_terminal_after_timeout(
        send_keys,
        capture_pane=_unexpected_probe(),
    )

    assert outcome.status is CheckpointQuiesceStatus.RECOVERED
    assert outcome.checkpointable is True
    assert outcome.reason is None
    assert send_keys.calls == [
        SendKeysCall(
            ("C-c",),
            {"block": False, "min_timeout_sec": 0.1},
        ),
        SendKeysCall(
            (":", "Enter"),
            {"block": True, "max_timeout_sec": 15.0},
        ),
    ]


@pytest.mark.asyncio
async def test_alive_but_slow_shell_prompt_does_not_abort_paid_episode() -> None:
    send_keys = FakeSendKeys(timeout_calls=(2,))
    pane_calls = 0

    async def capture_pane() -> str:
        nonlocal pane_calls
        pane_calls += 1
        return (
            'root@137798b67ea8:/app/output/workspace# (set -- "$?"; '
            'tmux wait -S driftlock-cdb118; exit "$1")\n'
            "root@137798b67ea8:/app/output/workspace# \n"
        )

    outcome = await quiesce_terminal_after_timeout(
        send_keys,
        capture_pane=capture_pane,
    )

    assert outcome.status is CheckpointQuiesceStatus.BOUNDARY_NOT_CHECKPOINTABLE
    assert outcome.checkpointable is False
    assert outcome.reason == (
        "checkpoint marker send raised TimeoutError; marker completion was not "
        "observed; pane capture shows a shell prompt"
    )
    assert pane_calls == 1
    assert send_keys.calls == [
        SendKeysCall(
            ("C-c",),
            {"block": False, "min_timeout_sec": 0.1},
        ),
        SendKeysCall(
            (":", "Enter"),
            {"block": True, "max_timeout_sec": 15.0},
        ),
    ]


@pytest.mark.asyncio
async def test_ambiguous_pane_is_absence_of_evidence_not_terminal_failure() -> None:
    send_keys = FakeSendKeys(timeout_calls=(2,))

    async def capture_pane() -> str:
        return "still rendering output without a recognizable prompt"

    outcome = await quiesce_terminal_after_timeout(
        send_keys,
        capture_pane=capture_pane,
    )

    assert outcome.status is CheckpointQuiesceStatus.BOUNDARY_NOT_CHECKPOINTABLE
    assert outcome.checkpointable is False
    assert outcome.reason == (
        "checkpoint marker send raised TimeoutError; marker completion was not "
        "observed; pane was captured without a recognizable shell prompt; absence "
        "of a prompt was not treated as evidence that the terminal session is gone"
    )


@pytest.mark.asyncio
async def test_positive_tmux_session_absence_raises_named_terminal_error() -> None:
    send_keys = FakeSendKeys(timeout_calls=(2,))
    liveness_calls = 0

    async def capture_pane() -> str:
        return ""

    async def session_is_alive() -> bool:
        nonlocal liveness_calls
        liveness_calls += 1
        return False

    with pytest.raises(DriftlockTerminalUnusableError) as caught:
        await quiesce_terminal_after_timeout(
            send_keys,
            capture_pane=capture_pane,
            session_is_alive=session_is_alive,
        )

    assert type(caught.value) is DriftlockTerminalUnusableError
    assert str(caught.value) == "terminal session is gone after the timed-out command"
    assert "agent" not in str(caught.value)
    assert liveness_calls == 1
    assert send_keys.calls == [
        SendKeysCall(
            ("C-c",),
            {"block": False, "min_timeout_sec": 0.1},
        ),
        SendKeysCall(
            (":", "Enter"),
            {"block": True, "max_timeout_sec": 15.0},
        ),
    ]


@pytest.mark.asyncio
async def test_missing_pane_capability_preserves_episode_with_reason() -> None:
    send_keys = FakeSendKeys(timeout_calls=(2,))

    outcome = await quiesce_terminal_after_timeout(send_keys)

    assert outcome.status is CheckpointQuiesceStatus.BOUNDARY_NOT_CHECKPOINTABLE
    assert outcome.checkpointable is False
    assert outcome.reason == (
        "checkpoint marker send raised TimeoutError; marker completion was not "
        "observed; no pane-capture capability was supplied, so terminal usability "
        "was not established"
    )


@pytest.mark.asyncio
async def test_interrupt_timeout_is_classified_by_policy_without_second_send() -> None:
    send_keys = FakeSendKeys(timeout_calls=(1,))

    async def capture_pane() -> str:
        return "root@container:/app# "

    outcome = await quiesce_terminal_after_timeout(
        send_keys,
        capture_pane=capture_pane,
    )

    assert outcome.status is CheckpointQuiesceStatus.BOUNDARY_NOT_CHECKPOINTABLE
    assert outcome.checkpointable is False
    assert outcome.reason == (
        "interrupt send raised TimeoutError; interrupt delivery was not established; "
        "pane capture shows a shell prompt"
    )
    assert send_keys.calls == [
        SendKeysCall(
            ("C-c",),
            {"block": False, "min_timeout_sec": 0.1},
        )
    ]


@pytest.mark.asyncio
async def test_unrelated_timeout_subclass_does_not_become_terminal_error() -> None:
    class PaneTransportTimeout(TimeoutError):
        pass

    send_keys = FakeSendKeys(timeout_calls=(2,))

    async def capture_pane() -> str:
        raise PaneTransportTimeout("capture transport failed")

    outcome = await quiesce_terminal_after_timeout(
        send_keys,
        capture_pane=capture_pane,
    )

    assert outcome.status is CheckpointQuiesceStatus.BOUNDARY_NOT_CHECKPOINTABLE
    assert outcome.reason == (
        "checkpoint marker send raised TimeoutError; marker completion was not "
        "observed; pane evidence raised PaneTransportTimeout, so no terminal "
        "usability conclusion was drawn"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError(), ValueError("broken")])
async def test_non_timeout_send_keys_errors_propagate(error: BaseException) -> None:
    send_keys = FakeSendKeys(errors={2: error})

    with pytest.raises(type(error)) as caught:
        await quiesce_terminal_after_timeout(send_keys)

    assert caught.value is error


@pytest.mark.asyncio
async def test_cancelled_pane_capture_propagates_unchanged() -> None:
    send_keys = FakeSendKeys(timeout_calls=(2,))
    cancelled = asyncio.CancelledError()

    async def capture_pane() -> str:
        raise cancelled

    with pytest.raises(asyncio.CancelledError) as caught:
        await quiesce_terminal_after_timeout(
            send_keys,
            capture_pane=capture_pane,
        )

    assert caught.value is cancelled


def test_quiesce_status_values_and_deadlines_are_independent_literals() -> None:
    values = [member.value for member in CheckpointQuiesceStatus.__members__.values()]

    assert values == ["recovered", "boundary_not_checkpointable"]
    assert len(values) == len(set(values))
    assert QUIESCE_HANDSHAKE_TIMEOUT_SECONDS == 15.0
    assert PANE_LIVENESS_TIMEOUT_SECONDS == 60.0
    assert 60.0 > 15.0
