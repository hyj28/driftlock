from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest

from driftlock.terminal_quiescence import (
    PANE_LIVENESS_TIMEOUT_SECONDS,
    QUIESCE_HANDSHAKE_TIMEOUT_SECONDS,
    CheckpointQuiesceOutcome,
    quiesce_terminal_after_timeout,
    record_checkpoint_quiesce_outcome,
)


@dataclass(frozen=True, slots=True)
class SendKeysCall:
    keys: tuple[str, ...]
    options: dict[str, Any]


class FakeSendKeys:
    def __init__(self, errors: dict[int, BaseException] | None = None) -> None:
        self.errors = errors or {}
        self.calls: list[SendKeysCall] = []

    async def __call__(self, keys: list[str], **options: Any) -> None:
        self.calls.append(SendKeysCall(tuple(keys), options))
        error = self.errors.get(len(self.calls))
        if error is not None:
            raise error


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

    assert outcome == CheckpointQuiesceOutcome(checkpointable=True)
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
async def test_alive_but_slow_shell_prompt_never_aborts_paid_episode() -> None:
    send_keys = FakeSendKeys({2: TimeoutError("marker never came back")})

    async def capture_pane() -> str:
        return (
            'root@137798b67ea8:/app/output/workspace# (set -- "$?"; '
            'tmux wait -S driftlock-cdb118; exit "$1")\n'
            "root@137798b67ea8:/app/output/workspace# \n"
        )

    outcome = await quiesce_terminal_after_timeout(
        send_keys,
        capture_pane=capture_pane,
    )

    assert outcome == CheckpointQuiesceOutcome(
        checkpointable=False,
        reason=(
            "checkpoint marker send raised TimeoutError; marker completion and "
            "shell-boundary recovery could not be established; pane capture "
            "returned non-empty content; terminal usability was not inferred from "
            "pane text"
        ),
    )
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
async def test_marker_runtime_error_is_an_uncheckpointable_boundary() -> None:
    send_keys = FakeSendKeys({2: RuntimeError("tmux channel failed")})

    async def capture_pane() -> str:
        return "still rendering output"

    outcome = await quiesce_terminal_after_timeout(
        send_keys,
        capture_pane=capture_pane,
    )

    assert outcome.checkpointable is False
    assert outcome.reason == (
        "checkpoint marker send raised RuntimeError; marker completion and "
        "shell-boundary recovery could not be established; pane capture returned "
        "non-empty content; terminal usability was not inferred from pane text"
    )
    assert len(send_keys.calls) == 2


@pytest.mark.asyncio
async def test_interrupt_runtime_error_is_handled_without_a_second_send() -> None:
    send_keys = FakeSendKeys(
        {1: RuntimeError("trial-7: failed to send non-blocking keys")}
    )

    async def capture_pane() -> str:
        return "root@container:/app# "

    outcome = await quiesce_terminal_after_timeout(
        send_keys,
        capture_pane=capture_pane,
    )

    assert outcome.checkpointable is False
    assert outcome.reason == (
        "interrupt send raised RuntimeError; interrupt delivery and shell-boundary "
        "recovery could not be established; pane capture returned non-empty content; "
        "terminal usability was not inferred from pane text"
    )
    assert send_keys.calls == [
        SendKeysCall(
            ("C-c",),
            {"block": False, "min_timeout_sec": 0.1},
        )
    ]


@pytest.mark.asyncio
async def test_empty_pane_does_not_claim_that_terminal_is_dead() -> None:
    send_keys = FakeSendKeys({2: TimeoutError("marker wait timed out")})

    async def capture_pane() -> str:
        return ""

    outcome = await quiesce_terminal_after_timeout(
        send_keys,
        capture_pane=capture_pane,
    )

    assert outcome.checkpointable is False
    assert outcome.reason == (
        "checkpoint marker send raised TimeoutError; marker completion and "
        "shell-boundary recovery could not be established; pane capture returned no "
        "content; terminal usability could not be determined"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capture_error",
    [TimeoutError("capture timed out"), RuntimeError("docker exec failed")],
)
async def test_pane_capture_failure_records_uncertainty(
    capture_error: Exception,
) -> None:
    send_keys = FakeSendKeys({2: TimeoutError("marker wait timed out")})

    async def capture_pane() -> str:
        raise capture_error

    outcome = await quiesce_terminal_after_timeout(
        send_keys,
        capture_pane=capture_pane,
    )

    assert outcome.checkpointable is False
    assert outcome.reason == (
        "checkpoint marker send raised TimeoutError; marker completion and "
        f"shell-boundary recovery could not be established; pane capture raised "
        f"{type(capture_error).__name__}, so terminal usability could not be "
        "determined"
    )


@pytest.mark.asyncio
async def test_missing_pane_capability_records_uncertainty() -> None:
    send_keys = FakeSendKeys({2: TimeoutError("marker wait timed out")})

    outcome = await quiesce_terminal_after_timeout(send_keys)

    assert outcome.checkpointable is False
    assert outcome.reason == (
        "checkpoint marker send raised TimeoutError; marker completion and "
        "shell-boundary recovery could not be established; no pane-capture capability "
        "was supplied, so terminal usability was not established"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError(), KeyboardInterrupt()])
async def test_process_control_send_errors_propagate(error: BaseException) -> None:
    send_keys = FakeSendKeys({1: error})

    with pytest.raises(type(error)) as caught:
        await quiesce_terminal_after_timeout(send_keys)

    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError(), KeyboardInterrupt()])
async def test_process_control_pane_errors_propagate(error: BaseException) -> None:
    send_keys = FakeSendKeys({2: TimeoutError("marker wait timed out")})

    async def capture_pane() -> str:
        raise error

    with pytest.raises(type(error)) as caught:
        await quiesce_terminal_after_timeout(
            send_keys,
            capture_pane=capture_pane,
        )

    assert caught.value is error


def test_outcome_recording_sets_and_clears_runtime_boundary_field() -> None:
    class Target:
        _driftlock_boundary_uncheckpointable_reason: str | None = "stale reason"

    target = Target()
    missed = CheckpointQuiesceOutcome(
        checkpointable=False,
        reason="marker completion could not be established",
    )

    record_checkpoint_quiesce_outcome(target, missed)

    assert target._driftlock_boundary_uncheckpointable_reason == (
        "marker completion could not be established"
    )

    record_checkpoint_quiesce_outcome(
        target,
        CheckpointQuiesceOutcome(checkpointable=True),
    )

    assert target._driftlock_boundary_uncheckpointable_reason is None


@pytest.mark.parametrize(
    "values",
    [
        {"checkpointable": True, "reason": "failure"},
        {"checkpointable": False},
    ],
)
def test_outcome_rejects_disagreeing_fields(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CheckpointQuiesceOutcome(**values)


def test_quiesce_deadlines_are_independent_literals() -> None:
    assert QUIESCE_HANDSHAKE_TIMEOUT_SECONDS == 15.0
    assert PANE_LIVENESS_TIMEOUT_SECONDS == 60.0
    assert 60.0 > 15.0
