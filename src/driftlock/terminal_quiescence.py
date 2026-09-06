"""Recovery policy for a terminal command that timed out at a checkpoint boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

# A tenth of a second lets tmux deliver the interrupt before the shell no-op is
# queued without turning signal delivery itself into another long blocking wait.
INTERRUPT_SETTLE_SECONDS = 0.1

# Fifteen seconds keeps the normal recovery path responsive while allowing the
# completion marker more scheduling headroom than the failed ten-second policy.
QUIESCE_HANDSHAKE_TIMEOUT_SECONDS = 15.0

# A loaded validation host runs four containers concurrently. One minute lets
# pane capture survive host contention without treating its result as a verdict.
PANE_LIVENESS_TIMEOUT_SECONDS = 60.0

# One interrupt targets the timed-out foreground process without risking a second
# signal after the shell has already returned and begun processing the marker.
_INTERRUPT_KEYS = ["C-c"]

# The shell no-op creates a command boundary, and Harbor appends its unique tmux
# completion marker when these keys are sent in blocking mode.
_SHELL_MARKER_KEYS = [":", "Enter"]

SendKeys = Callable[..., Awaitable[Any]]
CapturePane = Callable[[], Awaitable[str]]


class BoundaryReasonTarget(Protocol):
    """Object carrying the per-boundary reason consumed by the LHTB runtime."""

    _driftlock_boundary_uncheckpointable_reason: str | None


@dataclass(frozen=True, slots=True)
class CheckpointQuiesceOutcome:
    """Whether this boundary can be checkpointed, with evidence when it cannot."""

    checkpointable: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.checkpointable, bool):
            raise TypeError("checkpointable must be a boolean")
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("reason must be a string or None")
        if self.checkpointable and self.reason is not None:
            raise ValueError("a checkpointable boundary cannot have a failure reason")
        if not self.checkpointable and not self.reason:
            raise ValueError("an uncheckpointable boundary needs a failure reason")


def record_checkpoint_quiesce_outcome(
    target: BoundaryReasonTarget,
    outcome: CheckpointQuiesceOutcome,
) -> None:
    """Expose the policy result on the agent field read by the boundary runtime."""

    target._driftlock_boundary_uncheckpointable_reason = outcome.reason


async def _uncheckpointable_boundary(
    *,
    capture_pane: CapturePane | None,
    trigger_reason: str,
) -> CheckpointQuiesceOutcome:
    if capture_pane is None:
        return CheckpointQuiesceOutcome(
            checkpointable=False,
            reason=(
                f"{trigger_reason}; no pane-capture capability was supplied, so "
                "terminal usability was not established"
            ),
        )

    try:
        async with asyncio.timeout(PANE_LIVENESS_TIMEOUT_SECONDS):
            pane = await capture_pane()
    except Exception as error:
        return CheckpointQuiesceOutcome(
            checkpointable=False,
            reason=(
                f"{trigger_reason}; pane capture raised {type(error).__name__}, so "
                "terminal usability could not be determined"
            ),
        )

    if not isinstance(pane, str):
        return CheckpointQuiesceOutcome(
            checkpointable=False,
            reason=(
                f"{trigger_reason}; pane capture returned {type(pane).__name__} "
                "instead of text, so terminal usability could not be determined"
            ),
        )
    pane_reason = (
        "pane capture returned non-empty content; terminal usability was not "
        "inferred from pane text"
        if pane
        else (
            "pane capture returned no content; terminal usability could not be "
            "determined"
        )
    )
    return CheckpointQuiesceOutcome(
        checkpointable=False,
        reason=f"{trigger_reason}; {pane_reason}",
    )


async def quiesce_terminal_after_timeout(
    send_keys: SendKeys,
    *,
    capture_pane: CapturePane | None = None,
) -> CheckpointQuiesceOutcome:
    """Report whether a timed-out command reached an observable shell boundary.

    Harbor passes its tmux methods, while tests can supply deterministic async
    callables. Ordinary session failures make only this boundary uncheckpointable;
    terminal lifecycle decisions remain Harbor's responsibility.
    """

    try:
        await send_keys(
            list(_INTERRUPT_KEYS),
            block=False,
            min_timeout_sec=INTERRUPT_SETTLE_SECONDS,
        )
    except Exception as error:
        return await _uncheckpointable_boundary(
            capture_pane=capture_pane,
            trigger_reason=(
                "interrupt send raised "
                f"{type(error).__name__}; interrupt delivery and shell-boundary "
                "recovery could not be established"
            ),
        )

    try:
        await send_keys(
            list(_SHELL_MARKER_KEYS),
            block=True,
            max_timeout_sec=QUIESCE_HANDSHAKE_TIMEOUT_SECONDS,
        )
    except Exception as error:
        # Do not enqueue a second no-op: the first marker may only be late, and pane
        # capture provides independent evidence without leaving two markers pending.
        return await _uncheckpointable_boundary(
            capture_pane=capture_pane,
            trigger_reason=(
                "checkpoint marker send raised "
                f"{type(error).__name__}; marker completion and shell-boundary "
                "recovery could not be established"
            ),
        )
    return CheckpointQuiesceOutcome(
        checkpointable=True,
    )
