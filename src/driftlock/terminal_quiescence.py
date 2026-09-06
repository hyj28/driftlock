"""Recovery policy for a terminal command that timed out at a checkpoint boundary."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# A tenth of a second lets tmux deliver the interrupt before the shell no-op is
# queued without turning signal delivery itself into another long blocking wait.
INTERRUPT_SETTLE_SECONDS = 0.1

# Fifteen seconds keeps the normal recovery path responsive while allowing the
# completion marker more scheduling headroom than the failed ten-second policy.
QUIESCE_HANDSHAKE_TIMEOUT_SECONDS = 15.0

# A loaded validation host runs four containers concurrently. One minute gives
# pane capture and the fallback tmux session check time to survive host contention.
PANE_LIVENESS_TIMEOUT_SECONDS = 60.0

# One interrupt targets the timed-out foreground process without risking a second
# signal after the shell has already returned and begun processing the marker.
_INTERRUPT_KEYS = ["C-c"]

# The shell no-op creates a command boundary, and Harbor appends its unique tmux
# completion marker when these keys are sent in blocking mode.
_SHELL_MARKER_KEYS = [":", "Enter"]

# Shell prompts vary, but the observed LHTB prompt and conventional interactive
# Bash prompts end a line with ``#`` or ``$`` followed only by optional whitespace.
_SHELL_PROMPT_PATTERN = re.compile(r"(?m)^[^\r\n]*[#$][ \t]*$")

# This wording records uncertainty about marker completion without claiming that
# an arbitrary TimeoutError proves the marker itself failed to reach the shell.
_MARKER_TIMEOUT_REASON = (
    "checkpoint marker send raised TimeoutError; marker completion was not observed"
)

SendKeys = Callable[..., Awaitable[Any]]
CapturePane = Callable[[], Awaitable[str]]
SessionIsAlive = Callable[[], Awaitable[bool]]


class CheckpointQuiesceStatus(StrEnum):
    """What terminal recovery established after a command timeout."""

    RECOVERED = "recovered"
    BOUNDARY_NOT_CHECKPOINTABLE = "boundary_not_checkpointable"


@dataclass(frozen=True, slots=True)
class CheckpointQuiesceOutcome:
    """Checkpoint availability established without ending the episode."""

    status: CheckpointQuiesceStatus
    checkpointable: bool
    reason: str | None = None


class DriftlockTerminalUnusableError(RuntimeError):
    """Raised when tmux positively reports that the terminal session is gone."""


async def _uncheckpointable_boundary(
    *,
    capture_pane: CapturePane | None,
    session_is_alive: SessionIsAlive | None,
    trigger_reason: str,
) -> CheckpointQuiesceOutcome:
    if capture_pane is None:
        return CheckpointQuiesceOutcome(
            status=CheckpointQuiesceStatus.BOUNDARY_NOT_CHECKPOINTABLE,
            checkpointable=False,
            reason=(
                f"{trigger_reason}; no pane-capture capability was supplied, so "
                "terminal usability was not established"
            ),
        )

    try:
        async with asyncio.timeout(PANE_LIVENESS_TIMEOUT_SECONDS):
            pane = await capture_pane()
            if not isinstance(pane, str):
                raise TypeError("pane capture must return a string")
            if not pane and session_is_alive is not None:
                alive = await session_is_alive()
                if not isinstance(alive, bool):
                    raise TypeError("session liveness check must return a boolean")
                if not alive:
                    raise DriftlockTerminalUnusableError(
                        "terminal session is gone after the timed-out command"
                    )
    except TimeoutError as error:
        return CheckpointQuiesceOutcome(
            status=CheckpointQuiesceStatus.BOUNDARY_NOT_CHECKPOINTABLE,
            checkpointable=False,
            reason=(
                f"{trigger_reason}; pane evidence raised "
                f"{type(error).__name__}, so no terminal usability conclusion "
                "was drawn"
            ),
        )

    pane_reason = (
        "pane capture shows a shell prompt"
        if _SHELL_PROMPT_PATTERN.search(pane)
        else (
            "pane was captured without a recognizable shell prompt; absence of a "
            "prompt was not treated as evidence that the terminal session is gone"
        )
    )
    return CheckpointQuiesceOutcome(
        status=CheckpointQuiesceStatus.BOUNDARY_NOT_CHECKPOINTABLE,
        checkpointable=False,
        reason=f"{trigger_reason}; {pane_reason}",
    )


async def quiesce_terminal_after_timeout(
    send_keys: SendKeys,
    *,
    capture_pane: CapturePane | None = None,
    session_is_alive: SessionIsAlive | None = None,
) -> CheckpointQuiesceOutcome:
    """Interrupt a timed-out command and inspect tmux before declaring it unusable.

    Harbor passes its tmux methods, while tests can supply deterministic async
    callables. A missed send deadline is never treated as proof that the terminal
    is dead; only an explicit negative ``session_is_alive`` result is conclusive.
    """

    try:
        await send_keys(
            list(_INTERRUPT_KEYS),
            block=False,
            min_timeout_sec=INTERRUPT_SETTLE_SECONDS,
        )
    except TimeoutError as error:
        return await _uncheckpointable_boundary(
            capture_pane=capture_pane,
            session_is_alive=session_is_alive,
            trigger_reason=(
                "interrupt send raised "
                f"{type(error).__name__}; interrupt delivery was not established"
            ),
        )

    try:
        await send_keys(
            list(_SHELL_MARKER_KEYS),
            block=True,
            max_timeout_sec=QUIESCE_HANDSHAKE_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        # Do not enqueue a second no-op: the first marker may only be late, and pane
        # capture provides independent evidence without leaving two markers pending.
        return await _uncheckpointable_boundary(
            capture_pane=capture_pane,
            session_is_alive=session_is_alive,
            trigger_reason=_MARKER_TIMEOUT_REASON,
        )
    return CheckpointQuiesceOutcome(
        status=CheckpointQuiesceStatus.RECOVERED,
        checkpointable=True,
    )
