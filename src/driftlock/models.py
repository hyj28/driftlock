"""Public data models used by the driftlock runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class Verdict(StrEnum):
    """The fine judge's assessment of the current trajectory."""

    HEALTHY = "healthy"
    DRIFTED = "drifted"
    UNCERTAIN = "uncertain"


class RunStatus(StrEnum):
    """Why a runner invocation stopped."""

    COMPLETED = "completed"
    STEP_LIMIT = "step_limit"
    TOKEN_LIMIT = "token_limit"
    ROLLBACK_LIMIT = "rollback_limit"


class StepTokenBudgetExhausted(RuntimeError):
    """Raised before a provider call when no safe output-token allowance remains."""


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A durable workspace and agent-state snapshot."""

    checkpoint_id: str
    step: int
    created_at: datetime
    digest: str
    path: Path
    parent_id: str | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class DriftSignal:
    """A cheap heuristic indicating that a trajectory may have drifted."""

    kind: str
    detail: str
    lookback: int = 1

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """The observable result of one agent step.

    ``state`` must be JSON-serializable because it is stored with checkpoints.
    ``changed_paths`` and ``diff`` should describe only the just-finished step.
    """

    action: str
    state: Mapping[str, Any]
    changed_paths: tuple[str, ...] = ()
    diff: str = ""
    workspace_delta_observed: bool = True
    workspace_observation_error: str | None = None
    error: str | None = None
    reward: float | None = None
    tokens: int = 0
    completed: bool = False
    summary: str = ""

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens cannot be negative")
        if not isinstance(self.workspace_delta_observed, bool):
            raise TypeError("workspace_delta_observed must be a boolean")
        if self.workspace_observation_error is not None and not isinstance(
            self.workspace_observation_error, str
        ):
            raise TypeError("workspace_observation_error must be a string or None")


@dataclass(frozen=True, slots=True)
class StepRecord:
    """A step plus its physical and logical position in the run."""

    sequence: int
    logical_step: int
    attempt: int
    outcome: StepOutcome
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class StepContext:
    """Context supplied to the wrapped agent for its next step."""

    goal: str
    plan: str
    state: Mapping[str, Any]
    sequence: int
    logical_step: int
    attempt: int
    rollback_feedback: str | None
    tokens_remaining: int | None


@dataclass(frozen=True, slots=True)
class DriftContext:
    """Evidence supplied to the fine semantic judge."""

    goal: str
    plan: str
    checkpoint: Checkpoint
    signals: tuple[DriftSignal, ...]
    recent_steps: tuple[StepRecord, ...]
    diff: str
    tokens_remaining: int | None


@dataclass(frozen=True, slots=True)
class JudgeCompletion:
    """Raw fine-judge text together with its billed token usage."""

    text: str
    tokens: int = 0

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens cannot be negative")


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Structured fine-judge response."""

    verdict: Verdict
    reason: str
    confidence: float = 1.0
    tokens: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.tokens < 0:
            raise ValueError("tokens cannot be negative")


@dataclass(frozen=True, slots=True)
class RollbackRecord:
    """An intervention made by the runner."""

    sequence: int
    checkpoint_id: str
    signals: tuple[DriftSignal, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class RunResult:
    """Final state and accounting for a runner invocation."""

    status: RunStatus
    state: Mapping[str, Any]
    steps: tuple[StepRecord, ...]
    rollbacks: tuple[RollbackRecord, ...]
    checkpoints: tuple[Checkpoint, ...]
    tokens_used: int
    agent_tokens_used: int
    judge_tokens_used: int
