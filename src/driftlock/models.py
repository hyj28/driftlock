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


class FineJudgeStatus(StrEnum):
    """Whether a coarse trigger was evaluated by a configured fine judge."""

    VERDICT = "verdict"
    NOT_CONFIGURED = "not_configured"


class DriftTriggerOutcome(StrEnum):
    """What happened after the coarse detector fired."""

    ROLLED_BACK = "rolled_back"
    VETOED = "vetoed"
    ROLLBACK_LIMIT_REFUSED = "rollback_limit_refused"


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
    commands_run: int = 0
    commands_failed: int = 0
    tool_observations: tuple[str, ...] = ()
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
        for name, value in (
            ("commands_run", self.commands_run),
            ("commands_failed", self.commands_failed),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.commands_failed > self.commands_run:
            raise ValueError("commands_failed cannot exceed commands_run")
        if not isinstance(self.tool_observations, tuple) or any(
            not isinstance(observation, str) for observation in self.tool_observations
        ):
            raise TypeError("tool_observations must be a tuple of strings")


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
    tool_observations: tuple[str, ...] = ()


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
class DriftTriggerRecord:
    """A coarse trigger and the complete disposition of that intervention attempt."""

    sequence: int
    logical_step: int
    signals: tuple[DriftSignal, ...]
    judge_status: FineJudgeStatus
    judge_verdict: Verdict | None
    judge_reason: str | None
    outcome: DriftTriggerOutcome
    rollback_checkpoint_id: str | None = None
    rollback_checkpoint_step: int | None = None

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("sequence must be positive")
        if self.logical_step <= 0:
            raise ValueError("logical_step must be positive")
        if not self.signals:
            raise ValueError("a trigger record must contain at least one signal")
        if self.judge_status is FineJudgeStatus.NOT_CONFIGURED:
            if self.judge_verdict is not None or self.judge_reason is not None:
                raise ValueError("an unconfigured fine judge cannot have a verdict")
        elif self.judge_verdict is None or self.judge_reason is None:
            raise ValueError("a configured fine judge must have a verdict and reason")
        has_checkpoint_id = self.rollback_checkpoint_id is not None
        has_checkpoint_step = self.rollback_checkpoint_step is not None
        if has_checkpoint_id != has_checkpoint_step:
            raise ValueError(
                "rollback checkpoint id and step must be recorded together"
            )
        if self.outcome is DriftTriggerOutcome.ROLLED_BACK and not has_checkpoint_id:
            raise ValueError("a rollback must identify its target checkpoint")
        if self.outcome is not DriftTriggerOutcome.ROLLED_BACK and has_checkpoint_id:
            raise ValueError(
                "only a completed rollback may identify a target checkpoint"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by phase audit records."""
        checkpoint = None
        if self.rollback_checkpoint_id is not None:
            checkpoint = {
                "checkpoint_id": self.rollback_checkpoint_id,
                "step": self.rollback_checkpoint_step,
            }
        return {
            "sequence": self.sequence,
            "logical_step": self.logical_step,
            "signals": [
                {
                    "kind": signal.kind,
                    "detail": signal.detail,
                    "lookback": signal.lookback,
                }
                for signal in self.signals
            ],
            "judge": {
                "status": self.judge_status.value,
                "verdict": (
                    self.judge_verdict.value if self.judge_verdict is not None else None
                ),
                "reason": self.judge_reason,
            },
            "outcome": self.outcome.value,
            "rollback_checkpoint": checkpoint,
        }


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
    coarse_triggers: tuple[DriftTriggerRecord, ...] = ()

    @property
    def signal_counts(self) -> dict[str, dict[str, int]]:
        """Count signal occurrences by kind and upheld/vetoed disposition."""
        counts: dict[str, dict[str, int]] = {}
        for trigger in self.coarse_triggers:
            disposition = (
                "vetoed" if trigger.outcome is DriftTriggerOutcome.VETOED else "upheld"
            )
            for signal in trigger.signals:
                kind_counts = counts.setdefault(signal.kind, {"upheld": 0, "vetoed": 0})
                kind_counts[disposition] += 1
        return {kind: counts[kind] for kind in sorted(counts)}
