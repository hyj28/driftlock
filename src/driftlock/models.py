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


class JudgeReliabilityStatus(StrEnum):
    """Whether fine-judge availability made the run a valid measurement."""

    NOT_ASSESSED = "not_assessed"
    RELIABLE = "reliable"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class FineJudgeStatus(StrEnum):
    """Whether a coarse trigger was evaluated by a configured fine judge."""

    VERDICT = "verdict"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NOT_CONFIGURED = "not_configured"
    NOT_INVOKED = "not_invoked"


class DriftTriggerOutcome(StrEnum):
    """What happened after the coarse detector fired."""

    ROLLED_BACK = "rolled_back"
    VETOED = "vetoed"
    ROLLBACK_LIMIT_REFUSED = "rollback_limit_refused"
    SUPPRESSED = "suppressed"
    JUDGE_FAILED = "judge_failed"
    JUDGE_BUDGET_EXHAUSTED = "judge_budget_exhausted"


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
    unstable_paths: tuple[str, ...] = ()


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

    verdict: Verdict | None
    reason: str
    confidence: float = 1.0
    tokens: int = 0
    status: FineJudgeStatus = FineJudgeStatus.VERDICT

    def __post_init__(self) -> None:
        if self.status not in {
            FineJudgeStatus.VERDICT,
            FineJudgeStatus.FAILED,
            FineJudgeStatus.BUDGET_EXHAUSTED,
        }:
            raise ValueError("a judge result must describe an invoked fine judge")
        if (self.verdict is not None) != (self.status is FineJudgeStatus.VERDICT):
            raise ValueError("only an adjudicated fine judge may return a verdict")
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
        if self.judge_status is FineJudgeStatus.VERDICT:
            if self.judge_verdict is None or self.judge_reason is None:
                raise ValueError(
                    "an adjudicated fine judge must have a verdict and reason"
                )
        elif self.judge_status in {
            FineJudgeStatus.FAILED,
            FineJudgeStatus.BUDGET_EXHAUSTED,
        }:
            if self.judge_verdict is not None or self.judge_reason is None:
                raise ValueError(
                    "a failed fine judge must have a reason but no verdict"
                )
        elif self.judge_verdict is not None or self.judge_reason is not None:
            raise ValueError("a fine judge that did not run cannot have a verdict")
        if self.judge_status is FineJudgeStatus.VERDICT and (
            self.outcome
            in {
                DriftTriggerOutcome.JUDGE_FAILED,
                DriftTriggerOutcome.JUDGE_BUDGET_EXHAUSTED,
            }
        ):
            raise ValueError(
                "an adjudicated trigger cannot have a judge-failure outcome"
            )
        expected_failure_outcome = {
            FineJudgeStatus.FAILED: DriftTriggerOutcome.JUDGE_FAILED,
            FineJudgeStatus.BUDGET_EXHAUSTED: (
                DriftTriggerOutcome.JUDGE_BUDGET_EXHAUSTED
            ),
        }.get(self.judge_status)
        if (
            expected_failure_outcome is not None
            and self.outcome is not expected_failure_outcome
        ):
            raise ValueError("a judge failure must have its matching trigger outcome")
        if expected_failure_outcome is None and self.outcome in {
            DriftTriggerOutcome.JUDGE_FAILED,
            DriftTriggerOutcome.JUDGE_BUDGET_EXHAUSTED,
        }:
            raise ValueError("a judge-failure outcome requires a matching judge status")
        suppressed = self.outcome is DriftTriggerOutcome.SUPPRESSED
        not_invoked = self.judge_status is FineJudgeStatus.NOT_INVOKED
        if suppressed != not_invoked:
            raise ValueError(
                "a suppressed trigger is exactly one the fine judge never saw"
            )
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
    def judge_attempts(self) -> int:
        """Count coarse triggers escalated to a configured fine judge."""
        return sum(
            trigger.judge_status
            in {
                FineJudgeStatus.VERDICT,
                FineJudgeStatus.FAILED,
                FineJudgeStatus.BUDGET_EXHAUSTED,
            }
            for trigger in self.coarse_triggers
        )

    @property
    def judge_failures(self) -> int:
        """Count escalations that produced no fine-judge verdict."""
        return sum(
            trigger.judge_status
            in {FineJudgeStatus.FAILED, FineJudgeStatus.BUDGET_EXHAUSTED}
            for trigger in self.coarse_triggers
        )

    @property
    def judge_reliability(self) -> JudgeReliabilityStatus:
        """Classify the fine judge independently of why the run stopped."""
        return classify_judge_reliability(
            attempts=self.judge_attempts,
            failures=self.judge_failures,
        )

    @property
    def signal_counts(self) -> dict[str, dict[str, int]]:
        """Count signal occurrences by kind and disposition.

        ``suppressed`` counts triggers the coarse tier raised but never escalated,
        because every signal in them was corroborating-only. They are separate from
        ``vetoed`` because they cost no judge tokens and carry no verdict. Judge
        failures and output-budget exhaustion have separate buckets so neither can
        be mistaken for an adjudicated veto.
        """
        counts: dict[str, dict[str, int]] = {}
        dispositions = {
            DriftTriggerOutcome.VETOED: "vetoed",
            DriftTriggerOutcome.SUPPRESSED: "suppressed",
            DriftTriggerOutcome.JUDGE_FAILED: "judge_failed",
            DriftTriggerOutcome.JUDGE_BUDGET_EXHAUSTED: ("judge_budget_exhausted"),
        }
        for trigger in self.coarse_triggers:
            disposition = dispositions.get(trigger.outcome, "upheld")
            for signal in trigger.signals:
                kind_counts = counts.setdefault(
                    signal.kind,
                    {
                        "upheld": 0,
                        "vetoed": 0,
                        "suppressed": 0,
                        "judge_failed": 0,
                        "judge_budget_exhausted": 0,
                    },
                )
                kind_counts[disposition] += 1
        return {kind: counts[kind] for kind in sorted(counts)}


def classify_judge_reliability(
    *, attempts: int, failures: int
) -> JudgeReliabilityStatus:
    """Classify whether fine-judge failures invalidate a measurement."""
    if attempts < 0 or not 0 <= failures <= attempts:
        raise ValueError("judge failures must be between zero and attempts")
    if attempts == 0:
        return JudgeReliabilityStatus.NOT_ASSESSED
    if failures == 0:
        return JudgeReliabilityStatus.RELIABLE
    # Four calls is the first sample where one transient failure and repeated
    # failure are cleanly separated: 1/4 remains valid, while 3/4 means the
    # treatment adjudicated at most one quarter of escalations. The 75% boundary
    # rejects both observed broken rounds (117/137 and 33/34). An all-failed
    # smaller sample is explicitly inconclusive instead of guessed clean or dead.
    if attempts < 4:
        return (
            JudgeReliabilityStatus.INCONCLUSIVE
            if failures == attempts
            else JudgeReliabilityStatus.RELIABLE
        )
    if failures * 4 >= attempts * 3:
        return JudgeReliabilityStatus.FAILED
    return JudgeReliabilityStatus.RELIABLE


def merge_signal_counts(
    *summaries: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    """Add per-phase signal counts into one stable trial-level summary."""
    dispositions = (
        "upheld",
        "vetoed",
        "suppressed",
        "judge_failed",
        "judge_budget_exhausted",
    )
    merged: dict[str, dict[str, int]] = {}
    for summary in summaries:
        for kind, counts in summary.items():
            kind_counts = merged.setdefault(
                kind, {disposition: 0 for disposition in dispositions}
            )
            for disposition in dispositions:
                kind_counts[disposition] += counts.get(disposition, 0)
    return {kind: merged[kind] for kind in sorted(merged)}


def aggregate_run_summary(
    previous: Mapping[str, Any] | None, result: RunResult
) -> dict[str, Any]:
    """Merge one phase result into cumulative trial-level driftlock metadata."""
    prior = previous or {}

    def accumulated(name: str, current: int) -> int:
        value = prior.get(name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"prior driftlock {name} must be a non-negative integer")
        return value + current

    prior_signal_counts = prior.get("signal_counts", {})
    if not isinstance(prior_signal_counts, Mapping):
        raise ValueError("prior driftlock signal_counts must be a mapping")
    judge_attempts = accumulated("judge_attempts", result.judge_attempts)
    judge_failures = accumulated("judge_failures", result.judge_failures)
    return {
        "status": result.status.value,
        "judge_reliability": classify_judge_reliability(
            attempts=judge_attempts,
            failures=judge_failures,
        ).value,
        "judge_attempts": judge_attempts,
        "judge_failures": judge_failures,
        "steps": accumulated("steps", len(result.steps)),
        "rollbacks": accumulated("rollbacks", len(result.rollbacks)),
        "tokens_used": accumulated("tokens_used", result.tokens_used),
        "agent_tokens_used": accumulated("agent_tokens_used", result.agent_tokens_used),
        "judge_tokens_used": accumulated("judge_tokens_used", result.judge_tokens_used),
        "signal_counts": merge_signal_counts(
            prior_signal_counts,
            result.signal_counts,
        ),
    }
