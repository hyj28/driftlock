"""The progress-aware checkpoint and rollback control loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from driftlock.checkpoints import CheckpointStore
from driftlock.heuristics import HeuristicJudge
from driftlock.judges import FineJudge
from driftlock.models import (
    Checkpoint,
    DriftContext,
    DriftSignal,
    JudgeVerdict,
    RollbackRecord,
    RunResult,
    RunStatus,
    StepContext,
    StepOutcome,
    StepRecord,
    Verdict,
)

StepFunction = Callable[[StepContext], Awaitable[StepOutcome]]


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Execution and intervention limits."""

    max_steps: int = 100
    max_rollbacks: int = 3
    checkpoint_interval: int = 5
    recent_steps_for_judge: int = 12
    max_tokens: int | None = None
    rollback_on_uncertain: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_rollbacks < 0:
            raise ValueError("max_rollbacks cannot be negative")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.recent_steps_for_judge <= 0:
            raise ValueError("recent_steps_for_judge must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when set")


class DriftlockRunner:
    """Wrap an async agent step function with drift-aware rollback."""

    def __init__(
        self,
        checkpoint_store: CheckpointStore,
        coarse_judge: HeuristicJudge,
        *,
        fine_judge: FineJudge | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        self.checkpoint_store = checkpoint_store
        self.coarse_judge = coarse_judge
        self.fine_judge = fine_judge
        self.config = config or RunnerConfig()

    async def run(
        self,
        *,
        goal: str,
        step: StepFunction,
        initial_state: Mapping[str, Any],
        plan: str = "",
    ) -> RunResult:
        state = dict(initial_state)
        checkpoint = self.checkpoint_store.create(state, step=0, label="initial")
        checkpoints = [checkpoint]
        all_steps: list[StepRecord] = []
        active_steps: list[StepRecord] = []
        rollbacks: list[RollbackRecord] = []
        tokens_used = 0
        logical_step = 0
        attempt = 1
        rollback_feedback: str | None = None

        for sequence in range(1, self.config.max_steps + 1):
            context = StepContext(
                goal=goal,
                plan=plan,
                state=state,
                sequence=sequence,
                logical_step=logical_step + 1,
                attempt=attempt,
                rollback_feedback=rollback_feedback,
            )
            rollback_feedback = None
            try:
                outcome = await step(context)
            except Exception as error:  # Agent failures are trajectory evidence.
                outcome = StepOutcome(
                    action="<agent step raised>",
                    state=state,
                    error=f"{type(error).__name__}: {error}",
                )
            state = dict(outcome.state)
            tokens_used += outcome.tokens
            logical_step += 1
            record = StepRecord(
                sequence=sequence,
                logical_step=logical_step,
                attempt=attempt,
                outcome=outcome,
            )
            all_steps.append(record)
            active_steps.append(record)

            over_token_budget = (
                self.config.max_tokens is not None
                and tokens_used > self.config.max_tokens
            )
            if outcome.completed and not over_token_budget:
                return self._result(
                    RunStatus.COMPLETED,
                    state,
                    all_steps,
                    rollbacks,
                    checkpoints,
                    tokens_used,
                )
            if (
                self.config.max_tokens is not None
                and tokens_used >= self.config.max_tokens
            ):
                return self._result(
                    RunStatus.TOKEN_LIMIT,
                    state,
                    all_steps,
                    rollbacks,
                    checkpoints,
                    tokens_used,
                )

            signals = self.coarse_judge.evaluate(active_steps)
            if signals:
                verdict = await self._judge(
                    goal=goal,
                    plan=plan,
                    checkpoint=checkpoint,
                    signals=signals,
                    active_steps=active_steps,
                )
                should_rollback = verdict.verdict is Verdict.DRIFTED or (
                    verdict.verdict is Verdict.UNCERTAIN
                    and self.config.rollback_on_uncertain
                )
                if should_rollback:
                    if len(rollbacks) >= self.config.max_rollbacks:
                        return self._result(
                            RunStatus.ROLLBACK_LIMIT,
                            state,
                            all_steps,
                            rollbacks,
                            checkpoints,
                            tokens_used,
                        )
                    state = self.checkpoint_store.restore(checkpoint)
                    rollbacks.append(
                        RollbackRecord(
                            sequence=sequence,
                            checkpoint_id=checkpoint.checkpoint_id,
                            signals=signals,
                            reason=verdict.reason,
                        )
                    )
                    logical_step = checkpoint.step
                    attempt += 1
                    active_steps.clear()
                    rollback_feedback = verdict.reason
                    continue

            if logical_step - checkpoint.step >= self.config.checkpoint_interval:
                checkpoint = self.checkpoint_store.create(
                    state,
                    step=logical_step,
                    parent_id=checkpoint.checkpoint_id,
                    label="healthy",
                )
                checkpoints.append(checkpoint)
                active_steps.clear()

        return self._result(
            RunStatus.STEP_LIMIT,
            state,
            all_steps,
            rollbacks,
            checkpoints,
            tokens_used,
        )

    async def _judge(
        self,
        *,
        goal: str,
        plan: str,
        checkpoint: Checkpoint,
        signals: tuple[DriftSignal, ...],
        active_steps: list[StepRecord],
    ) -> JudgeVerdict:
        if self.fine_judge is None:
            return JudgeVerdict(
                Verdict.DRIFTED,
                "coarse heuristic fired in heuristics-only mode",
            )
        recent = tuple(active_steps[-self.config.recent_steps_for_judge :])
        return await self.fine_judge.judge(
            DriftContext(
                goal=goal,
                plan=plan,
                checkpoint=checkpoint,
                signals=signals,
                recent_steps=recent,
                diff=recent[-1].outcome.diff if recent else "",
            )
        )

    @staticmethod
    def _result(
        status: RunStatus,
        state: Mapping[str, Any],
        steps: list[StepRecord],
        rollbacks: list[RollbackRecord],
        checkpoints: list[Checkpoint],
        tokens_used: int,
    ) -> RunResult:
        return RunResult(
            status=status,
            state=dict(state),
            steps=tuple(steps),
            rollbacks=tuple(rollbacks),
            checkpoints=tuple(checkpoints),
            tokens_used=tokens_used,
        )
