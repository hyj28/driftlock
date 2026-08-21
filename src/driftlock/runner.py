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
    StepTokenBudgetExhausted,
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
    checkpoint_on_exit: bool = False

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
        checkpoint = await self._create_checkpoint(state, step=0, label="initial")
        checkpoints = [checkpoint]
        checkpoint_lineage = [checkpoint]
        checkpoint_histories: dict[str, list[StepRecord]] = {
            checkpoint.checkpoint_id: []
        }
        all_steps: list[StepRecord] = []
        recent_steps: list[StepRecord] = []
        rollbacks: list[RollbackRecord] = []
        agent_tokens_used = 0
        judge_tokens_used = 0
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
                tokens_remaining=self._tokens_remaining(
                    agent_tokens_used + judge_tokens_used
                ),
            )
            rollback_feedback = None
            try:
                outcome = await step(context)
            except StepTokenBudgetExhausted:
                return await self._finish(
                    RunStatus.TOKEN_LIMIT,
                    state,
                    all_steps,
                    rollbacks,
                    checkpoints,
                    agent_tokens_used,
                    judge_tokens_used,
                    current_checkpoint=checkpoint,
                    logical_step=logical_step,
                )
            state = dict(outcome.state)
            agent_tokens_used += outcome.tokens
            logical_step += 1
            record = StepRecord(
                sequence=sequence,
                logical_step=logical_step,
                attempt=attempt,
                outcome=outcome,
            )
            all_steps.append(record)
            recent_steps.append(record)
            history_limit = max(
                self.coarse_judge.history_window,
                self.config.recent_steps_for_judge,
            )
            recent_steps = recent_steps[-history_limit:]

            tokens_used = agent_tokens_used + judge_tokens_used
            over_token_budget = (
                self.config.max_tokens is not None
                and tokens_used > self.config.max_tokens
            )
            if outcome.completed and not over_token_budget:
                return await self._finish(
                    RunStatus.COMPLETED,
                    state,
                    all_steps,
                    rollbacks,
                    checkpoints,
                    agent_tokens_used,
                    judge_tokens_used,
                    current_checkpoint=checkpoint,
                    logical_step=logical_step,
                )
            if (
                self.config.max_tokens is not None
                and tokens_used >= self.config.max_tokens
            ):
                return await self._finish(
                    RunStatus.TOKEN_LIMIT,
                    state,
                    all_steps,
                    rollbacks,
                    checkpoints,
                    agent_tokens_used,
                    judge_tokens_used,
                    current_checkpoint=checkpoint,
                    logical_step=logical_step,
                )

            signals = self.coarse_judge.evaluate(recent_steps)
            checkpoint_is_healthy = not signals
            if signals:
                rollback_checkpoint = self._select_rollback_checkpoint(
                    checkpoint_lineage,
                    signals,
                    logical_step,
                )
                verdict = await self._judge(
                    goal=goal,
                    plan=plan,
                    checkpoint=rollback_checkpoint,
                    signals=signals,
                    recent_steps=recent_steps,
                    tokens_used=agent_tokens_used + judge_tokens_used,
                )
                judge_tokens_used += verdict.tokens
                should_rollback = verdict.verdict is Verdict.DRIFTED or (
                    verdict.verdict is Verdict.UNCERTAIN
                    and self.config.rollback_on_uncertain
                )
                if should_rollback:
                    if len(rollbacks) >= self.config.max_rollbacks:
                        return await self._finish(
                            RunStatus.ROLLBACK_LIMIT,
                            state,
                            all_steps,
                            rollbacks,
                            checkpoints,
                            agent_tokens_used,
                            judge_tokens_used,
                            current_checkpoint=checkpoint,
                            logical_step=logical_step,
                        )
                    checkpoint = rollback_checkpoint
                    state = await self._restore_checkpoint(checkpoint)
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
                    checkpoint_index = checkpoint_lineage.index(checkpoint)
                    checkpoint_lineage = checkpoint_lineage[: checkpoint_index + 1]
                    recent_steps = list(checkpoint_histories[checkpoint.checkpoint_id])
                    rollback_feedback = verdict.reason
                    if self._budget_exhausted(agent_tokens_used + judge_tokens_used):
                        return await self._finish(
                            RunStatus.TOKEN_LIMIT,
                            state,
                            all_steps,
                            rollbacks,
                            checkpoints,
                            agent_tokens_used,
                            judge_tokens_used,
                            current_checkpoint=checkpoint,
                            logical_step=logical_step,
                        )
                    continue
                checkpoint_is_healthy = verdict.verdict is Verdict.HEALTHY

            if self._budget_exhausted(agent_tokens_used + judge_tokens_used):
                return await self._finish(
                    RunStatus.TOKEN_LIMIT,
                    state,
                    all_steps,
                    rollbacks,
                    checkpoints,
                    agent_tokens_used,
                    judge_tokens_used,
                    current_checkpoint=checkpoint,
                    logical_step=logical_step,
                )

            if (
                checkpoint_is_healthy
                and logical_step - checkpoint.step >= self.config.checkpoint_interval
            ):
                checkpoint = await self._create_checkpoint(
                    state,
                    step=logical_step,
                    parent_id=checkpoint.checkpoint_id,
                    label="accepted",
                )
                checkpoints.append(checkpoint)
                checkpoint_lineage.append(checkpoint)
                checkpoint_histories[checkpoint.checkpoint_id] = list(recent_steps)

        return await self._finish(
            RunStatus.STEP_LIMIT,
            state,
            all_steps,
            rollbacks,
            checkpoints,
            agent_tokens_used,
            judge_tokens_used,
            current_checkpoint=checkpoint,
            logical_step=logical_step,
        )

    async def _judge(
        self,
        *,
        goal: str,
        plan: str,
        checkpoint: Checkpoint,
        signals: tuple[DriftSignal, ...],
        recent_steps: list[StepRecord],
        tokens_used: int,
    ) -> JudgeVerdict:
        if self.fine_judge is None:
            return JudgeVerdict(
                Verdict.DRIFTED,
                "coarse heuristic fired in heuristics-only mode",
            )
        recent = tuple(recent_steps[-self.config.recent_steps_for_judge :])
        return await self.fine_judge.judge(
            DriftContext(
                goal=goal,
                plan=plan,
                checkpoint=checkpoint,
                signals=signals,
                recent_steps=recent,
                diff=recent[-1].outcome.diff if recent else "",
                tokens_remaining=self._tokens_remaining(tokens_used),
            )
        )

    def _tokens_remaining(self, tokens_used: int) -> int | None:
        if self.config.max_tokens is None:
            return None
        return max(0, self.config.max_tokens - tokens_used)

    def _budget_exhausted(self, tokens_used: int) -> bool:
        return (
            self.config.max_tokens is not None and tokens_used >= self.config.max_tokens
        )

    @staticmethod
    def _select_rollback_checkpoint(
        checkpoint_lineage: list[Checkpoint],
        signals: tuple[DriftSignal, ...],
        logical_step: int,
    ) -> Checkpoint:
        suspicious_start = logical_step - max(signal.lookback for signal in signals) + 1
        candidates = [
            checkpoint
            for checkpoint in checkpoint_lineage
            if checkpoint.step < suspicious_start
        ]
        return candidates[-1] if candidates else checkpoint_lineage[0]

    async def _create_checkpoint(
        self,
        state: Mapping[str, Any],
        *,
        step: int,
        parent_id: str | None = None,
        label: str | None = None,
    ) -> Checkpoint:
        result = self.checkpoint_store.create(
            state,
            step=step,
            parent_id=parent_id,
            label=label,
        )
        if isinstance(result, Checkpoint):
            return result
        return await result

    async def _restore_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
        result = self.checkpoint_store.restore(checkpoint)
        if isinstance(result, dict):
            return result
        return await result

    @staticmethod
    def _result(
        status: RunStatus,
        state: Mapping[str, Any],
        steps: list[StepRecord],
        rollbacks: list[RollbackRecord],
        checkpoints: list[Checkpoint],
        agent_tokens_used: int,
        judge_tokens_used: int,
    ) -> RunResult:
        return RunResult(
            status=status,
            state=dict(state),
            steps=tuple(steps),
            rollbacks=tuple(rollbacks),
            checkpoints=tuple(checkpoints),
            tokens_used=agent_tokens_used + judge_tokens_used,
            agent_tokens_used=agent_tokens_used,
            judge_tokens_used=judge_tokens_used,
        )

    async def _finish(
        self,
        status: RunStatus,
        state: Mapping[str, Any],
        steps: list[StepRecord],
        rollbacks: list[RollbackRecord],
        checkpoints: list[Checkpoint],
        agent_tokens_used: int,
        judge_tokens_used: int,
        *,
        current_checkpoint: Checkpoint,
        logical_step: int,
    ) -> RunResult:
        if self.config.checkpoint_on_exit and current_checkpoint.step != logical_step:
            terminal = await self._create_checkpoint(
                state,
                step=logical_step,
                parent_id=current_checkpoint.checkpoint_id,
                label="terminal",
            )
            checkpoints.append(terminal)
        return self._result(
            status,
            state,
            steps,
            rollbacks,
            checkpoints,
            agent_tokens_used,
            judge_tokens_used,
        )
