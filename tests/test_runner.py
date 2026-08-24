from __future__ import annotations

from pathlib import Path

import pytest

from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.judges import JudgeTokenBudgetExhausted
from driftlock.models import (
    DriftContext,
    DriftSignal,
    DriftTriggerOutcome,
    DriftTriggerRecord,
    FineJudgeStatus,
    JudgeReliabilityStatus,
    JudgeVerdict,
    RunStatus,
    StepContext,
    StepOutcome,
    StepTokenBudgetExhausted,
    Verdict,
    classify_judge_reliability,
)
from driftlock.runner import DriftlockRunner, RunnerConfig


class HealthyJudge:
    async def judge(self, context: DriftContext) -> JudgeVerdict:
        return JudgeVerdict(Verdict.HEALTHY, "exploration is still on goal")


class SequencedJudge:
    def __init__(self, *verdicts: JudgeVerdict) -> None:
        self.verdicts = list(verdicts)

    async def judge(self, context: DriftContext) -> JudgeVerdict:
        return self.verdicts.pop(0)


class RaisingJudge:
    async def judge(self, context: DriftContext) -> JudgeVerdict:
        raise TimeoutError("judge provider unavailable")


def _store(tmp_path: Path) -> tuple[Path, DirectoryCheckpointStore]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("healthy", encoding="utf-8")
    return workspace, DirectoryCheckpointStore(workspace, tmp_path / "snapshots")


def _quick_coarse_judge() -> HeuristicJudge:
    # corroborating_signals is emptied deliberately: these tests exercise what
    # happens once a signal reaches the fine judge, so every kind must be able to
    # get there. The corroborating-only gate has its own tests below.
    return HeuristicJudge(
        HeuristicConfig(
            no_change_steps=2,
            loop_window=2,
            loop_repetitions=2,
            error_window=2,
            reward_stall_steps=2,
            corroborating_signals=frozenset(),
        )
    )


async def test_runner_rolls_back_workspace_and_state_then_retries(
    tmp_path: Path,
) -> None:
    workspace, store = _store(tmp_path)

    async def agent_step(context: StepContext) -> StepOutcome:
        if context.attempt == 1:
            (workspace / "answer.txt").write_text("drifted", encoding="utf-8")
            return StepOutcome(
                action=f"wander {context.logical_step}",
                state={"value": "drifted"},
                tokens=10,
            )
        assert context.rollback_feedback is not None
        assert context.state == {"value": "healthy"}
        assert (workspace / "answer.txt").read_text(encoding="utf-8") == "healthy"
        return StepOutcome(
            action="finish",
            state={"value": "solved"},
            changed_paths=("answer.txt",),
            tokens=5,
            completed=True,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        config=RunnerConfig(max_steps=5, max_rollbacks=1, checkpoint_interval=5),
    ).run(goal="stay healthy", step=agent_step, initial_state={"value": "healthy"})

    assert result.status is RunStatus.COMPLETED
    assert result.state == {"value": "solved"}
    assert len(result.rollbacks) == 1
    assert [record.sequence for record in result.steps] == [1, 2, 3]
    assert [record.logical_step for record in result.steps] == [1, 2, 1]
    assert [record.attempt for record in result.steps] == [1, 1, 2]
    assert [record.sequence for record in result.rollbacks] == [2]
    assert result.tokens_used == 25
    assert result.agent_tokens_used == 25
    assert result.judge_tokens_used == 0

    assert len(result.coarse_triggers) == 1
    trigger = result.coarse_triggers[0]
    assert trigger.sequence == 2
    assert trigger.logical_step == 2
    assert trigger.judge_status is FineJudgeStatus.NOT_CONFIGURED
    assert trigger.judge_verdict is None
    assert trigger.judge_reason is None
    assert trigger.outcome is DriftTriggerOutcome.ROLLED_BACK
    assert trigger.rollback_checkpoint_id == result.checkpoints[0].checkpoint_id
    assert trigger.rollback_checkpoint_step == 0
    assert trigger.to_dict()["signals"] == [
        {
            "kind": "no_file_change",
            "detail": "no files changed in the last 2 steps",
            "lookback": 2,
        }
    ]
    assert trigger.to_dict()["judge"] == {
        "status": "not_configured",
        "verdict": None,
        "reason": None,
    }
    assert result.signal_counts == {
        "no_file_change": {
            "upheld": 1,
            "vetoed": 0,
            "suppressed": 0,
            "judge_failed": 0,
            "judge_budget_exhausted": 0,
        }
    }


async def test_fine_judge_can_veto_a_coarse_signal(tmp_path: Path) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"think {context.logical_step}",
            state={"turn": context.logical_step},
            completed=context.logical_step == 3,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=HealthyJudge(),
        config=RunnerConfig(max_steps=3, checkpoint_interval=10),
    ).run(goal="reason first", step=agent_step, initial_state={})

    assert result.status is RunStatus.COMPLETED
    assert result.rollbacks == ()
    assert len(result.coarse_triggers) == 1
    trigger = result.coarse_triggers[0]
    assert trigger.sequence == 2
    assert trigger.logical_step == 2
    assert trigger.judge_status is FineJudgeStatus.VERDICT
    assert trigger.judge_verdict is Verdict.HEALTHY
    assert trigger.judge_reason == "exploration is still on goal"
    assert trigger.outcome is DriftTriggerOutcome.VETOED
    assert trigger.rollback_checkpoint_id is None
    assert trigger.rollback_checkpoint_step is None
    assert trigger.to_dict()["signals"] == [
        {
            "kind": "no_file_change",
            "detail": "no files changed in the last 2 steps",
            "lookback": 2,
        }
    ]
    assert result.signal_counts == {
        "no_file_change": {
            "upheld": 0,
            "vetoed": 1,
            "suppressed": 0,
            "judge_failed": 0,
            "judge_budget_exhausted": 0,
        }
    }


async def test_all_failed_small_sample_is_inconclusive(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"think {context.logical_step}",
            state={"turn": context.logical_step},
            completed=context.logical_step == 3,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=RaisingJudge(),
        config=RunnerConfig(max_steps=3, checkpoint_interval=10),
    ).run(goal="reason first", step=agent_step, initial_state={})

    assert result.status is RunStatus.COMPLETED
    assert result.judge_reliability is JudgeReliabilityStatus.INCONCLUSIVE
    assert len(result.steps) == 3
    assert result.state == {"turn": 3}
    assert result.rollbacks == ()
    assert [trigger.judge_status for trigger in result.coarse_triggers] == [
        FineJudgeStatus.FAILED,
    ]
    assert [trigger.outcome for trigger in result.coarse_triggers] == [
        DriftTriggerOutcome.JUDGE_FAILED,
    ]
    assert result.coarse_triggers[0].to_dict()["judge"] == {
        "status": "failed",
        "verdict": None,
        "reason": "fine judge raised unexpectedly: judge provider unavailable",
    }
    assert result.signal_counts == {
        "no_file_change": {
            "upheld": 0,
            "vetoed": 0,
            "suppressed": 0,
            "judge_failed": 1,
            "judge_budget_exhausted": 0,
        }
    }


async def test_one_judge_failure_does_not_invalidate_a_later_adjudication(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)

    class IntermittentJudge:
        def __init__(self) -> None:
            self.calls = 0

        async def judge(self, context: DriftContext) -> JudgeVerdict:
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("one bad call")
            return JudgeVerdict(Verdict.UNCERTAIN, "model found mixed evidence")

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"think {context.logical_step}",
            state={"turn": context.logical_step},
            completed=context.logical_step == 4,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=IntermittentJudge(),
        config=RunnerConfig(
            max_steps=4,
            checkpoint_interval=10,
            rollback_on_uncertain=False,
        ),
    ).run(goal="reason first", step=agent_step, initial_state={})

    assert result.status is RunStatus.COMPLETED
    assert result.judge_reliability is JudgeReliabilityStatus.RELIABLE
    assert result.rollbacks == ()
    assert [trigger.judge_status for trigger in result.coarse_triggers] == [
        FineJudgeStatus.FAILED,
        FineJudgeStatus.VERDICT,
    ]
    assert result.coarse_triggers[1].judge_verdict is Verdict.UNCERTAIN
    assert [trigger.outcome for trigger in result.coarse_triggers] == [
        DriftTriggerOutcome.JUDGE_FAILED,
        DriftTriggerOutcome.VETOED,
    ]
    assert result.signal_counts == {
        "no_file_change": {
            "upheld": 0,
            "vetoed": 1,
            "suppressed": 0,
            "judge_failed": 1,
            "judge_budget_exhausted": 0,
        }
    }


async def test_uncertain_verdict_rolls_back_when_configured(tmp_path: Path) -> None:
    _workspace, store = _store(tmp_path)
    judge = SequencedJudge(JudgeVerdict(Verdict.UNCERTAIN, "evidence is mixed"))

    async def agent_step(context: StepContext) -> StepOutcome:
        if context.attempt == 1:
            return StepOutcome(
                action=f"think {context.logical_step}",
                state={"attempt": 1},
            )
        return StepOutcome(
            action="finish after conservative rollback",
            state={"done": True},
            changed_paths=("answer.txt",),
            completed=True,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=judge,
        config=RunnerConfig(
            max_steps=3,
            max_rollbacks=1,
            checkpoint_interval=10,
            rollback_on_uncertain=True,
        ),
    ).run(goal="reason first", step=agent_step, initial_state={})

    assert result.status is RunStatus.COMPLETED
    assert result.judge_reliability is JudgeReliabilityStatus.RELIABLE
    assert len(result.rollbacks) == 1
    assert result.coarse_triggers[0].judge_verdict is Verdict.UNCERTAIN
    assert result.coarse_triggers[0].outcome is DriftTriggerOutcome.ROLLED_BACK


async def test_token_limit_preserves_failed_judge_reliability(tmp_path: Path) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"think {context.logical_step}",
            state={"turn": context.logical_step},
            tokens=1,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=RaisingJudge(),
        config=RunnerConfig(max_steps=10, max_tokens=6, checkpoint_interval=100),
    ).run(goal="reason first", step=agent_step, initial_state={})

    assert result.status is RunStatus.TOKEN_LIMIT
    assert result.tokens_used == 6
    assert result.judge_attempts == 4
    assert result.judge_failures == 4
    assert result.judge_reliability is JudgeReliabilityStatus.FAILED


async def test_judge_budget_exhaustion_has_its_own_trigger_and_count(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)

    class BudgetExhaustedJudge:
        async def judge(self, context: DriftContext) -> JudgeVerdict:
            raise JudgeTokenBudgetExhausted("prompt consumes the remaining budget")

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"think {context.logical_step}",
            state={"turn": context.logical_step},
            completed=context.logical_step == 3,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=BudgetExhaustedJudge(),
        config=RunnerConfig(max_steps=3, checkpoint_interval=10),
    ).run(goal="reason first", step=agent_step, initial_state={})

    assert result.status is RunStatus.COMPLETED
    assert result.judge_reliability is JudgeReliabilityStatus.INCONCLUSIVE
    assert result.coarse_triggers[0].judge_status is FineJudgeStatus.BUDGET_EXHAUSTED
    assert result.coarse_triggers[0].judge_verdict is None
    assert (
        result.coarse_triggers[0].outcome is DriftTriggerOutcome.JUDGE_BUDGET_EXHAUSTED
    )
    assert result.signal_counts == {
        "no_file_change": {
            "upheld": 0,
            "vetoed": 0,
            "suppressed": 0,
            "judge_failed": 0,
            "judge_budget_exhausted": 1,
        }
    }


async def test_repeated_all_failed_calls_reject_the_measurement(tmp_path: Path) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"think {context.logical_step}",
            state={"turn": context.logical_step},
            completed=context.logical_step == 6,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=RaisingJudge(),
        config=RunnerConfig(max_steps=6, checkpoint_interval=100),
    ).run(goal="reason first", step=agent_step, initial_state={})

    assert result.status is RunStatus.COMPLETED
    assert result.judge_reliability is JudgeReliabilityStatus.FAILED
    assert len(result.coarse_triggers) == 4
    assert all(
        trigger.judge_status is FineJudgeStatus.FAILED
        for trigger in result.coarse_triggers
    )


async def test_real_34_call_shape_with_33_failures_rejects_the_measurement(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)
    failed = JudgeVerdict(
        None,
        "invalid response",
        confidence=0.0,
        status=FineJudgeStatus.FAILED,
    )
    judge = SequencedJudge(
        *([failed] * 33),
        JudgeVerdict(Verdict.HEALTHY, "one response parsed"),
    )

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"think {context.logical_step}",
            state={"turn": context.logical_step},
            completed=context.logical_step == 36,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=judge,
        config=RunnerConfig(max_steps=36, checkpoint_interval=100),
    ).run(goal="reason first", step=agent_step, initial_state={})

    assert result.status is RunStatus.COMPLETED
    assert result.judge_reliability is JudgeReliabilityStatus.FAILED
    assert len(result.coarse_triggers) == 34
    assert (
        sum(
            trigger.judge_status is FineJudgeStatus.FAILED
            for trigger in result.coarse_triggers
        )
        == 33
    )
    assert (
        sum(
            trigger.judge_status is FineJudgeStatus.VERDICT
            for trigger in result.coarse_triggers
        )
        == 1
    )


@pytest.mark.parametrize(
    ("attempts", "failures", "expected"),
    [
        (0, 0, JudgeReliabilityStatus.NOT_ASSESSED),
        (1, 1, JudgeReliabilityStatus.INCONCLUSIVE),
        (2, 1, JudgeReliabilityStatus.RELIABLE),
        (3, 3, JudgeReliabilityStatus.INCONCLUSIVE),
        (3, 2, JudgeReliabilityStatus.RELIABLE),
        (4, 1, JudgeReliabilityStatus.RELIABLE),
        (4, 2, JudgeReliabilityStatus.RELIABLE),
        (4, 3, JudgeReliabilityStatus.FAILED),
        (34, 33, JudgeReliabilityStatus.FAILED),
        (137, 117, JudgeReliabilityStatus.FAILED),
    ],
)
def test_judge_failure_policy_boundaries(
    attempts: int, failures: int, expected: JudgeReliabilityStatus
) -> None:
    assert classify_judge_reliability(attempts=attempts, failures=failures) is expected


async def test_fine_judge_upheld_trigger_records_verdict_and_rollback_target(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)
    judge = SequencedJudge(
        JudgeVerdict(Verdict.DRIFTED, "the trajectory repeated", tokens=7)
    )

    async def agent_step(context: StepContext) -> StepOutcome:
        if context.attempt == 1:
            return StepOutcome(
                action=f"inspect {context.logical_step}",
                state={"attempt": 1},
                tokens=3,
            )
        return StepOutcome(
            action="finish",
            state={"done": True},
            changed_paths=("answer.txt",),
            tokens=5,
            completed=True,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=judge,
        config=RunnerConfig(max_steps=3, max_rollbacks=1, checkpoint_interval=5),
    ).run(goal="finish", step=agent_step, initial_state={})

    assert result.status is RunStatus.COMPLETED
    assert len(result.rollbacks) == 1
    assert result.tokens_used == 18
    assert result.agent_tokens_used == 11
    assert result.judge_tokens_used == 7
    assert len(result.coarse_triggers) == 1
    trigger = result.coarse_triggers[0]
    assert trigger.sequence == 2
    assert trigger.logical_step == 2
    assert trigger.judge_status is FineJudgeStatus.VERDICT
    assert trigger.judge_verdict is Verdict.DRIFTED
    assert trigger.judge_reason == "the trajectory repeated"
    assert trigger.outcome is DriftTriggerOutcome.ROLLED_BACK
    assert trigger.rollback_checkpoint_id == result.checkpoints[0].checkpoint_id
    assert trigger.rollback_checkpoint_step == 0


async def test_rollback_limit_records_refused_trigger_without_a_rollback(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"inspect {context.logical_step}",
            state={"turn": context.logical_step},
            tokens=4,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        config=RunnerConfig(max_steps=3, max_rollbacks=0, checkpoint_interval=5),
    ).run(goal="finish", step=agent_step, initial_state={})

    assert result.status is RunStatus.ROLLBACK_LIMIT
    assert [record.logical_step for record in result.steps] == [1, 2]
    assert result.rollbacks == ()
    assert result.tokens_used == 8
    assert result.agent_tokens_used == 8
    assert result.judge_tokens_used == 0
    assert len(result.coarse_triggers) == 1
    trigger = result.coarse_triggers[0]
    assert trigger.sequence == 2
    assert trigger.logical_step == 2
    assert trigger.judge_status is FineJudgeStatus.NOT_CONFIGURED
    assert trigger.judge_verdict is None
    assert trigger.judge_reason is None
    assert trigger.outcome is DriftTriggerOutcome.ROLLBACK_LIMIT_REFUSED
    assert trigger.rollback_checkpoint_id is None
    assert trigger.rollback_checkpoint_step is None
    assert result.signal_counts == {
        "no_file_change": {
            "upheld": 1,
            "vetoed": 0,
            "suppressed": 0,
            "judge_failed": 0,
            "judge_budget_exhausted": 0,
        }
    }


async def test_signal_counts_split_upheld_and_vetoed_triggers(tmp_path: Path) -> None:
    _workspace, store = _store(tmp_path)
    judge = SequencedJudge(
        JudgeVerdict(Verdict.HEALTHY, "first trigger is useful exploration"),
        JudgeVerdict(Verdict.DRIFTED, "second trigger confirms drift"),
    )

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"inspect {context.sequence}",
            state={"turn": context.sequence},
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=judge,
        config=RunnerConfig(max_steps=3, max_rollbacks=1, checkpoint_interval=5),
    ).run(goal="finish", step=agent_step, initial_state={})

    assert [trigger.outcome.value for trigger in result.coarse_triggers] == [
        "vetoed",
        "rolled_back",
    ]
    assert result.signal_counts == {
        "no_file_change": {
            "upheld": 1,
            "vetoed": 1,
            "suppressed": 0,
            "judge_failed": 0,
            "judge_budget_exhausted": 0,
        }
    }


async def test_runner_passes_bounded_tool_observations_to_fine_judge(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)
    contexts: list[DriftContext] = []

    class CapturingJudge:
        async def judge(self, context: DriftContext) -> JudgeVerdict:
            contexts.append(context)
            return JudgeVerdict(Verdict.HEALTHY, "test failures show progress")

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"test attempt {context.logical_step}",
            state={},
            tool_observations=(
                f"run_shell:\nexit_code: 1\nstderr:\nfailure-{context.logical_step}-"
                + "x" * 1_200,
            ),
        )

    await DriftlockRunner(
        store,
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=9,
                loop_window=10,
                loop_repetitions=10,
                error_window=10,
                command_failure_window=10,
                reward_stall_steps=10,
                corroborating_signals=frozenset(),
            )
        ),
        fine_judge=CapturingJudge(),
        config=RunnerConfig(max_steps=9, checkpoint_interval=10),
    ).run(goal="fix tests", step=agent_step, initial_state={})

    assert len(contexts) == 1
    assert len(contexts[0].tool_observations) == 8
    assert sum(map(len, contexts[0].tool_observations)) == 8_000
    assert contexts[0].tool_observations[0].startswith("step 2:\nrun_shell:")
    assert contexts[0].tool_observations[-1].startswith("step 9:\nrun_shell:")


async def test_runner_enforces_token_budget(tmp_path: Path) -> None:
    _workspace, store = _store(tmp_path)
    remaining: list[int | None] = []

    async def agent_step(context: StepContext) -> StepOutcome:
        remaining.append(context.tokens_remaining)
        assert context.tokens_remaining is not None
        return StepOutcome(
            action=f"work {context.sequence}",
            state={},
            changed_paths=("work.txt",),
            tokens=min(7, context.tokens_remaining),
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        config=RunnerConfig(max_steps=10, max_tokens=10),
    ).run(goal="bounded", step=agent_step, initial_state={})

    assert result.status is RunStatus.TOKEN_LIMIT
    assert len(result.steps) == 2
    assert result.tokens_used == 10
    assert result.agent_tokens_used == 10
    assert result.judge_tokens_used == 0
    assert remaining == [10, 3]


async def test_runner_stops_cleanly_when_step_preflight_exhausts_budget(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(_context: StepContext) -> StepOutcome:
        raise StepTokenBudgetExhausted("input consumes the remaining budget")

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        config=RunnerConfig(max_steps=3, max_tokens=10),
    ).run(goal="bounded", step=agent_step, initial_state={"safe": True})

    assert result.status is RunStatus.TOKEN_LIMIT
    assert result.state == {"safe": True}
    assert result.steps == ()
    assert result.tokens_used == 0


async def test_completion_cannot_claim_success_over_token_budget(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(_context: StepContext) -> StepOutcome:
        return StepOutcome(action="finish", state={}, tokens=11, completed=True)

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        config=RunnerConfig(max_steps=1, max_tokens=10),
    ).run(goal="bounded", step=agent_step, initial_state={})

    assert result.status is RunStatus.TOKEN_LIMIT


async def test_runner_creates_periodic_healthy_checkpoints(tmp_path: Path) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"work {context.logical_step}",
            state={"turn": context.logical_step},
            changed_paths=(f"file-{context.logical_step}",),
            completed=context.logical_step == 5,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        config=RunnerConfig(max_steps=5, checkpoint_interval=2),
    ).run(goal="checkpoint", step=agent_step, initial_state={})

    assert result.status is RunStatus.COMPLETED
    assert [checkpoint.step for checkpoint in result.checkpoints] == [0, 2, 4]
    assert result.checkpoints[1].parent_id == result.checkpoints[0].checkpoint_id
    assert result.coarse_triggers == ()
    assert result.signal_counts == {}


async def test_uncertain_verdict_never_advances_healthy_checkpoint(
    tmp_path: Path,
) -> None:
    workspace, store = _store(tmp_path)
    judge = SequencedJudge(
        JudgeVerdict(Verdict.UNCERTAIN, "provider timeout"),
        JudgeVerdict(Verdict.DRIFTED, "confirmed drift"),
    )

    async def agent_step(context: StepContext) -> StepOutcome:
        (workspace / "answer.txt").write_text(
            f"bad-{context.sequence}", encoding="utf-8"
        )
        return StepOutcome(action=f"wander {context.sequence}", state={"bad": True})

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=judge,
        config=RunnerConfig(max_steps=3, max_rollbacks=1, checkpoint_interval=2),
    ).run(goal="stay healthy", step=agent_step, initial_state={"bad": False})

    assert len(result.rollbacks) == 1
    assert [checkpoint.step for checkpoint in result.checkpoints] == [0]
    assert result.state == {"bad": False}
    assert (workspace / "answer.txt").read_text(encoding="utf-8") == "healthy"


async def test_default_action_loop_detector_survives_checkpoint_boundary(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(_context: StepContext) -> StepOutcome:
        return StepOutcome(
            action="run the same command",
            state={},
            changed_paths=("changing.log",),
        )

    result = await DriftlockRunner(
        store,
        HeuristicJudge(),
        config=RunnerConfig(max_steps=6, max_rollbacks=1, checkpoint_interval=5),
    ).run(goal="avoid loops", step=agent_step, initial_state={})

    assert len(result.rollbacks) == 1
    assert [checkpoint.step for checkpoint in result.checkpoints] == [0, 5]
    assert result.rollbacks[0].checkpoint_id == result.checkpoints[0].checkpoint_id
    assert any(signal.kind == "action_loop" for signal in result.rollbacks[0].signals)


async def test_long_detector_window_does_not_block_periodic_checkpoints(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)
    coarse = HeuristicJudge(
        HeuristicConfig(
            no_change_steps=2,
            loop_window=100,
            loop_repetitions=100,
            error_window=100,
            reward_stall_steps=100,
            corroborating_signals=frozenset(),
        )
    )

    async def agent_step(context: StepContext) -> StepOutcome:
        is_stalled = context.logical_step >= 9
        return StepOutcome(
            action=f"step {context.logical_step}",
            state={"n": context.logical_step},
            changed_paths=() if is_stalled else ("progress.txt",),
        )

    result = await DriftlockRunner(
        store,
        coarse,
        config=RunnerConfig(max_steps=10, max_rollbacks=1, checkpoint_interval=2),
    ).run(goal="keep progress", step=agent_step, initial_state={"n": 0})

    assert [checkpoint.step for checkpoint in result.checkpoints] == [0, 2, 4, 6, 8]
    assert len(result.rollbacks) == 1
    assert result.state == {"n": 8}
    assert result.rollbacks[0].checkpoint_id == result.checkpoints[-1].checkpoint_id


async def test_fine_judge_tokens_are_included_in_budget(tmp_path: Path) -> None:
    _workspace, store = _store(tmp_path)
    judge = SequencedJudge(
        JudgeVerdict(Verdict.HEALTHY, "still useful", tokens=3),
    )

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"think {context.sequence}",
            state={},
            tokens=2,
        )

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        fine_judge=judge,
        config=RunnerConfig(max_steps=4, max_tokens=7),
    ).run(goal="budget judge", step=agent_step, initial_state={})

    assert result.status is RunStatus.TOKEN_LIMIT
    assert result.agent_tokens_used == 4
    assert result.judge_tokens_used == 3
    assert result.tokens_used == 7


async def test_unexpected_agent_exception_is_not_misreported_as_zero_cost(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(_context: StepContext) -> StepOutcome:
        raise RuntimeError("adapter lost usage accounting")

    runner = DriftlockRunner(store, _quick_coarse_judge())

    with pytest.raises(RuntimeError, match="usage accounting"):
        await runner.run(goal="account exactly", step=agent_step, initial_state={})


class CountingJudge:
    """A fine judge that would always roll back, and counts how often it is asked."""

    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, context: DriftContext) -> JudgeVerdict:
        self.calls += 1
        return JudgeVerdict(Verdict.DRIFTED, "off goal", tokens=17)


async def test_corroborating_only_trigger_is_recorded_without_calling_the_judge(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)
    judge = CountingJudge()

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action=f"read file {context.logical_step}",
            state={"turn": context.logical_step},
            tokens=5,
            completed=context.logical_step == 3,
        )

    result = await DriftlockRunner(
        store,
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=2,
                loop_window=100,
                loop_repetitions=100,
                error_window=100,
                command_failure_window=100,
                reward_stall_steps=100,
            )
        ),
        fine_judge=judge,
        config=RunnerConfig(max_steps=3, checkpoint_interval=10),
    ).run(goal="explore before writing", step=agent_step, initial_state={})

    assert judge.calls == 0
    assert result.status is RunStatus.COMPLETED
    assert result.rollbacks == ()
    assert result.judge_tokens_used == 0
    assert result.tokens_used == 15

    assert len(result.coarse_triggers) == 1
    trigger = result.coarse_triggers[0]
    assert trigger.logical_step == 2
    assert [signal.kind for signal in trigger.signals] == ["no_file_change"]
    assert trigger.outcome is DriftTriggerOutcome.SUPPRESSED
    assert trigger.judge_status is FineJudgeStatus.NOT_INVOKED
    assert trigger.judge_verdict is None
    assert trigger.judge_reason is None
    assert trigger.to_dict()["judge"] == {
        "status": "not_invoked",
        "verdict": None,
        "reason": None,
    }
    assert result.signal_counts == {
        "no_file_change": {
            "upheld": 0,
            "vetoed": 0,
            "suppressed": 1,
            "judge_failed": 0,
            "judge_budget_exhausted": 0,
        }
    }


async def test_a_second_signal_lifts_the_same_trajectory_to_the_judge(
    tmp_path: Path,
) -> None:
    _workspace, store = _store(tmp_path)
    judge = CountingJudge()

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(
            action="ls -la",
            state={"turn": context.logical_step},
            tokens=5,
            completed=context.logical_step == 3,
        )

    result = await DriftlockRunner(
        store,
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=2,
                loop_window=2,
                loop_repetitions=2,
                error_window=100,
                command_failure_window=100,
                reward_stall_steps=100,
            )
        ),
        fine_judge=judge,
        config=RunnerConfig(max_steps=3, max_rollbacks=1, checkpoint_interval=10),
    ).run(goal="stop repeating yourself", step=agent_step, initial_state={})

    assert judge.calls == 1
    assert len(result.rollbacks) == 1
    assert result.coarse_triggers[0].outcome is DriftTriggerOutcome.ROLLED_BACK
    assert {signal.kind for signal in result.coarse_triggers[0].signals} == {
        "no_file_change",
        "action_loop",
    }
    assert result.signal_counts["action_loop"] == {
        "upheld": 1,
        "vetoed": 0,
        "suppressed": 0,
        "judge_failed": 0,
        "judge_budget_exhausted": 0,
    }


def test_a_suppressed_trigger_cannot_carry_a_verdict() -> None:
    signal = DriftSignal("no_file_change", "no files changed in the last 2 steps", 2)
    with pytest.raises(ValueError, match="a fine judge that did not run"):
        DriftTriggerRecord(
            sequence=2,
            logical_step=2,
            signals=(signal,),
            judge_status=FineJudgeStatus.NOT_INVOKED,
            judge_verdict=Verdict.HEALTHY,
            judge_reason="looks fine",
            outcome=DriftTriggerOutcome.SUPPRESSED,
        )


def test_a_judged_trigger_cannot_be_recorded_as_suppressed() -> None:
    signal = DriftSignal("no_file_change", "no files changed in the last 2 steps", 2)
    with pytest.raises(ValueError, match="never saw"):
        DriftTriggerRecord(
            sequence=2,
            logical_step=2,
            signals=(signal,),
            judge_status=FineJudgeStatus.VERDICT,
            judge_verdict=Verdict.HEALTHY,
            judge_reason="looks fine",
            outcome=DriftTriggerOutcome.SUPPRESSED,
        )


def test_an_unasked_judge_cannot_be_recorded_as_a_veto() -> None:
    signal = DriftSignal("no_file_change", "no files changed in the last 2 steps", 2)
    with pytest.raises(ValueError, match="never saw"):
        DriftTriggerRecord(
            sequence=2,
            logical_step=2,
            signals=(signal,),
            judge_status=FineJudgeStatus.NOT_INVOKED,
            judge_verdict=None,
            judge_reason=None,
            outcome=DriftTriggerOutcome.VETOED,
        )


def test_a_failed_judge_cannot_carry_a_model_verdict() -> None:
    signal = DriftSignal("no_file_change", "no files changed in the last 2 steps", 2)
    with pytest.raises(ValueError, match="reason but no verdict"):
        DriftTriggerRecord(
            sequence=2,
            logical_step=2,
            signals=(signal,),
            judge_status=FineJudgeStatus.FAILED,
            judge_verdict=Verdict.UNCERTAIN,
            judge_reason="provider failed",
            outcome=DriftTriggerOutcome.JUDGE_FAILED,
        )


def test_a_failed_judge_requires_its_matching_trigger_outcome() -> None:
    signal = DriftSignal("no_file_change", "no files changed in the last 2 steps", 2)
    with pytest.raises(ValueError, match="matching trigger outcome"):
        DriftTriggerRecord(
            sequence=2,
            logical_step=2,
            signals=(signal,),
            judge_status=FineJudgeStatus.BUDGET_EXHAUSTED,
            judge_verdict=None,
            judge_reason="no output room",
            outcome=DriftTriggerOutcome.JUDGE_FAILED,
        )
