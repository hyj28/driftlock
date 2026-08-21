from __future__ import annotations

from pathlib import Path

import pytest

from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.models import (
    DriftContext,
    JudgeVerdict,
    RunStatus,
    StepContext,
    StepOutcome,
    StepTokenBudgetExhausted,
    Verdict,
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


def _store(tmp_path: Path) -> tuple[Path, DirectoryCheckpointStore]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("healthy", encoding="utf-8")
    return workspace, DirectoryCheckpointStore(workspace, tmp_path / "snapshots")


def _quick_coarse_judge() -> HeuristicJudge:
    return HeuristicJudge(
        HeuristicConfig(
            no_change_steps=2,
            loop_window=2,
            loop_repetitions=2,
            error_window=2,
            reward_stall_steps=2,
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
    assert [record.attempt for record in result.steps] == [1, 1, 2]
    assert result.tokens_used == 25


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
