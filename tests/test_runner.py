from __future__ import annotations

from pathlib import Path

from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.models import (
    DriftContext,
    JudgeVerdict,
    RunStatus,
    StepContext,
    StepOutcome,
    Verdict,
)
from driftlock.runner import DriftlockRunner, RunnerConfig


class HealthyJudge:
    async def judge(self, context: DriftContext) -> JudgeVerdict:
        return JudgeVerdict(Verdict.HEALTHY, "exploration is still on goal")


def _store(tmp_path: Path) -> tuple[Path, DirectoryCheckpointStore]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("healthy", encoding="utf-8")
    return workspace, DirectoryCheckpointStore(workspace, tmp_path / "snapshots")


def _quick_coarse_judge() -> HeuristicJudge:
    return HeuristicJudge(
        HeuristicConfig(
            no_change_steps=2,
            loop_window=5,
            loop_repetitions=5,
            error_window=5,
            reward_stall_steps=5,
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


async def test_runner_enforces_token_budget(tmp_path: Path) -> None:
    _workspace, store = _store(tmp_path)

    async def agent_step(context: StepContext) -> StepOutcome:
        return StepOutcome(action="work", state={}, tokens=7)

    result = await DriftlockRunner(
        store,
        _quick_coarse_judge(),
        config=RunnerConfig(max_steps=10, max_tokens=10),
    ).run(goal="bounded", step=agent_step, initial_state={})

    assert result.status is RunStatus.TOKEN_LIMIT
    assert len(result.steps) == 2
    assert result.tokens_used == 14


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
