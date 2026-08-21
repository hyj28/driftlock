from __future__ import annotations

from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.models import StepOutcome, StepRecord


def _record(
    sequence: int,
    *,
    action: str = "work",
    changed: bool = True,
    error: str | None = None,
    reward: float | None = None,
) -> StepRecord:
    return StepRecord(
        sequence=sequence,
        logical_step=sequence,
        attempt=1,
        outcome=StepOutcome(
            action=action,
            state={},
            changed_paths=("file",) if changed else (),
            error=error,
            reward=reward,
        ),
    )


def test_detects_all_coarse_failure_modes() -> None:
    judge = HeuristicJudge(
        HeuristicConfig(
            no_change_steps=3,
            loop_window=3,
            loop_repetitions=3,
            error_window=3,
            error_rate=2 / 3,
            reward_stall_steps=3,
        )
    )
    steps = [
        _record(i, action="  RUN   tests ", changed=False, error="failed", reward=0.2)
        for i in range(1, 4)
    ]

    assert {signal.kind for signal in judge.evaluate(steps)} == {
        "no_file_change",
        "action_loop",
        "error_spike",
        "reward_stall",
    }


def test_healthy_progress_does_not_trigger() -> None:
    judge = HeuristicJudge(
        HeuristicConfig(
            no_change_steps=2,
            loop_window=3,
            loop_repetitions=3,
            error_window=3,
            reward_stall_steps=3,
        )
    )
    steps = [
        _record(1, action="inspect", reward=0.1),
        _record(2, action="edit", reward=0.2),
        _record(3, action="test", reward=0.3),
    ]

    assert judge.evaluate(steps) == ()


def _command_record(
    sequence: int, *, commands_run: int = 1, commands_failed: int = 1
) -> StepRecord:
    return StepRecord(
        sequence=sequence,
        logical_step=sequence,
        attempt=1,
        outcome=StepOutcome(
            action=f"run command {sequence}",
            state={},
            changed_paths=(f"file-{sequence}",),
            commands_run=commands_run,
            commands_failed=commands_failed,
        ),
    )


def test_detects_sustained_command_failure() -> None:
    steps = [_command_record(sequence) for sequence in range(1, 9)]

    signals = HeuristicJudge().evaluate(steps)

    assert [(signal.kind, signal.lookback) for signal in signals] == [
        ("sustained_command_failure", 8)
    ]


def test_sustained_command_failure_requires_every_command_to_fail() -> None:
    steps = [_command_record(sequence) for sequence in range(1, 9)]
    steps[4] = _command_record(5, commands_run=2, commands_failed=1)

    assert HeuristicJudge().evaluate(steps) == ()


def test_step_without_a_command_is_not_counted_as_command_failure() -> None:
    steps = [_command_record(sequence) for sequence in range(1, 9)]
    steps[4] = _command_record(5, commands_run=0, commands_failed=0)

    assert HeuristicJudge().evaluate(steps) == ()
