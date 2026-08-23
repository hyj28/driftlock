from __future__ import annotations

import pytest

from driftlock.heuristics import SIGNAL_KINDS, HeuristicConfig, HeuristicJudge
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


def test_no_file_change_alone_does_not_initiate_a_fine_review() -> None:
    judge = HeuristicJudge(HeuristicConfig(no_change_steps=3))
    steps = [
        _record(sequence, action=f"read file {sequence}", changed=False)
        for sequence in range(1, 4)
    ]

    signals = judge.evaluate(steps)

    assert [signal.kind for signal in signals] == ["no_file_change"]
    assert judge.initiates_review(signals) is False


def test_no_file_change_alongside_another_signal_initiates_a_fine_review() -> None:
    judge = HeuristicJudge(
        HeuristicConfig(no_change_steps=3, loop_window=3, loop_repetitions=3)
    )
    steps = [
        _record(sequence, action="ls -la", changed=False) for sequence in range(1, 4)
    ]

    signals = judge.evaluate(steps)

    assert {signal.kind for signal in signals} == {"no_file_change", "action_loop"}
    assert judge.initiates_review(signals) is True


def test_the_corroborating_set_is_configurable() -> None:
    signals = HeuristicJudge(HeuristicConfig(no_change_steps=3)).evaluate(
        [
            _record(sequence, action=f"read {sequence}", changed=False)
            for sequence in range(1, 4)
        ]
    )

    assert (
        HeuristicJudge(
            HeuristicConfig(corroborating_signals=frozenset())
        ).initiates_review(signals)
        is True
    )
    assert (
        HeuristicJudge(
            HeuristicConfig(corroborating_signals=frozenset({"action_loop"}))
        ).initiates_review(signals)
        is True
    )


def test_corroborating_signals_must_name_real_detectors() -> None:
    with pytest.raises(ValueError, match="unknown corroborating signal kinds: typo"):
        HeuristicConfig(corroborating_signals=frozenset({"typo"}))


def test_at_least_one_kind_must_be_able_to_initiate_a_review() -> None:
    with pytest.raises(ValueError, match="at least one signal kind"):
        HeuristicConfig(corroborating_signals=SIGNAL_KINDS)


def test_signal_kinds_lists_every_kind_the_detector_can_emit() -> None:
    # SIGNAL_KINDS is what validates a corroborating set, so a detector added
    # without updating it would silently become unblockable.
    stalled = [
        _record(sequence, action="  RUN   tests ", changed=False, error="x", reward=0.2)
        for sequence in range(1, 9)
    ]
    wedged = [_command_record(sequence) for sequence in range(1, 9)]
    judge = HeuristicJudge(
        HeuristicConfig(
            no_change_steps=3,
            loop_window=3,
            loop_repetitions=3,
            error_window=3,
            error_rate=2 / 3,
            reward_stall_steps=3,
            corroborating_signals=frozenset(),
        )
    )
    emitted = {signal.kind for signal in judge.evaluate(stalled)} | {
        signal.kind for signal in HeuristicJudge().evaluate(wedged)
    }

    assert emitted == SIGNAL_KINDS
