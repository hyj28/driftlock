from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from driftlock.lhtb_experiment import main
from driftlock.skill_admission import SkillLibrary
from driftlock.skill_distillation import Skill, serialize_skill
from driftlock.skill_validation import (
    DEFAULT_MAX_CONCURRENT_TRIALS,
    ValidationTrial,
    ValidationTrialResult,
    ValidationTrialStatus,
    plan_skill_validation,
    run_skill_validation,
)


def _validation_plan(tmp_path: Path, *, candidate_count: int = 1):
    root = tmp_path / "LHTB"
    task = root / "tasks" / "task-0"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        "[task]\nname = 'long-horizon-terminal-bench/task-0'\n",
        encoding="utf-8",
    )
    candidate_file = tmp_path / "candidates.json"
    candidate_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "skill-distillation",
                "candidates": [
                    {
                        "candidate_id": f"candidate-{index}",
                        "arm": "localized" if index == 0 else "baseline",
                        "skill": serialize_skill(
                            Skill(
                                activation=f"When candidate {index} applies.",
                                execution=f"Use only candidate {index}.",
                                termination="Stop after validation.",
                            )
                        ),
                        "paired_deltas": [],
                        "task_name": "long-horizon-terminal-bench/task-0",
                    }
                    for index in range(candidate_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    return plan_skill_validation(candidate_file, root), root, candidate_file


def _measured(trial: ValidationTrial) -> ValidationTrialResult:
    return ValidationTrialResult(
        status=ValidationTrialStatus.MEASURED,
        reward=0.4 if trial.condition == "without_skill" else 0.5,
        injected_candidate_ids=SkillLibrary(trial.library_dir).admitted_skill_ids(),
    )


class _TrackingRunner:
    def __init__(self) -> None:
        self.calls: list[ValidationTrial] = []
        self.in_flight = 0
        self.maximum_in_flight = 0

    async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
        self.calls.append(trial)
        self.in_flight += 1
        self.maximum_in_flight = max(self.maximum_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            return _measured(trial)
        finally:
            self.in_flight -= 1


@pytest.mark.asyncio
async def test_bound_four_over_twenty_trials_is_measured_and_checkpointed(
    tmp_path: Path,
) -> None:
    plan, _, _ = _validation_plan(tmp_path)
    runner = _TrackingRunner()
    output = tmp_path / "validated.json"

    report = await run_skill_validation(
        plan,
        output,
        runner=runner,
        work_dir=tmp_path / "work",
        max_concurrent_trials=4,
    )

    assert runner.maximum_in_flight == 4
    assert len(runner.calls) == 20
    assert report["validation"]["summary"]["measured_trial_count"] == 20
    assert report["validation"]["summary"]["pending_trial_count"] == 0
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.asyncio
async def test_bound_one_preserves_sequential_order_and_report(tmp_path: Path) -> None:
    plan, _, _ = _validation_plan(tmp_path)
    sequential_runner = _TrackingRunner()
    sequential = await run_skill_validation(
        plan,
        tmp_path / "sequential.json",
        runner=sequential_runner,
        work_dir=tmp_path / "sequential-work",
        max_concurrent_trials=1,
    )
    concurrent_runner = _TrackingRunner()
    concurrent = await run_skill_validation(
        plan,
        tmp_path / "concurrent.json",
        runner=concurrent_runner,
        work_dir=tmp_path / "concurrent-work",
        max_concurrent_trials=4,
    )

    assert sequential_runner.maximum_in_flight == 1
    assert [
        (trial.candidate_id, trial.replicate_index) for trial in sequential_runner.calls
    ] == [
        (None, 1),
        (None, 2),
        (None, 3),
        (None, 4),
        (None, 5),
        (None, 6),
        (None, 7),
        (None, 8),
        (None, 9),
        (None, 10),
        ("candidate-0", 1),
        ("candidate-0", 2),
        ("candidate-0", 3),
        ("candidate-0", 4),
        ("candidate-0", 5),
        ("candidate-0", 6),
        ("candidate-0", 7),
        ("candidate-0", 8),
        ("candidate-0", 9),
        ("candidate-0", 10),
    ]
    assert sequential == concurrent


@pytest.mark.asyncio
async def test_interrupt_loses_only_four_in_flight_and_resume_runs_only_four(
    tmp_path: Path,
) -> None:
    plan, _, _ = _validation_plan(tmp_path)
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"

    class BlockingRunner:
        def __init__(self) -> None:
            self.blocked = 0
            self.all_blocked = asyncio.Event()

        async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
            if trial.condition == "with_skill" and trial.replicate_index >= 7:
                self.blocked += 1
                if self.blocked == 4:
                    self.all_blocked.set()
                await asyncio.Event().wait()
            await asyncio.sleep(0)
            return _measured(trial)

    first = BlockingRunner()
    running = asyncio.create_task(
        run_skill_validation(
            plan,
            output,
            runner=first,
            work_dir=work_dir,
            max_concurrent_trials=4,
        )
    )
    await asyncio.wait_for(first.all_blocked.wait(), timeout=1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    assert checkpoint["validation"]["summary"]["completed_attempt_count"] == 16
    assert checkpoint["validation"]["summary"]["pending_trial_count"] == 4

    resumed = _TrackingRunner()
    report = await run_skill_validation(
        plan,
        output,
        runner=resumed,
        work_dir=work_dir,
        max_concurrent_trials=4,
    )
    assert [trial.replicate_index for trial in resumed.calls] == [7, 8, 9, 10]
    assert report["validation"]["summary"]["reused_trial_count"] == 16
    assert report["validation"]["summary"]["new_attempt_count"] == 4


@pytest.mark.asyncio
async def test_controls_complete_once_before_shared_task_treatments(
    tmp_path: Path,
) -> None:
    plan, _, _ = _validation_plan(tmp_path, candidate_count=2)

    class PhaseRunner:
        def __init__(self) -> None:
            self.control_completions = 0
            self.calls: list[ValidationTrial] = []
            self.treatment_start_control_completions: list[int] = []

        async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
            self.calls.append(trial)
            if trial.condition == "with_skill":
                self.treatment_start_control_completions.append(
                    self.control_completions
                )
            await asyncio.sleep(0)
            if trial.condition == "without_skill":
                self.control_completions += 1
            else:
                assert self.control_completions == 10
            return _measured(trial)

    runner = PhaseRunner()
    await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
        max_concurrent_trials=4,
    )

    assert Counter(trial.condition for trial in runner.calls) == {
        "without_skill": 10,
        "with_skill": 20,
    }
    assert all(trial.condition == "without_skill" for trial in runner.calls[:10])
    assert runner.treatment_start_control_completions == [10] * 20


@pytest.mark.asyncio
async def test_failed_control_still_runs_treatments_but_leaves_delta_unmeasured(
    tmp_path: Path,
) -> None:
    plan, _, _ = _validation_plan(tmp_path)

    class FailedControlRunner:
        def __init__(self) -> None:
            self.treatment_replicates: list[int] = []

        async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
            if trial.condition == "with_skill":
                self.treatment_replicates.append(trial.replicate_index)
            await asyncio.sleep(0)
            if trial.condition == "without_skill" and trial.replicate_index == 1:
                return ValidationTrialResult(
                    status=ValidationTrialStatus.FAILED,
                    reason="synthetic control failure",
                )
            return _measured(trial)

    runner = FailedControlRunner()
    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
        max_concurrent_trials=4,
    )

    assert runner.treatment_replicates == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    first = report["validation"]["paired_measurements"][0]
    assert first["control_reward"] is None
    assert first["treatment_reward"] == 0.5
    assert first["delta"] is None
    assert first["measured"] is False


@pytest.mark.asyncio
async def test_overlapping_candidates_keep_distinct_staged_libraries(
    tmp_path: Path,
) -> None:
    plan, _, _ = _validation_plan(tmp_path, candidate_count=2)

    class IsolationRunner:
        def __init__(self) -> None:
            self.first_candidate_waiting = asyncio.Event()
            self.second_candidate_started = asyncio.Event()
            self.overlapped = False
            self.paths: dict[str, Path] = {}

        async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
            ids = SkillLibrary(trial.library_dir).admitted_skill_ids()
            if trial.candidate_id is not None:
                assert ids == (trial.candidate_id,)
                self.paths[trial.candidate_id] = trial.library_dir
            if trial.candidate_id == "candidate-0" and trial.replicate_index == 10:
                self.first_candidate_waiting.set()
                await self.second_candidate_started.wait()
            elif trial.candidate_id == "candidate-1" and trial.replicate_index == 1:
                assert self.first_candidate_waiting.is_set()
                self.overlapped = True
                self.second_candidate_started.set()
            await asyncio.sleep(0)
            return _measured(trial)

    runner = IsolationRunner()
    await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
        max_concurrent_trials=4,
    )

    assert runner.overlapped is True
    assert runner.paths == {
        "candidate-0": tmp_path / "work" / "libraries" / "candidate-0",
        "candidate-1": tmp_path / "work" / "libraries" / "candidate-1",
    }


@pytest.mark.asyncio
async def test_one_raised_trial_does_not_cancel_concurrent_siblings(
    tmp_path: Path,
) -> None:
    plan, _, _ = _validation_plan(tmp_path)

    class RaisingRunner:
        async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
            await asyncio.sleep(0)
            if trial.condition == "without_skill" and trial.replicate_index == 2:
                raise RuntimeError("synthetic runner exception")
            return _measured(trial)

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=RaisingRunner(),
        work_dir=tmp_path / "work",
        max_concurrent_trials=4,
    )

    summary = report["validation"]["summary"]
    assert summary["completed_attempt_count"] == 20
    assert summary["measured_trial_count"] == 19
    assert summary["failed_attempt_count"] == 1
    failed = [
        attempt
        for attempt in report["validation"]["attempts"]
        if attempt["status"] == "failed"
    ]
    assert failed[0]["failure_kind"] == "trial_runner"


def test_cli_dry_run_reports_operator_concurrency_and_runs_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, root, candidate_file = _validation_plan(tmp_path)
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"

    assert DEFAULT_MAX_CONCURRENT_TRIALS == 4
    assert (
        main(
            [
                "validate-skills",
                str(candidate_file),
                "--lhtb-dir",
                str(root),
                "--output",
                str(output),
                "--work-dir",
                str(work_dir),
                "--max-concurrent-trials",
                "7",
                "--dry-run",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "up to 7 concurrent trial(s)" in printed
    assert "dry run; no trials run and no files written" in printed
    assert not output.exists()
    assert not work_dir.exists()
