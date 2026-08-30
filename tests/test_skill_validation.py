from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

import driftlock.lhtb_experiment as experiment
from driftlock.lhtb_experiment import main
from driftlock.skill_admission import (
    SkillLibrary,
    decide_skill_admission,
    load_admission_candidates,
)
from driftlock.skill_distillation import Skill, serialize_skill
from driftlock.skill_validation import (
    VALIDATION_PROCEDURE_ID,
    ValidationTrial,
    ValidationTrialResult,
    ValidationTrialStatus,
    plan_skill_validation,
    run_skill_validation,
)


def _skill(label: str) -> Skill:
    return Skill(
        activation=f"When validation task {label} is active.",
        execution=f"Apply only the isolated {label} procedure.",
        termination="Stop after the task verifier produces its reward.",
    )


def _candidate_file(
    tmp_path: Path, arms: tuple[str, ...] = ("localized", "baseline")
) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "skill-distillation",
                "candidates": [
                    {
                        "candidate_id": f"candidate-{index}",
                        "arm": arm,
                        "skill": serialize_skill(_skill(f"candidate-{index}")),
                        "paired_deltas": [],
                        "source_marker": f"source-{index}",
                    }
                    for index, arm in enumerate(arms)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _task_file(tmp_path: Path, count: int) -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps({"selected_tasks": [f"task-{index}" for index in range(count)]}),
        encoding="utf-8",
    )
    return path


class _FakeRunner:
    def __init__(
        self,
        rewards: dict[tuple[str | None, str], float],
        *,
        failures: set[tuple[str | None, str]] | None = None,
    ) -> None:
        self.rewards = rewards
        self.failures = failures or set()
        self.calls: list[ValidationTrial] = []

    async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
        self.calls.append(trial)
        key = (trial.candidate_id, trial.task_name)
        library_ids = SkillLibrary(trial.library_dir).admitted_skill_ids()
        if key in self.failures:
            return ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason="synthetic upstream 429",
                injected_candidate_ids=library_ids,
                audit={"library_candidate_ids": list(library_ids)},
            )
        return ValidationTrialResult(
            status=ValidationTrialStatus.MEASURED,
            reward=self.rewards[key],
            injected_candidate_ids=library_ids,
            audit={"library_candidate_ids": list(library_ids)},
        )


def _three_task_rewards() -> dict[tuple[str | None, str], float]:
    return {
        (None, "task-0"): 0.2,
        (None, "task-1"): 0.4,
        (None, "task-2"): 0.6,
        ("candidate-0", "task-0"): 0.3,
        ("candidate-0", "task-1"): 0.35,
        ("candidate-0", "task-2"): 0.9,
        ("candidate-1", "task-0"): 0.2,
        ("candidate-1", "task-1"): 0.8,
        ("candidate-1", "task-2"): 0.1,
    }


@pytest.mark.asyncio
async def test_dry_run_reports_exact_cost_and_breakdown_without_writes(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _task_file(tmp_path, 3))
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"

    class ExplodingRunner:
        async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
            del trial
            raise AssertionError("dry run ran a trial")

    report = await run_skill_validation(
        plan,
        output,
        runner=ExplodingRunner(),
        work_dir=work_dir,
        dry_run=True,
    )

    assert report["validation"]["summary"] == {
        "planned_trial_count": 9,
        "completed_attempt_count": 0,
        "measured_trial_count": 0,
        "pending_trial_count": 9,
        "shared_control_trial_count": 3,
        "treatment_trial_count": 6,
        "reused_trial_count": 0,
        "new_attempt_count": 0,
        "failed_attempt_count": 0,
        "estimated_cost_per_trial_usd": 0.151,
        "estimated_planned_cost_usd": 1.359,
        "estimated_pending_cost_usd": 1.359,
        "status_counts": {},
    }
    assert not output.exists()
    assert not work_dir.exists()


def test_validate_skills_cli_dry_run_prints_candidate_task_costs_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidates = _candidate_file(tmp_path)
    tasks = _task_file(tmp_path, 3)
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"

    assert (
        main(
            [
                "validate-skills",
                str(candidates),
                str(tasks),
                "--output",
                str(output),
                "--work-dir",
                str(work_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    printed = capsys.readouterr().out
    assert (
        "9 trial(s) planned: 3 shared no-skill + 6 with-skill; estimated "
        "$1.359 at $0.151/trial" in printed
    )
    assert "candidate-0 [localized]:" in printed
    assert "candidate-1 [baseline]:" in printed
    assert printed.count("task-2: 1") == 3
    assert "dry run; no trials run and no files written" in printed
    assert not output.exists()
    assert not work_dir.exists()


@pytest.mark.asyncio
async def test_fake_rewards_produce_hand_computed_paired_delta_literals(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _task_file(tmp_path, 3))
    runner = _FakeRunner(_three_task_rewards())
    output = tmp_path / "validated.json"

    report = await run_skill_validation(
        plan,
        output,
        runner=runner,
        work_dir=tmp_path / "work",
    )

    assert report["candidates"][0]["paired_deltas"] == pytest.approx([0.1, -0.05, 0.3])
    assert report["candidates"][1]["paired_deltas"] == pytest.approx([0.0, 0.4, -0.5])
    assert report["candidates"][0]["source_marker"] == "source-0"
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.asyncio
async def test_shared_no_skill_baseline_runs_once_per_task_and_libraries_are_isolated(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _task_file(tmp_path, 3))
    runner = _FakeRunner(_three_task_rewards())

    await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
    )

    controls = [trial for trial in runner.calls if trial.candidate_id is None]
    assert [trial.task_name for trial in controls] == ["task-0", "task-1", "task-2"]
    assert len(controls) == 3
    for trial in runner.calls:
        assert SkillLibrary(trial.library_dir).admitted_skill_ids() == (
            (trial.candidate_id,) if trial.candidate_id is not None else ()
        )


@pytest.mark.asyncio
async def test_failed_trial_is_null_and_candidate_is_incomplete_while_run_continues(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _task_file(tmp_path, 3))
    runner = _FakeRunner(_three_task_rewards(), failures={("candidate-0", "task-1")})
    output = tmp_path / "validated.json"

    report = await run_skill_validation(
        plan,
        output,
        runner=runner,
        work_dir=tmp_path / "work",
    )

    assert report["candidates"][0]["paired_deltas"] == pytest.approx([0.1, None, 0.3])
    assert report["candidates"][1]["paired_deltas"] == pytest.approx([0.0, 0.4, -0.5])
    candidates = load_admission_candidates(output)
    decision = decide_skill_admission(candidates[0])
    assert decision["status"] == "incomplete"
    assert decision["measurement"]["measured_task_count"] == 2
    assert decision["measurement"]["paired_deltas"] == pytest.approx([0.1, None, 0.3])
    assert len(runner.calls) == 9


@pytest.mark.asyncio
async def test_all_treatments_for_one_candidate_can_fail_without_affecting_another(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _task_file(tmp_path, 3))
    failures = {("candidate-0", f"task-{index}") for index in range(3)}
    runner = _FakeRunner(_three_task_rewards(), failures=failures)

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
    )

    assert report["candidates"][0]["paired_deltas"] == [None, None, None]
    assert report["candidates"][1]["paired_deltas"] == pytest.approx([0.0, 0.4, -0.5])
    assert report["validation"]["summary"]["pending_trial_count"] == 3


@pytest.mark.asyncio
async def test_interrupted_run_resumes_without_repeating_completed_trials(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _task_file(tmp_path, 3))
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"
    first = _FakeRunner(_three_task_rewards())
    original_run = first.run

    async def interrupt_on_fourth(trial: ValidationTrial) -> ValidationTrialResult:
        if len(first.calls) == 3:
            raise KeyboardInterrupt
        return await original_run(trial)

    first.run = interrupt_on_fourth  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        await run_skill_validation(plan, output, runner=first, work_dir=work_dir)
    assert len(first.calls) == 3

    resumed = _FakeRunner(_three_task_rewards())
    report = await run_skill_validation(plan, output, runner=resumed, work_dir=work_dir)

    assert len(resumed.calls) == 6
    assert all(trial.candidate_id is not None for trial in resumed.calls)
    assert report["validation"]["summary"]["reused_trial_count"] == 3
    assert report["validation"]["summary"]["new_attempt_count"] == 6
    assert [attempt["reused"] for attempt in report["validation"]["attempts"][:3]] == [
        True,
        True,
        True,
    ]


@pytest.mark.asyncio
async def test_both_candidate_arms_use_one_validation_procedure(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _task_file(tmp_path, 1))
    rewards = {
        (None, "task-0"): 0.2,
        ("candidate-0", "task-0"): 0.3,
        ("candidate-1", "task-0"): 0.4,
    }
    runner = _FakeRunner(rewards)

    await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
    )

    treatments = [trial for trial in runner.calls if trial.candidate_id is not None]
    assert [trial.distillation_arm for trial in treatments] == [
        "localized",
        "baseline",
    ]
    assert {trial.procedure_id for trial in treatments} == {VALIDATION_PROCEDURE_ID}
    assert type(treatments[0]) is type(treatments[1]) is ValidationTrial


@pytest.mark.asyncio
async def test_injection_mismatch_discards_reward_instead_of_measuring_wrong_skill(
    tmp_path: Path,
) -> None:
    candidates = _candidate_file(tmp_path, arms=("localized",))
    plan = plan_skill_validation(candidates, _task_file(tmp_path, 1))

    class MismatchRunner(_FakeRunner):
        async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
            result = await super().run(trial)
            if trial.candidate_id is not None:
                return ValidationTrialResult(
                    status=ValidationTrialStatus.MEASURED,
                    reward=result.reward,
                    injected_candidate_ids=("some-other-candidate",),
                )
            return result

    runner = MismatchRunner({(None, "task-0"): 0.2, ("candidate-0", "task-0"): 0.5})

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
    )

    assert report["candidates"][0]["paired_deltas"] == [None]
    attempt = report["validation"]["attempts"][-1]
    assert attempt["status"] == "failed"
    assert attempt["reward"] is None
    assert "skill injection mismatch" in attempt["reason"]


@pytest.mark.asyncio
async def test_ten_task_output_feeds_admit_skills_without_transformation(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _task_file(tmp_path, 10))
    rewards: dict[tuple[str | None, str], float] = {}
    for index in range(10):
        task = f"task-{index}"
        rewards[(None, task)] = 0.4
        rewards[("candidate-0", task)] = 0.5 if index < 9 else 0.4
        rewards[("candidate-1", task)] = 0.5 if index < 8 else 0.4
    output = tmp_path / "validated.json"
    await run_skill_validation(
        plan,
        output,
        runner=_FakeRunner(rewards),
        work_dir=tmp_path / "work",
    )

    admission_output = tmp_path / "admission.json"
    assert (
        main(
            [
                "admit-skills",
                str(output),
                "--library-dir",
                str(tmp_path / "admitted-library"),
                "--output",
                str(admission_output),
            ]
        )
        == 0
    )

    admission = json.loads(admission_output.read_text(encoding="utf-8"))
    assert admission["tested_candidate_count"] == 2
    assert admission["incomplete_candidate_count"] == 0
    assert admission["admitted_candidate_count"] == 1
    assert admission["rejected_candidate_count"] == 1


def test_real_cli_path_can_be_driven_with_offline_trial_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_rewards = _three_task_rewards()
    fake = _FakeRunner(plan_rewards)
    preflight_calls = 0
    probe_calls = 0

    def fake_preflight(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal preflight_calls
        preflight_calls += 1
        return {}

    def fake_probe(*args: object, **kwargs: object) -> None:
        nonlocal probe_calls
        probe_calls += 1

    monkeypatch.setattr(experiment, "preflight", fake_preflight)
    monkeypatch.setattr(experiment, "probe_provider", fake_probe)
    monkeypatch.setattr(
        experiment, "_HarborSkillValidationRunner", lambda **kwargs: fake
    )
    output = tmp_path / "validated.json"

    assert (
        main(
            [
                "validate-skills",
                str(_candidate_file(tmp_path)),
                str(_task_file(tmp_path, 3)),
                "--output",
                str(output),
                "--work-dir",
                str(tmp_path / "work"),
                "--skill-embedder",
                "offline_embedder:embed",
            ]
        )
        == 0
    )

    assert preflight_calls == 1
    assert probe_calls == 2
    assert Counter(trial.condition for trial in fake.calls) == {
        "without_skill": 3,
        "with_skill": 6,
    }
    assert load_admission_candidates(output)[0].paired_deltas == pytest.approx(
        (0.1, -0.05, 0.3)
    )
