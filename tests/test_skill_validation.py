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


def _lhtb_tree(tmp_path: Path, *tasks: str) -> Path:
    root = tmp_path / "LHTB"
    for task in tasks:
        task_dir = root / "tasks" / task
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            f"[task]\nname = 'long-horizon-terminal-bench/{task}'\n",
            encoding="utf-8",
        )
    return root


def _candidate_file(
    tmp_path: Path,
    arms: tuple[str, ...] = ("localized", "baseline"),
    task_names: tuple[str, ...] = ("task-0", "task-1"),
) -> Path:
    assert len(arms) == len(task_names)
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
                        "task_name": f"long-horizon-terminal-bench/{task_names[index]}",
                        "source_marker": f"source-{index}",
                    }
                    for index, arm in enumerate(arms)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _rewards(
    task_names: tuple[str, ...] = ("task-0", "task-1"),
) -> dict[tuple[str | None, str, int], float]:
    rewards: dict[tuple[str | None, str, int], float] = {}
    for task_name in dict.fromkeys(task_names):
        for replicate_index in range(1, 11):
            rewards[(None, task_name, replicate_index)] = 0.4
    for candidate_index, task_name in enumerate(task_names):
        for replicate_index in range(1, 11):
            rewards[(f"candidate-{candidate_index}", task_name, replicate_index)] = 0.5
    return rewards


class _FakeRunner:
    def __init__(
        self,
        rewards: dict[tuple[str | None, str, int], float],
        *,
        failures: set[tuple[str | None, str, int]] | None = None,
    ) -> None:
        self.rewards = rewards
        self.failures = failures or set()
        self.calls: list[ValidationTrial] = []

    async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
        self.calls.append(trial)
        key = (trial.candidate_id, trial.task_name, trial.replicate_index)
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


@pytest.mark.asyncio
async def test_dry_run_reports_exact_cost_and_breakdown_without_writes(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0", "task-1")
    plan = plan_skill_validation(_candidate_file(tmp_path), root)
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
        "planned_trial_count": 40,
        "completed_attempt_count": 0,
        "measured_trial_count": 0,
        "pending_trial_count": 40,
        "distinct_source_task_count": 2,
        "observations_per_candidate": 10,
        "paired_observation_count": 20,
        "shared_control_trial_count": 20,
        "treatment_trial_count": 20,
        "reused_trial_count": 0,
        "new_attempt_count": 0,
        "failed_attempt_count": 0,
        "estimated_cost_per_trial_usd": 0.151,
        "estimated_planned_cost_usd": 6.04,
        "estimated_pending_cost_usd": 6.04,
        "status_counts": {},
    }
    assert not output.exists()
    assert not work_dir.exists()


def test_validate_skills_cli_dry_run_prints_candidate_task_costs_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _lhtb_tree(tmp_path, "task-0", "task-1")
    candidates = _candidate_file(tmp_path)
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"

    assert (
        main(
            [
                "validate-skills",
                str(candidates),
                "--lhtb-dir",
                str(root),
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
        "40 trial(s) planned: 20 shared no-skill + 20 with-skill; estimated "
        "$6.040 at $0.151/trial" in printed
    )
    assert "candidate-0 [localized] from long-horizon-terminal-bench/task-0:" in printed
    assert "candidate-1 [baseline] from long-horizon-terminal-bench/task-1:" in printed
    assert "task-0 replicate 10: 1 with-skill trial" in printed
    assert "dry run; no trials run and no files written" in printed
    assert not output.exists()
    assert not work_dir.exists()


@pytest.mark.asyncio
async def test_fake_rewards_produce_hand_computed_paired_delta_literals(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0", "task-1")
    plan = plan_skill_validation(_candidate_file(tmp_path), root)
    rewards = _rewards()
    rewards[("candidate-0", "task-0", 2)] = 0.35
    rewards[("candidate-0", "task-0", 3)] = 0.7
    rewards[("candidate-1", "task-1", 2)] = 0.8
    rewards[("candidate-1", "task-1", 3)] = 0.1
    output = tmp_path / "validated.json"

    report = await run_skill_validation(
        plan,
        output,
        runner=_FakeRunner(rewards),
        work_dir=tmp_path / "work",
    )

    assert report["candidates"][0]["paired_deltas"] == pytest.approx(
        [0.1, -0.05, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    assert report["candidates"][1]["paired_deltas"] == pytest.approx(
        [0.1, 0.4, -0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    assert report["candidates"][0]["source_marker"] == "source-0"
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.asyncio
async def test_shared_no_skill_baseline_runs_once_per_task_and_libraries_are_isolated(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0", "task-1")
    plan = plan_skill_validation(_candidate_file(tmp_path), root)
    runner = _FakeRunner(_rewards())

    await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
    )

    controls = [trial for trial in runner.calls if trial.candidate_id is None]
    assert Counter(trial.task_name for trial in controls) == {
        "task-0": 10,
        "task-1": 10,
    }
    assert len(controls) == 20
    for trial in runner.calls:
        assert SkillLibrary(trial.library_dir).admitted_skill_ids() == (
            (trial.candidate_id,) if trial.candidate_id is not None else ()
        )


@pytest.mark.asyncio
async def test_failed_trial_is_null_and_candidate_is_incomplete_while_run_continues(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0", "task-1")
    plan = plan_skill_validation(_candidate_file(tmp_path), root)
    runner = _FakeRunner(_rewards(), failures={("candidate-0", "task-0", 2)})
    output = tmp_path / "validated.json"

    report = await run_skill_validation(
        plan,
        output,
        runner=runner,
        work_dir=tmp_path / "work",
    )

    assert report["candidates"][0]["paired_deltas"] == pytest.approx(
        [0.1, None, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    assert report["candidates"][1]["paired_deltas"] == pytest.approx([0.1] * 10)
    decision = decide_skill_admission(load_admission_candidates(output)[0])
    assert decision["status"] == "incomplete"
    assert decision["measurement"]["measured_task_count"] == 9
    assert decision["measurement"]["paired_deltas"] == pytest.approx(
        [0.1, None, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    assert len(runner.calls) == 40


@pytest.mark.asyncio
async def test_all_treatments_for_one_candidate_can_fail_without_affecting_another(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0", "task-1")
    plan = plan_skill_validation(_candidate_file(tmp_path), root)
    failures = {("candidate-0", "task-0", index) for index in range(1, 11)}
    runner = _FakeRunner(_rewards(), failures=failures)

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
    )

    assert report["candidates"][0]["paired_deltas"] == [None] * 10
    assert report["candidates"][1]["paired_deltas"] == pytest.approx([0.1] * 10)
    assert report["validation"]["summary"]["pending_trial_count"] == 10


@pytest.mark.asyncio
async def test_interrupted_run_resumes_without_repeating_completed_trials(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0", "task-1")
    plan = plan_skill_validation(_candidate_file(tmp_path), root)
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"
    first = _FakeRunner(_rewards())
    original_run = first.run

    async def interrupt_on_fourth(trial: ValidationTrial) -> ValidationTrialResult:
        if len(first.calls) == 3:
            raise KeyboardInterrupt
        return await original_run(trial)

    first.run = interrupt_on_fourth  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        await run_skill_validation(plan, output, runner=first, work_dir=work_dir)
    assert len(first.calls) == 3

    resumed = _FakeRunner(_rewards())
    report = await run_skill_validation(plan, output, runner=resumed, work_dir=work_dir)

    assert len(resumed.calls) == 37
    assert {trial.trial_id for trial in first.calls}.isdisjoint(
        trial.trial_id for trial in resumed.calls
    )
    assert report["validation"]["summary"]["reused_trial_count"] == 3
    assert report["validation"]["summary"]["new_attempt_count"] == 37
    assert [attempt["reused"] for attempt in report["validation"]["attempts"][:3]] == [
        True,
        True,
        True,
    ]


@pytest.mark.asyncio
async def test_both_candidate_arms_use_one_validation_procedure(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0")
    candidates = _candidate_file(tmp_path, task_names=("task-0", "task-0"))
    payload = json.loads(candidates.read_text(encoding="utf-8"))
    payload["candidates"][1]["skill"] = payload["candidates"][0]["skill"]
    candidates.write_text(json.dumps(payload), encoding="utf-8")
    plan = plan_skill_validation(candidates, root)
    runner = _FakeRunner(_rewards(("task-0", "task-0")))

    await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
    )

    treatments = [trial for trial in runner.calls if trial.candidate_id is not None]
    assert [trial.distillation_arm for trial in treatments] == [
        "localized",
        "localized",
        "localized",
        "localized",
        "localized",
        "localized",
        "localized",
        "localized",
        "localized",
        "localized",
        "baseline",
        "baseline",
        "baseline",
        "baseline",
        "baseline",
        "baseline",
        "baseline",
        "baseline",
        "baseline",
        "baseline",
    ]
    assert {trial.procedure_id for trial in treatments} == {VALIDATION_PROCEDURE_ID}
    assert {type(trial) for trial in treatments} == {ValidationTrial}
    assert [
        (trial.task_name, trial.replicate_index, trial.condition, trial.procedure_id)
        for trial in treatments[:10]
    ] == [
        (trial.task_name, trial.replicate_index, trial.condition, trial.procedure_id)
        for trial in treatments[10:]
    ]


@pytest.mark.asyncio
async def test_injection_mismatch_discards_reward_instead_of_measuring_wrong_skill(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0")
    candidates = _candidate_file(tmp_path, arms=("localized",), task_names=("task-0",))
    plan = plan_skill_validation(candidates, root)

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

    runner = MismatchRunner(_rewards(("task-0",)))

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
    )

    assert report["candidates"][0]["paired_deltas"] == [None] * 10
    attempt = report["validation"]["attempts"][-1]
    assert attempt["status"] == "failed"
    assert attempt["reward"] is None
    assert "skill injection mismatch" in attempt["reason"]


@pytest.mark.asyncio
async def test_ten_task_output_feeds_admit_skills_without_transformation(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0", "task-1")
    plan = plan_skill_validation(_candidate_file(tmp_path), root)
    rewards = _rewards()
    rewards[("candidate-0", "task-0", 10)] = 0.4
    rewards[("candidate-1", "task-1", 9)] = 0.4
    rewards[("candidate-1", "task-1", 10)] = 0.4
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
    root = _lhtb_tree(tmp_path, "task-0", "task-1")
    fake = _FakeRunner(_rewards())
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
                "--lhtb-dir",
                str(root),
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
        "without_skill": 20,
        "with_skill": 20,
    }
    assert load_admission_candidates(output)[0].paired_deltas == pytest.approx(
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    )
