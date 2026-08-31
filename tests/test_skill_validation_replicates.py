from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from driftlock.lhtb_experiment import main
from driftlock.skill_admission import SkillLibrary, load_admission_candidates
from driftlock.skill_distillation import Skill, serialize_skill
from driftlock.skill_validation import (
    VALIDATION_PROCEDURE_ID,
    ValidationTrial,
    ValidationTrialResult,
    ValidationTrialStatus,
    plan_skill_validation,
    run_skill_validation,
)

_V2_PROCEDURE_ID = "shared-replicated-single-candidate-paired-v2"


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


def _candidate_file(tmp_path: Path, task_names: tuple[str, ...]) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "skill-distillation",
                "candidates": [
                    {
                        "candidate_id": f"candidate-{index}",
                        "arm": "localized" if index % 2 == 0 else "baseline",
                        "skill": serialize_skill(
                            Skill(
                                activation=f"When candidate {index} applies.",
                                execution=f"Use candidate {index} only.",
                                termination="Stop after the verifier returns.",
                            )
                        ),
                        "task_name": f"long-horizon-terminal-bench/{task_name}",
                        "paired_deltas": [],
                    }
                    for index, task_name in enumerate(task_names)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _reward_table(
    task_names: tuple[str, ...],
    *,
    control: float = 0.4,
    treatment: float = 0.5,
) -> dict[tuple[str | None, str, int], float]:
    rewards: dict[tuple[str | None, str, int], float] = {}
    for task_name in dict.fromkeys(task_names):
        for replicate_index in range(1, 11):
            rewards[(None, task_name, replicate_index)] = control
    for candidate_index, task_name in enumerate(task_names):
        for replicate_index in range(1, 11):
            rewards[(f"candidate-{candidate_index}", task_name, replicate_index)] = (
                treatment
            )
    return rewards


class _ReplicateRunner:
    def __init__(
        self,
        rewards: dict[tuple[str | None, str, int], float],
        *,
        failures: set[tuple[str | None, str, int]] | None = None,
        never_inject: set[str] | None = None,
    ) -> None:
        self.rewards = rewards
        self.failures = failures or set()
        self.never_inject = never_inject or set()
        self.calls: list[ValidationTrial] = []

    async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
        self.calls.append(trial)
        key = (trial.candidate_id, trial.task_name, trial.replicate_index)
        library_ids = SkillLibrary(trial.library_dir).admitted_skill_ids()
        injected = () if trial.candidate_id in self.never_inject else library_ids
        if key in self.failures:
            return ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason="synthetic failure",
                injected_candidate_ids=injected,
            )
        return ValidationTrialResult(
            status=ValidationTrialStatus.MEASURED,
            reward=self.rewards[key],
            injected_candidate_ids=injected,
        )


def test_validation_procedure_id_pins_replicated_trial_semantics() -> None:
    assert (
        VALIDATION_PROCEDURE_ID
        == "shared-own-task-replicated-single-candidate-paired-v3"
    )


def test_five_tasks_two_replicates_plan_ten_shared_controls_and_twenty_treatments(
    tmp_path: Path,
) -> None:
    task_names = (
        "alp-paper-reproduction",
        "alp-paper-reproduction",
        "alp-paper-reproduction",
        "alp-paper-reproduction",
        "riscv-core-debug",
        "riscv-core-debug",
        "riscv-core-debug",
        "riscv-core-debug",
        "riscv-core-debug",
        "riscv-core-debug",
        "spice-ephemeris-regression",
        "spice-ephemeris-regression",
        "spice-ephemeris-regression",
        "spice-ephemeris-regression",
    )
    root = _lhtb_tree(
        tmp_path,
        "alp-paper-reproduction",
        "riscv-core-debug",
        "spice-ephemeris-regression",
    )
    plan = plan_skill_validation(_candidate_file(tmp_path, task_names), root)

    work = plan.work_items(tmp_path / "work")

    assert plan.replicate_count == 10
    assert plan.observation_count == 10
    assert plan.shared_control_trial_count == 30
    assert plan.treatment_trial_count == 140
    assert plan.planned_trial_count == 170
    assert Counter(item.condition for item in work) == {
        "without_skill": 30,
        "with_skill": 140,
    }
    controls = [item for item in work if item.condition == "without_skill"]
    assert Counter((item.task_name, item.replicate_index) for item in controls) == {
        ("alp-paper-reproduction", 1): 1,
        ("alp-paper-reproduction", 2): 1,
        ("alp-paper-reproduction", 3): 1,
        ("alp-paper-reproduction", 4): 1,
        ("alp-paper-reproduction", 5): 1,
        ("alp-paper-reproduction", 6): 1,
        ("alp-paper-reproduction", 7): 1,
        ("alp-paper-reproduction", 8): 1,
        ("alp-paper-reproduction", 9): 1,
        ("alp-paper-reproduction", 10): 1,
        ("riscv-core-debug", 1): 1,
        ("riscv-core-debug", 2): 1,
        ("riscv-core-debug", 3): 1,
        ("riscv-core-debug", 4): 1,
        ("riscv-core-debug", 5): 1,
        ("riscv-core-debug", 6): 1,
        ("riscv-core-debug", 7): 1,
        ("riscv-core-debug", 8): 1,
        ("riscv-core-debug", 9): 1,
        ("riscv-core-debug", 10): 1,
        ("spice-ephemeris-regression", 1): 1,
        ("spice-ephemeris-regression", 2): 1,
        ("spice-ephemeris-regression", 3): 1,
        ("spice-ephemeris-regression", 4): 1,
        ("spice-ephemeris-regression", 5): 1,
        ("spice-ephemeris-regression", 6): 1,
        ("spice-ephemeris-regression", 7): 1,
        ("spice-ephemeris-regression", 8): 1,
        ("spice-ephemeris-regression", 9): 1,
        ("spice-ephemeris-regression", 10): 1,
    }
    assert len({item.trial_id for item in controls}) == 30


def test_fourteen_candidate_dry_run_reports_replicate_breakdown_and_22_65_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task_names = (
        "alp",
        "alp",
        "alp",
        "alp",
        "riscv",
        "riscv",
        "riscv",
        "riscv",
        "riscv",
        "riscv",
        "spice",
        "spice",
        "spice",
        "spice",
    )
    root = _lhtb_tree(tmp_path, "alp", "riscv", "spice")
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"

    assert (
        main(
            [
                "validate-skills",
                str(_candidate_file(tmp_path, task_names)),
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
        "170 trial(s) planned: 30 shared no-skill + 140 with-skill; estimated "
        "$25.670 at $0.151/trial; 3 distinct source task(s) x 10 replicate(s) = "
        "30 shared control trial(s); 14 candidate(s) x 10 own-task observation(s) "
        "= 140 paired observation(s)" in printed
    )
    assert "alp replicate 1: 1 trial" in printed
    assert "spice replicate 10: 1 trial" in printed
    assert not output.exists()
    assert not work_dir.exists()


@pytest.mark.asyncio
async def test_same_task_replicates_have_distinct_controls_and_observations(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0")
    plan = plan_skill_validation(_candidate_file(tmp_path, ("task-0",)), root)
    rewards = _reward_table(("task-0",), control=0.2, treatment=0.5)
    rewards[(None, "task-0", 2)] = 0.7
    rewards[("candidate-0", "task-0", 2)] = 0.6

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=_ReplicateRunner(rewards),
        work_dir=tmp_path / "work",
    )

    measurements = report["validation"]["paired_measurements"]
    assert report["candidates"][0]["paired_deltas"] == pytest.approx(
        [0.3, -0.1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    )
    assert [item["replicate_index"] for item in measurements] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert len({item["control_trial_id"] for item in measurements}) == 10


@pytest.mark.asyncio
async def test_failed_treatment_replicate_leaves_other_replicate_measured(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0")
    plan = plan_skill_validation(_candidate_file(tmp_path, ("task-0",)), root)
    runner = _ReplicateRunner(
        _reward_table(("task-0",), control=0.4, treatment=0.7),
        failures={("candidate-0", "task-0", 1)},
    )

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
    )

    assert report["candidates"][0]["paired_deltas"] == pytest.approx(
        [None, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    )
    measurements = report["validation"]["paired_measurements"]
    assert [item["measured"] for item in measurements] == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert report["validation"]["summary"]["pending_trial_count"] == 1


@pytest.mark.asyncio
async def test_control_is_shared_by_candidates_within_each_task_replicate(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0")
    plan = plan_skill_validation(_candidate_file(tmp_path, ("task-0", "task-0")), root)
    runner = _ReplicateRunner(_reward_table(("task-0", "task-0")))

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
    )

    assert Counter(call.condition for call in runner.calls) == {
        "without_skill": 10,
        "with_skill": 20,
    }
    measurements = report["validation"]["paired_measurements"]
    assert [item["control_trial_id"] for item in measurements[:10]] == [
        item["control_trial_id"] for item in measurements[10:]
    ]


@pytest.mark.asyncio
async def test_different_source_tasks_have_disjoint_controls_and_no_cross_pairing(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a", "task-b")
    plan = plan_skill_validation(_candidate_file(tmp_path, ("task-a", "task-b")), root)
    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=_ReplicateRunner(_reward_table(("task-a", "task-b"))),
        work_dir=tmp_path / "work",
    )

    measurements = report["validation"]["paired_measurements"]
    assert [item["task_name"] for item in measurements[:10]] == ["task-a"] * 10
    assert [item["task_name"] for item in measurements[10:]] == ["task-b"] * 10
    assert {item["control_trial_id"] for item in measurements[:10]}.isdisjoint(
        item["control_trial_id"] for item in measurements[10:]
    )


@pytest.mark.asyncio
async def test_never_injected_treatments_are_measured_zeroes_with_attribution(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "alp")
    plan = plan_skill_validation(_candidate_file(tmp_path, ("alp",)), root)
    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=_ReplicateRunner(
            _reward_table(("alp",), control=0.4, treatment=0.4),
            never_inject={"candidate-0"},
        ),
        work_dir=tmp_path / "work",
    )

    assert report["candidates"][0]["paired_deltas"] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    summary = report["candidates"][0]["validation_observation_summary"]
    assert summary["skill_application"] == "never_injected"
    assert summary["skill_injected_observation_count"] == 0
    assert summary["no_skill_injected_observation_count"] == 10
    measurements = report["validation"]["paired_measurements"]
    assert {item["attribution"] for item in measurements} == {"no_skill_injected"}


def test_candidate_missing_task_name_is_refused_with_named_reason(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0")
    candidates = _candidate_file(tmp_path, ("task-0",))
    payload = json.loads(candidates.read_text(encoding="utf-8"))
    del payload["candidates"][0]["task_name"]
    candidates.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing_task_name"):
        plan_skill_validation(candidates, root)


def test_candidate_unknown_task_name_is_refused_with_named_reason(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0")
    candidates = _candidate_file(tmp_path, ("unknown-task",))

    with pytest.raises(ValueError, match="unresolvable_task_name"):
        plan_skill_validation(candidates, root)


@pytest.mark.asyncio
async def test_replicated_run_resumes_without_repaying_completed_trials(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0")
    plan = plan_skill_validation(_candidate_file(tmp_path, ("task-0", "task-0")), root)
    rewards = _reward_table(("task-0", "task-0"))
    first = _ReplicateRunner(rewards)
    original_run = first.run

    async def interrupt_on_fourth(trial: ValidationTrial) -> ValidationTrialResult:
        if len(first.calls) == 3:
            raise KeyboardInterrupt
        return await original_run(trial)

    first.run = interrupt_on_fourth  # type: ignore[method-assign]
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"
    with pytest.raises(KeyboardInterrupt):
        await run_skill_validation(plan, output, runner=first, work_dir=work_dir)

    resumed = _ReplicateRunner(rewards)
    report = await run_skill_validation(plan, output, runner=resumed, work_dir=work_dir)

    assert len(first.calls) == 3
    assert len(resumed.calls) == 27
    assert {call.trial_id for call in first.calls}.isdisjoint(
        call.trial_id for call in resumed.calls
    )
    assert report["validation"]["summary"]["reused_trial_count"] == 3
    assert report["validation"]["summary"]["new_attempt_count"] == 27


@pytest.mark.asyncio
async def test_changed_replicate_count_refuses_existing_cohort(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0", "task-1")
    candidates = _candidate_file(tmp_path, ("task-0",))
    first_plan = plan_skill_validation(candidates, root)
    output = tmp_path / "validated.json"
    await run_skill_validation(
        first_plan,
        output,
        runner=_ReplicateRunner(_reward_table(("task-0",))),
        work_dir=tmp_path / "work",
    )
    changed_payload = json.loads(candidates.read_text(encoding="utf-8"))
    changed_payload["candidates"][0]["task_name"] = "long-horizon-terminal-bench/task-1"
    candidates.write_text(json.dumps(changed_payload), encoding="utf-8")
    changed_plan = plan_skill_validation(candidates, root)
    changed_runner = _ReplicateRunner({})

    with pytest.raises(ValueError, match="different run"):
        await run_skill_validation(
            changed_plan,
            output,
            runner=changed_runner,
            work_dir=tmp_path / "work",
        )
    assert changed_runner.calls == []


@pytest.mark.asyncio
async def test_resume_refuses_v1_procedure_output_before_reusing_trials(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0")
    candidates = _candidate_file(tmp_path, ("task-0",))
    plan = plan_skill_validation(candidates, root)
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"
    report = await run_skill_validation(
        plan,
        output,
        runner=_ReplicateRunner(_reward_table(("task-0",))),
        work_dir=work_dir,
    )
    report["validation"]["plan"]["procedure_id"] = _V2_PROCEDURE_ID
    output.write_text(json.dumps(report), encoding="utf-8")
    resumed = _ReplicateRunner({})

    with pytest.raises(ValueError, match="different run"):
        await run_skill_validation(plan, output, runner=resumed, work_dir=work_dir)
    assert resumed.calls == []


@pytest.mark.asyncio
async def test_five_by_two_output_is_accepted_directly_by_admit_skills(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-0")
    plan = plan_skill_validation(_candidate_file(tmp_path, ("task-0", "task-0")), root)
    rewards = _reward_table(("task-0", "task-0"), control=0.4, treatment=0.5)
    rewards[("candidate-0", "task-0", 10)] = 0.4
    rewards[("candidate-1", "task-0", 9)] = 0.4
    rewards[("candidate-1", "task-0", 10)] = 0.4
    output = tmp_path / "validated.json"

    await run_skill_validation(
        plan,
        output,
        runner=_ReplicateRunner(rewards),
        work_dir=tmp_path / "work",
    )

    loaded = load_admission_candidates(output)
    assert len(loaded[0].paired_deltas) == 10
    admission_output = tmp_path / "admission.json"
    assert (
        main(
            [
                "admit-skills",
                str(output),
                "--library-dir",
                str(tmp_path / "library"),
                "--output",
                str(admission_output),
            ]
        )
        == 0
    )
    admission = json.loads(admission_output.read_text(encoding="utf-8"))
    assert admission["admitted_candidate_count"] == 1
    assert admission["rejected_candidate_count"] == 1
