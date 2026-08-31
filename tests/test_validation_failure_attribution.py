from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import driftlock.lhtb_experiment as experiment
from driftlock.skill_distillation import Skill, serialize_skill
from driftlock.skill_validation import (
    ValidationFailureKind,
    ValidationTrial,
    ValidationTrialResult,
    ValidationTrialStatus,
    plan_skill_validation,
    run_skill_validation,
)


def _lhtb_tree(tmp_path: Path) -> Path:
    root = tmp_path / "LHTB"
    task_dir = root / "tasks" / "task-0"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        "[task]\nname = 'long-horizon-terminal-bench/task-0'\n",
        encoding="utf-8",
    )
    return root


def _candidate_file(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "skill-distillation",
                "candidates": [
                    {
                        "candidate_id": "candidate-0",
                        "arm": "localized",
                        "skill": serialize_skill(
                            Skill(
                                activation="When task 0 is active.",
                                execution="Apply candidate 0 only.",
                                termination="Stop after the verifier returns.",
                            )
                        ),
                        "paired_deltas": [],
                        "task_name": "long-horizon-terminal-bench/task-0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _trial(tmp_path: Path, *, treatment: bool = False) -> ValidationTrial:
    candidate_id = "candidate-0" if treatment else None
    available = ("candidate-0",) if treatment else ()
    library_dir = tmp_path / ("candidate-library" if treatment else "empty-library")
    library_dir.mkdir(exist_ok=True)
    return ValidationTrial(
        trial_id="treatment-1" if treatment else "control-1",
        task_name="task-0",
        replicate_index=1,
        condition="with_skill" if treatment else "without_skill",
        distillation_arm="localized",
        library_dir=library_dir,
        candidate_id=candidate_id,
        available_candidate_ids=available,
    )


def _write_run_record(
    job_dir: Path,
    candidate_ids: list[str],
    *,
    trial_name: str = "trial-0",
) -> Path:
    path = job_dir / trial_name / "agent" / "driftlock-result.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "skill_layer": {
                    "distillation_arm": "localized",
                    "injection": {
                        "status": "injected" if candidate_ids else "not_injected",
                        "candidate_ids": candidate_ids,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_job_reward(job_dir: Path, reward: float | None) -> None:
    by_reward = {} if reward is None else {str(reward): ["trial-0"]}
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "stats": {
                    "evals": {"evaluation": {"reward_stats": {"reward": by_reward}}}
                }
            }
        ),
        encoding="utf-8",
    )


def _runner(tmp_path: Path) -> experiment._HarborSkillValidationRunner:
    return experiment._HarborSkillValidationRunner(
        lhtb_dir=_lhtb_tree(tmp_path),
        work_dir=tmp_path / "work",
        skill_embedder_import_path="offline_embedder:embed",
        model="offline-model",
        provider="offline-provider",
        api_base="http://offline.invalid/v1",
        judge_api_base=None,
        judge_provider="offline-judge-provider",
        timeout_sec=60,
        max_total_tokens=100,
    )


@pytest.mark.parametrize("with_trial_directory", [False, True])
def test_missing_run_record_names_the_absent_result(
    tmp_path: Path, with_trial_directory: bool
) -> None:
    job_dir = tmp_path / "job"
    if with_trial_directory:
        (job_dir / "trial-0" / "agent").mkdir(parents=True)

    with pytest.raises(ValueError) as caught:
        experiment._validation_injection_evidence(job_dir)

    assert str(caught.value) == (
        "validation trial did not produce a result: no run record was found"
    )
    assert "skill" not in str(caught.value)


def test_empty_injection_is_valid_evidence_and_two_run_records_are_ambiguous(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    first = _write_run_record(job_dir, [])

    candidate_ids, audit = experiment._validation_injection_evidence(job_dir)

    assert candidate_ids == ()
    assert audit == {
        "record": str(first),
        "status": "not_injected",
        "candidate_ids": [],
        "distillation_arm": "localized",
    }

    _write_run_record(job_dir, [], trial_name="trial-1")
    with pytest.raises(ValueError) as caught:
        experiment._validation_injection_evidence(job_dir)
    assert str(caught.value) == "validation job has 2 run records; expected one"


@pytest.mark.parametrize("treatment", [False, True])
@pytest.mark.asyncio
async def test_ran_with_empty_injection_is_measured_for_both_conditions(
    tmp_path: Path, treatment: bool
) -> None:
    runner = _runner(tmp_path)
    trial = _trial(tmp_path, treatment=treatment)
    job_dir = runner.work_dir / "jobs" / trial.job_name
    job_dir.mkdir(parents=True)
    _write_run_record(job_dir, [])
    _write_job_reward(job_dir, 0.75)

    result = await runner.run(trial)

    assert result.status is ValidationTrialStatus.MEASURED
    assert result.reward == 0.75
    assert result.injected_candidate_ids == ()
    assert result.failure_kind is None


@pytest.mark.asyncio
async def test_runner_attributes_no_run_record_to_missing_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)
    trial = _trial(tmp_path)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=130)

    monkeypatch.setattr(experiment, "_pinned_harbor_command", lambda: ["harbor"])
    monkeypatch.setattr(experiment.subprocess, "run", fake_run)

    result = await runner.run(trial)

    assert result.status is ValidationTrialStatus.FAILED
    assert result.reward is None
    assert result.reason == (
        "validation trial did not produce a result: no run record was found"
    )
    assert result.failure_kind is ValidationFailureKind.DID_NOT_PRODUCE_RESULT


@pytest.mark.asyncio
async def test_runner_reaches_existing_no_reward_failure_with_a_run_record(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    trial = _trial(tmp_path, treatment=True)
    job_dir = runner.work_dir / "jobs" / trial.job_name
    job_dir.mkdir(parents=True)
    _write_run_record(job_dir, [])
    _write_job_reward(job_dir, None)

    result = await runner.run(trial)

    assert result.status is ValidationTrialStatus.FAILED
    assert result.reward is None
    assert result.reason == "validation job produced no reward (job recovered)"
    assert result.failure_kind is ValidationFailureKind.NO_REWARD


@pytest.mark.asyncio
async def test_runner_attributes_invalid_injection_evidence_to_the_skill_layer(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    trial = _trial(tmp_path, treatment=True)
    job_dir = runner.work_dir / "jobs" / trial.job_name
    job_dir.mkdir(parents=True)
    run_record = _write_run_record(job_dir, [])
    run_record.write_text(
        json.dumps(
            {
                "skill_layer": {
                    "distillation_arm": "localized",
                    "injection": {
                        "status": "injected",
                        "candidate_ids": "candidate-0",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    _write_job_reward(job_dir, 0.75)

    result = await runner.run(trial)

    assert result.status is ValidationTrialStatus.FAILED
    assert result.reward is None
    assert result.reason.startswith("validation skill-layer evidence is unusable:")
    assert result.failure_kind is ValidationFailureKind.SKILL_LAYER_EVIDENCE


@pytest.mark.asyncio
async def test_report_preserves_structured_failure_attribution_and_isolation(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _lhtb_tree(tmp_path))

    class MixedRunner:
        async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
            if trial.condition == "without_skill" and trial.replicate_index == 1:
                return ValidationTrialResult(
                    status=ValidationTrialStatus.FAILED,
                    reason="validation trial did not produce a result",
                    failure_kind=ValidationFailureKind.DID_NOT_PRODUCE_RESULT,
                )
            if trial.condition == "without_skill" and trial.replicate_index == 2:
                return ValidationTrialResult(
                    status=ValidationTrialStatus.FAILED,
                    reason="validation job produced no reward (process exited 1)",
                    failure_kind=ValidationFailureKind.NO_REWARD,
                )
            if trial.condition == "without_skill" and trial.replicate_index == 3:
                return ValidationTrialResult(
                    status=ValidationTrialStatus.FAILED,
                    reason="validation skill-layer evidence is unusable",
                    failure_kind=ValidationFailureKind.SKILL_LAYER_EVIDENCE,
                )
            if trial.condition == "with_skill" and trial.replicate_index == 4:
                injected_ids: tuple[str, ...] = ()
            elif trial.condition == "with_skill" and trial.replicate_index == 5:
                injected_ids = ("candidate-from-another-library",)
            else:
                injected_ids = trial.available_candidate_ids
            return ValidationTrialResult(
                status=ValidationTrialStatus.MEASURED,
                reward=0.6 if trial.condition == "with_skill" else 0.4,
                injected_candidate_ids=injected_ids,
            )

    output = tmp_path / "validated.json"
    report = await run_skill_validation(
        plan,
        output,
        runner=MixedRunner(),
        work_dir=tmp_path / "validation-work",
    )

    attempts = report["validation"]["attempts"]
    assert [attempts[index]["failure_kind"] for index in range(3)] == [
        "did_not_produce_result",
        "no_reward",
        "skill_layer_evidence",
    ]
    mismatch = next(
        attempt
        for attempt in attempts
        if attempt["condition"] == "with_skill" and attempt["replicate_index"] == 5
    )
    assert mismatch["failure_kind"] == "skill_injection_mismatch"
    assert mismatch["reason"] == (
        "skill injection mismatch: available candidate ids were ['candidate-0'], "
        "observed ['candidate-from-another-library']"
    )
    measurement = report["validation"]["paired_measurements"][3]
    assert measurement["attribution"] == "no_skill_injected"
    assert measurement["delta"] == pytest.approx(0.2)
    assert json.loads(output.read_text(encoding="utf-8")) == report
