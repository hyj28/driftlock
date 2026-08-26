from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import driftlock.lhtb_experiment as experiment
from driftlock.checkpoint_scoring import (
    SCORE_REPORT_NAME,
    assemble_scored_timelines,
    build_checkpoint_replay_config,
    enumerate_retained_checkpoints,
    extract_job_trial_rewards,
    load_completed_scores,
    write_score_report,
)
from driftlock.lhtb_experiment import main


def _task_tree(tmp_path: Path, *tasks: str) -> Path:
    root = tmp_path / "LHTB"
    for task in tasks:
        directory = root / "tasks" / task
        directory.mkdir(parents=True)
        (directory / "task.toml").write_text(
            f"[task]\nname = 'long-horizon-terminal-bench/{task}'\n",
            encoding="utf-8",
        )
    return root


def _trial(job: Path, trial_name: str, task: str) -> Path:
    trial = job / trial_name
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": f"long-horizon-terminal-bench/{task}",
                "metrics": None,
                "config": {
                    "agent": {
                        "import_path": "driftlock.harbor_agent:LHTBDriftlockAgent",
                        "model_name": "openrouter/source-model",
                        "env": {
                            "HB_CONTINUE_MODE": "same_conversation",
                            "DRIFTLOCK_EXPERIMENT_FINGERPRINT": "a" * 64,
                        },
                        "kwargs": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return trial


def _checkpoint(trial: Path, *, phase: int, step: int, checkpoint_id: str) -> Path:
    directory = (
        trial
        / ".driftlock-checkpoints"
        / f"phase-{phase}"
        / "checkpoints"
        / checkpoint_id
    )
    directory.mkdir(parents=True)
    archive = f"archive-{phase}-{step}".encode()
    state_text = json.dumps({"step": step})
    digest = hashlib.sha256(archive)
    digest.update(b"\0state\0")
    digest.update(state_text.encode())
    (directory / "workspace.tar.gz").write_bytes(archive)
    (directory / "state.json").write_text(state_text, encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "step": step,
                "created_at": "2026-08-25T10:00:00+00:00",
                "digest": digest.hexdigest(),
                "parent_id": None,
                "label": f"step-{step}",
                "remote_workspace": "/app",
            }
        ),
        encoding="utf-8",
    )
    return directory


def _job_result(job: Path, rewards: dict[str, list[str]]) -> None:
    job.mkdir(parents=True, exist_ok=True)
    (job / "result.json").write_text(
        json.dumps(
            {
                "stats": {
                    "evals": {
                        "agent__model__tasks": {"reward_stats": {"reward": rewards}}
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _source_job(tmp_path: Path) -> tuple[Path, Path]:
    root = _task_tree(tmp_path, "task-a", "task-b")
    job = tmp_path / "source-job"
    first = _trial(job, "task-a__trial", "task-a")
    _checkpoint(first, phase=1, step=10, checkpoint_id="b" * 32)
    _checkpoint(first, phase=0, step=5, checkpoint_id="a" * 32)
    _trial(job, "task-b__trial", "task-b")
    _job_result(job, {"0.625": ["task-a__trial"]})
    return root, job


def test_enumeration_reports_phases_steps_missing_checkpoints_and_timed_out_reward(
    tmp_path: Path,
) -> None:
    _, job = _source_job(tmp_path)

    plan = enumerate_retained_checkpoints(job)

    assert plan.trials_without_checkpoints == ("task-b__trial",)
    assert [
        (checkpoint.phase, checkpoint.step) for checkpoint in plan.trials[0].checkpoints
    ] == [(0, 5), (1, 10)]
    assert plan.trials[0].final_reward == 0.625
    assert plan.trials[1].final_reward is None


def test_enumeration_refuses_job_with_no_retained_checkpoints(tmp_path: Path) -> None:
    job = tmp_path / "source-job"
    _trial(job, "task-a__trial", "task-a")
    _job_result(job, {"0.25": ["task-a__trial"]})

    with pytest.raises(ValueError, match="no retained checkpoints"):
        enumerate_retained_checkpoints(job)


def test_job_reward_stats_are_authoritative_when_trial_metrics_are_absent() -> None:
    payload = {
        "stats": {
            "evals": {
                "first": {
                    "reward_stats": {
                        "reward": {
                            "0.712136": ["timed-out__trial"],
                            "0.3": ["completed__trial"],
                        }
                    }
                }
            }
        }
    }

    assert extract_job_trial_rewards(payload) == {
        "timed-out__trial": 0.712136,
        "completed__trial": 0.3,
    }


def test_job_reward_stats_reject_conflicting_eval_entries() -> None:
    payload = {
        "stats": {
            "evals": {
                "first": {"reward_stats": {"reward": {"0.25": ["trial"]}}},
                "second": {"reward_stats": {"reward": {"0.75": ["trial"]}}},
            }
        }
    }

    with pytest.raises(ValueError, match="conflicting job-level rewards"):
        extract_job_trial_rewards(payload)


def test_timeline_reports_checkpoint_scores_final_reward_and_headroom(
    tmp_path: Path,
) -> None:
    _, job = _source_job(tmp_path)
    plan = enumerate_retained_checkpoints(job)
    first, second = plan.trials[0].checkpoints

    report = assemble_scored_timelines(
        plan, {first.candidate_id: 0.375, second.candidate_id: 0.875}
    )

    trial = report["trials"][0]
    assert [(item["step"], item["reward"]) for item in trial["checkpoints"]] == [
        (5, 0.375),
        (10, 0.875),
    ]
    assert trial["final_reward"] == 0.625
    assert trial["best_checkpoint_reward"] == 0.875
    assert trial["headroom"] == 0.25
    unknown = report["trials"][1]
    assert unknown["final_reward"] is None
    assert "headroom" not in unknown


def test_replay_config_has_integrity_parameters_and_no_provider_access(
    tmp_path: Path,
) -> None:
    root, job = _source_job(tmp_path)
    replay = enumerate_retained_checkpoints(job).checkpoints[0]

    config = build_checkpoint_replay_config(
        lhtb_dir=root, jobs_dir=tmp_path / "scored" / "jobs", replay=replay
    )

    agent = config["agents"][0]
    assert agent["import_path"] == ("driftlock.harbor_agent:LHTBCheckpointScoringAgent")
    assert agent["kwargs"] == {
        "driftlock_scoring_checkpoint_dir": str(replay.checkpoint_dir),
        "driftlock_scoring_checkpoint_digest": replay.digest,
        "driftlock_scoring_expected_workspace": "/app",
    }
    assert "llm_call_kwargs" not in agent["kwargs"]
    assert "api_base" not in agent["kwargs"]


def test_partial_report_round_trip_loads_only_completed_scores(tmp_path: Path) -> None:
    _, job = _source_job(tmp_path)
    plan = enumerate_retained_checkpoints(job)
    first = plan.checkpoints[0]
    report_path = tmp_path / SCORE_REPORT_NAME

    write_score_report(report_path, plan, {first.candidate_id: 0.5})

    assert load_completed_scores(report_path, plan) == {first.candidate_id: 0.5}


def test_score_checkpoints_cli_resumes_without_rerunning_completed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, job = _source_job(tmp_path)
    output = tmp_path / "scored"
    calls: list[list[str]] = []
    preflight_calls: list[dict[str, object]] = []

    def fake_preflight(*args: object, **kwargs: object) -> dict[str, object]:
        preflight_calls.append({"args": args, **kwargs})
        return {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        calls.append(command)
        config = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        _job_result(
            Path(config["jobs_dir"]) / config["job_name"],
            {"0.5": [f"{config['job_name']}__trial"]},
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment, "preflight", fake_preflight)
    monkeypatch.setattr(
        experiment, "_pinned_harbor_command", lambda: ["/pinned/bin/harbor"]
    )
    monkeypatch.setattr(experiment.subprocess, "run", fake_run)

    arguments = [
        "score-checkpoints",
        "--lhtb-dir",
        str(root),
        "--source-job-dir",
        str(job),
        "--output-dir",
        str(output),
    ]
    assert main(arguments) == 0
    assert len(calls) == 2
    assert preflight_calls[0]["require_credential"] is False

    assert main(arguments) == 0
    assert len(calls) == 2
    report = json.loads((output / SCORE_REPORT_NAME).read_text(encoding="utf-8"))
    assert report["scored_checkpoint_count"] == 2


def test_score_checkpoints_cli_refuses_missing_job_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "score-checkpoints",
                "--source-job-dir",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "output"),
                "--dry-run",
            ]
        )

    assert raised.value.code == 2
    assert "source job directory does not exist" in capsys.readouterr().err
