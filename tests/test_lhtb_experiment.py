from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftlock.lhtb_experiment import build_job_config, main, select_tasks


def _lhtb_tree(tmp_path: Path, *tasks: str) -> Path:
    root = tmp_path / "LHTB"
    for task in tasks:
        task_dir = root / "tasks" / task
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text("version = '1'\n", encoding="utf-8")
    return root


def test_build_driftlock_config_has_total_budget_and_no_retries(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a", "task-b")
    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name="screen-driftlock",
        arm="driftlock",
        tasks=["task-a", "task-b", "task-a"],
        max_total_tokens=123_456,
    )

    agent = config["agents"][0]
    assert agent["import_path"] == "driftlock.harbor_agent:LHTBDriftlockAgent"
    assert agent["kwargs"]["driftlock_max_tokens"] == 123_456
    assert agent["kwargs"]["enable_summarize"] is False
    assert "num_retries" not in agent["kwargs"]["llm_call_kwargs"]
    assert "max_retries" not in agent["kwargs"]["llm_call_kwargs"]
    assert config["retry"]["max_retries"] == 0
    assert config["datasets"][0]["task_names"] == ["task-a", "task-b"]


def test_build_stock_config_matches_leaderboard_retry_behavior(tmp_path: Path) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name="stock-smoke",
        arm="stock",
        tasks=["task-a"],
    )

    agent = config["agents"][0]
    assert agent["name"] == "terminus-2"
    assert agent["kwargs"]["enable_summarize"] is True
    assert agent["kwargs"]["llm_call_kwargs"]["num_retries"] == 4


@pytest.mark.parametrize("task", ["../escape", "task/*", "", "."])
def test_build_config_rejects_unsafe_or_unknown_task(tmp_path: Path, task: str) -> None:
    root = _lhtb_tree(tmp_path, "known")
    with pytest.raises(ValueError):
        build_job_config(
            lhtb_dir=root,
            jobs_dir=tmp_path / "jobs",
            job_name="safe",
            arm="stock",
            tasks=[task],
        )


def _trial(job: Path, name: str, task: str, reward: float | None) -> None:
    trial = job / name
    trial.mkdir(parents=True)
    verifier = None if reward is None else {"rewards": {"reward": reward}}
    payload = {
        "task_name": task,
        "verifier_result": verifier,
        "exception_info": (
            {"exception_type": "AgentTimeoutError"} if reward is None else None
        ),
    }
    (trial / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_select_tasks_uses_measured_mean_partial_credit(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _trial(job, "a-1", "a", 0.2)
    _trial(job, "a-2", "a", 0.6)
    _trial(job, "b-1", "b", 0.8)
    _trial(job, "solved-1", "solved", 0.95)
    _trial(job, "zero-1", "zero", 0.0)
    _trial(job, "failed-1", "failed", None)

    report = select_tasks([job], limit=2)

    assert report["selected_tasks"] == ["b", "a"]
    assert report["eligible"][1]["mean_reward"] == pytest.approx(0.4)
    assert report["eligible"][1]["attempts"] == 2
    assert {item["task"] for item in report["failures"]} == {"failed"}


def test_prepare_cli_writes_json_without_credentials_or_harbor(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    output = tmp_path / "job.json"

    assert (
        main(
            [
                "prepare",
                "--lhtb-dir",
                str(root),
                "--jobs-dir",
                str(tmp_path / "jobs"),
                "--config",
                str(output),
                "--job-name",
                "prepared",
                "--arm",
                "driftlock",
                "--tasks",
                "task-a",
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["job_name"] == "prepared"
