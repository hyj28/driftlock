from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import driftlock.lhtb_experiment as experiment
from driftlock.lhtb_experiment import (
    PreflightError,
    build_job_config,
    main,
    select_tasks,
)


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
    assert agent["kwargs"]["driftlock_judge_model"] == experiment.DEFAULT_JUDGE_MODEL
    assert agent["kwargs"]["driftlock_judge_api_base"] == experiment.DEFAULT_API_BASE
    assert agent["kwargs"]["enable_summarize"] is False
    assert agent["env"]["HB_CONTINUE_MODE"] == "same_conversation"
    assert len(agent["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"]) == 64
    assert "num_retries" not in agent["kwargs"]["llm_call_kwargs"]
    assert "max_retries" not in agent["kwargs"]["llm_call_kwargs"]
    assert config["retry"]["max_retries"] == 0
    assert config["datasets"][0]["task_names"] == ["task-a", "task-b"]


@pytest.mark.parametrize(
    ("arm", "import_path", "has_fine_judge"),
    [
        (
            "retry",
            "driftlock.harbor_agent:LHTBBlindRetryAgent",
            False,
        ),
        (
            "driftlock-heuristic",
            "driftlock.harbor_agent:LHTBDriftlockAgent",
            False,
        ),
        (
            "driftlock",
            "driftlock.harbor_agent:LHTBDriftlockAgent",
            True,
        ),
    ],
)
def test_build_controlled_arm_configs(
    tmp_path: Path,
    arm: str,
    import_path: str,
    has_fine_judge: bool,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name=f"arm-{arm}",
        arm=arm,
        tasks=["task-a"],
        max_total_tokens=999,
        judge_api_base="https://judge.invalid/v1",
    )

    agent = config["agents"][0]
    assert agent["import_path"] == import_path
    assert agent["env"]["HB_CONTINUE_MODE"] == "same_conversation"
    assert agent["kwargs"]["driftlock_max_tokens"] == 999
    assert ("driftlock_judge_model" in agent["kwargs"]) is has_fine_judge
    if has_fine_judge:
        assert (
            agent["kwargs"]["driftlock_judge_model"] == experiment.DEFAULT_JUDGE_MODEL
        )
        assert agent["kwargs"]["driftlock_judge_api_base"] == "https://judge.invalid/v1"


def test_oracle_cannot_be_misrepresented_as_an_online_agent(tmp_path: Path) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    with pytest.raises(ValueError, match="hidden-verifier checkpoint replay"):
        build_job_config(
            lhtb_dir=root,
            jobs_dir=tmp_path / "jobs",
            job_name="not-an-oracle",
            arm="oracle",
            tasks=["task-a"],
        )


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
    assert agent["env"]["HB_CONTINUE_MODE"] == "fresh"
    assert len(agent["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"]) == 64
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


@pytest.mark.parametrize(
    ("arm", "expected_mode"),
    [
        ("driftlock", "same_conversation"),
        ("driftlock-heuristic", "same_conversation"),
        ("retry", "same_conversation"),
        ("stock", None),
    ],
)
def test_run_uses_current_python_harbor_and_pins_continuation_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    expected_mode: str | None,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(experiment, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        experiment,
        "_pinned_harbor_command",
        lambda: ["/pinned/python", "/pinned/bin/harbor"],
    )
    monkeypatch.setenv("HB_CONTINUE_MODE", "ambient-wrong-mode")
    monkeypatch.setenv("HB_PROCESS_REWARD", "30,300")

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    arguments = [
        "run",
        "--lhtb-dir",
        str(root),
        "--config",
        str(tmp_path / f"{arm}.json"),
        "--job-name",
        f"{arm}-run",
        "--arm",
        arm,
        "--tasks",
        "task-a",
    ]
    if arm == "stock":
        arguments.append("--ack-unbounded-stock-tokens")

    assert main(arguments) == 0
    assert calls[0]["command"][:2] == [
        "/pinned/python",
        "/pinned/bin/harbor",
    ]
    child_env = calls[0]["env"]
    assert isinstance(child_env, dict)
    assert child_env.get("HB_CONTINUE_MODE") == expected_mode
    assert "HB_PROCESS_REWARD" not in child_env


def test_checkout_validation_rejects_task_and_non_patch_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "LHTB"
    patched = root / "harbor" / "patched.py"
    task = root / "tasks" / "task-a" / "task.toml"
    patched.parent.mkdir(parents=True)
    task.parent.mkdir(parents=True)
    patched.write_text("base\n", encoding="utf-8")
    task.write_text("base\n", encoding="utf-8")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "Test"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "base"],
    ):
        subprocess.run(command, cwd=root, check=True)
    patched.write_text("expected patch\n", encoding="utf-8")
    monkeypatch.setattr(
        experiment,
        "_PATCHED_HARBOR_SHA256",
        {"harbor/patched.py": experiment._file_sha256(patched)},
    )

    experiment._validate_checkout_contents(root)

    task.write_text("modified task\n", encoding="utf-8")
    with pytest.raises(PreflightError, match="task tree"):
        experiment._validate_checkout_contents(root)
    task.write_text("base\n", encoding="utf-8")
    patched.write_text("unexpected Harbor edit\n", encoding="utf-8")
    with pytest.raises(PreflightError, match="companion patch"):
        experiment._validate_checkout_contents(root)


def test_patched_harbor_manifest_contains_sha256_values() -> None:
    assert experiment._PATCHED_HARBOR_SHA256
    assert all(
        len(value) == 64
        and value.isascii()
        and all(character in "0123456789abcdef" for character in value)
        for value in experiment._PATCHED_HARBOR_SHA256.values()
    )
