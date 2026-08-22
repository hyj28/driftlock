from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import driftlock.lhtb_experiment as experiment
from driftlock.lhtb_analysis import task_directory_sha256
from driftlock.lhtb_experiment import (
    PreflightError,
    build_job_config,
    main,
    prepare_oracle_replays,
    select_tasks,
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


def test_checkpoint_retention_is_explicit_and_only_for_driftlock(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name="source",
        arm="driftlock",
        tasks=["task-a"],
        retain_checkpoints=True,
    )
    assert config["agents"][0]["kwargs"]["driftlock_retain_checkpoints"] is True
    with pytest.raises(ValueError, match="requires a driftlock arm"):
        build_job_config(
            lhtb_dir=root,
            jobs_dir=tmp_path / "jobs",
            job_name="invalid",
            arm="retry",
            tasks=["task-a"],
            retain_checkpoints=True,
        )


@pytest.mark.parametrize("arm", ["native-driftlock-heuristic", "native-driftlock"])
def test_native_checkpoint_retention_is_rejected_before_job_generation(
    tmp_path: Path, arm: str
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")

    with pytest.raises(ValueError, match="native oracle replay is future work"):
        build_job_config(
            lhtb_dir=root,
            jobs_dir=tmp_path / "jobs",
            job_name="unsupported-native-source",
            arm=arm,
            tasks=["task-a"],
            retain_checkpoints=True,
        )


def _retained_source_trial(root: Path, job: Path) -> tuple[Path, str]:
    trial_id = str(uuid4())
    trial = job / "task-a.1-of-1.2026-01-01"
    checkpoint_id = "c" * 32
    checkpoint = (
        trial / ".driftlock-checkpoints" / "phase-0" / "checkpoints" / checkpoint_id
    )
    checkpoint.mkdir(parents=True)
    archive = b"checkpoint archive"
    state_text = '{"conversation":[]}'
    digest = hashlib.sha256(archive)
    digest.update(b"\0state\0")
    digest.update(state_text.encode())
    (checkpoint / "workspace.tar.gz").write_bytes(archive)
    (checkpoint / "state.json").write_text(state_text, encoding="utf-8")
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "step": 5,
                "created_at": datetime.now(UTC).isoformat(),
                "digest": digest.hexdigest(),
                "parent_id": None,
                "label": "step-5",
                "remote_workspace": "/app",
            }
        ),
        encoding="utf-8",
    )
    audit_dir = trial / "agent"
    audit_dir.mkdir()
    (audit_dir / "driftlock-result.json").write_text(
        json.dumps(
            {
                "phases": [
                    {
                        "phase": 0,
                        "checkpoint_dir": str(checkpoint.parent.parent.resolve()),
                        "checkpoints_retained": True,
                        "status": "completed",
                        "checkpoint_count": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = {
        "id": trial_id,
        "task_name": "long-horizon-terminal-bench/task-a",
        "task_checksum": task_directory_sha256(root / "tasks" / "task-a"),
        "agent_info": {
            "name": "driftlock-terminus-2",
            "version": "0.1.0",
            "model_info": {
                "provider": "openrouter",
                "name": "deepseek/deepseek-v4-flash-0731",
            },
        },
        "config": {
            "agent": {
                "import_path": "driftlock.harbor_agent:LHTBDriftlockAgent",
                "model_name": experiment.DEFAULT_MODEL,
                "env": {
                    "HB_CONTINUE_MODE": "same_conversation",
                    "DRIFTLOCK_EXPERIMENT_FINGERPRINT": (
                        experiment.lhtb_experiment_fingerprint()
                    ),
                },
                "kwargs": {
                    "enable_summarize": False,
                    "driftlock_max_tokens": 10_000,
                    "driftlock_max_steps": 500,
                    "driftlock_max_rollbacks": 3,
                    "driftlock_checkpoint_interval": 5,
                    "driftlock_retain_checkpoints": True,
                },
            }
        },
        "agent_result": {
            "n_input_tokens": 100,
            "n_cache_tokens": 20,
            "n_output_tokens": 10,
            "cost_usd": 0.5,
        },
    }
    result_file = trial / "result.json"
    result_file.write_text(json.dumps(result), encoding="utf-8")
    return result_file, trial_id


def test_prepare_oracle_replays_generates_one_isolated_job_per_checkpoint(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    source = tmp_path / "source-job"
    result_file, trial_id = _retained_source_trial(root, source)

    manifest = prepare_oracle_replays(
        lhtb_dir=root,
        source_job_dir=source,
        output_dir=tmp_path / "oracle",
    )

    assert manifest["candidate_count"] == 1
    candidate = manifest["candidates"][0]
    assert candidate["source_trial_id"] == trial_id
    assert candidate["usage_policy"] == "full-source-trial-conservative"
    config = json.loads(Path(candidate["config"]).read_text())
    agent = config["agents"][0]
    assert agent["import_path"] == "driftlock.harbor_agent:LHTBCheckpointReplayOracle"
    assert agent["env"]["HB_CONTINUE_MODE"] == "same_conversation"
    assert agent["kwargs"]["driftlock_source_result"] == str(result_file.resolve())
    assert agent["kwargs"]["driftlock_source_usage"]["input_tokens"] == 100
    assert config["n_concurrent_trials"] == 1
    assert config["retry"] == {"max_retries": 0}


def test_prepare_oracle_replays_reports_native_incompatibility_explicitly(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    source = tmp_path / "source-job"
    result_file, _ = _retained_source_trial(root, source)
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    payload["config"]["agent"]["import_path"] = (
        "driftlock.harbor_native_agent:LHTBNativeDriftlockAgent"
    )
    result_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match="native retained checkpoints cannot be used for oracle replay"
    ):
        prepare_oracle_replays(
            lhtb_dir=root,
            source_job_dir=source,
            output_dir=tmp_path / "oracle",
        )


def test_oracle_run_needs_docker_but_not_provider_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "oracle"
    calls: list[dict[str, object]] = []

    def fake_preflight(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append({"kind": "preflight", "args": args, **kwargs})
        return {}

    monkeypatch.setattr(experiment, "preflight", fake_preflight)
    monkeypatch.setattr(
        experiment,
        "prepare_oracle_replays",
        lambda **kwargs: {
            "candidate_count": 1,
            "candidates": [{"config": str(tmp_path / "candidate.json")}],
        },
    )
    monkeypatch.setattr(
        experiment,
        "_pinned_harbor_command",
        lambda: ["/pinned/python", "/pinned/harbor"],
    )
    monkeypatch.setenv("HB_PROCESS_REWARD", "unsafe")

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append({"kind": "run", "command": command, **kwargs})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)

    assert (
        main(
            [
                "oracle-run",
                "--lhtb-dir",
                str(root),
                "--source-job-dir",
                str(source),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    preflight_call = calls[0]
    assert preflight_call["require_credential"] is False
    run_call = calls[1]
    assert run_call["command"][-3:] == [
        "run",
        "-c",
        str(tmp_path / "candidate.json"),
    ]
    child_env = run_call["env"]
    assert isinstance(child_env, dict)
    assert child_env["HB_CONTINUE_MODE"] == "same_conversation"
    assert "HB_PROCESS_REWARD" not in child_env


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
