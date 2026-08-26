from __future__ import annotations

import hashlib
import io
import json
import subprocess
import urllib.error
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
    assert agent["kwargs"]["driftlock_judge_max_output_tokens"] == 8192
    assert agent["kwargs"]["enable_summarize"] is False
    assert agent["env"]["HB_CONTINUE_MODE"] == "same_conversation"
    assert len(agent["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"]) == 64
    assert "num_retries" not in agent["kwargs"]["llm_call_kwargs"]
    assert "max_retries" not in agent["kwargs"]["llm_call_kwargs"]
    assert config["retry"]["max_retries"] == 0
    assert config["datasets"][0]["task_names"] == ["task-a", "task-b"]


@pytest.mark.parametrize(
    "arm",
    [
        "driftlock-heuristic",
        "driftlock",
        "native-driftlock-heuristic",
        "native-driftlock",
    ],
)
def test_build_driftlock_config_records_every_detector_threshold(
    tmp_path: Path, arm: str
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")

    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name=f"detectors-{arm}",
        arm=arm,
        tasks=["task-a"],
    )

    kwargs = config["agents"][0]["kwargs"]
    assert {
        name: kwargs[name]
        for name in (
            "driftlock_no_change_steps",
            "driftlock_loop_window",
            "driftlock_loop_repetitions",
            "driftlock_error_window",
            "driftlock_error_rate",
            "driftlock_command_failure_window",
            "driftlock_command_failure_rate",
            "driftlock_reward_stall_steps",
            "driftlock_reward_epsilon",
            "driftlock_corroborating_signals",
        )
    } == {
        "driftlock_no_change_steps": 4,
        "driftlock_loop_window": 6,
        "driftlock_loop_repetitions": 3,
        "driftlock_error_window": 5,
        "driftlock_error_rate": 0.6,
        "driftlock_command_failure_window": 8,
        "driftlock_command_failure_rate": 1.0,
        "driftlock_reward_stall_steps": 5,
        "driftlock_reward_epsilon": 0.000001,
        "driftlock_corroborating_signals": ["no_file_change"],
    }


def test_build_stock_config_has_no_detector_thresholds(tmp_path: Path) -> None:
    root = _lhtb_tree(tmp_path, "task-a")

    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name="stock-without-detectors",
        arm="stock",
        tasks=["task-a"],
    )

    kwargs = config["agents"][0]["kwargs"]
    assert not any(name.startswith("driftlock_") for name in kwargs)


@pytest.mark.parametrize(
    "arm",
    [
        "stock",
        "retry",
        "driftlock-heuristic",
        "driftlock",
        "native-driftlock-heuristic",
        "native-driftlock",
    ],
)
def test_every_paid_agent_arm_has_strict_provider_routing(
    tmp_path: Path, arm: str
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")

    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name=f"provider-{arm}",
        arm=arm,
        tasks=["task-a"],
    )

    assert config["agents"][0]["kwargs"]["llm_call_kwargs"]["extra_body"] == {
        "provider": {"only": ["deepinfra/fp8"], "allow_fallbacks": False}
    }


def test_agent_and_judge_provider_routes_are_independently_configurable(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")

    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name="separate-providers",
        arm="driftlock",
        tasks=["task-a"],
        provider="streamlake/fp8",
        judge_provider="alibaba",
    )

    kwargs = config["agents"][0]["kwargs"]
    assert kwargs["llm_call_kwargs"]["extra_body"] == {
        "provider": {"only": ["streamlake/fp8"], "allow_fallbacks": False}
    }
    assert kwargs["driftlock_judge_llm_call_kwargs"] == {
        "extra_body": {"provider": {"only": ["alibaba"], "allow_fallbacks": False}}
    }


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
        assert agent["kwargs"]["driftlock_judge_max_output_tokens"] == 8192


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


def _retained_source_trial(
    root: Path,
    job: Path,
    *,
    retain_checkpoints: bool = True,
    checkpoints_by_phase: dict[int, int] | None = None,
    fingerprint: str | None = None,
) -> tuple[Path, str]:
    trial_id = str(uuid4())
    trial = job / "task-a.1-of-1.2026-01-01"
    checkpoint_counts = {0: 1} if checkpoints_by_phase is None else checkpoints_by_phase
    checkpoint_ids = iter(("c" * 32, "d" * 32, "e" * 32, "f" * 32))
    phase_records = []
    for phase, checkpoint_count in sorted(checkpoint_counts.items()):
        checkpoints = (
            trial / ".driftlock-checkpoints" / f"phase-{phase}" / "checkpoints"
        )
        checkpoints.mkdir(parents=True)
        for index in range(checkpoint_count):
            checkpoint_id = next(checkpoint_ids)
            checkpoint = checkpoints / checkpoint_id
            checkpoint.mkdir()
            archive = f"checkpoint archive {phase} {index}".encode()
            state_text = '{"conversation":[]}'
            digest = hashlib.sha256(archive)
            digest.update(b"\0state\0")
            digest.update(state_text.encode())
            step = phase * 10 + index * 5 + 5
            (checkpoint / "workspace.tar.gz").write_bytes(archive)
            (checkpoint / "state.json").write_text(state_text, encoding="utf-8")
            (checkpoint / "manifest.json").write_text(
                json.dumps(
                    {
                        "checkpoint_id": checkpoint_id,
                        "step": step,
                        "created_at": datetime.now(UTC).isoformat(),
                        "digest": digest.hexdigest(),
                        "parent_id": None,
                        "label": f"step-{step}",
                        "remote_workspace": "/app",
                    }
                ),
                encoding="utf-8",
            )
        phase_records.append(
            {
                "phase": phase,
                "checkpoint_dir": str(checkpoints.parent.resolve()),
                "checkpoints_retained": True,
                "status": "completed",
                "checkpoint_count": checkpoint_count,
            }
        )
    audit_dir = trial / "agent"
    audit_dir.mkdir(parents=True)
    (audit_dir / "driftlock-result.json").write_text(
        json.dumps({"phases": phase_records}),
        encoding="utf-8",
    )
    source_fingerprint = fingerprint or experiment.lhtb_experiment_fingerprint()
    source_environment = {
        "HB_CONTINUE_MODE": "same_conversation",
        "DRIFTLOCK_EXPERIMENT_FINGERPRINT": source_fingerprint,
    }
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
                "env": source_environment,
                "kwargs": {
                    "enable_summarize": False,
                    "driftlock_max_tokens": 10_000,
                    "driftlock_max_steps": 500,
                    "driftlock_max_rollbacks": 3,
                    "driftlock_checkpoint_interval": 5,
                    "llm_call_kwargs": {
                        "extra_body": {
                            "provider": {
                                "only": ["deepinfra/fp8"],
                                "allow_fallbacks": False,
                            }
                        }
                    },
                    **(
                        {"driftlock_retain_checkpoints": True}
                        if retain_checkpoints
                        else {}
                    ),
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
    (job / "lock.json").write_text(
        json.dumps({"trials": [{"agent": {"env": source_environment}}]}),
        encoding="utf-8",
    )
    return result_file, trial_id


def _replace_source_usage_with_trajectory(result_file: Path) -> None:
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    payload["agent_result"] = None
    result_file.write_text(json.dumps(payload), encoding="utf-8")
    agent_dir = result_file.parent / "agent"
    for number in range(2):
        (agent_dir / f"episode-{number}").mkdir()
    (agent_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "source": "agent",
                        "metrics": {
                            "prompt_tokens": 11,
                            "cached_tokens": 4,
                            "completion_tokens": 3,
                            "cost_usd": 0.125,
                        },
                    },
                    {
                        "source": "agent",
                        "metrics": {
                            "prompt_tokens": 13,
                            "cached_tokens": 5,
                            "completion_tokens": 4,
                            "cost_usd": 0.25,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


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
    assert candidate["checkpoint_phase"] == 0
    assert candidate["checkpoint_coverage"] == {
        "retained_phases": [0],
        "scope": "whole-trial",
        "whole_trial": True,
    }
    config = json.loads(Path(candidate["config"]).read_text())
    agent = config["agents"][0]
    assert agent["import_path"] == "driftlock.harbor_agent:LHTBCheckpointReplayOracle"
    assert agent["env"]["HB_CONTINUE_MODE"] == "same_conversation"
    assert agent["kwargs"]["driftlock_source_result"] == str(result_file.resolve())
    assert agent["kwargs"]["driftlock_source_usage"]["input_tokens"] == 100
    assert agent["kwargs"]["driftlock_source_usage_source"] == "agent_result"
    assert agent["kwargs"]["driftlock_checkpoint_coverage"] == {
        "retained_phases": [0],
        "scope": "whole-trial",
        "whole_trial": True,
    }
    assert config["n_concurrent_trials"] == 1
    assert config["retry"] == {"max_retries": 0}


def test_prepare_oracle_replays_records_whole_trial_multiphase_coverage(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    source = tmp_path / "source-job"
    _retained_source_trial(root, source, checkpoints_by_phase={0: 1, 1: 1})

    manifest = prepare_oracle_replays(
        lhtb_dir=root,
        source_job_dir=source,
        output_dir=tmp_path / "oracle",
    )

    assert manifest["candidate_count"] == 2
    assert [item["checkpoint_phase"] for item in manifest["candidates"]] == [0, 1]
    assert [item["checkpoint_coverage"] for item in manifest["candidates"]] == [
        {
            "retained_phases": [0, 1],
            "scope": "whole-trial",
            "whole_trial": True,
        },
        {
            "retained_phases": [0, 1],
            "scope": "whole-trial",
            "whole_trial": True,
        },
    ]


def test_prepare_oracle_replays_accepts_unflagged_phase_zero_checkpoints(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    source = tmp_path / "source-job"
    _retained_source_trial(
        root,
        source,
        retain_checkpoints=False,
        checkpoints_by_phase={0: 2},
    )

    manifest = prepare_oracle_replays(
        lhtb_dir=root,
        source_job_dir=source,
        output_dir=tmp_path / "oracle",
    )

    assert manifest["candidate_count"] == 2
    assert [item["checkpoint_phase"] for item in manifest["candidates"]] == [0, 0]
    assert [item["checkpoint_step"] for item in manifest["candidates"]] == [5, 10]
    assert manifest["candidates"][0]["checkpoint_coverage"] == {
        "retained_phases": [0],
        "scope": "prefix",
        "whole_trial": False,
    }
    first_config = json.loads(
        Path(manifest["candidates"][0]["config"]).read_text(encoding="utf-8")
    )
    assert first_config["agents"][0]["kwargs"]["driftlock_checkpoint_coverage"] == {
        "retained_phases": [0],
        "scope": "prefix",
        "whole_trial": False,
    }


@pytest.mark.parametrize("fingerprint", [None, "abc"])
def test_prepare_oracle_replays_refuses_missing_or_malformed_source_fingerprint(
    tmp_path: Path, fingerprint: str | None
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    source = tmp_path / "source-job"
    result_file, _ = _retained_source_trial(root, source)
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    environment = payload["config"]["agent"]["env"]
    if fingerprint is None:
        del environment["DRIFTLOCK_EXPERIMENT_FINGERPRINT"]
    else:
        environment["DRIFTLOCK_EXPERIMENT_FINGERPRINT"] = fingerprint
    result_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing or malformed build fingerprint"):
        prepare_oracle_replays(
            lhtb_dir=root,
            source_job_dir=source,
            output_dir=tmp_path / "oracle",
        )


@pytest.mark.parametrize("fingerprint", [None, "abc"])
def test_prepare_oracle_replays_refuses_missing_or_malformed_lock_fingerprint(
    tmp_path: Path, fingerprint: str | None
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    source = tmp_path / "source-job"
    _retained_source_trial(root, source)
    lock_file = source / "lock.json"
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    environment = lock["trials"][0]["agent"]["env"]
    if fingerprint is None:
        del environment["DRIFTLOCK_EXPERIMENT_FINGERPRINT"]
    else:
        environment["DRIFTLOCK_EXPERIMENT_FINGERPRINT"] = fingerprint
    lock_file.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="missing or malformed build fingerprint in source Harbor lock audit",
    ):
        prepare_oracle_replays(
            lhtb_dir=root,
            source_job_dir=source,
            output_dir=tmp_path / "oracle",
        )


def test_prepare_oracle_replays_refuses_result_and_lock_fingerprint_disagreement(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    source = tmp_path / "source-job"
    _retained_source_trial(root, source, fingerprint="a" * 64)
    lock_file = source / "lock.json"
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    lock["trials"][0]["agent"]["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"] = "b" * 64
    lock_file.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(
        ValueError, match="result fingerprint disagrees with Harbor lock audit"
    ):
        prepare_oracle_replays(
            lhtb_dir=root,
            source_job_dir=source,
            output_dir=tmp_path / "oracle",
        )


@pytest.mark.parametrize("checkpoints_by_phase", [{0: 0}, {}])
def test_prepare_oracle_replays_refuses_when_no_checkpoint_bundle_exists(
    tmp_path: Path, checkpoints_by_phase: dict[int, int]
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    source = tmp_path / "source-job"
    result_file, _ = _retained_source_trial(
        root,
        source,
        retain_checkpoints=False,
        checkpoints_by_phase=checkpoints_by_phase,
    )
    expected_pattern = (
        result_file.parent / ".driftlock-checkpoints" / "phase-*" / "checkpoints" / "*"
    )

    with pytest.raises(ValueError, match="no loadable retained checkpoints") as raised:
        prepare_oracle_replays(
            lhtb_dir=root,
            source_job_dir=source,
            output_dir=tmp_path / "oracle",
        )

    assert str(expected_pattern) in str(raised.value)


def test_oracle_prepare_cli_accepts_unflagged_retained_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    source = tmp_path / "source-job"
    _retained_source_trial(root, source, retain_checkpoints=False, fingerprint="a" * 64)
    result_file = next(source.glob("*/result.json"))
    _replace_source_usage_with_trajectory(result_file)
    output = tmp_path / "oracle"

    exit_code = main(
        [
            "oracle-prepare",
            "--lhtb-dir",
            str(root),
            "--source-job-dir",
            str(source),
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        f"wrote 1 replay configs and {output / 'oracle-manifest.json'}\n"
    )
    manifest = json.loads((output / "oracle-manifest.json").read_text())
    assert manifest["candidates"][0]["source_usage"] == {
        "input_tokens": 24,
        "cache_tokens": 9,
        "output_tokens": 7,
        "cost_usd": 0.375,
    }
    assert manifest["candidates"][0]["source_usage_source"] == "trajectory"
    assert manifest["candidates"][0]["source_fingerprint"] == "a" * 64
    assert manifest["experiment_fingerprint"] == "a" * 64
    assert manifest["candidates"][0]["checkpoint_coverage"] == {
        "retained_phases": [0],
        "scope": "prefix",
        "whole_trial": False,
    }
    config = json.loads(Path(manifest["candidates"][0]["config"]).read_text())
    assert config["agents"][0]["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"] == ("a" * 64)
    assert config["agents"][0]["kwargs"]["driftlock_source_usage"] == {
        "input_tokens": 24,
        "cache_tokens": 9,
        "output_tokens": 7,
        "cost_usd": 0.375,
    }
    assert config["agents"][0]["kwargs"]["driftlock_source_usage_source"] == (
        "trajectory"
    )


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
        "--no-provider-probe",
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


class _FakeHTTPResponse:
    def __init__(self, body: bytes = b"{}") -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_provider_probe_asks_the_pinned_provider_for_one_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")
    seen: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def opener(request: object, timeout: float = 0.0) -> _FakeHTTPResponse:
        seen.append(
            (
                request.full_url,
                json.loads(request.data),
                dict(request.headers),
            )
        )
        return _FakeHTTPResponse()

    experiment.probe_provider(
        model="openrouter/deepseek/deepseek-v4-flash-0731",
        provider="deepinfra/fp8",
        api_base="https://openrouter.ai/api/v1",
        opener=opener,
    )

    assert len(seen) == 1
    url, body, headers = seen[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    # The openrouter/ prefix is a LiteLLM routing convention, not part of the id.
    assert body["model"] == "deepseek/deepseek-v4-flash-0731"
    assert body["max_tokens"] == 1
    assert body["provider"] == {
        "only": ["deepinfra/fp8"],
        "allow_fallbacks": False,
    }
    assert headers["Authorization"] == "Bearer secret-value"


def test_provider_probe_reports_a_rate_limited_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")

    def opener(request: object, timeout: float = 0.0) -> _FakeHTTPResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error":{"code":429,"metadata":{"raw":"rate-limited"}}}'),
        )

    with pytest.raises(experiment.PreflightError) as caught:
        experiment.probe_provider(
            model="openrouter/deepseek/deepseek-v4-flash-0731",
            provider="baidu/fp8",
            api_base="https://openrouter.ai/api/v1",
            opener=opener,
        )

    message = str(caught.value)
    assert "baidu/fp8" in message
    assert "429" in message
    assert "rate-limited" in message


def test_run_probes_both_providers_before_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    monkeypatch.setattr(experiment, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        experiment, "_pinned_harbor_command", lambda: ["/pinned/bin/harbor"]
    )
    probed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        experiment,
        "probe_provider",
        lambda **kwargs: probed.append((kwargs["model"], kwargs["provider"])),
    )
    started: list[list[str]] = []
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda command, **kwargs: (
            started.append(command) or SimpleNamespace(returncode=0)
        ),
    )
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "run",
                "--lhtb-dir",
                str(root),
                "--jobs-dir",
                str(tmp_path / "jobs"),
                "--job-name",
                "probed",
                "--arm",
                "driftlock",
                "--tasks",
                "task-a",
            ]
        )
        == 0
    )

    assert probed == [
        ("openrouter/deepseek/deepseek-v4-flash-0731", "deepinfra/fp8"),
        ("openrouter/deepseek/deepseek-v4-pro-0813", "alibaba"),
    ]
    assert len(started) == 1


def test_run_does_not_probe_a_judge_the_arm_never_pays_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    monkeypatch.setattr(experiment, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        experiment, "_pinned_harbor_command", lambda: ["/pinned/bin/harbor"]
    )
    probed: list[str] = []
    monkeypatch.setattr(
        experiment,
        "probe_provider",
        lambda **kwargs: probed.append(kwargs["provider"]),
    )
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "run",
                "--lhtb-dir",
                str(root),
                "--jobs-dir",
                str(tmp_path / "jobs"),
                "--job-name",
                "heuristic-probed",
                "--arm",
                "driftlock-heuristic",
                "--tasks",
                "task-a",
            ]
        )
        == 0
    )

    assert probed == ["deepinfra/fp8"]


def test_a_dead_provider_stops_the_run_before_harbor_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    monkeypatch.setattr(experiment, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        experiment, "_pinned_harbor_command", lambda: ["/pinned/bin/harbor"]
    )

    def dead(**kwargs: object) -> None:
        raise experiment.PreflightError("pinned provider 'x' answered HTTP 429")

    monkeypatch.setattr(experiment, "probe_provider", dead)
    started: list[list[str]] = []
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda command, **kwargs: (
            started.append(command) or SimpleNamespace(returncode=0)
        ),
    )
    monkeypatch.chdir(tmp_path)

    # A failed precondition exits through argparse, like every other one.
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "run",
                "--lhtb-dir",
                str(root),
                "--jobs-dir",
                str(tmp_path / "jobs"),
                "--job-name",
                "doomed",
                "--arm",
                "stock",
                "--ack-unbounded-stock-tokens",
                "--tasks",
                "task-a",
            ]
        )

    assert caught.value.code == 2
    assert started == []


def test_two_jobs_do_not_share_a_config_path_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Concurrent arms each write a config and then hand its path to Harbor. A
    # shared default path means the last writer decides what every arm runs.
    root = _lhtb_tree(tmp_path, "task-a")
    monkeypatch.setattr(experiment, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        experiment, "_pinned_harbor_command", lambda: ["/pinned/bin/harbor"]
    )
    configs: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        configs.append(command[command.index("-c") + 1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    for arm, job in (("stock", "round-stock"), ("driftlock", "round-driftlock")):
        arguments = [
            "run",
            "--no-provider-probe",
            "--lhtb-dir",
            str(root),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--job-name",
            job,
            "--arm",
            arm,
            "--tasks",
            "task-a",
        ]
        if arm == "stock":
            arguments.append("--ack-unbounded-stock-tokens")
        assert main(arguments) == 0

    assert len(set(configs)) == 2
    for path, job in zip(configs, ("round-stock", "round-driftlock"), strict=True):
        assert json.loads(Path(path).read_text())["job_name"] == job


def test_run_refuses_a_config_another_writer_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lhtb_tree(tmp_path, "task-a")
    shared = tmp_path / "shared.json"
    monkeypatch.setattr(experiment, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        experiment, "_pinned_harbor_command", lambda: ["/pinned/bin/harbor"]
    )
    started: list[list[str]] = []
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda command, **kwargs: (
            started.append(command) or SimpleNamespace(returncode=0)
        ),
    )

    original_write = Path.write_text

    def racing_write(self: Path, *args: object, **kwargs: object) -> int:
        written = original_write(self, *args, **kwargs)
        if self == shared.resolve():
            # Another arm wins the race between our write and Harbor's read.
            original_write(
                self, json.dumps({"job_name": "someone-else"}), encoding="utf-8"
            )
        return written

    monkeypatch.setattr(Path, "write_text", racing_write)

    with pytest.raises(SystemExit, match="another job is using the same --config"):
        main(
            [
                "run",
                "--no-provider-probe",
                "--lhtb-dir",
                str(root),
                "--jobs-dir",
                str(tmp_path / "jobs"),
                "--config",
                str(shared),
                "--job-name",
                "mine",
                "--arm",
                "driftlock",
                "--tasks",
                "task-a",
            ]
        )

    assert started == []


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
