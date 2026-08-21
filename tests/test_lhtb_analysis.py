from __future__ import annotations

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from driftlock.lhtb import LHTB_REPOSITORY_REVISION, lhtb_experiment_fingerprint
from driftlock.lhtb_analysis import (
    _task_directory_sha256,
    analyze_jobs,
    goal_drift_actions,
    goal_drift_inaction,
    parse_arm_directories,
)
from driftlock.lhtb_experiment import main


def _task(root: Path, name: str, *, expert_minutes: int, category: str) -> None:
    task = root / "tasks" / name
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        "\n".join(
            (
                "version = '1'",
                "[task]",
                f"name = 'long-horizon-terminal-bench/{name}'",
                "[metadata]",
                f"expert_time_estimate_min = {expert_minutes}",
                f"category = '{category}'",
                "",
            )
        ),
        encoding="utf-8",
    )


def _result(
    job: Path,
    trial: str,
    task: str,
    reward: float | None,
    *,
    checksum: str | None = None,
    model: str = "deepseek/deepseek-v4-pro",
    usage: dict[str, int | float] | None = None,
    exception: str | None = None,
    arm: str | None = None,
) -> Path:
    directory = job / trial
    directory.mkdir(parents=True)
    resolved_arm = arm or job.name
    result_task = task if "/" in task else f"long-horizon-terminal-bench/{task}"
    agent_config: dict[str, object] = {
        "name": None,
        "import_path": "driftlock.harbor_agent:LHTBDriftlockAgent",
        "model_name": "openrouter/deepseek/deepseek-v4-pro",
        "override_timeout_sec": 5400,
        "override_setup_timeout_sec": None,
        "max_timeout_sec": None,
        "env": {
            "HB_CONTINUE_MODE": "same_conversation",
            "DRIFTLOCK_EXPERIMENT_FINGERPRINT": lhtb_experiment_fingerprint(),
        },
        "kwargs": {
            "api_base": "https://openrouter.ai/api/v1",
            "parser_name": "json",
            "temperature": 0.7,
            "record_terminal_session": True,
            "llm_call_kwargs": {
                "temperature": 0.7,
                "max_tokens": 8192,
                "timeout": 240,
            },
            "model_info": {
                "max_input_tokens": 128000,
                "max_output_tokens": 8192,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
            "enable_summarize": False,
            "driftlock_max_tokens": 10_000,
            "driftlock_max_steps": 500,
            "driftlock_max_rollbacks": 3,
            "driftlock_checkpoint_interval": 5,
        },
    }
    agent_name = "driftlock-terminus-2"
    if resolved_arm == "stock":
        agent_name = "terminus-2"
        agent_config = {
            "name": "terminus-2",
            "import_path": None,
            "model_name": "openrouter/deepseek/deepseek-v4-pro",
            "override_timeout_sec": 5400,
            "override_setup_timeout_sec": None,
            "max_timeout_sec": None,
            "env": {
                "HB_CONTINUE_MODE": "fresh",
                "DRIFTLOCK_EXPERIMENT_FINGERPRINT": (lhtb_experiment_fingerprint()),
            },
            "kwargs": {
                "api_base": "https://openrouter.ai/api/v1",
                "parser_name": "json",
                "temperature": 0.7,
                "record_terminal_session": True,
                "model_info": {
                    "max_input_tokens": 128000,
                    "max_output_tokens": 8192,
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
                "enable_summarize": True,
                "proactive_summarization_threshold": 8000,
                "llm_call_kwargs": {
                    "temperature": 0.7,
                    "max_tokens": 8192,
                    "timeout": 240,
                    "num_retries": 4,
                },
            },
        }
    elif resolved_arm == "retry":
        agent_name = "compute-matched-blind-retry-terminus-2"
        agent_config["import_path"] = "driftlock.harbor_agent:LHTBBlindRetryAgent"
    elif resolved_arm == "driftlock":
        kwargs = agent_config["kwargs"]
        assert isinstance(kwargs, dict)
        kwargs.update(
            {
                "driftlock_judge_model": ("openrouter/deepseek/deepseek-v4-flash-0731"),
                "driftlock_judge_api_base": "https://judge.invalid/v1",
                "driftlock_judge_max_output_tokens": 512,
            }
        )
    payload = {
        "id": str(uuid5(NAMESPACE_URL, f"{job.resolve()}:{trial}")),
        "task_name": result_task,
        "trial_name": trial,
        "task_checksum": checksum or f"checksum-{task}",
        "agent_info": {
            "name": agent_name,
            "version": "2.0.0" if resolved_arm == "stock" else "0.1.0",
            "model_info": {"provider": "openrouter", "name": model},
        },
        "config": {
            "job_id": _job_id(job),
            "trial_name": trial,
            "agent": agent_config,
            "environment": {
                "type": "docker",
                "import_path": None,
                "force_build": True,
                "delete": True,
                "override_cpus": None,
                "override_memory_mb": None,
                "override_storage_mb": None,
                "override_gpus": None,
                "suppress_override_warnings": False,
                "mounts": None,
                "env": {},
                "kwargs": {},
            },
            "verifier": {
                "override_timeout_sec": None,
                "max_timeout_sec": None,
                "env": {},
                "disable": False,
            },
            "timeout_multiplier": 1.0,
            "agent_timeout_multiplier": None,
            "verifier_timeout_multiplier": None,
            "agent_setup_timeout_multiplier": None,
            "environment_build_timeout_multiplier": None,
            "artifacts": [],
        },
        "agent_result": usage
        or {
            "n_input_tokens": 100,
            "n_cache_tokens": 40,
            "n_output_tokens": 20,
            "cost_usd": 0.25,
        },
        "verifier_result": (
            None if reward is None else {"rewards": {"reward": reward}}
        ),
        "exception_info": (
            {"exception_type": exception} if exception is not None else None
        ),
        "started_at": "2026-08-20T10:00:00+00:00",
        "finished_at": "2026-08-20T10:02:00+00:00",
    }
    result = directory / "result.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result


def _job_summary(
    job: Path,
    n_total_trials: int,
    *,
    completed: int | None = None,
    errors: int = 0,
    retries: int = 0,
) -> None:
    usage: dict[str, int | float] = {
        "n_input_tokens": 0,
        "n_cache_tokens": 0,
        "n_output_tokens": 0,
        "cost_usd": 0.0,
    }
    for result_file in job.glob("*/result.json"):
        trial = json.loads(result_file.read_text())
        direct = trial.get("agent_result")
        if isinstance(direct, dict):
            contexts = [direct]
        else:
            contexts = [
                step["agent_result"]
                for step in trial.get("step_results", [])
                if isinstance(step, dict) and isinstance(step.get("agent_result"), dict)
            ]
        for context in contexts:
            for field in usage:
                usage[field] += context[field]
    lock_trials = []
    for result_file in sorted(job.glob("*/result.json")):
        trial = json.loads(result_file.read_text())
        config = trial["config"]
        lock_trials.append(
            {
                "task": {
                    "name": trial["task_name"].split("/")[-1],
                    "type": "local",
                    "digest": f"sha256:{trial['task_checksum']}",
                    "source": None,
                    "path": f"/tasks/{trial['task_name'].split('/')[-1]}",
                    "git_url": None,
                    "git_commit_id": None,
                },
                "timeout_multiplier": config["timeout_multiplier"],
                "agent_timeout_multiplier": config["agent_timeout_multiplier"],
                "verifier_timeout_multiplier": config["verifier_timeout_multiplier"],
                "agent_setup_timeout_multiplier": config[
                    "agent_setup_timeout_multiplier"
                ],
                "environment_build_timeout_multiplier": config[
                    "environment_build_timeout_multiplier"
                ],
                "agent": config["agent"],
                "environment": config["environment"],
                "verifier": config["verifier"],
            }
        )
    lock = {
        "schema_version": 1,
        "created_at": "2026-08-20T09:00:00+00:00",
        "harbor": {
            "version": "0.7.0",
            "git_commit_hash": LHTB_REPOSITORY_REVISION,
            "is_editable": True,
        },
        "invocation": ["harbor", "run", "-c", "driftlock-job.json"],
        "n_concurrent_trials": 1,
        "retry": {
            "max_retries": 0,
            "include_exceptions": None,
            "exclude_exceptions": [
                "AgentTimeoutError",
                "VerifierTimeoutError",
                "RewardFileNotFoundError",
                "RewardFileEmptyError",
                "VerifierOutputParseError",
            ],
            "wait_multiplier": 1.0,
            "min_wait_sec": 1.0,
            "max_wait_sec": 60.0,
        },
        "trials": lock_trials,
    }
    (job / "lock.json").write_text(json.dumps(_without_none(lock)), encoding="utf-8")
    payload = {
        "id": _job_id(job),
        "started_at": "2026-08-20T09:00:00+00:00",
        "finished_at": "2026-08-20T11:00:00+00:00",
        "n_total_trials": n_total_trials,
        "stats": {
            "n_completed_trials": (n_total_trials if completed is None else completed),
            "n_errored_trials": errors,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "n_retries": retries,
            **usage,
        },
    }
    (job / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def _job_id(job: Path) -> str:
    identifiers = {"stock": 1, "driftlock": 2, "retry": 3}
    value = identifiers.get(job.name, 99)
    return f"00000000-0000-0000-0000-{value:012d}"


def _without_none(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def _complete_jobs(tmp_path: Path) -> tuple[Path, Path, Path]:
    lhtb = tmp_path / "LHTB"
    _task(lhtb, "short", expert_minutes=60, category="build")
    _task(lhtb, "long", expert_minutes=240, category="debug")
    stock = tmp_path / "stock"
    driftlock = tmp_path / "driftlock"
    short_checksum = _task_directory_sha256(lhtb / "tasks" / "short")
    long_checksum = _task_directory_sha256(lhtb / "tasks" / "long")
    _result(stock, "short-1", "short", 1.0, checksum=short_checksum)
    _result(stock, "long-1", "long", 0.0, checksum=long_checksum)
    _result(
        driftlock,
        "short-1",
        "short",
        1.0,
        checksum=short_checksum,
        usage={
            "n_input_tokens": 80,
            "n_cache_tokens": 40,
            "n_output_tokens": 10,
            "cost_usd": 0.1,
        },
    )
    _result(
        driftlock,
        "long-1",
        "long",
        1.0,
        checksum=long_checksum,
        usage={
            "n_input_tokens": 120,
            "n_cache_tokens": 60,
            "n_output_tokens": 30,
            "cost_usd": 0.3,
        },
    )
    _job_summary(stock, 2)
    _job_summary(driftlock, 2)
    return lhtb, stock, driftlock


def test_goal_drift_formulas_match_published_definitions() -> None:
    assert goal_drift_actions(0.7, 0.4) == pytest.approx(0.3)
    assert goal_drift_actions(0.4, 0.7) == 0
    assert goal_drift_inaction(0.2, 0.6) == pytest.approx(0.4)
    assert goal_drift_inaction(0.6, 0.2) == 0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        goal_drift_actions(1.1, 0.2)


def test_parse_arm_directories_is_strict() -> None:
    assert parse_arm_directories(["stock=/a", "driftlock=/b"]) == {
        "stock": Path("/a"),
        "driftlock": Path("/b"),
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_arm_directories(["stock=/a", "stock=/b"])
    with pytest.raises(ValueError, match="unknown"):
        parse_arm_directories(["invented=/a"])
    with pytest.raises(ValueError, match="ARM=JOB_DIR"):
        parse_arm_directories(["stock"])


def test_task_checksum_matches_harbor_dirhash_protocol(tmp_path: Path) -> None:
    task = tmp_path / "task"
    (task / "nested").mkdir(parents=True)
    (task / "a.txt").write_text("alpha\n", encoding="utf-8")
    (task / "nested" / "b.bin").write_bytes(b"\x00beta")

    assert _task_directory_sha256(task) == (
        "51e6bc9129e8b8237e4c94c464058306cf66e836d26c78b32883b852f0e751da"
    )


def test_analyze_complete_matrix_reports_curves_costs_and_provenance(
    tmp_path: Path,
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    assert report["matrix"]["complete"] is True
    assert report["job_summaries"]["stock"]["n_total_trials"] == 2
    assert len(report["job_summaries"]["stock"]["result_sha256"]) == 64
    assert report["matrix"]["agent_model"] == ("openrouter/deepseek/deepseek-v4-pro")
    assert report["matrix"]["agent_versions"] == {
        "stock": "2.0.0",
        "driftlock": "0.1.0",
    }
    assert len(report["matrix"]["experiment_signature_sha256"]) == 64
    assert report["arms"]["stock"]["mean_reward"] == 0.5
    assert report["arms"]["stock"][
        "failure_slope_per_task_length_doubling"
    ] == pytest.approx(0.5)
    drift = report["arms"]["driftlock"]
    assert drift["solved_rate"] == 1.0
    assert drift["input_tokens"] == 200
    assert drift["cache_tokens"] == 100
    assert drift["output_tokens"] == 40
    assert drift["total_tokens"] == 240
    assert drift["cache_hit_rate"] == 0.5
    assert drift["cost_usd"] == pytest.approx(0.4)
    assert drift["mean_duration_sec"] == 120
    assert [item["task"] for item in drift["task_curve"]] == [
        "long-horizon-terminal-bench/short",
        "long-horizon-terminal-bench/long",
    ]
    comparison = report["paired_vs_stock"]["driftlock"]
    assert comparison["mean_task_reward_delta"] == 0.5
    assert comparison["mean_task_failure_rate_delta"] == -0.5
    assert comparison["failure_slope_delta"] == pytest.approx(-0.5)
    assert report["planned_arm_coverage"]["missing"] == [
        "retry",
        "driftlock-heuristic",
        "oracle",
    ]
    assert report["goal_drift_metrics"]["status"] == ("requires_domain_annotations")
    trial = drift["trials"][0]
    assert Path(trial["result_file"]).is_absolute()
    assert len(trial["result_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("matrix", "attempt matrix"),
        ("checksum", "does not match checkout"),
        ("model", "config model does not match"),
    ],
)
def test_analyze_rejects_noncomparable_arms(
    tmp_path: Path, mutation: str, message: str
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    if mutation == "matrix":
        _result(
            driftlock,
            "long-2",
            "long",
            1.0,
            checksum=_task_directory_sha256(lhtb / "tasks" / "long"),
        )
        _job_summary(driftlock, 3)
    elif mutation == "checksum":
        payload_path = driftlock / "long-1" / "result.json"
        payload = json.loads(payload_path.read_text())
        payload["task_checksum"] = "different"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        payload_path = driftlock / "long-1" / "result.json"
        payload = json.loads(payload_path.read_text())
        payload["agent_info"]["model_info"]["name"] = "another-model"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_rejects_swapped_arm_labels(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)

    with pytest.raises(ValueError, match="non-stock agent config"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": driftlock, "driftlock": stock},
        )


def test_analyze_binds_trial_to_job_summary_and_directory(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    payload["config"]["job_id"] = _job_id(stock)
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different Harbor job"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )

    payload["config"]["job_id"] = _job_id(driftlock)
    payload["trial_name"] = "stale-name"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="trial_name does not match directory"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_requires_globally_unique_trial_ids(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    first = json.loads((driftlock / "short-1" / "result.json").read_text())
    second_path = driftlock / "long-1" / "result.json"
    second = json.loads(second_path.read_text())
    second["id"] = first["id"]
    second_path.write_text(json.dumps(second), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate Harbor trial id"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("retry", "forbidden retries"),
        ("concurrency", "different Harbor job-level settings"),
        ("harbor", "pinned editable build"),
        ("fingerprint", "invalid trial provenance"),
    ],
)
def test_analyze_validates_canonical_job_lock(
    tmp_path: Path, mutation: str, message: str
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    lock_path = driftlock / "lock.json"
    lock = json.loads(lock_path.read_text())
    if mutation == "retry":
        lock["retry"]["max_retries"] = 1
    elif mutation == "concurrency":
        lock["n_concurrent_trials"] = 2
    elif mutation == "harbor":
        lock["harbor"]["git_commit_hash"] = "0" * 40
    else:
        lock["trials"][0]["agent"]["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"] = "0" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_distinguishes_heuristic_and_fine_judge_arms(
    tmp_path: Path,
) -> None:
    lhtb, stock, _ = _complete_jobs(tmp_path)
    heuristic = tmp_path / "driftlock-heuristic"
    for task, reward in (("short", 1.0), ("long", 0.5)):
        _result(
            heuristic,
            f"{task}-1",
            task,
            reward,
            checksum=_task_directory_sha256(lhtb / "tasks" / task),
        )
    _job_summary(heuristic, 2)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock-heuristic": heuristic},
    )
    assert "driftlock-heuristic" in report["arms"]

    with pytest.raises(ValueError, match="fine judge"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": heuristic},
        )


def test_analyze_requires_compute_matched_controlled_budgets(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    retry = tmp_path / "retry"
    for task, reward in (("short", 1.0), ("long", 0.0)):
        result = _result(
            retry,
            f"{task}-1",
            task,
            reward,
            checksum=_task_directory_sha256(lhtb / "tasks" / task),
        )
        payload = json.loads(result.read_text())
        payload["config"]["agent"]["kwargs"]["driftlock_max_tokens"] = 9_000
        result.write_text(json.dumps(payload), encoding="utf-8")
    _job_summary(retry, 2)

    with pytest.raises(ValueError, match="share one total-token budget"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={
                "stock": stock,
                "retry": retry,
                "driftlock": driftlock,
            },
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("temperature", "frozen harness"),
        ("api_base", "canonical Harbor lock"),
        ("environment", "environment differs"),
        ("version", "wrong agent config"),
        ("judge_model", "fine judge"),
    ],
)
def test_analyze_rejects_non_treatment_config_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    if mutation == "temperature":
        payload["config"]["agent"]["kwargs"]["temperature"] = 0
    elif mutation == "api_base":
        payload["config"]["agent"]["kwargs"]["api_base"] = (
            "https://different.invalid/v1"
        )
    elif mutation == "environment":
        payload["config"]["environment"]["delete"] = False
    elif mutation == "version":
        payload["agent_info"]["version"] = "9.9.9"
    else:
        payload["config"]["agent"]["kwargs"]["driftlock_judge_model"] = (
            "openrouter/another-judge"
        )
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_rejects_infrastructure_failure_instead_of_scoring_it(
    tmp_path: Path,
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    _result(
        stock,
        "failed",
        "short",
        None,
        checksum=_task_directory_sha256(lhtb / "tasks" / "short"),
        exception="AgentTimeoutError",
    )
    _job_summary(stock, 3, errors=1)

    with pytest.raises(ValueError, match="errored trials"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_requires_canonical_reward_key(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    payload["verifier_result"]["rewards"] = {
        "style": 0.9,
        "correctness": 0.1,
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical verifier reward"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_rejects_incomplete_harbor_job(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    _job_summary(driftlock, 2, completed=1)

    with pytest.raises(ValueError, match="job is incomplete"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_rejects_harbor_retries(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    _job_summary(driftlock, 2, retries=1)

    with pytest.raises(ValueError, match="forbidden retries"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_reconciles_trial_and_job_usage(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    summary_path = driftlock / "result.json"
    summary = json.loads(summary_path.read_text())
    summary["stats"]["n_input_tokens"] += 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="input_tokens total does not match"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_accepts_providerless_harbor_model_identity(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    for job in (stock, driftlock):
        for result_file in job.glob("*/result.json"):
            payload = json.loads(result_file.read_text())
            payload["agent_info"]["model_info"] = {
                "provider": None,
                "name": "gpt-5.4",
            }
            payload["config"]["agent"]["model_name"] = "gpt-5.4"
            result_file.write_text(json.dumps(payload), encoding="utf-8")
        _job_summary(job, 2)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )
    assert report["matrix"]["agent_model"] == "gpt-5.4"


def test_analyze_rejects_invalid_usage(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    payload["agent_result"]["n_cache_tokens"] = 101
    payload["agent_result"]["n_input_tokens"] = 100
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cache tokens exceed input tokens"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_aggregates_harbor_multi_step_usage(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    payload["agent_result"] = None
    payload["step_results"] = [
        {
            "agent_result": {
                "n_input_tokens": 50,
                "n_cache_tokens": 20,
                "n_output_tokens": 5,
                "cost_usd": 0.1,
            }
        },
        {
            "agent_result": {
                "n_input_tokens": 70,
                "n_cache_tokens": 40,
                "n_output_tokens": 25,
                "cost_usd": 0.2,
            }
        },
    ]
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    assert report["arms"]["driftlock"]["input_tokens"] == 200
    assert report["arms"]["driftlock"]["output_tokens"] == 40
    assert report["arms"]["driftlock"]["cost_usd"] == pytest.approx(0.4)


def test_analyze_can_explicitly_report_an_incomplete_matrix(tmp_path: Path) -> None:
    lhtb = tmp_path / "LHTB"
    _task(lhtb, "stock-only", expert_minutes=10, category="one")
    _task(lhtb, "other-only", expert_minutes=20, category="two")
    stock = tmp_path / "stock"
    retry = tmp_path / "retry"
    _result(
        stock,
        "one",
        "stock-only",
        1.0,
        checksum=_task_directory_sha256(lhtb / "tasks" / "stock-only"),
    )
    _result(
        retry,
        "two",
        "other-only",
        0.0,
        checksum=_task_directory_sha256(lhtb / "tasks" / "other-only"),
    )
    _job_summary(stock, 1)
    _job_summary(retry, 1)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "retry": retry},
        require_complete_matrix=False,
    )

    assert report["matrix"]["complete"] is False
    comparison = report["paired_vs_stock"]["retry"]
    assert comparison["shared_task_count"] == 0
    assert comparison["mean_task_reward_delta"] is None
    assert comparison["mean_total_tokens_per_trial_delta"] is None
    assert comparison["mean_cost_usd_per_trial_delta"] is None
    assert comparison["failure_slope_delta"] is None
    assert comparison["aggregate_workload_comparable"] is False


def test_analyze_cli_writes_report(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    output = tmp_path / "nested" / "analysis.json"

    assert (
        main(
            [
                "analyze",
                "--lhtb-dir",
                str(lhtb),
                "--arm-dir",
                f"stock={stock}",
                "--arm-dir",
                f"driftlock={driftlock}",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["schema_version"] == 1
