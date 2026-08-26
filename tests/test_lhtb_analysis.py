from __future__ import annotations

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from driftlock.lhtb import (
    LHTB_REPOSITORY_REVISION,
    lhtb_experiment_fingerprint,
    lhtb_runtime_fingerprint,
)
from driftlock.lhtb_analysis import (
    _task_directory_sha256,
    _trajectory_usage,
    _validate_arm_identity,
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
    model: str = "deepseek/deepseek-v4-flash-0731",
    usage: dict[str, int | float] | None = None,
    exception: str | None = None,
    arm: str | None = None,
    provider: str = "baidu/fp8",
) -> Path:
    directory = job / trial
    directory.mkdir(parents=True)
    resolved_arm = arm or job.name
    result_task = task if "/" in task else f"long-horizon-terminal-bench/{task}"
    agent_config: dict[str, object] = {
        "name": None,
        "import_path": "driftlock.harbor_agent:LHTBDriftlockAgent",
        "model_name": "openrouter/deepseek/deepseek-v4-flash-0731",
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
                "extra_body": {
                    "provider": {
                        "only": [provider],
                        "allow_fallbacks": False,
                    }
                },
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
            "model_name": "openrouter/deepseek/deepseek-v4-flash-0731",
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
                    "extra_body": {
                        "provider": {
                            "only": [provider],
                            "allow_fallbacks": False,
                        }
                    },
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
                "driftlock_judge_model": ("openrouter/deepseek/deepseek-v4-pro-0813"),
                "driftlock_judge_api_base": "https://judge.invalid/v1",
                "driftlock_judge_max_output_tokens": 8192,
                "driftlock_judge_llm_call_kwargs": {
                    "extra_body": {
                        "provider": {
                            "only": ["alibaba"],
                            "allow_fallbacks": False,
                        }
                    }
                },
            }
        )
    elif resolved_arm in {"native-driftlock-heuristic", "native-driftlock"}:
        agent_name = "driftlock-native-tool-agent"
        agent_config["import_path"] = (
            "driftlock.harbor_native_agent:LHTBNativeDriftlockAgent"
        )
        if resolved_arm == "native-driftlock":
            kwargs = agent_config["kwargs"]
            assert isinstance(kwargs, dict)
            kwargs.update(
                {
                    "driftlock_judge_model": (
                        "openrouter/deepseek/deepseek-v4-pro-0813"
                    ),
                    "driftlock_judge_api_base": "https://judge.invalid/v1",
                    "driftlock_judge_max_output_tokens": 8192,
                    "driftlock_judge_llm_call_kwargs": {
                        "extra_body": {
                            "provider": {
                                "only": ["alibaba"],
                                "allow_fallbacks": False,
                            }
                        }
                    },
                }
            )
    if resolved_arm in {
        "driftlock-heuristic",
        "driftlock",
        "native-driftlock-heuristic",
        "native-driftlock",
    }:
        kwargs = agent_config["kwargs"]
        assert isinstance(kwargs, dict)
        kwargs.update(
            {
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


def _trajectory_steps() -> list[dict[str, object]]:
    return [
        {"step_id": 1, "source": "user", "message": "task"},
        {
            "step_id": 2,
            "source": "agent",
            "message": "first",
            "metrics": {
                "prompt_tokens": 11,
                "cached_tokens": 7,
                "completion_tokens": 3,
                "cost_usd": 0.125,
            },
        },
        {
            "step_id": 3,
            "source": "agent",
            "message": "second",
            "metrics": {
                "prompt_tokens": 13,
                "cached_tokens": 9,
                "completion_tokens": 5,
                "cost_usd": 0.25,
            },
        },
        {
            "step_id": 4,
            "source": "agent",
            "message": "zero-token final entry",
            "metrics": {
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            },
        },
    ]


def _complete_jobs(
    tmp_path: Path, *, provider: str = "baidu/fp8"
) -> tuple[Path, Path, Path]:
    lhtb = tmp_path / "LHTB"
    _task(lhtb, "short", expert_minutes=60, category="build")
    _task(lhtb, "long", expert_minutes=240, category="debug")
    stock = tmp_path / "stock"
    driftlock = tmp_path / "driftlock"
    short_checksum = _task_directory_sha256(lhtb / "tasks" / "short")
    long_checksum = _task_directory_sha256(lhtb / "tasks" / "long")
    _result(
        stock,
        "short-1",
        "short",
        1.0,
        checksum=short_checksum,
        provider=provider,
    )
    _result(
        stock,
        "long-1",
        "long",
        0.0,
        checksum=long_checksum,
        provider=provider,
    )
    _result(
        driftlock,
        "short-1",
        "short",
        1.0,
        checksum=short_checksum,
        provider=provider,
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
        provider=provider,
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


def _set_graded_timeout(result_file: Path, *, reward: float) -> None:
    payload = json.loads(result_file.read_text())
    payload["agent_result"] = None
    payload["verifier_result"] = {"rewards": {"reward": reward}}
    payload["exception_info"] = {
        "exception_type": "AgentTimeoutError",
        "exception_message": "Agent execution timed out after 5400.0 seconds",
    }
    result_file.write_text(json.dumps(payload), encoding="utf-8")
    trajectory = result_file.parent / "agent" / "trajectory.json"
    trajectory.parent.mkdir()
    trajectory.write_text(
        json.dumps({"schema_version": "ATIF-v1.0", "steps": _trajectory_steps()}),
        encoding="utf-8",
    )


def _four_arm_jobs(tmp_path: Path, *, task_count: int) -> tuple[Path, dict[str, Path]]:
    lhtb = tmp_path / "LHTB"
    arms = {
        "stock": tmp_path / "stock",
        "retry": tmp_path / "retry",
        "driftlock-heuristic": tmp_path / "driftlock-heuristic",
        "driftlock": tmp_path / "driftlock",
    }
    reward_offsets = {
        "stock": 0.0,
        "retry": 0.1,
        "driftlock-heuristic": 0.2,
        "driftlock": 0.3,
    }
    for number in range(1, task_count + 1):
        task = f"task-{number}"
        _task(
            lhtb,
            task,
            expert_minutes=number * 10,
            category=f"category-{number}",
        )
        checksum = _task_directory_sha256(lhtb / "tasks" / task)
        for arm, job in arms.items():
            _result(
                job,
                f"{task}-1",
                task,
                number / 20 + reward_offsets[arm],
                checksum=checksum,
                arm=arm,
            )
    for job in arms.values():
        _job_summary(job, task_count)
    return lhtb, arms


def _set_dead_trial(result_file: Path, exception_type: str) -> None:
    payload = json.loads(result_file.read_text())
    payload["agent_result"] = None
    payload["verifier_result"] = None
    payload["exception_info"] = {"exception_type": exception_type}
    result_file.write_text(json.dumps(payload), encoding="utf-8")
    trajectory = result_file.parent / "agent" / "trajectory.json"
    trajectory.parent.mkdir()
    trajectory.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.0",
                "steps": [
                    {"step_id": 1, "source": "user", "message": "task"},
                    {
                        "step_id": 2,
                        "source": "agent",
                        "message": "provider failed",
                        "metrics": {
                            "prompt_tokens": 15,
                            "cached_tokens": 5,
                            "completion_tokens": 5,
                            "cost_usd": 0.05,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


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
    assert report["matrix"]["agent_model"] == (
        "openrouter/deepseek/deepseek-v4-flash-0731"
    )
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


def test_analyze_clean_round_reports_empty_task_exclusions(tmp_path: Path) -> None:
    lhtb, arms = _four_arm_jobs(tmp_path, task_count=4)

    report = analyze_jobs(lhtb_dir=lhtb, arm_directories=arms)

    assert report["task_exclusions"] == {
        "excluded_task_count": 0,
        "excluded_tasks": [],
        "dead_trial_counts_by_arm": {
            "stock": 0,
            "retry": 0,
            "driftlock-heuristic": 0,
            "driftlock": 0,
        },
        "surviving_task_count": 4,
    }
    assert all(
        "dead_exception_type" not in trial
        for arm in report["arms"].values()
        for trial in arm["trials"]
    )


def test_analyze_excludes_one_dead_task_from_every_arm_and_keeps_its_spend(
    tmp_path: Path,
) -> None:
    lhtb, arms = _four_arm_jobs(tmp_path, task_count=4)
    _set_dead_trial(arms["driftlock"] / "task-2-1" / "result.json", "APIError")
    _job_summary(arms["driftlock"], 4, errors=1)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories=arms,
        exclude_dead_tasks=True,
    )

    assert report["task_exclusions"] == {
        "excluded_task_count": 1,
        "excluded_tasks": [
            {
                "task": "long-horizon-terminal-bench/task-2",
                "reason": "non_timeout_trial_exception",
                "deaths": [{"arm": "driftlock", "exception_type": "APIError"}],
            }
        ],
        "dead_trial_counts_by_arm": {
            "stock": 0,
            "retry": 0,
            "driftlock-heuristic": 0,
            "driftlock": 1,
        },
        "surviving_task_count": 3,
    }
    assert report["matrix"]["attempts_per_task"] == {
        "stock": {
            "long-horizon-terminal-bench/task-1": 1,
            "long-horizon-terminal-bench/task-3": 1,
            "long-horizon-terminal-bench/task-4": 1,
        },
        "retry": {
            "long-horizon-terminal-bench/task-1": 1,
            "long-horizon-terminal-bench/task-3": 1,
            "long-horizon-terminal-bench/task-4": 1,
        },
        "driftlock-heuristic": {
            "long-horizon-terminal-bench/task-1": 1,
            "long-horizon-terminal-bench/task-3": 1,
            "long-horizon-terminal-bench/task-4": 1,
        },
        "driftlock": {
            "long-horizon-terminal-bench/task-1": 1,
            "long-horizon-terminal-bench/task-3": 1,
            "long-horizon-terminal-bench/task-4": 1,
        },
    }
    expected_reported_usage = {
        "stock": (400, 160, 80, 1.0),
        "retry": (400, 160, 80, 1.0),
        "driftlock-heuristic": (400, 160, 80, 1.0),
        "driftlock": (315, 125, 65, 0.8),
    }
    expected_harbor_usage = {
        "stock": (400, 160, 80, 1.0),
        "retry": (400, 160, 80, 1.0),
        "driftlock-heuristic": (400, 160, 80, 1.0),
        "driftlock": (300, 120, 60, 0.75),
    }
    expected_rewards = {
        "stock": 0.13333333333333333,
        "retry": 0.23333333333333334,
        "driftlock-heuristic": 0.3333333333333333,
        "driftlock": 0.43333333333333335,
    }
    for arm in arms:
        input_tokens, cache_tokens, output_tokens, cost_usd = expected_reported_usage[
            arm
        ]
        harbor_input, harbor_cache, harbor_output, harbor_cost = expected_harbor_usage[
            arm
        ]
        assert report["arms"][arm]["task_count"] == 3
        assert report["arms"][arm]["trial_count"] == 4
        assert report["arms"][arm]["billed_trial_count"] == 4
        assert report["arms"][arm]["analyzed_trial_count"] == 3
        assert report["arms"][arm]["usage_totals_population"] == ("all_billed_trials")
        assert report["arms"][arm]["mean_reward"] == pytest.approx(
            expected_rewards[arm]
        )
        assert report["arms"][arm]["input_tokens"] == input_tokens
        assert report["arms"][arm]["cache_tokens"] == cache_tokens
        assert report["arms"][arm]["output_tokens"] == output_tokens
        assert report["arms"][arm]["cost_usd"] == cost_usd
        assert report["job_summaries"][arm]["usage"] == {
            "input_tokens": harbor_input,
            "cache_tokens": harbor_cache,
            "output_tokens": harbor_output,
            "cost_usd": harbor_cost,
        }
        assert report["arms"][arm]["mean_total_tokens_per_analyzed_trial"] == 120
        assert report["arms"][arm]["mean_cost_usd_per_analyzed_trial"] == 0.25
    assert report["arms"]["stock"]["mean_total_tokens_per_billed_trial"] == 120
    assert report["arms"]["stock"]["mean_cost_usd_per_billed_trial"] == 0.25
    assert report["arms"]["driftlock"]["mean_total_tokens_per_billed_trial"] == 95
    assert report["arms"]["driftlock"]["mean_cost_usd_per_billed_trial"] == 0.2
    assert report["arms"]["driftlock"]["trajectory_reconstructed_usage"] == {
        "input_tokens": 15,
        "cache_tokens": 5,
        "output_tokens": 5,
        "cost_usd": 0.05,
    }
    expected_reward_deltas = {
        "retry": 0.1,
        "driftlock-heuristic": 0.2,
        "driftlock": 0.3,
    }
    for arm, comparison in report["paired_vs_stock"].items():
        assert comparison["shared_task_count"] == 3
        assert comparison["mean_task_reward_delta"] == pytest.approx(
            expected_reward_deltas[arm]
        )
        assert [item["task"] for item in comparison["tasks"]] == [
            "long-horizon-terminal-bench/task-1",
            "long-horizon-terminal-bench/task-3",
            "long-horizon-terminal-bench/task-4",
        ]
        assert comparison["aggregate_workload_comparable"] is True
        assert comparison["aggregate_delta_status"] == (
            "complete_surviving_task_attempt_matrix"
        )
        assert comparison["workload_population"] == "reward_analysis_trials"
        assert comparison["mean_total_tokens_per_trial_delta"] == 0
        assert comparison["mean_cost_usd_per_trial_delta"] == 0
        assert comparison["mean_total_tokens_per_surviving_trial_delta"] == 0
        assert comparison["mean_cost_usd_per_surviving_trial_delta"] == 0


def test_excluded_round_workload_matches_round_where_task_was_never_run(
    tmp_path: Path,
) -> None:
    excluded_lhtb, excluded_arms = _four_arm_jobs(tmp_path / "excluded", task_count=4)
    _set_dead_trial(excluded_arms["driftlock"] / "task-2-1" / "result.json", "APIError")
    _job_summary(excluded_arms["driftlock"], 4, errors=1)
    excluded_report = analyze_jobs(
        lhtb_dir=excluded_lhtb,
        arm_directories=excluded_arms,
        exclude_dead_tasks=True,
    )

    never_run_lhtb, never_run_arms = _four_arm_jobs(
        tmp_path / "never-run", task_count=4
    )
    for job in never_run_arms.values():
        (job / "task-2-1" / "result.json").unlink()
        _job_summary(job, 3)
    never_run_report = analyze_jobs(
        lhtb_dir=never_run_lhtb,
        arm_directories=never_run_arms,
    )

    excluded = excluded_report["paired_vs_stock"]["driftlock"]
    never_run = never_run_report["paired_vs_stock"]["driftlock"]
    assert excluded["aggregate_workload_comparable"] is True
    assert excluded["mean_total_tokens_per_trial_delta"] == 0
    assert excluded["mean_cost_usd_per_trial_delta"] == 0
    assert never_run["aggregate_workload_comparable"] is True
    assert never_run["mean_total_tokens_per_trial_delta"] == 0
    assert never_run["mean_cost_usd_per_trial_delta"] == 0
    assert (
        excluded["mean_total_tokens_per_trial_delta"]
        == never_run["mean_total_tokens_per_trial_delta"]
    )
    assert (
        excluded["mean_cost_usd_per_trial_delta"]
        == never_run["mean_cost_usd_per_trial_delta"]
    )


def test_dead_task_exclusion_preserves_raw_incomplete_matrix(tmp_path: Path) -> None:
    lhtb, arms = _four_arm_jobs(tmp_path, task_count=5)
    checksum = _task_directory_sha256(lhtb / "tasks" / "task-5")
    extra = _result(
        arms["driftlock"],
        "task-5-2",
        "task-5",
        0.9,
        checksum=checksum,
        arm="driftlock",
    )
    _set_dead_trial(extra, "APIError")
    _job_summary(arms["driftlock"], 6, errors=1)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories=arms,
        require_complete_matrix=False,
        exclude_dead_tasks=True,
    )

    matrix = report["matrix"]
    assert matrix["pre_exclusion_complete"] is False
    assert matrix["post_exclusion_complete"] is True
    assert matrix["complete"] is False
    assert (
        matrix["pre_exclusion_attempts_per_task"]["stock"][
            "long-horizon-terminal-bench/task-5"
        ]
        == 1
    )
    assert (
        matrix["pre_exclusion_attempts_per_task"]["driftlock"][
            "long-horizon-terminal-bench/task-5"
        ]
        == 2
    )
    assert (
        "long-horizon-terminal-bench/task-5"
        not in matrix["attempts_per_task"]["driftlock"]
    )
    comparison = report["paired_vs_stock"]["driftlock"]
    assert comparison["aggregate_workload_comparable"] is False
    assert comparison["aggregate_delta_status"] == (
        "unavailable_for_incomplete_task_attempt_matrix"
    )
    assert comparison["mean_total_tokens_per_trial_delta"] is None
    assert comparison["mean_cost_usd_per_trial_delta"] is None


@pytest.mark.parametrize(
    ("deaths", "expected_counts", "offender"),
    [
        (
            {
                "driftlock-heuristic": ("task-1", "task-2"),
                "driftlock": ("task-3",),
            },
            "stock=0, retry=0, driftlock-heuristic=2, driftlock=1",
            "driftlock-heuristic=2",
        ),
        (
            {
                "retry": ("task-1", "task-2", "task-3"),
                "driftlock": ("task-4", "task-5", "task-6", "task-7", "task-8"),
            },
            "stock=0, retry=3, driftlock-heuristic=0, driftlock=5",
            "driftlock=5",
        ),
    ],
)
def test_analyze_rejects_real_concentrated_dead_trial_shapes(
    tmp_path: Path,
    deaths: dict[str, tuple[str, ...]],
    expected_counts: str,
    offender: str,
) -> None:
    lhtb, arms = _four_arm_jobs(tmp_path, task_count=8)
    for arm, tasks in deaths.items():
        for task in tasks:
            _set_dead_trial(arms[arm] / f"{task}-1" / "result.json", "APIError")
        _job_summary(arms[arm], 8, errors=len(tasks))

    with pytest.raises(ValueError) as caught:
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories=arms,
            exclude_dead_tasks=True,
        )

    message = str(caught.value)
    assert expected_counts in message
    assert offender in message
    assert "possible arm defect" in message


def test_analyze_accepts_one_dead_trial_per_arm(tmp_path: Path) -> None:
    lhtb, arms = _four_arm_jobs(tmp_path, task_count=8)
    for number, (_arm, job) in enumerate(arms.items(), start=1):
        _set_dead_trial(job / f"task-{number}-1" / "result.json", "APIError")
        _job_summary(job, 8, errors=1)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories=arms,
        exclude_dead_tasks=True,
    )

    assert report["task_exclusions"]["dead_trial_counts_by_arm"] == {
        "stock": 1,
        "retry": 1,
        "driftlock-heuristic": 1,
        "driftlock": 1,
    }
    assert report["task_exclusions"]["excluded_task_count"] == 4
    assert report["task_exclusions"]["surviving_task_count"] == 4
    assert all(
        comparison["shared_task_count"] == 4
        for comparison in report["paired_vs_stock"].values()
    )


def test_analyze_rejects_too_few_tasks_after_spread_deaths(tmp_path: Path) -> None:
    lhtb, arms = _four_arm_jobs(tmp_path, task_count=4)
    for number, arm in enumerate(("stock", "retry", "driftlock-heuristic"), start=1):
        _set_dead_trial(arms[arm] / f"task-{number}-1" / "result.json", "APIError")
        _job_summary(arms[arm], 4, errors=1)

    with pytest.raises(
        ValueError,
        match="dead-task exclusion leaves 1 surviving tasks; at least 3 are required",
    ):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories=arms,
            exclude_dead_tasks=True,
        )


def test_planned_arm_coverage_includes_native_arm_present_in_report(
    tmp_path: Path,
) -> None:
    lhtb = tmp_path / "LHTB"
    _task(lhtb, "short", expert_minutes=60, category="build")
    checksum = _task_directory_sha256(lhtb / "tasks" / "short")
    stock = tmp_path / "stock"
    native = tmp_path / "native-driftlock"
    _result(stock, "short-1", "short", 0.0, checksum=checksum)
    _result(
        native,
        "short-1",
        "short",
        1.0,
        checksum=checksum,
        arm="native-driftlock",
    )
    _job_summary(stock, 1)
    _job_summary(native, 1)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "native-driftlock": native},
    )

    assert report["planned_arm_coverage"]["present"] == [
        "stock",
        "native-driftlock",
    ]
    assert "native-driftlock" in report["arms"]
    assert "native-driftlock" in report["paired_vs_stock"]


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


def test_arm_identity_accepts_matching_provider_and_names_mismatch(
    tmp_path: Path,
) -> None:
    job = tmp_path / "driftlock"
    result_file = _result(job, "short-1", "short", 1.0)
    payload = json.loads(result_file.read_text())

    budget, signature = _validate_arm_identity(
        payload,
        "driftlock",
        "openrouter/deepseek/deepseek-v4-flash-0731",
        result_file,
        expected_provider="baidu/fp8",
        expected_judge_provider="alibaba",
    )

    assert budget == 10_000
    assert len(signature) == 64
    with pytest.raises(ValueError) as caught:
        _validate_arm_identity(
            payload,
            "driftlock",
            "openrouter/deepseek/deepseek-v4-flash-0731",
            result_file,
            expected_provider="sail-research/fp4",
            expected_judge_provider="alibaba",
        )
    message = str(caught.value)
    assert "sail-research/fp4" in message
    assert "baidu/fp8" in message


def test_analyze_accepts_matching_detector_thresholds_across_controlled_arms(
    tmp_path: Path,
) -> None:
    lhtb = tmp_path / "LHTB"
    _task(lhtb, "short", expert_minutes=60, category="build")
    checksum = _task_directory_sha256(lhtb / "tasks" / "short")
    stock = tmp_path / "stock"
    heuristic = tmp_path / "driftlock-heuristic"
    driftlock = tmp_path / "driftlock"
    _result(stock, "short-1", "short", 0.0, checksum=checksum)
    _result(heuristic, "short-1", "short", 1.0, checksum=checksum)
    _result(driftlock, "short-1", "short", 1.0, checksum=checksum)
    _job_summary(stock, 1)
    _job_summary(heuristic, 1)
    _job_summary(driftlock, 1)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={
            "stock": stock,
            "driftlock-heuristic": heuristic,
            "driftlock": driftlock,
        },
    )

    assert report["matrix"]["complete"] is True


def test_analyze_names_detector_threshold_and_both_values_on_mismatch(
    tmp_path: Path,
) -> None:
    lhtb = tmp_path / "LHTB"
    _task(lhtb, "short", expert_minutes=60, category="build")
    checksum = _task_directory_sha256(lhtb / "tasks" / "short")
    stock = tmp_path / "stock"
    heuristic = tmp_path / "driftlock-heuristic"
    driftlock = tmp_path / "driftlock"
    _result(stock, "short-1", "short", 0.0, checksum=checksum)
    _result(heuristic, "short-1", "short", 1.0, checksum=checksum)
    driftlock_result = _result(driftlock, "short-1", "short", 1.0, checksum=checksum)
    payload = json.loads(driftlock_result.read_text())
    payload["config"]["agent"]["kwargs"]["driftlock_no_change_steps"] = 7
    driftlock_result.write_text(json.dumps(payload), encoding="utf-8")
    _job_summary(stock, 1)
    _job_summary(heuristic, 1)
    _job_summary(driftlock, 1)

    with pytest.raises(ValueError) as caught:
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={
                "stock": stock,
                "driftlock-heuristic": heuristic,
                "driftlock": driftlock,
            },
        )

    message = str(caught.value)
    assert "driftlock_no_change_steps" in message
    assert "driftlock-heuristic=4" in message
    assert "driftlock=7" in message


def test_analyze_rejects_arms_that_gate_different_signals(tmp_path: Path) -> None:
    lhtb = tmp_path / "LHTB"
    _task(lhtb, "short", expert_minutes=60, category="build")
    checksum = _task_directory_sha256(lhtb / "tasks" / "short")
    stock = tmp_path / "stock"
    heuristic = tmp_path / "driftlock-heuristic"
    driftlock = tmp_path / "driftlock"
    _result(stock, "short-1", "short", 0.0, checksum=checksum)
    _result(heuristic, "short-1", "short", 1.0, checksum=checksum)
    driftlock_result = _result(driftlock, "short-1", "short", 1.0, checksum=checksum)
    payload = json.loads(driftlock_result.read_text())
    payload["config"]["agent"]["kwargs"]["driftlock_corroborating_signals"] = []
    driftlock_result.write_text(json.dumps(payload), encoding="utf-8")
    _job_summary(stock, 1)
    _job_summary(heuristic, 1)
    _job_summary(driftlock, 1)

    with pytest.raises(ValueError) as caught:
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={
                "stock": stock,
                "driftlock-heuristic": heuristic,
                "driftlock": driftlock,
            },
        )

    message = str(caught.value)
    assert "driftlock_corroborating_signals" in message
    assert "driftlock-heuristic=('no_file_change',)" in message
    assert "driftlock=()" in message


def test_analyze_accepts_the_same_gate_written_in_a_different_order(
    tmp_path: Path,
) -> None:
    lhtb = tmp_path / "LHTB"
    _task(lhtb, "short", expert_minutes=60, category="build")
    checksum = _task_directory_sha256(lhtb / "tasks" / "short")
    stock = tmp_path / "stock"
    heuristic = tmp_path / "driftlock-heuristic"
    driftlock = tmp_path / "driftlock"
    _result(stock, "short-1", "short", 0.0, checksum=checksum)
    for job in (heuristic, driftlock):
        result_file = _result(job, "short-1", "short", 1.0, checksum=checksum)
        payload = json.loads(result_file.read_text())
        kwargs = payload["config"]["agent"]["kwargs"]
        kwargs["driftlock_corroborating_signals"] = (
            ["no_file_change", "reward_stall"]
            if job is heuristic
            else ["reward_stall", "no_file_change"]
        )
        result_file.write_text(json.dumps(payload), encoding="utf-8")
    _job_summary(stock, 1)
    _job_summary(heuristic, 1)
    _job_summary(driftlock, 1)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={
            "stock": stock,
            "driftlock-heuristic": heuristic,
            "driftlock": driftlock,
        },
    )

    # Accepted: order is not part of the configuration. The neighbouring test
    # shows a genuinely different set is still rejected.
    assert sorted(report["arms"]) == ["driftlock", "driftlock-heuristic", "stock"]


def test_analyze_rejects_a_corroborating_gate_that_is_not_a_list_of_strings(
    tmp_path: Path,
) -> None:
    job = tmp_path / "driftlock"
    result_file = _result(job, "short-1", "short", 1.0)
    payload = json.loads(result_file.read_text())
    payload["config"]["agent"]["kwargs"]["driftlock_corroborating_signals"] = (
        "no_file_change"
    )
    result_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid driftlock_corroborating_signals"):
        _validate_arm_identity(
            payload,
            "driftlock",
            "openrouter/deepseek/deepseek-v4-flash-0731",
            result_file,
        )


def test_arm_identity_rejects_legacy_config_without_detector_thresholds(
    tmp_path: Path,
) -> None:
    job = tmp_path / "driftlock"
    result_file = _result(job, "short-1", "short", 1.0)
    payload = json.loads(result_file.read_text())
    kwargs = payload["config"]["agent"]["kwargs"]
    for name in tuple(kwargs):
        if name.startswith("driftlock_") and name not in {
            "driftlock_max_tokens",
            "driftlock_max_steps",
            "driftlock_max_rollbacks",
            "driftlock_checkpoint_interval",
            "driftlock_judge_model",
            "driftlock_judge_api_base",
            "driftlock_judge_max_output_tokens",
            "driftlock_judge_llm_call_kwargs",
        }:
            del kwargs[name]

    with pytest.raises(ValueError, match="missing detector setting"):
        _validate_arm_identity(
            payload,
            "driftlock",
            "openrouter/deepseek/deepseek-v4-flash-0731",
            result_file,
        )


def test_analyze_rejects_trial_provider_different_from_canonical_arm_lock(
    tmp_path: Path,
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    routing = payload["config"]["agent"]["kwargs"]["llm_call_kwargs"]
    routing["extra_body"]["provider"]["only"] = ["streamlake/fp8"]
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )
    message = str(caught.value)
    assert "baidu/fp8" in message
    assert "streamlake/fp8" in message


def test_analyze_rejects_different_providers_between_arms(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    for result_file in driftlock.glob("*/result.json"):
        payload = json.loads(result_file.read_text())
        routing = payload["config"]["agent"]["kwargs"]["llm_call_kwargs"]
        routing["extra_body"]["provider"]["only"] = ["streamlake/fp8"]
        result_file.write_text(json.dumps(payload), encoding="utf-8")
    _job_summary(driftlock, 2)

    with pytest.raises(ValueError) as caught:
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )
    message = str(caught.value)
    assert "different agent providers" in message
    assert "baidu/fp8" in message
    assert "streamlake/fp8" in message


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
        ("fingerprint", "mix driftlock builds"),
        ("no-fingerprint", "missing or malformed build fingerprint"),
        ("short-fingerprint", "missing or malformed build fingerprint"),
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
    elif mutation == "no-fingerprint":
        del lock["trials"][0]["agent"]["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"]
    elif mutation == "short-fingerprint":
        lock["trials"][0]["agent"]["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"] = "abc"
    else:
        lock["trials"][0]["agent"]["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"] = "0" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_treats_harbor_retry_exception_order_as_set_order(
    tmp_path: Path,
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    lock_path = driftlock / "lock.json"
    lock = json.loads(lock_path.read_text())
    lock["retry"]["exclude_exceptions"] = list(
        reversed(lock["retry"]["exclude_exceptions"])
    )
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    assert report["matrix"]["complete"] is True


def test_analyze_rejects_a_different_retry_exception_set(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    lock_path = driftlock / "lock.json"
    lock = json.loads(lock_path.read_text())
    lock["retry"]["exclude_exceptions"][-1] = "FutureHarborError"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="different Harbor job-level settings"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def _restamp_build(job: Path, fingerprint: str) -> None:
    """Rewrite a job's recorded build fingerprint in the lock and every trial."""
    lock_path = job / "lock.json"
    lock = json.loads(lock_path.read_text())
    for trial in lock["trials"]:
        trial["agent"]["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"] = fingerprint
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    for result_file in job.glob("*/result.json"):
        payload = json.loads(result_file.read_text())
        payload["config"]["agent"]["env"]["DRIFTLOCK_EXPERIMENT_FINGERPRINT"] = (
            fingerprint
        )
        result_file.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_rejects_two_arms_produced_by_different_builds(
    tmp_path: Path,
) -> None:
    # The hazard the fingerprint exists for: comparing an arm built by one
    # revision against an arm built by another. Each lock is internally
    # consistent here, so only a cross-arm check can catch it -- the old
    # per-lock comparison against the installed build never looked across arms.
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    _restamp_build(driftlock, "b" * 64)

    with pytest.raises(ValueError) as caught:
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )

    message = str(caught.value)
    assert "experiment arms mix driftlock builds" in message
    assert "b" * 64 in message
    assert "not comparable" in message


def test_analyze_records_whether_the_reader_was_the_producing_build(
    tmp_path: Path,
) -> None:
    # Analysing with a different build is reported, not refused: making it fatal
    # meant an analyzer bug permanently locked out the run it affected.
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    for job in (stock, driftlock):
        _restamp_build(job, "c" * 64)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    matrix = report["matrix"]
    assert matrix["driftlock_build_fingerprint"] == "c" * 64
    assert matrix["analyzer_build_fingerprint"] == lhtb_runtime_fingerprint()
    assert matrix["analyzed_by_producing_build"] is False


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


def test_analyze_accepts_graded_timeout_and_reconstructs_exact_usage(
    tmp_path: Path,
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    _set_graded_timeout(stock / "long-1" / "result.json", reward=0.15)
    _job_summary(stock, 2, errors=1)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    stock_report = report["arms"]["stock"]
    assert stock_report["trial_count"] == 2
    assert stock_report["graded_at_time_cap_count"] == 1
    assert stock_report["input_tokens"] == 124
    assert stock_report["cache_tokens"] == 56
    assert stock_report["output_tokens"] == 28
    assert stock_report["cost_usd"] == 0.625
    assert stock_report["trajectory_reconstructed_usage"] == {
        "input_tokens": 24,
        "cache_tokens": 16,
        "output_tokens": 8,
        "cost_usd": 0.375,
    }
    timeout_trial = next(
        trial
        for trial in stock_report["trials"]
        if trial["task"] == "long-horizon-terminal-bench/long"
    )
    assert timeout_trial["reward"] == 0.15
    assert timeout_trial["graded_at_time_cap"] is True
    assert timeout_trial["usage_source"] == "trajectory"
    assert timeout_trial["input_tokens"] == 24
    assert timeout_trial["cache_tokens"] == 16
    assert timeout_trial["output_tokens"] == 8
    assert timeout_trial["cost_usd"] == 0.375
    assert report["matrix"]["complete"] is True
    assert (
        report["paired_vs_stock"]["driftlock"]["aggregate_workload_comparable"] is True
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing trajectory needed to reconstruct agent usage"),
        # A cache field that is present but unusable is still a hard error; only
        # an entirely absent one is read as "no cache hit". See the test below.
        ("bad-cache-type", "cached_tokens"),
        ("negative-cache", "cached_tokens"),
        # Prompt and completion stay required. Cost recovery additionally needs
        # an audited provider; this fixture deliberately uses unaudited Baidu.
        ("no-prompt", "prompt_tokens"),
        ("no-completion", "completion_tokens"),
        ("no-cost", "cost_usd"),
    ],
)
def test_analyze_rejects_unusable_trajectory_needed_for_timeout_usage(
    tmp_path: Path, mutation: str, message: str
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    result_file = stock / "long-1" / "result.json"
    _set_graded_timeout(result_file, reward=0.15)
    trajectory = result_file.parent / "agent" / "trajectory.json"
    if mutation == "missing":
        trajectory.unlink()
    else:
        metrics: dict[str, object] = {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "cached_tokens": 4,
            "cost_usd": 0.125,
        }
        if mutation == "bad-cache-type":
            metrics["cached_tokens"] = "4"
        elif mutation == "negative-cache":
            metrics["cached_tokens"] = -1
        elif mutation == "no-prompt":
            del metrics["prompt_tokens"]
        elif mutation == "no-completion":
            del metrics["completion_tokens"]
        elif mutation == "no-cost":
            del metrics["cost_usd"]
        trajectory.write_text(
            json.dumps({"steps": [{"source": "agent", "metrics": metrics}]}),
            encoding="utf-8",
        )
    _job_summary(stock, 2, errors=1)

    with pytest.raises(ValueError, match=message):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_a_step_that_reports_no_cache_read_is_counted_as_zero_cache(
    tmp_path: Path,
) -> None:
    # A call that hits no cache may omit cached_tokens entirely; DeepInfra did
    # this on 2026-08-24 and it rejected an otherwise clean four-arm round.
    # Reading it as zero attributes the whole prompt to full-price input, so an
    # omission can never make a billed call look cheaper than it was.
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    result_file = stock / "long-1" / "result.json"
    _set_graded_timeout(result_file, reward=0.15)
    trajectory = result_file.parent / "agent" / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "source": "agent",
                        "metrics": {
                            "prompt_tokens": 20,
                            "completion_tokens": 5,
                            "cached_tokens": 12,
                            "cost_usd": 0.25,
                        },
                    },
                    {
                        "source": "agent",
                        "metrics": {
                            "prompt_tokens": 4,
                            "completion_tokens": 3,
                            "cost_usd": 0.125,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    _job_summary(stock, 2, errors=1)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    trial = next(
        trial
        for trial in report["arms"]["stock"]["trials"]
        if trial["task"] == "long-horizon-terminal-bench/long"
    )
    assert trial["usage_source"] == "trajectory"
    assert trial["input_tokens"] == 24
    assert trial["cache_tokens"] == 12
    assert trial["output_tokens"] == 8
    assert trial["cost_usd"] == 0.375


def test_missing_trajectory_cost_is_imputed_and_reported_per_arm(
    tmp_path: Path,
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path, provider="deepinfra/fp8")
    result_file = stock / "long-1" / "result.json"
    _set_graded_timeout(result_file, reward=0.15)
    trajectory = result_file.parent / "agent" / "trajectory.json"
    payload = json.loads(trajectory.read_text())
    del payload["steps"][2]["metrics"]["cost_usd"]
    trajectory.write_text(json.dumps(payload), encoding="utf-8")
    _job_summary(stock, 2, errors=1)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    stock_report = report["arms"]["stock"]
    assert stock_report["cost_usd"] == pytest.approx(0.37500194)
    assert stock_report["trajectory_reconstructed_usage"] == {
        "input_tokens": 24,
        "cache_tokens": 16,
        "output_tokens": 8,
        "cost_usd": pytest.approx(0.12500194),
    }
    assert stock_report["trajectory_cost_imputation"] == {
        "step_count": 1,
        "cost_usd": pytest.approx(0.00000194),
    }
    assert report["arms"]["driftlock"]["trajectory_cost_imputation"] == {
        "step_count": 0,
        "cost_usd": 0.0,
    }


def test_trajectory_usage_imputes_only_an_absent_cost_from_audited_rates(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "trial" / "result.json"
    trajectory = result_file.parent / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "source": "agent",
                        "metrics": {
                            "prompt_tokens": 20,
                            "cached_tokens": 12,
                            "completion_tokens": 5,
                        },
                    },
                    {
                        "source": "agent",
                        "metrics": {
                            "prompt_tokens": 4,
                            "cached_tokens": 0,
                            "completion_tokens": 3,
                            "cost_usd": 0.25,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    usage = _trajectory_usage(result_file, provider="deepinfra/fp8")

    assert usage == {
        "input_tokens": 24,
        "cache_tokens": 12,
        "output_tokens": 8,
        "cost_usd": pytest.approx(0.2500025),
        "imputed_cost_steps": 1,
        "imputed_cost_usd": pytest.approx(0.0000025),
    }


def test_trajectory_usage_refuses_to_reuse_pricing_for_a_new_provider(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "trial" / "result.json"
    trajectory = result_file.parent / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "source": "agent",
                        "metrics": {
                            "prompt_tokens": 20,
                            "cached_tokens": 12,
                            "completion_tokens": 5,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"provider 'future-provider/fp8'.*no audited",
    ):
        _trajectory_usage(result_file, provider="future-provider/fp8")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_tokens", None),
        ("cached_tokens", None),
        ("completion_tokens", "3"),
        ("cost_usd", None),
    ],
)
def test_trajectory_usage_rejects_present_but_unusable_metrics(
    tmp_path: Path, field: str, value: object
) -> None:
    result_file = tmp_path / "trial" / "result.json"
    trajectory = result_file.parent / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    metrics: dict[str, object] = {
        "prompt_tokens": 20,
        "cached_tokens": 12,
        "completion_tokens": 5,
        "cost_usd": 0.25,
    }
    metrics[field] = value
    trajectory.write_text(
        json.dumps({"steps": [{"source": "agent", "metrics": metrics}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        _trajectory_usage(result_file, provider="deepinfra/fp8")


def test_trajectory_usage_rejects_a_materially_truncated_episode_segment(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "trial" / "result.json"
    agent_dir = result_file.parent / "agent"
    agent_dir.mkdir(parents=True)
    for number in range(46):
        (agent_dir / f"episode-{number}").mkdir()
    steps = [
        {
            "source": "agent",
            "metrics": {
                "prompt_tokens": 1,
                "cached_tokens": 0,
                "completion_tokens": 1,
                "cost_usd": 0.000001,
            },
        }
        for _ in range(11)
    ]
    (agent_dir / "trajectory.json").write_text(
        json.dumps({"steps": steps}), encoding="utf-8"
    )

    with pytest.raises(
        ValueError,
        match=r"contains 11 agent steps for 46 provider-call episode directories",
    ):
        _trajectory_usage(result_file, provider="deepinfra/fp8")


def test_trajectory_usage_accepts_round_five_complete_episode_coverage(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "trial" / "result.json"
    agent_dir = result_file.parent / "agent"
    agent_dir.mkdir(parents=True)
    for number in range(50):
        (agent_dir / f"episode-{number}").mkdir()
    steps = [
        {
            "source": "agent",
            "metrics": {
                "prompt_tokens": 2,
                "cached_tokens": 1,
                "completion_tokens": 1,
                "cost_usd": 0.01,
            },
        }
        for _ in range(49)
    ]
    (agent_dir / "trajectory.json").write_text(
        json.dumps({"steps": steps}), encoding="utf-8"
    )

    usage = _trajectory_usage(result_file, provider="deepinfra/fp8")

    assert usage == {
        "input_tokens": 98,
        "cache_tokens": 49,
        "output_tokens": 49,
        "cost_usd": pytest.approx(0.49),
        "imputed_cost_steps": 0,
        "imputed_cost_usd": 0.0,
    }


def test_analyze_rejects_non_timeout_exception_by_type(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    result_file = stock / "long-1" / "result.json"
    payload = json.loads(result_file.read_text())
    payload["exception_info"] = {"exception_type": "RuntimeError"}
    result_file.write_text(json.dumps(payload), encoding="utf-8")
    _job_summary(stock, 2, errors=1)

    with pytest.raises(ValueError, match="RuntimeError"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_reports_different_timeout_counts_per_arm(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    _set_graded_timeout(stock / "short-1" / "result.json", reward=0.4)
    _set_graded_timeout(stock / "long-1" / "result.json", reward=0.2)
    _job_summary(stock, 2, errors=2)

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    assert report["arms"]["stock"]["graded_at_time_cap_count"] == 2
    assert report["arms"]["driftlock"]["graded_at_time_cap_count"] == 0


def test_analyze_prefers_agent_result_over_trajectory_usage(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    result_file = driftlock / "long-1" / "result.json"
    trajectory = result_file.parent / "agent" / "trajectory.json"
    trajectory.parent.mkdir()
    trajectory.write_text("not JSON", encoding="utf-8")

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    trial = next(
        trial
        for trial in report["arms"]["driftlock"]["trials"]
        if trial["task"] == "long-horizon-terminal-bench/long"
    )
    assert trial["usage_source"] == "agent_result"
    assert trial["input_tokens"] == 120
    assert trial["cache_tokens"] == 60
    assert trial["output_tokens"] == 30
    assert trial["cost_usd"] == 0.3


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


@pytest.mark.parametrize(
    "status",
    ["judge_failed", "judge_inconclusive", "future_unclassified_status"],
)
def test_analyze_rejects_non_measurable_driftlock_status(
    tmp_path: Path, status: str
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    payload["agent_result"]["metadata"] = {
        "driftlock": {"status": status},
        "termination_reason": f"driftlock_{status}",
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-measurable driftlock run status"):
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )


def test_analyze_reports_validated_driftlock_status(tmp_path: Path) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    payload["agent_result"]["metadata"] = {
        "driftlock": {
            "status": "completed",
            "judge_reliability": "reliable",
            "judge_attempts": 4,
            "judge_failures": 1,
        },
        "termination_reason": "confirmed_task_complete",
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    trial = next(
        trial
        for trial in report["arms"]["driftlock"]["trials"]
        if trial["task"] == "long-horizon-terminal-bench/long"
    )
    assert trial["driftlock_run_statuses"] == ("completed",)
    assert trial["judge_reliability"] == "reliable"


@pytest.mark.parametrize(
    ("status", "termination_reason"),
    [
        ("completed", "confirmed_task_complete"),
        ("completed", "max_turns"),
        ("completed", "driftlock_output_length_boundary"),
        ("step_limit", "driftlock_step_limit"),
        ("token_limit", "driftlock_token_limit"),
        ("rollback_limit", "driftlock_rollback_limit"),
    ],
)
def test_analyze_accepts_producer_status_termination_pairings(
    tmp_path: Path, status: str, termination_reason: str
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    payload["agent_result"]["metadata"] = {
        "driftlock": {"status": status},
        "termination_reason": termination_reason,
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    report = analyze_jobs(
        lhtb_dir=lhtb,
        arm_directories={"stock": stock, "driftlock": driftlock},
    )

    trial = next(
        trial
        for trial in report["arms"]["driftlock"]["trials"]
        if trial["task"] == "long-horizon-terminal-bench/long"
    )
    assert trial["driftlock_run_statuses"] == (status,)


@pytest.mark.parametrize(
    ("status", "termination_reason"),
    [
        ("completed", "future_harbor_reason"),
        ("rollback_limit", "max_turns"),
        ("token_limit", "confirmed_task_complete"),
    ],
)
def test_analyze_rejects_unknown_or_contradictory_status_termination_pairings(
    tmp_path: Path, status: str, termination_reason: str
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    payload["agent_result"]["metadata"] = {
        "driftlock": {"status": status},
        "termination_reason": termination_reason,
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        analyze_jobs(
            lhtb_dir=lhtb,
            arm_directories={"stock": stock, "driftlock": driftlock},
        )

    message = str(caught.value)
    assert repr(status) in message
    assert repr(termination_reason) in message


@pytest.mark.parametrize(
    ("status", "termination", "reliability", "attempts", "failures"),
    [
        ("token_limit", "driftlock_token_limit", "failed", 13, 12),
        ("completed", "confirmed_task_complete", "inconclusive", 1, 1),
    ],
)
def test_analyze_rejects_invalid_cumulative_judge_reliability(
    tmp_path: Path,
    status: str,
    termination: str,
    reliability: str,
    attempts: int,
    failures: int,
) -> None:
    lhtb, stock, driftlock = _complete_jobs(tmp_path)
    payload_path = driftlock / "long-1" / "result.json"
    payload = json.loads(payload_path.read_text())
    payload["agent_result"]["metadata"] = {
        "driftlock": {
            "status": status,
            "judge_reliability": reliability,
            "judge_attempts": attempts,
            "judge_failures": failures,
        },
        "termination_reason": termination,
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-measurable judge reliability"):
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


def test_analyze_cli_explicitly_enables_dead_task_exclusion(tmp_path: Path) -> None:
    lhtb, arms = _four_arm_jobs(tmp_path, task_count=4)
    _set_dead_trial(arms["retry"] / "task-1-1" / "result.json", "APIError")
    _job_summary(arms["retry"], 4, errors=1)
    output = tmp_path / "analysis.json"
    arguments = [
        "analyze",
        "--lhtb-dir",
        str(lhtb),
        "--exclude-dead-tasks",
        "--output",
        str(output),
    ]
    for arm, job in arms.items():
        arguments.extend(("--arm-dir", f"{arm}={job}"))

    assert main(arguments) == 0

    assert json.loads(output.read_text())["task_exclusions"] == {
        "excluded_task_count": 1,
        "excluded_tasks": [
            {
                "task": "long-horizon-terminal-bench/task-1",
                "reason": "non_timeout_trial_exception",
                "deaths": [{"arm": "retry", "exception_type": "APIError"}],
            }
        ],
        "dead_trial_counts_by_arm": {
            "stock": 0,
            "retry": 1,
            "driftlock-heuristic": 0,
            "driftlock": 0,
        },
        "surviving_task_count": 3,
    }
