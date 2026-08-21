"""Strict, dependency-free analysis of completed LHTB Harbor jobs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

ANALYSIS_ARMS = (
    "stock",
    "retry",
    "driftlock-heuristic",
    "driftlock",
    "oracle",
)
SOLVE_THRESHOLD = 0.95


def goal_drift_actions(
    baseline_aligned_action_share: float,
    evaluation_aligned_action_share: float,
) -> float:
    """Return Arike et al.'s commission metric for compatible annotations."""
    baseline = _share(baseline_aligned_action_share, "baseline action share")
    evaluation = _share(evaluation_aligned_action_share, "evaluation action share")
    return max(0.0, baseline - evaluation)


def goal_drift_inaction(
    baseline_residual_misaligned_share: float,
    evaluation_residual_misaligned_share: float,
) -> float:
    """Return Arike et al.'s omission metric for compatible annotations."""
    baseline = _share(baseline_residual_misaligned_share, "baseline residual share")
    evaluation = _share(
        evaluation_residual_misaligned_share, "evaluation residual share"
    )
    return max(0.0, evaluation - baseline)


def parse_arm_directories(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable ``ARM=JOB_DIR`` command-line values."""
    if not values:
        raise ValueError("at least one --arm-dir ARM=JOB_DIR is required")
    result: dict[str, Path] = {}
    for value in values:
        arm, separator, raw_path = value.partition("=")
        if separator != "=" or not raw_path:
            raise ValueError(f"invalid arm directory {value!r}; expected ARM=JOB_DIR")
        if arm not in ANALYSIS_ARMS:
            raise ValueError(
                f"unknown analysis arm {arm!r}; expected one of "
                + ", ".join(ANALYSIS_ARMS)
            )
        if arm in result:
            raise ValueError(f"duplicate arm directory: {arm}")
        result[arm] = Path(raw_path)
    return result


def analyze_jobs(
    *,
    lhtb_dir: Path,
    arm_directories: Mapping[str, Path],
    solve_threshold: float = SOLVE_THRESHOLD,
    require_complete_matrix: bool = True,
) -> dict[str, Any]:
    """Build an auditable multi-arm report from Harbor ``result.json`` files.

    Strict mode rejects missing rewards, missing usage, task checksum disagreement,
    model disagreement, and unequal per-task attempt counts. Infrastructure failures
    therefore cannot silently become model failures or disappear from denominators.
    """
    if not 0 < solve_threshold <= 1:
        raise ValueError("solve_threshold must be in (0, 1]")
    if "stock" not in arm_directories:
        raise ValueError("analysis requires a stock baseline arm")
    if len(arm_directories) < 2:
        raise ValueError("analysis requires stock and at least one comparison arm")
    unknown = sorted(set(arm_directories) - set(ANALYSIS_ARMS))
    if unknown:
        raise ValueError("unknown analysis arms: " + ", ".join(unknown))
    root = lhtb_dir.expanduser().resolve()
    task_root = root / "tasks"
    if not task_root.is_dir():
        raise FileNotFoundError(task_root)

    trials_by_arm: dict[str, list[dict[str, Any]]] = {}
    seen_results: set[Path] = set()
    for arm, raw_directory in arm_directories.items():
        directory = raw_directory.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = sorted(directory.rglob("result.json"))
        if not files:
            raise ValueError(f"arm {arm!r} contains no Harbor result.json files")
        trials: list[dict[str, Any]] = []
        for result_file in files:
            resolved = result_file.resolve()
            if resolved in seen_results:
                raise ValueError(
                    f"result file is assigned to multiple arms: {resolved}"
                )
            seen_results.add(resolved)
            trials.append(
                _load_trial(
                    result_file=resolved,
                    arm=arm,
                    task_root=task_root,
                    solve_threshold=solve_threshold,
                )
            )
        trials_by_arm[arm] = trials

    matrix = _validate_matrix(
        trials_by_arm, require_complete_matrix=require_complete_matrix
    )
    arm_reports = {
        arm: _aggregate_arm(trials, solve_threshold=solve_threshold)
        for arm, trials in trials_by_arm.items()
    }
    stock = arm_reports["stock"]
    comparisons = {
        arm: _compare_to_stock(stock, report)
        for arm, report in arm_reports.items()
        if arm != "stock"
    }
    return {
        "schema_version": 1,
        "solve_threshold": solve_threshold,
        "sources": {
            arm: str(path.expanduser().resolve())
            for arm, path in arm_directories.items()
        },
        "matrix": matrix,
        "arms": arm_reports,
        "paired_vs_stock": comparisons,
        "planned_arm_coverage": {
            "present": [arm for arm in ANALYSIS_ARMS if arm in arm_reports],
            "missing": [arm for arm in ANALYSIS_ARMS if arm not in arm_reports],
        },
        "goal_drift_metrics": {
            "status": "requires_domain_annotations",
            "source": "https://arxiv.org/abs/2505.02709",
            "detail": (
                "GD_actions and GD_inaction require paired aligned-action budget "
                "shares and residual misaligned-holding shares. Generic LHTB "
                "result.json files do not contain those task-specific labels, so "
                "this report does not fabricate proxy values."
            ),
        },
    }


def _load_trial(
    *,
    result_file: Path,
    arm: str,
    task_root: Path,
    solve_threshold: float,
) -> dict[str, Any]:
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {result_file}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Harbor result must be an object: {result_file}")
    task = data.get("task_name")
    if not isinstance(task, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", task
    ):
        raise ValueError(f"invalid task_name in {result_file}")
    reward = _reward(data, result_file)
    checksum = data.get("task_checksum")
    if not isinstance(checksum, str) or not checksum:
        raise ValueError(f"missing task_checksum in {result_file}")
    model = _model_identity(data, result_file)
    usage = _usage(data, result_file)
    task_metadata = _task_metadata(task_root, task)
    duration = _duration_seconds(data, result_file)
    return {
        "arm": arm,
        "task": task,
        "reward": reward,
        "solved": reward >= solve_threshold,
        "task_checksum": checksum,
        "model": model,
        "expert_time_estimate_min": task_metadata["expert_time_estimate_min"],
        "category": task_metadata["category"],
        "input_tokens": usage["input_tokens"],
        "cache_tokens": usage["cache_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        "cost_usd": usage["cost_usd"],
        "duration_sec": duration,
        "result_file": str(result_file),
        "result_sha256": _file_sha256(result_file),
    }


def _reward(data: dict[str, Any], result_file: Path) -> float:
    verifier = data.get("verifier_result")
    if not isinstance(verifier, dict):
        exception = data.get("exception_info")
        detail = _exception_name(exception)
        raise ValueError(f"missing verifier result in {result_file} ({detail})")
    rewards = verifier.get("rewards")
    if not isinstance(rewards, dict) or not rewards:
        raise ValueError(f"missing verifier rewards in {result_file}")
    raw = rewards.get("reward", next(iter(rewards.values())))
    return _unit_interval(raw, f"reward in {result_file}")


def _model_identity(data: dict[str, Any], result_file: Path) -> str:
    agent = data.get("agent_info")
    model = agent.get("model_info") if isinstance(agent, dict) else None
    name = model.get("name") if isinstance(model, dict) else None
    provider = model.get("provider") if isinstance(model, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError(f"missing agent model identity in {result_file}")
    if provider is None:
        provider = "unknown"
    if not isinstance(provider, str) or not provider:
        raise ValueError(f"invalid agent model provider in {result_file}")
    return f"{provider}/{name}"


def _usage(data: dict[str, Any], result_file: Path) -> dict[str, int | float]:
    direct = data.get("agent_result")
    step_results = data.get("step_results")
    if isinstance(direct, dict) and isinstance(step_results, list) and step_results:
        raise ValueError(
            f"result has both direct and step agent contexts: {result_file}"
        )
    if isinstance(direct, dict):
        contexts = [direct]
    elif isinstance(step_results, list):
        contexts = [
            step.get("agent_result")
            for step in step_results
            if isinstance(step, dict) and isinstance(step.get("agent_result"), dict)
        ]
    else:
        contexts = []
    if not contexts:
        raise ValueError(f"missing agent usage context in {result_file}")
    totals: dict[str, int | float] = {
        "input_tokens": 0,
        "cache_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    fields = {
        "input_tokens": "n_input_tokens",
        "cache_tokens": "n_cache_tokens",
        "output_tokens": "n_output_tokens",
        "cost_usd": "cost_usd",
    }
    for context in contexts:
        assert isinstance(context, dict)
        for output_name, source_name in fields.items():
            raw = context.get(source_name)
            if output_name == "cost_usd":
                value = _nonnegative_number(raw, f"{source_name} in {result_file}")
            else:
                value = _nonnegative_integer(raw, f"{source_name} in {result_file}")
            totals[output_name] += value
    if totals["cache_tokens"] > totals["input_tokens"]:
        raise ValueError(f"cache tokens exceed input tokens in {result_file}")
    return totals


def _task_metadata(task_root: Path, task: str) -> dict[str, Any]:
    path = task_root / task / "task.toml"
    if not path.is_file():
        raise ValueError(f"task result does not exist in LHTB checkout: {task}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid task metadata for {task}: {error}") from error
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"task {task} lacks [metadata]")
    expert_time = _positive_number(
        metadata.get("expert_time_estimate_min"),
        f"expert_time_estimate_min for {task}",
    )
    category = metadata.get("category")
    if not isinstance(category, str) or not category:
        raise ValueError(f"task {task} lacks metadata.category")
    return {"expert_time_estimate_min": expert_time, "category": category}


def _duration_seconds(data: dict[str, Any], result_file: Path) -> float | None:
    started = data.get("started_at")
    finished = data.get("finished_at")
    if started is None and finished is None:
        return None
    if not isinstance(started, str) or not isinstance(finished, str):
        raise ValueError(f"incomplete trial timestamps in {result_file}")
    try:
        start_time = datetime.fromisoformat(started.replace("Z", "+00:00"))
        finish_time = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        duration = (finish_time - start_time).total_seconds()
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid trial timestamps in {result_file}") from error
    if duration < 0:
        raise ValueError(f"trial finished before it started in {result_file}")
    return duration


def _validate_matrix(
    trials_by_arm: Mapping[str, Sequence[dict[str, Any]]],
    *,
    require_complete_matrix: bool,
) -> dict[str, Any]:
    task_counts: dict[str, dict[str, int]] = {}
    task_checksums: dict[str, set[str]] = defaultdict(set)
    models: set[str] = set()
    for arm, trials in trials_by_arm.items():
        counts: dict[str, int] = defaultdict(int)
        for trial in trials:
            task = trial["task"]
            counts[task] += 1
            task_checksums[task].add(trial["task_checksum"])
            models.add(trial["model"])
        task_counts[arm] = dict(sorted(counts.items()))
    mismatched_checksums = sorted(
        task for task, checksums in task_checksums.items() if len(checksums) != 1
    )
    if mismatched_checksums:
        raise ValueError(
            "task checksum differs across trials: " + ", ".join(mismatched_checksums)
        )
    if len(models) != 1:
        raise ValueError(
            "experiment arms use different agent models: " + ", ".join(sorted(models))
        )
    reference_arm = "stock"
    reference = task_counts[reference_arm]
    mismatched_arms = [
        arm for arm, counts in task_counts.items() if counts != reference
    ]
    complete = not mismatched_arms
    if require_complete_matrix and not complete:
        raise ValueError(
            "arm/task attempt matrix differs from stock: " + ", ".join(mismatched_arms)
        )
    return {
        "complete": complete,
        "reference_arm": reference_arm,
        "attempts_per_task": task_counts,
        "task_checksums": {
            task: next(iter(checksums)) for task, checksums in task_checksums.items()
        },
        "agent_model": next(iter(models)),
    }


def _aggregate_arm(
    trials: Sequence[dict[str, Any]], *, solve_threshold: float
) -> dict[str, Any]:
    task_trials: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        task_trials[trial["task"]].append(trial)
    task_curve = []
    for task, values in task_trials.items():
        rewards = [trial["reward"] for trial in values]
        failure_rate = sum(reward < solve_threshold for reward in rewards) / len(
            rewards
        )
        task_curve.append(
            {
                "task": task,
                "category": values[0]["category"],
                "expert_time_estimate_min": values[0]["expert_time_estimate_min"],
                "attempts": len(values),
                "mean_reward": _mean(rewards),
                "failure_rate": failure_rate,
            }
        )
    task_curve.sort(key=lambda item: (item["expert_time_estimate_min"], item["task"]))
    input_tokens = sum(trial["input_tokens"] for trial in trials)
    cache_tokens = sum(trial["cache_tokens"] for trial in trials)
    output_tokens = sum(trial["output_tokens"] for trial in trials)
    total_tokens = input_tokens + output_tokens
    cost = sum(trial["cost_usd"] for trial in trials)
    durations = [
        trial["duration_sec"] for trial in trials if trial["duration_sec"] is not None
    ]
    return {
        "trial_count": len(trials),
        "task_count": len(task_trials),
        "mean_reward": _mean([trial["reward"] for trial in trials]),
        "solved_rate": _mean([float(trial["solved"]) for trial in trials]),
        "input_tokens": input_tokens,
        "cache_tokens": cache_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "mean_total_tokens_per_trial": total_tokens / len(trials),
        "cache_hit_rate": cache_tokens / input_tokens if input_tokens else None,
        "cost_usd": cost,
        "mean_cost_usd_per_trial": cost / len(trials),
        "mean_duration_sec": _mean(durations) if durations else None,
        "duration_observation_count": len(durations),
        "failure_slope_per_task_length_doubling": _slope(
            [math.log2(item["expert_time_estimate_min"]) for item in task_curve],
            [item["failure_rate"] for item in task_curve],
        ),
        "task_curve": task_curve,
        "trials": list(trials),
    }


def _compare_to_stock(stock: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    stock_tasks = {item["task"]: item for item in stock["task_curve"]}
    arm_tasks = {item["task"]: item for item in arm["task_curve"]}
    shared = sorted(set(stock_tasks) & set(arm_tasks))
    task_deltas = [
        {
            "task": task,
            "reward_delta": arm_tasks[task]["mean_reward"]
            - stock_tasks[task]["mean_reward"],
            "failure_rate_delta": arm_tasks[task]["failure_rate"]
            - stock_tasks[task]["failure_rate"],
        }
        for task in shared
    ]
    return {
        "shared_task_count": len(shared),
        "mean_task_reward_delta": _mean_or_none(
            [item["reward_delta"] for item in task_deltas]
        ),
        "mean_task_failure_rate_delta": _mean_or_none(
            [item["failure_rate_delta"] for item in task_deltas]
        ),
        "mean_total_tokens_per_trial_delta": arm["mean_total_tokens_per_trial"]
        - stock["mean_total_tokens_per_trial"],
        "mean_cost_usd_per_trial_delta": arm["mean_cost_usd_per_trial"]
        - stock["mean_cost_usd_per_trial"],
        "failure_slope_delta": _optional_delta(
            arm["failure_slope_per_task_length_doubling"],
            stock["failure_slope_per_task_length_doubling"],
        ),
        "tasks": task_deltas,
    }


def _slope(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        return None
    return (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )


def _optional_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _share(value: float, name: str) -> float:
    return _unit_interval(value, name)


def _unit_interval(value: Any, name: str) -> float:
    number = _nonnegative_number(value, name)
    if number > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _positive_number(value: Any, name: str) -> float:
    number = _nonnegative_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute a mean over no values")
    return sum(values) / len(values)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return _mean(values) if values else None


def _exception_name(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("exception_type"), str):
        return value["exception_type"]
    return "no exception metadata"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
