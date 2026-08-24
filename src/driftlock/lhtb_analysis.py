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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import UUID

from driftlock.heuristics import HeuristicConfig
from driftlock.judges import DEFAULT_JUDGE_MAX_OUTPUT_TOKENS
from driftlock.lhtb import (
    LHTB_REPOSITORY_REVISION,
    lhtb_runtime_fingerprint,
    openrouter_provider_from_call_kwargs,
    task_directory_sha256,
)
from driftlock.models import (
    JudgeReliabilityStatus,
    RunStatus,
    classify_judge_reliability,
)
from driftlock.oracle import (
    OracleCheckpointError,
    ReplayUsage,
    load_remote_checkpoint_bundle,
    load_source_trial_provenance,
    validate_checkpoint_source_audit,
)

ANALYSIS_ARMS = (
    "stock",
    "retry",
    "driftlock-heuristic",
    "driftlock",
    "oracle",
)
NATIVE_ANALYSIS_ARMS = ("native-driftlock-heuristic", "native-driftlock")
_ACCEPTED_ANALYSIS_ARMS = (*ANALYSIS_ARMS, *NATIVE_ANALYSIS_ARMS)
SOLVE_THRESHOLD = 0.95
_FINE_JUDGE_MODEL = "openrouter/deepseek/deepseek-v4-pro-0813"
_DRIFTLOCK_DETECTOR_ARMS = frozenset(
    {
        "driftlock-heuristic",
        "driftlock",
        "native-driftlock-heuristic",
        "native-driftlock",
    }
)
_DRIFTLOCK_DETECTOR_FIELDS = (
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


_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")


def _recorded_fingerprint(environment: Any, context: str) -> str:
    """Read a trial's recorded build fingerprint without assuming which build.

    This deliberately does not compare against the installed build. Comparability
    is a property of the *data*: every trial in a comparison must carry the same
    fingerprint, which `_require_one_build` enforces globally. Requiring the
    reader to be the writer as well made an analyzer bug permanently unfixable
    for the run it affected, and blocked analysis on any other machine, without
    protecting anything the global check does not already cover.
    """
    value = environment.get("DRIFTLOCK_EXPERIMENT_FINGERPRINT") if environment else None
    if not isinstance(value, str) or not _FINGERPRINT_PATTERN.fullmatch(value):
        raise ValueError(f"missing or malformed build fingerprint in {context}")
    return value


def _require_one_build(fingerprints: set[str], *, context: str) -> str:
    if len(fingerprints) != 1:
        raise ValueError(
            f"{context} mix driftlock builds: "
            + ", ".join(sorted(fingerprints))
            + " -- trials produced by different code are not comparable"
        )
    return next(iter(fingerprints))


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
        if arm not in _ACCEPTED_ANALYSIS_ARMS:
            raise ValueError(
                f"unknown analysis arm {arm!r}; expected one of "
                + ", ".join(_ACCEPTED_ANALYSIS_ARMS)
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
    unknown = sorted(set(arm_directories) - set(_ACCEPTED_ANALYSIS_ARMS))
    if unknown:
        raise ValueError("unknown analysis arms: " + ", ".join(unknown))
    root = lhtb_dir.expanduser().resolve()
    task_root = root / "tasks"
    if not task_root.is_dir():
        raise FileNotFoundError(task_root)
    task_index = _task_index(task_root)

    trials_by_arm: dict[str, list[dict[str, Any]]] = {}
    job_summaries: dict[str, dict[str, Any]] = {}
    task_metadata_cache: dict[str, dict[str, Any]] = {}
    seen_results: set[Path] = set()
    seen_trial_ids: set[str] = set()
    for arm, raw_directory in arm_directories.items():
        directory = raw_directory.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        job_summaries[arm] = _load_job_summary(directory, arm, task_index)
        files = sorted(directory.glob("*/result.json"))
        if not files:
            raise ValueError(f"arm {arm!r} contains no Harbor result.json files")
        if len(files) != job_summaries[arm]["n_total_trials"]:
            raise ValueError(
                f"arm {arm!r} has {len(files)} trial result files but its Harbor "
                f"job summary declares {job_summaries[arm]['n_total_trials']}"
            )
        trials: list[dict[str, Any]] = []
        for result_file in files:
            resolved = result_file.resolve()
            if resolved in seen_results:
                raise ValueError(
                    f"result file is assigned to multiple arms: {resolved}"
                )
            seen_results.add(resolved)
            trial = _load_trial(
                result_file=resolved,
                arm=arm,
                job_id=job_summaries[arm]["job_id"],
                task_index=task_index,
                task_metadata_cache=task_metadata_cache,
                solve_threshold=solve_threshold,
                expected_provider=job_summaries[arm]["agent_provider"],
                expected_judge_provider=job_summaries[arm]["judge_provider"],
            )
            if trial["trial_id"] in seen_trial_ids:
                raise ValueError(f"duplicate Harbor trial id: {trial['trial_id']}")
            seen_trial_ids.add(trial["trial_id"])
            trials.append(trial)
        graded_at_time_cap_count = sum(trial["graded_at_time_cap"] for trial in trials)
        if graded_at_time_cap_count != job_summaries[arm]["n_errored_trials"]:
            raise ValueError(
                f"arm {arm!r} Harbor job summary declares "
                f"{job_summaries[arm]['n_errored_trials']} errored trials but "
                f"trial results contain {graded_at_time_cap_count} graded "
                "AgentTimeoutError trials"
            )
        actual_task_counts: dict[str, int] = defaultdict(int)
        for trial in trials:
            actual_task_counts[trial["task"]] += 1
        if (
            dict(sorted(actual_task_counts.items()))
            != job_summaries[arm]["lock_task_counts"]
        ):
            raise ValueError(f"arm {arm!r} trial results do not match Harbor lock")
        if (
            sorted(trial["lock_trial_signature"] for trial in trials)
            != job_summaries[arm]["lock_trial_signatures"]
        ):
            raise ValueError(
                f"arm {arm!r} trial configs do not match canonical Harbor lock"
            )
        trials_by_arm[arm] = trials

    matrix = _validate_matrix(
        trials_by_arm, require_complete_matrix=require_complete_matrix
    )
    # The guard that actually matters: every trial being compared must have been
    # produced by one build. Previously this was only implied by each lock being
    # matched against the *installed* build, so it disappeared entirely whenever
    # the analyzer itself changed -- and it never covered cross-arm agreement.
    # Checked before the lock signature, which subsumes it but reports it as a
    # generic settings difference.
    build = _require_one_build(
        {
            summary["driftlock_experiment_fingerprint"]
            for summary in job_summaries.values()
        },
        context="experiment arms",
    )
    lock_signatures = {
        summary["lock_signature_sha256"] for summary in job_summaries.values()
    }
    if len(lock_signatures) != 1:
        raise ValueError("experiment arms use different Harbor job-level settings")
    matrix["job_lock_signature_sha256"] = next(iter(lock_signatures))
    analyzer = lhtb_runtime_fingerprint()
    matrix["driftlock_build_fingerprint"] = build
    matrix["analyzer_build_fingerprint"] = analyzer
    matrix["analyzed_by_producing_build"] = build == analyzer
    arm_reports = {
        arm: _aggregate_arm(trials, solve_threshold=solve_threshold)
        for arm, trials in trials_by_arm.items()
    }
    for arm, report in arm_reports.items():
        _validate_job_usage(arm, job_summaries[arm], report)
    stock = arm_reports["stock"]
    comparisons = {
        arm: _compare_to_stock(
            stock,
            report,
            comparable=(
                matrix["attempts_per_task"][arm] == matrix["attempts_per_task"]["stock"]
            ),
        )
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
        "job_summaries": job_summaries,
        "matrix": matrix,
        "arms": arm_reports,
        "paired_vs_stock": comparisons,
        "planned_arm_coverage": {
            "present": [arm for arm in _ACCEPTED_ANALYSIS_ARMS if arm in arm_reports],
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


def _load_job_summary(
    directory: Path, arm: str, task_index: Mapping[str, Path]
) -> dict[str, Any]:
    result_file = directory / "result.json"
    if not result_file.is_file():
        raise ValueError(f"arm {arm!r} lacks its Harbor job-level result.json")
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {result_file}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Harbor job result must be an object: {result_file}")
    job_id = _uuid_string(data.get("id"), f"job id in {result_file}")
    n_total = _positive_integer(
        data.get("n_total_trials"), f"n_total_trials in {result_file}"
    )
    finished = data.get("finished_at")
    if not isinstance(finished, str):
        raise ValueError(f"Harbor job is not finished: {result_file}")
    try:
        datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid finished_at in {result_file}") from error
    stats = data.get("stats")
    if not isinstance(stats, dict):
        raise ValueError(f"missing Harbor job stats in {result_file}")
    counts = {
        name: _nonnegative_integer(stats.get(name), f"{name} in {result_file}")
        for name in (
            "n_completed_trials",
            "n_errored_trials",
            "n_running_trials",
            "n_pending_trials",
            "n_cancelled_trials",
        )
    }
    if counts["n_completed_trials"] != n_total:
        raise ValueError(
            f"Harbor job is incomplete in {result_file}: completed "
            f"{counts['n_completed_trials']} of {n_total} trials"
        )
    unfinished = {
        name: value
        for name, value in counts.items()
        if name in {"n_running_trials", "n_pending_trials"} and value
    }
    if unfinished:
        raise ValueError(f"Harbor job still has active trials in {result_file}")
    if counts["n_cancelled_trials"]:
        raise ValueError(f"Harbor job contains cancelled trials in {result_file}")
    retries = _nonnegative_integer(
        stats.get("n_retries"), f"n_retries in {result_file}"
    )
    if retries:
        raise ValueError(f"Harbor job used forbidden retries in {result_file}")
    usage = {
        "input_tokens": _nonnegative_integer(
            stats.get("n_input_tokens"), f"n_input_tokens in {result_file}"
        ),
        "cache_tokens": _nonnegative_integer(
            stats.get("n_cache_tokens"), f"n_cache_tokens in {result_file}"
        ),
        "output_tokens": _nonnegative_integer(
            stats.get("n_output_tokens"), f"n_output_tokens in {result_file}"
        ),
        "cost_usd": _nonnegative_number(
            stats.get("cost_usd"), f"cost_usd in {result_file}"
        ),
    }
    if usage["cache_tokens"] > usage["input_tokens"]:
        raise ValueError(f"job cache tokens exceed input tokens in {result_file}")
    lock = _load_job_lock(directory, arm, n_total, task_index)
    return {
        "result_file": str(result_file),
        "result_sha256": _file_sha256(result_file),
        "job_id": job_id,
        "n_total_trials": n_total,
        "n_errored_trials": counts["n_errored_trials"],
        "n_retries": retries,
        "usage": usage,
        "finished_at": finished,
        **lock,
    }


def _load_job_lock(
    directory: Path,
    arm: str,
    n_total: int,
    task_index: Mapping[str, Path],
) -> dict[str, Any]:
    lock_file = directory / "lock.json"
    if not lock_file.is_file():
        raise ValueError(f"arm {arm!r} lacks its canonical Harbor lock.json")
    try:
        data = json.loads(lock_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {lock_file}: {error}") from error
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"invalid Harbor job lock in {lock_file}")
    harbor = data.get("harbor")
    if not isinstance(harbor, dict) or (
        harbor.get("version") != "0.7.0"
        or harbor.get("git_commit_hash") != LHTB_REPOSITORY_REVISION
        or harbor.get("is_editable") is not True
    ):
        raise ValueError(f"Harbor lock is not the pinned editable build: {lock_file}")
    concurrency = _positive_integer(
        data.get("n_concurrent_trials"), f"n_concurrent_trials in {lock_file}"
    )
    retry = data.get("retry")
    if not isinstance(retry, dict) or retry.get("max_retries") != 0:
        raise ValueError(f"Harbor lock enables forbidden retries: {lock_file}")
    trials = data.get("trials")
    if not isinstance(trials, list) or len(trials) != n_total:
        raise ValueError(f"Harbor lock trial count mismatch in {lock_file}")
    lock_fingerprints: set[str] = set()
    basename_index = {path.name: name for name, path in task_index.items()}
    task_counts: dict[str, int] = defaultdict(int)
    trial_signatures: list[str] = []
    agent_providers: set[str] = set()
    judge_providers: set[str] = set()
    for trial in trials:
        task = trial.get("task") if isinstance(trial, dict) else None
        agent = trial.get("agent") if isinstance(trial, dict) else None
        name = task.get("name") if isinstance(task, dict) else None
        raw_task_path = task.get("path") if isinstance(task, dict) else None
        digest = task.get("digest") if isinstance(task, dict) else None
        environment = agent.get("env") if isinstance(agent, dict) else None
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or not isinstance(environment, dict)
        ):
            raise ValueError(f"invalid trial provenance in Harbor lock: {lock_file}")
        lock_fingerprints.add(
            _recorded_fingerprint(environment, f"Harbor lock {lock_file}")
        )
        canonical_name = name if name in task_index else basename_index.get(name)
        if canonical_name is None:
            raise ValueError(f"unknown task in Harbor lock: {name}")
        expected_directory = task_index[canonical_name].name
        if (
            not isinstance(raw_task_path, str)
            or Path(raw_task_path).name != expected_directory
        ):
            raise ValueError(f"task path mismatch in Harbor lock: {lock_file}")
        task_counts[canonical_name] += 1
        trial_signatures.append(_lock_trial_signature(canonical_name, trial))
        if arm != "oracle":
            agent_providers.add(_agent_provider(agent, lock_file))
        if arm in {"driftlock", "native-driftlock"}:
            judge_providers.add(_judge_provider(agent, lock_file))
    if len(agent_providers) > 1:
        raise ValueError(
            f"arm {arm!r} Harbor lock uses different agent providers: "
            + ", ".join(sorted(agent_providers))
        )
    if len(judge_providers) > 1:
        raise ValueError(
            f"arm {arm!r} Harbor lock uses different judge providers: "
            + ", ".join(sorted(judge_providers))
        )
    fingerprint = _require_one_build(
        lock_fingerprints, context=f"trials in Harbor lock {lock_file}"
    )
    signature_payload = {
        "schema_version": 1,
        "harbor": harbor,
        "n_concurrent_trials": concurrency,
        "retry": retry,
        "driftlock_experiment_fingerprint": fingerprint,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "lock_file": str(lock_file),
        "lock_sha256": _file_sha256(lock_file),
        "lock_signature_sha256": signature,
        "lock_task_counts": dict(sorted(task_counts.items())),
        "lock_trial_signatures": sorted(trial_signatures),
        "n_concurrent_trials": concurrency,
        "driftlock_experiment_fingerprint": fingerprint,
        "agent_provider": (next(iter(agent_providers)) if agent_providers else None),
        "judge_provider": (next(iter(judge_providers)) if judge_providers else None),
    }


def _validate_job_usage(
    arm: str, summary: dict[str, Any], report: dict[str, Any]
) -> None:
    usage = summary["usage"]
    reconstructed = report["trajectory_reconstructed_usage"]
    for name in ("input_tokens", "cache_tokens", "output_tokens"):
        if usage[name] != report[name] - reconstructed[name]:
            raise ValueError(
                f"arm {arm!r} trial {name} total does not match Harbor job summary"
            )
    if not math.isclose(
        usage["cost_usd"],
        report["cost_usd"] - reconstructed["cost_usd"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"arm {arm!r} trial cost total does not match Harbor job summary"
        )


def _load_trial(
    *,
    result_file: Path,
    arm: str,
    job_id: str,
    task_index: Mapping[str, Path],
    task_metadata_cache: dict[str, dict[str, Any]],
    solve_threshold: float,
    expected_provider: str | None,
    expected_judge_provider: str | None,
) -> dict[str, Any]:
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {result_file}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Harbor result must be an object: {result_file}")
    trial_id = _uuid_string(data.get("id"), f"trial id in {result_file}")
    _validate_trial_provenance(data, result_file, job_id)
    task = data.get("task_name")
    if not isinstance(task, str) or task not in task_index:
        raise ValueError(f"invalid task_name in {result_file}")
    graded_at_time_cap = _graded_at_time_cap(data, result_file)
    reward = _reward(data, result_file)
    checksum = data.get("task_checksum")
    if not isinstance(checksum, str) or not checksum:
        raise ValueError(f"missing task_checksum in {result_file}")
    model = _model_identity(data, result_file)
    total_token_budget, experiment_signature = _validate_arm_identity(
        data,
        arm,
        model,
        result_file,
        expected_provider=expected_provider,
        expected_judge_provider=expected_judge_provider,
    )
    provider, judge_provider = _trial_providers(data, arm, result_file)
    usage, usage_source = _usage(data, result_file)
    driftlock_run_statuses, judge_reliability = _driftlock_run_metadata(
        data, arm, result_file
    )
    task_metadata = task_metadata_cache.get(task)
    if task_metadata is None:
        task_metadata = _task_metadata(task_index[task], task)
        task_metadata_cache[task] = task_metadata
    if checksum != task_metadata["task_checksum"]:
        raise ValueError(
            f"task checksum in {result_file} does not match checkout task {task}"
        )
    duration = _duration_seconds(data, result_file)
    return {
        "arm": arm,
        "trial_id": trial_id,
        "task": task,
        "reward": reward,
        "solved": reward >= solve_threshold,
        "graded_at_time_cap": graded_at_time_cap,
        "task_checksum": checksum,
        "model": model,
        "provider": provider,
        "judge_provider": judge_provider,
        "agent_version": data["agent_info"]["version"],
        "total_token_budget": total_token_budget,
        "detector_settings": _detector_settings(
            kwargs=data["config"]["agent"]["kwargs"], arm=arm, result_file=result_file
        ),
        "experiment_signature": experiment_signature,
        "lock_trial_signature": _lock_trial_signature(task, data["config"]),
        "expert_time_estimate_min": task_metadata["expert_time_estimate_min"],
        "category": task_metadata["category"],
        "input_tokens": usage["input_tokens"],
        "cache_tokens": usage["cache_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        "cost_usd": usage["cost_usd"],
        "usage_source": usage_source,
        "driftlock_run_statuses": driftlock_run_statuses,
        "judge_reliability": judge_reliability,
        "duration_sec": duration,
        "result_file": str(result_file),
        "result_sha256": _file_sha256(result_file),
    }


def _driftlock_run_metadata(
    data: dict[str, Any], arm: str, result_file: Path
) -> tuple[tuple[str, ...], str | None]:
    """Validate terminal and cumulative judge statuses before analysis."""
    if arm not in _DRIFTLOCK_DETECTOR_ARMS:
        return (), None
    direct = data.get("agent_result")
    step_results = data.get("step_results")
    if isinstance(direct, dict):
        contexts = [direct]
    elif isinstance(step_results, list):
        contexts = [
            step["agent_result"]
            for step in step_results
            if isinstance(step, dict) and isinstance(step.get("agent_result"), dict)
        ]
    else:
        contexts = []

    # This is intentionally an allow-list. A newly added RunStatus must be
    # classified here before analysis can treat it as a valid measurement.
    measurable = {
        RunStatus.COMPLETED.value,
        RunStatus.STEP_LIMIT.value,
        RunStatus.TOKEN_LIMIT.value,
        RunStatus.ROLLBACK_LIMIT.value,
    }
    statuses: list[str] = []
    reliabilities: list[str] = []
    for context in contexts:
        metadata = context.get("metadata")
        driftlock = metadata.get("driftlock") if isinstance(metadata, dict) else None
        if driftlock is None:
            continue
        if not isinstance(driftlock, dict) or not isinstance(
            driftlock.get("status"), str
        ):
            raise ValueError(f"invalid driftlock run status in {result_file}")
        status = driftlock["status"]
        if status not in measurable:
            raise ValueError(
                f"non-measurable driftlock run status {status!r} in {result_file}"
            )
        expected_termination = (
            "confirmed_task_complete"
            if status == RunStatus.COMPLETED.value
            else f"driftlock_{status}"
        )
        if metadata.get("termination_reason") != expected_termination:
            raise ValueError(
                f"driftlock termination reason disagrees with status in {result_file}"
            )
        statuses.append(status)
        reliability = driftlock.get("judge_reliability")
        if reliability is None:
            continue
        if not isinstance(reliability, str):
            raise ValueError(f"invalid judge reliability in {result_file}")
        attempts = driftlock.get("judge_attempts")
        failures = driftlock.get("judge_failures")
        if (
            not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or not isinstance(failures, int)
            or isinstance(failures, bool)
        ):
            raise ValueError(f"invalid judge reliability counts in {result_file}")
        try:
            expected_reliability = classify_judge_reliability(
                attempts=attempts,
                failures=failures,
            ).value
        except ValueError as error:
            raise ValueError(
                f"invalid judge reliability counts in {result_file}: {error}"
            ) from error
        if reliability != expected_reliability:
            raise ValueError(
                f"judge reliability disagrees with its counts in {result_file}"
            )
        reliabilities.append(reliability)

    final_reliability = reliabilities[-1] if reliabilities else None
    if final_reliability in {
        JudgeReliabilityStatus.FAILED.value,
        JudgeReliabilityStatus.INCONCLUSIVE.value,
    }:
        raise ValueError(
            f"non-measurable judge reliability {final_reliability!r} in {result_file}"
        )
    return tuple(statuses), final_reliability


def _lock_trial_signature(task_name: str, config: dict[str, Any]) -> str:
    payload = {
        "task": task_name,
        "timeout_multiplier": config.get("timeout_multiplier"),
        "agent_timeout_multiplier": config.get("agent_timeout_multiplier"),
        "verifier_timeout_multiplier": config.get("verifier_timeout_multiplier"),
        "agent_setup_timeout_multiplier": config.get("agent_setup_timeout_multiplier"),
        "environment_build_timeout_multiplier": config.get(
            "environment_build_timeout_multiplier"
        ),
        "agent": config.get("agent"),
        "environment": config.get("environment"),
        "verifier": config.get("verifier"),
    }
    serialized = json.dumps(
        _without_none(payload), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def _validate_trial_provenance(
    data: dict[str, Any], result_file: Path, job_id: str
) -> None:
    trial_name = data.get("trial_name")
    if not isinstance(trial_name, str) or trial_name != result_file.parent.name:
        raise ValueError(f"trial_name does not match directory in {result_file}")
    config = data.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"missing trial config in {result_file}")
    config_trial_name = config.get("trial_name")
    if config_trial_name != trial_name:
        raise ValueError(f"config trial_name mismatch in {result_file}")
    config_job_id = _uuid_string(
        config.get("job_id"), f"trial config job_id in {result_file}"
    )
    if config_job_id != job_id:
        raise ValueError(f"trial belongs to a different Harbor job: {result_file}")


def _agent_provider(agent: Any, source: Path) -> str:
    kwargs = agent.get("kwargs") if isinstance(agent, dict) else None
    call_kwargs = kwargs.get("llm_call_kwargs") if isinstance(kwargs, dict) else None
    if not isinstance(call_kwargs, dict):
        raise ValueError(f"missing llm_call_kwargs in {source}")
    return openrouter_provider_from_call_kwargs(
        call_kwargs, source=f"llm_call_kwargs in {source}"
    )


def _judge_provider(agent: Any, source: Path) -> str:
    kwargs = agent.get("kwargs") if isinstance(agent, dict) else None
    call_kwargs = (
        kwargs.get("driftlock_judge_llm_call_kwargs")
        if isinstance(kwargs, dict)
        else None
    )
    if not isinstance(call_kwargs, dict) or set(call_kwargs) != {"extra_body"}:
        raise ValueError(
            f"fine judge driftlock_judge_llm_call_kwargs in {source} must contain "
            "exactly extra_body"
        )
    return openrouter_provider_from_call_kwargs(
        call_kwargs, source=f"driftlock_judge_llm_call_kwargs in {source}"
    )


def _trial_providers(
    data: dict[str, Any], arm: str, result_file: Path
) -> tuple[str, str | None]:
    config = data.get("config")
    agent = config.get("agent") if isinstance(config, dict) else None
    if arm != "oracle":
        provider = _agent_provider(agent, result_file)
        judge_provider = (
            _judge_provider(agent, result_file)
            if arm in {"driftlock", "native-driftlock"}
            else None
        )
        return provider, judge_provider

    kwargs = agent.get("kwargs") if isinstance(agent, dict) else None
    if not isinstance(kwargs, dict):
        raise ValueError(f"missing oracle agent config in {result_file}")
    try:
        source = load_source_trial_provenance(
            kwargs["driftlock_source_result"],
            expected_sha256=kwargs["driftlock_source_result_sha256"],
        )
    except (KeyError, OracleCheckpointError) as error:
        raise ValueError(f"oracle source provider is invalid: {result_file}") from error
    source_agent = source.data["config"]["agent"]
    provider = _agent_provider(source_agent, source.result_path)
    source_kwargs = source_agent["kwargs"]
    judge_provider = (
        _judge_provider(source_agent, source.result_path)
        if "driftlock_judge_model" in source_kwargs
        else None
    )
    return provider, judge_provider


def _validate_arm_identity(
    data: dict[str, Any],
    arm: str,
    model: str,
    result_file: Path,
    *,
    expected_provider: str | None = None,
    expected_judge_provider: str | None = None,
) -> tuple[int | None, str]:
    agent_info = data.get("agent_info")
    info_name = agent_info.get("name") if isinstance(agent_info, dict) else None
    info_version = agent_info.get("version") if isinstance(agent_info, dict) else None
    config = data.get("config")
    agent = config.get("agent") if isinstance(config, dict) else None
    if not isinstance(agent, dict):
        raise ValueError(f"missing trial agent config in {result_file}")
    if agent.get("model_name") != model:
        raise ValueError(f"agent config model does not match result in {result_file}")
    kwargs = agent.get("kwargs")
    environment = agent.get("env")
    if not isinstance(kwargs, dict) or not isinstance(environment, dict):
        raise ValueError(f"invalid trial agent config in {result_file}")
    if arm == "oracle":
        return _validate_oracle_identity(
            data=data,
            config=config,
            agent=agent,
            kwargs=kwargs,
            environment=environment,
            model=model,
            result_file=result_file,
        )

    base_kwargs = {
        "api_base",
        "parser_name",
        "temperature",
        "record_terminal_session",
        "llm_call_kwargs",
        "model_info",
    }
    llm_kwargs = kwargs.get("llm_call_kwargs")
    model_info = kwargs.get("model_info")
    if not isinstance(llm_kwargs, dict):
        raise ValueError(f"invalid llm_call_kwargs in {result_file}")
    provider = openrouter_provider_from_call_kwargs(
        llm_kwargs, source=f"llm_call_kwargs in {result_file}"
    )
    if expected_provider is not None and provider != expected_provider:
        raise ValueError(
            f"arm {arm!r} expected provider {expected_provider!r} but trial "
            f"recorded {provider!r} in {result_file}"
        )
    common_valid = (
        isinstance(agent.get("override_timeout_sec"), (int, float))
        and not isinstance(agent.get("override_timeout_sec"), bool)
        and agent["override_timeout_sec"] > 0
        and agent.get("override_setup_timeout_sec") is None
        and agent.get("max_timeout_sec") is None
        and isinstance(kwargs.get("api_base"), str)
        and bool(kwargs["api_base"])
        and kwargs.get("parser_name") == "json"
        and kwargs.get("temperature") == 0.7
        and kwargs.get("record_terminal_session") is True
        and llm_kwargs.get("temperature") == 0.7
        and llm_kwargs.get("max_tokens") == 8192
        and llm_kwargs.get("timeout") == 240
        and model_info
        == {
            "max_input_tokens": 128000,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }
    )
    if not common_valid:
        raise ValueError(f"agent config differs from the frozen harness: {result_file}")
    experiment_signature = _experiment_signature(config, kwargs, model, result_file)

    if arm == "stock":
        expected_environment = {
            "HB_CONTINUE_MODE": "fresh",
            "DRIFTLOCK_EXPERIMENT_FINGERPRINT": _recorded_fingerprint(
                environment, f"arm {arm!r} config in {result_file}"
            ),
        }
        valid = (
            info_name == "terminus-2"
            and info_version == "2.0.0"
            and agent.get("name") == "terminus-2"
            and agent.get("import_path") is None
            and environment == expected_environment
            and set(kwargs)
            == base_kwargs | {"enable_summarize", "proactive_summarization_threshold"}
            and kwargs.get("enable_summarize") is True
            and kwargs.get("proactive_summarization_threshold") == 8000
            and set(llm_kwargs)
            == {
                "temperature",
                "max_tokens",
                "timeout",
                "extra_body",
                "num_retries",
            }
            and llm_kwargs.get("num_retries") == 4
        )
        if not valid:
            raise ValueError(f"arm 'stock' has a non-stock agent config: {result_file}")
        return None, experiment_signature

    expected_import_path: str
    expected_name: str
    if arm == "retry":
        expected_import_path = "driftlock.harbor_agent:LHTBBlindRetryAgent"
        expected_name = "compute-matched-blind-retry-terminus-2"
    elif arm in {"driftlock-heuristic", "driftlock"}:
        expected_import_path = "driftlock.harbor_agent:LHTBDriftlockAgent"
        expected_name = "driftlock-terminus-2"
    elif arm in {"native-driftlock-heuristic", "native-driftlock"}:
        expected_import_path = "driftlock.harbor_native_agent:LHTBNativeDriftlockAgent"
        expected_name = "driftlock-native-tool-agent"
    else:
        raise ValueError(f"unsupported online arm: {arm}")

    valid = (
        info_name == expected_name
        and info_version == _installed_driftlock_version()
        and agent.get("name") is None
        and agent.get("import_path") == expected_import_path
        and environment
        == {
            "HB_CONTINUE_MODE": "same_conversation",
            "DRIFTLOCK_EXPERIMENT_FINGERPRINT": _recorded_fingerprint(
                environment, f"arm {arm!r} config in {result_file}"
            ),
        }
        and kwargs.get("enable_summarize") is False
        and set(llm_kwargs) == {"temperature", "max_tokens", "timeout", "extra_body"}
    )
    if not valid:
        raise ValueError(f"arm {arm!r} has the wrong agent config: {result_file}")
    budget = _positive_integer(
        kwargs.get("driftlock_max_tokens"),
        f"driftlock_max_tokens in {result_file}",
    )
    for field, expected in (
        ("driftlock_max_steps", 500),
        ("driftlock_max_rollbacks", 3),
        ("driftlock_checkpoint_interval", 5),
    ):
        if kwargs.get(field) != expected:
            raise ValueError(f"unexpected {field} in {result_file}")
    judge_fields = {
        "driftlock_judge_model",
        "driftlock_judge_api_base",
        "driftlock_judge_max_output_tokens",
        "driftlock_judge_llm_call_kwargs",
    }
    expected_keys = base_kwargs | {
        "enable_summarize",
        "driftlock_max_tokens",
        "driftlock_max_steps",
        "driftlock_max_rollbacks",
        "driftlock_checkpoint_interval",
    }
    if arm in _DRIFTLOCK_DETECTOR_ARMS:
        _detector_settings(kwargs=kwargs, arm=arm, result_file=result_file)
        expected_keys.update(_DRIFTLOCK_DETECTOR_FIELDS)
    if "driftlock_retain_checkpoints" in kwargs:
        if (
            arm not in {"driftlock-heuristic", "driftlock"}
            or kwargs["driftlock_retain_checkpoints"] is not True
        ):
            raise ValueError(f"arm {arm!r} has invalid checkpoint retention")
        expected_keys.add("driftlock_retain_checkpoints")
    if arm in {"driftlock", "native-driftlock"}:
        judge_provider = _judge_provider(agent, result_file)
        if (
            kwargs.get("driftlock_judge_model") != _FINE_JUDGE_MODEL
            or not isinstance(kwargs.get("driftlock_judge_api_base"), str)
            or not kwargs["driftlock_judge_api_base"]
            or kwargs.get("driftlock_judge_max_output_tokens")
            != DEFAULT_JUDGE_MAX_OUTPUT_TOKENS
        ):
            raise ValueError(f"driftlock arm lacks its fine judge: {result_file}")
        if (
            expected_judge_provider is not None
            and judge_provider != expected_judge_provider
        ):
            raise ValueError(
                f"arm {arm!r} expected judge provider "
                f"{expected_judge_provider!r} but trial recorded "
                f"{judge_provider!r} in {result_file}"
            )
        expected_keys |= judge_fields
    elif judge_fields & kwargs.keys():
        raise ValueError(
            f"arm {arm!r} unexpectedly enables a fine judge: {result_file}"
        )
    if set(kwargs) != expected_keys:
        raise ValueError(f"arm {arm!r} has unexpected agent settings: {result_file}")
    return budget, experiment_signature


def _detector_settings(
    *, kwargs: dict[str, Any], arm: str, result_file: Path
) -> dict[str, Any] | None:
    if arm not in _DRIFTLOCK_DETECTOR_ARMS:
        return None
    missing = [field for field in _DRIFTLOCK_DETECTOR_FIELDS if field not in kwargs]
    if missing:
        raise ValueError(
            f"arm {arm!r} is missing detector setting {missing[0]} in {result_file}"
        )
    settings: dict[str, Any] = {
        field: kwargs[field] for field in _DRIFTLOCK_DETECTOR_FIELDS
    }
    raw_corroborating = settings["driftlock_corroborating_signals"]
    if not isinstance(raw_corroborating, (list, tuple)) or not all(
        isinstance(kind, str) for kind in raw_corroborating
    ):
        raise ValueError(
            f"arm {arm!r} has invalid driftlock_corroborating_signals in {result_file}"
        )
    # Hashable and order-independent: two arms that list the same kinds in a
    # different order are the same configuration, and the cross-arm comparison
    # below puts these into a set.
    settings["driftlock_corroborating_signals"] = tuple(sorted(set(raw_corroborating)))
    try:
        HeuristicConfig(
            no_change_steps=settings["driftlock_no_change_steps"],
            loop_window=settings["driftlock_loop_window"],
            loop_repetitions=settings["driftlock_loop_repetitions"],
            error_window=settings["driftlock_error_window"],
            error_rate=settings["driftlock_error_rate"],
            command_failure_window=settings["driftlock_command_failure_window"],
            command_failure_rate=settings["driftlock_command_failure_rate"],
            reward_stall_steps=settings["driftlock_reward_stall_steps"],
            reward_epsilon=settings["driftlock_reward_epsilon"],
            corroborating_signals=frozenset(
                settings["driftlock_corroborating_signals"]
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"arm {arm!r} has invalid detector settings in {result_file}: {error}"
        ) from error
    return settings


def _validate_oracle_identity(
    *,
    data: dict[str, Any],
    config: dict[str, Any],
    agent: dict[str, Any],
    kwargs: dict[str, Any],
    environment: dict[str, Any],
    model: str,
    result_file: Path,
) -> tuple[None, str]:
    expected_keys = {
        "driftlock_oracle_mode",
        "driftlock_checkpoint_dir",
        "driftlock_checkpoint_digest",
        "driftlock_expected_workspace",
        "driftlock_source_trial_id",
        "driftlock_source_task_name",
        "driftlock_source_result",
        "driftlock_source_result_sha256",
        "driftlock_source_audit",
        "driftlock_source_audit_sha256",
        "driftlock_source_usage",
    }
    info = data.get("agent_info")
    if (
        not isinstance(info, dict)
        or info.get("name") != "driftlock-checkpoint-replay-oracle"
        or info.get("version") != _installed_driftlock_version()
        or agent.get("name") is not None
        or agent.get("import_path")
        != "driftlock.harbor_agent:LHTBCheckpointReplayOracle"
        or environment
        != {
            "HB_CONTINUE_MODE": "same_conversation",
            "DRIFTLOCK_EXPERIMENT_FINGERPRINT": _recorded_fingerprint(
                environment, f"oracle config in {result_file}"
            ),
        }
        or kwargs.get("driftlock_oracle_mode") != "isolated-checkpoint-replay"
        or set(kwargs) != expected_keys
    ):
        raise ValueError(f"oracle result has the wrong replay config: {result_file}")
    timeout = agent.get("override_timeout_sec")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise ValueError(f"oracle result has an invalid timeout: {result_file}")
    _validate_trial_environment(config, result_file)
    try:
        source = load_source_trial_provenance(
            kwargs["driftlock_source_result"],
            expected_sha256=kwargs["driftlock_source_result_sha256"],
        )
        usage = ReplayUsage.from_mapping(kwargs["driftlock_source_usage"])
        bundle = load_remote_checkpoint_bundle(
            kwargs["driftlock_checkpoint_dir"],
            expected_digest=kwargs["driftlock_checkpoint_digest"],
            expected_workspace=kwargs["driftlock_expected_workspace"],
        )
        phase = validate_checkpoint_source_audit(
            bundle.checkpoint.path,
            source_result=source.result_path,
            source_audit=kwargs["driftlock_source_audit"],
            expected_audit_sha256=kwargs["driftlock_source_audit_sha256"],
        )
    except (KeyError, OracleCheckpointError, ValueError) as error:
        raise ValueError(
            f"oracle replay provenance is invalid: {result_file}"
        ) from error
    if (
        source.trial_id != kwargs["driftlock_source_trial_id"]
        or source.task_name != kwargs["driftlock_source_task_name"]
        or source.task_name != data.get("task_name")
        or source.model_name != agent.get("model_name")
        or source.model_name != model
        or source.usage != usage
    ):
        raise ValueError(f"oracle replay differs from source trial: {result_file}")
    direct = data.get("agent_result")
    metadata = direct.get("metadata") if isinstance(direct, dict) else None
    audit = metadata.get("oracle") if isinstance(metadata, dict) else None
    if (
        not isinstance(audit, dict)
        or metadata.get("termination_reason") != "oracle_checkpoint_replay"
        or audit.get("source_trial_id") != source.trial_id
        or audit.get("source_task_name") != source.task_name
        or audit.get("source_result_sha256") != source.result_sha256
        or audit.get("source_audit_sha256") != kwargs["driftlock_source_audit_sha256"]
        or audit.get("checkpoint_id") != bundle.checkpoint.checkpoint_id
        or audit.get("checkpoint_digest") != bundle.checkpoint.digest
        or audit.get("workspace") != bundle.remote_workspace
        or audit.get("source_usage") != usage.as_dict()
        or audit.get("source_phase") != phase
        or audit.get("usage_policy") != "full-source-trial-conservative"
    ):
        raise ValueError(f"oracle runtime audit is inconsistent: {result_file}")
    source_kwargs = source.data["config"]["agent"]["kwargs"]
    source_arm = (
        "driftlock"
        if "driftlock_judge_model" in source_kwargs
        else "driftlock-heuristic"
    )
    _, experiment_signature = _validate_arm_identity(
        source.data,
        source_arm,
        model,
        source.result_path,
        expected_provider=_agent_provider(
            source.data["config"]["agent"], source.result_path
        ),
        expected_judge_provider=(
            _judge_provider(source.data["config"]["agent"], source.result_path)
            if source_arm == "driftlock"
            else None
        ),
    )
    return None, experiment_signature


def _experiment_signature(
    config: dict[str, Any],
    kwargs: dict[str, Any],
    model: str,
    result_file: Path,
) -> str:
    environment, verifier, multipliers = _validate_trial_environment(
        config, result_file
    )

    agent = config["agent"]
    signature = {
        "model": model,
        "agent_timeout_sec": agent["override_timeout_sec"],
        "api_base": kwargs["api_base"],
        "parser_name": kwargs["parser_name"],
        "temperature": kwargs["temperature"],
        "record_terminal_session": kwargs["record_terminal_session"],
        "llm_call_kwargs": {
            key: kwargs["llm_call_kwargs"][key]
            for key in ("temperature", "max_tokens", "timeout", "extra_body")
        },
        "model_info": kwargs["model_info"],
        "environment": environment,
        "verifier": verifier,
        "multipliers": multipliers,
    }
    serialized = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_trial_environment(
    config: dict[str, Any], result_file: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    environment = config.get("environment")
    verifier = config.get("verifier")
    expected_environment = {
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
    }
    expected_verifier = {
        "override_timeout_sec": None,
        "max_timeout_sec": None,
        "env": {},
        "disable": False,
    }
    if environment != expected_environment or verifier != expected_verifier:
        raise ValueError(
            f"trial environment differs from frozen harness: {result_file}"
        )
    multipliers = {
        key: config.get(key)
        for key in (
            "timeout_multiplier",
            "agent_timeout_multiplier",
            "verifier_timeout_multiplier",
            "agent_setup_timeout_multiplier",
            "environment_build_timeout_multiplier",
        )
    }
    if multipliers != {
        "timeout_multiplier": 1.0,
        "agent_timeout_multiplier": None,
        "verifier_timeout_multiplier": None,
        "agent_setup_timeout_multiplier": None,
        "environment_build_timeout_multiplier": None,
    }:
        raise ValueError(
            f"trial timeout config differs from frozen harness: {result_file}"
        )
    if config.get("artifacts") != []:
        raise ValueError(f"trial artifacts differ from frozen harness: {result_file}")
    return environment, verifier, multipliers


def _reward(data: dict[str, Any], result_file: Path) -> float:
    verifier = data.get("verifier_result")
    if not isinstance(verifier, dict):
        exception = data.get("exception_info")
        detail = _exception_name(exception)
        raise ValueError(
            f"Harbor job contains errored trials: missing verifier result in "
            f"{result_file} ({detail})"
        )
    rewards = verifier.get("rewards")
    if not isinstance(rewards, dict) or "reward" not in rewards:
        raise ValueError(f"missing canonical verifier reward in {result_file}")
    raw = rewards["reward"]
    return _unit_interval(raw, f"reward in {result_file}")


def _graded_at_time_cap(data: dict[str, Any], result_file: Path) -> bool:
    exception = data.get("exception_info")
    if exception is None:
        return False
    exception_type = _exception_name(exception)
    if exception_type != "AgentTimeoutError":
        raise ValueError(
            f"trial raised {exception_type} instead of reaching the accepted "
            f"agent time cap in {result_file}"
        )
    return True


def _model_identity(data: dict[str, Any], result_file: Path) -> str:
    agent = data.get("agent_info")
    model = agent.get("model_info") if isinstance(agent, dict) else None
    name = model.get("name") if isinstance(model, dict) else None
    provider = model.get("provider") if isinstance(model, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError(f"missing agent model identity in {result_file}")
    if provider is None:
        return name
    if not isinstance(provider, str) or not provider:
        raise ValueError(f"invalid agent model provider in {result_file}")
    return f"{provider}/{name}"


def _usage(
    data: dict[str, Any], result_file: Path
) -> tuple[dict[str, int | float], str]:
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
        return _trajectory_usage(result_file), "trajectory"
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
    return totals, "agent_result" if isinstance(direct, dict) else "step_results"


def _trajectory_usage(result_file: Path) -> dict[str, int | float]:
    trajectory_file = result_file.parent / "agent" / "trajectory.json"
    try:
        data = json.loads(trajectory_file.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"missing trajectory needed to reconstruct agent usage: {trajectory_file}"
        ) from error
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(
            f"unreadable trajectory needed to reconstruct agent usage: "
            f"{trajectory_file}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON in trajectory needed to reconstruct agent usage: "
            f"{trajectory_file}: {error}"
        ) from error
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise ValueError(
            f"malformed trajectory needed to reconstruct agent usage: "
            f"{trajectory_file} must contain a steps array"
        )
    steps = data["steps"]
    if not steps:
        raise ValueError(
            f"malformed trajectory needed to reconstruct agent usage: "
            f"{trajectory_file} has no steps"
        )
    totals: dict[str, int | float] = {
        "input_tokens": 0,
        "cache_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    fields = {
        "input_tokens": "prompt_tokens",
        "cache_tokens": "cached_tokens",
        "output_tokens": "completion_tokens",
        "cost_usd": "cost_usd",
    }
    for index, step in enumerate(steps):
        context = f"step {index} metrics in {trajectory_file}"
        if not isinstance(step, dict):
            raise ValueError(f"malformed trajectory {context}: step must be an object")
        metrics = step.get("metrics")
        if metrics is None and (
            step.get("source") in {"system", "user"}
            or (step.get("source") == "agent" and step.get("llm_call_count") == 0)
        ):
            continue
        if not isinstance(metrics, dict):
            raise ValueError(
                f"malformed trajectory {context}: metrics must be an object"
            )
        for output_name, source_name in fields.items():
            raw = metrics.get(source_name)
            if raw is None and output_name == "cache_tokens":
                # A call with no cache hit may omit the field entirely. Reading
                # that as zero is the conservative direction: it attributes the
                # whole prompt to full-price input rather than to cheap cache
                # reads, so an omission can never make a billed call look
                # cheaper than it was. Every other field stays required --
                # a missing prompt, completion or cost is real information loss.
                value: int | float = 0
            elif output_name == "cost_usd":
                value = _nonnegative_number(raw, f"{source_name} in {context}")
            else:
                value = _nonnegative_integer(raw, f"{source_name} in {context}")
            totals[output_name] += value
    if totals["cache_tokens"] > totals["input_tokens"]:
        raise ValueError(
            f"trajectory cache tokens exceed input tokens in {trajectory_file}"
        )
    return totals


def _task_index(task_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for directory in sorted(task_root.iterdir()):
        if not directory.is_dir():
            continue
        path = directory / "task.toml"
        if not path.is_file():
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"invalid task metadata in {path}: {error}") from error
        task_section = data.get("task")
        name = task_section.get("name") if isinstance(task_section, dict) else None
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", name
        ):
            raise ValueError(f"invalid [task].name in {path}")
        if name in result:
            raise ValueError(f"duplicate LHTB task name: {name}")
        result[name] = directory
    if not result:
        raise ValueError(f"no named LHTB tasks found in {task_root}")
    return result


def _task_metadata(directory: Path, task: str) -> dict[str, Any]:
    path = directory / "task.toml"
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
    return {
        "expert_time_estimate_min": expert_time,
        "category": category,
        "task_checksum": task_directory_sha256(directory),
    }


_task_directory_sha256 = task_directory_sha256  # historical alias


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
    providers: set[str] = set()
    judge_providers: set[str] = set()
    experiment_signatures: set[str] = set()
    arm_budgets: dict[str, set[int]] = defaultdict(set)
    arm_versions: dict[str, set[str]] = defaultdict(set)
    detector_values: dict[str, dict[str, set[Any]]] = {
        field: defaultdict(set) for field in _DRIFTLOCK_DETECTOR_FIELDS
    }
    for arm, trials in trials_by_arm.items():
        counts: dict[str, int] = defaultdict(int)
        for trial in trials:
            task = trial["task"]
            counts[task] += 1
            task_checksums[task].add(trial["task_checksum"])
            models.add(trial["model"])
            providers.add(trial["provider"])
            if trial["judge_provider"] is not None:
                judge_providers.add(trial["judge_provider"])
            experiment_signatures.add(trial["experiment_signature"])
            arm_versions[arm].add(trial["agent_version"])
            if trial["total_token_budget"] is not None:
                arm_budgets[arm].add(trial["total_token_budget"])
            settings = trial["detector_settings"]
            if settings is not None:
                for field in _DRIFTLOCK_DETECTOR_FIELDS:
                    detector_values[field][arm].add(settings[field])
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
    if len(providers) != 1:
        raise ValueError(
            "experiment arms use different agent providers: "
            + ", ".join(sorted(providers))
        )
    if len(judge_providers) > 1:
        raise ValueError(
            "fine-judge arms use different providers: "
            + ", ".join(sorted(judge_providers))
        )
    for field, values_by_arm in detector_values.items():
        distinct_values = {
            value for arm_values in values_by_arm.values() for value in arm_values
        }
        if len(distinct_values) > 1:
            details = ", ".join(
                f"{arm}={'/'.join(repr(value) for value in sorted(values))}"
                for arm, values in sorted(values_by_arm.items())
            )
            raise ValueError(f"controlled arms use different {field}: {details}")
    if len(experiment_signatures) != 1:
        raise ValueError("experiment arms use different non-treatment configurations")
    inconsistent_version_arms = sorted(
        arm for arm, versions in arm_versions.items() if len(versions) != 1
    )
    if inconsistent_version_arms:
        raise ValueError(
            "arm uses inconsistent agent versions: "
            + ", ".join(inconsistent_version_arms)
        )
    inconsistent_budget_arms = sorted(
        arm for arm, budgets in arm_budgets.items() if len(budgets) != 1
    )
    if inconsistent_budget_arms:
        raise ValueError(
            "arm uses inconsistent total-token budgets: "
            + ", ".join(inconsistent_budget_arms)
        )
    controlled_budgets = {
        next(iter(budgets))
        for arm, budgets in arm_budgets.items()
        if arm
        in {
            "retry",
            "driftlock-heuristic",
            "driftlock",
            "native-driftlock-heuristic",
            "native-driftlock",
        }
        and budgets
    }
    if len(controlled_budgets) > 1:
        raise ValueError("controlled arms do not share one total-token budget")
    # The guard that actually matters: every trial being compared must have been
    # produced by one build. Previously this was only implied by each lock being
    # matched against the installed build, so it silently disappeared whenever the
    # analyzer itself changed.
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
        "agent_provider": next(iter(providers)),
        "fine_judge_provider": (
            next(iter(judge_providers)) if judge_providers else None
        ),
        "experiment_signature_sha256": next(iter(experiment_signatures)),
        "agent_versions": {
            arm: next(iter(versions)) for arm, versions in arm_versions.items()
        },
        "controlled_total_token_budget": (
            next(iter(controlled_budgets)) if controlled_budgets else None
        ),
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
    reconstructed_trials = [
        trial for trial in trials if trial["usage_source"] == "trajectory"
    ]
    reconstructed_usage = {
        "input_tokens": sum(trial["input_tokens"] for trial in reconstructed_trials),
        "cache_tokens": sum(trial["cache_tokens"] for trial in reconstructed_trials),
        "output_tokens": sum(trial["output_tokens"] for trial in reconstructed_trials),
        "cost_usd": sum(trial["cost_usd"] for trial in reconstructed_trials),
    }
    durations = [
        trial["duration_sec"] for trial in trials if trial["duration_sec"] is not None
    ]
    return {
        "trial_count": len(trials),
        "graded_at_time_cap_count": sum(
            trial["graded_at_time_cap"] for trial in trials
        ),
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
        "trajectory_reconstructed_usage": reconstructed_usage,
        "mean_duration_sec": _mean(durations) if durations else None,
        "duration_observation_count": len(durations),
        "failure_slope_per_task_length_doubling": _slope(
            [math.log2(item["expert_time_estimate_min"]) for item in task_curve],
            [item["failure_rate"] for item in task_curve],
        ),
        "task_curve": task_curve,
        "trials": list(trials),
    }


def _compare_to_stock(
    stock: dict[str, Any], arm: dict[str, Any], *, comparable: bool
) -> dict[str, Any]:
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
        "aggregate_workload_comparable": comparable,
        "aggregate_delta_status": (
            "complete_task_attempt_matrix"
            if comparable
            else "unavailable_for_incomplete_task_attempt_matrix"
        ),
        "mean_total_tokens_per_trial_delta": (
            arm["mean_total_tokens_per_trial"] - stock["mean_total_tokens_per_trial"]
            if comparable
            else None
        ),
        "mean_cost_usd_per_trial_delta": (
            arm["mean_cost_usd_per_trial"] - stock["mean_cost_usd_per_trial"]
            if comparable
            else None
        ),
        "failure_slope_delta": (
            _optional_delta(
                arm["failure_slope_per_task_length_doubling"],
                stock["failure_slope_per_task_length_doubling"],
            )
            if comparable
            else None
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


def _positive_integer(value: Any, name: str) -> int:
    number = _nonnegative_integer(value, name)
    if number == 0:
        raise ValueError(f"{name} must be positive")
    return number


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


def _uuid_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UUID string")
    try:
        return str(UUID(value))
    except ValueError as error:
        raise ValueError(f"{name} must be a UUID string") from error


def _installed_driftlock_version() -> str:
    try:
        return version("driftlock")
    except PackageNotFoundError as error:
        raise ValueError("driftlock distribution metadata is unavailable") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
