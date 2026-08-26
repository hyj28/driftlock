"""Localize non-improving work in checkpoint-scoring timelines.

The input is the JSON-compatible report assembled by
:func:`driftlock.checkpoint_scoring.assemble_scored_timelines`.  This module is
deliberately Harbor-free: it judges only the measurements already present in a
scored timeline and refuses when those measurements cannot support localization.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

LOCALIZATION_REPORT_NAME = "checkpoint-localization.json"

# Two scored points, as in epidemic-inverse-control-audit, provide only one
# before/after gap and cannot establish where a *stretch* stopped improving.
# Three is therefore the smallest admissible series.  This admits the four-point
# alp-paper-reproduction regression rather than hiding it behind a sample-size
# heuristic, while the independent uniformity check refuses 2048 and sudoku.
_MIN_SCORED_CHECKPOINTS = 3

# Verifier rewards are ratios and partial-credit fractions: round five contains
# decimal forms such as riscv's 0.389091 and spice's 0.7575757575757576.  A JSON
# precision mismatch can perturb those by roughly 1e-16.  This absolute tolerance
# absorbs that representation noise while sitting ten orders of magnitude below
# the smallest real round-five delta (0.03), and far below spice's -0.0909 and
# alp's -0.2 regressions.
_SCORE_ABS_TOLERANCE = 1e-12


def localize_scored_timeline(timeline: Mapping[str, Any]) -> dict[str, Any]:
    """Judge one scorer timeline and return segments or an explicit refusal."""

    if not isinstance(timeline, Mapping):
        raise ValueError("scored timeline must be an object")
    trial_name = _required_string(timeline, "trial_name", "scored timeline")
    task_name = _required_string(timeline, "task_name", "scored timeline")
    raw_checkpoints = timeline.get("checkpoints")
    if not isinstance(raw_checkpoints, list):
        raise ValueError(f"timeline {trial_name!r} checkpoints must be a list")

    checkpoints = [
        _checkpoint(item, index, trial_name)
        for index, item in enumerate(raw_checkpoints)
    ]
    final_reward = _optional_reward(
        timeline.get("final_reward"), f"timeline {trial_name!r} final"
    )
    scored = [point for point in checkpoints if point["reward"] is not None]
    result: dict[str, Any] = {
        "task_name": task_name,
        "trial_name": trial_name,
        "status": "usable",
        "checkpoint_count": len(checkpoints),
        "scored_checkpoint_count": len(scored),
        "unscored_checkpoint_count": len(checkpoints) - len(scored),
        "final_reward": final_reward,
        "segments": [],
    }

    try:
        _validate_order(checkpoints, trial_name)
    except ValueError as error:
        result["status"] = "refused"
        result["refusal"] = {
            "reason": "malformed_timeline",
            "detail": str(error),
        }
        return result

    if len(scored) < _MIN_SCORED_CHECKPOINTS:
        result["status"] = "refused"
        result["refusal"] = {
            "reason": "insufficient_scored_checkpoints",
            "detail": (
                f"only {len(scored)} of {len(checkpoints)} checkpoints are scored; "
                f"at least {_MIN_SCORED_CHECKPOINTS} scored checkpoints are needed "
                "to localize a non-improving stretch"
            ),
        }
        return result

    observed_rewards = [point["reward"] for point in scored]
    if max(observed_rewards) - min(observed_rewards) <= _SCORE_ABS_TOLERANCE:
        only_reward = scored[0]["reward"]
        # A final reward is an independently scored endpoint, not another
        # retained checkpoint.  Even when it differs (2048 is the real case), it
        # cannot say which checkpoint gap was useful and therefore cannot rescue
        # an otherwise uniform localization signal.
        final_note = (
            "; the independently measured final reward does not locate variation "
            "among the retained checkpoints"
            if final_reward is not None
            else ""
        )
        result["status"] = "refused"
        result["refusal"] = {
            "reason": "uniform_checkpoint_scores",
            "detail": (
                f"all {len(scored)} scored checkpoints have reward "
                f"{only_reward:g}; a uniform checkpoint series cannot distinguish "
                "genuinely stalled work from a verifier that is insensitive to "
                f"intermediate state{final_note}"
            ),
        }
        return result

    result["segments"] = _segments(checkpoints)
    return result


def assemble_localization_report(
    scored_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Localize every task in one checkpoint-scoring report with coverage."""

    if not isinstance(scored_report, Mapping):
        raise ValueError("checkpoint score report must be an object")
    if (
        scored_report.get("schema_version") != 1
        or scored_report.get("mode") != "checkpoint-scoring"
    ):
        raise ValueError("input is not a schema-version-1 checkpoint-scoring report")
    raw_trials = scored_report.get("trials")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise ValueError("checkpoint score report trials must be a non-empty list")

    tasks = [localize_scored_timeline(trial) for trial in raw_trials]
    task_names = [task["task_name"] for task in tasks]
    if len(set(task_names)) != len(task_names):
        raise ValueError(
            "checkpoint score report has multiple timelines for one task; "
            "task-level coverage would be ambiguous"
        )
    usable = sum(task["status"] == "usable" for task in tasks)
    reason_counts = Counter(
        task["refusal"]["reason"] for task in tasks if task["status"] == "refused"
    )
    task_count = len(tasks)
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": "checkpoint-localization",
        "task_count": task_count,
        "usable_task_count": usable,
        "refused_task_count": task_count - usable,
        "coverage": usable / task_count,
        "refusal_reason_counts": dict(sorted(reason_counts.items())),
        "segment_count": sum(len(task["segments"]) for task in tasks),
        "tasks": tasks,
    }
    source_job_dir = scored_report.get("source_job_dir")
    if isinstance(source_job_dir, str):
        report["source_job_dir"] = source_job_dir
    return report


def load_and_localize_score_report(path: Path | str) -> dict[str, Any]:
    """Read a current scorer report or the older round-five probe format."""

    score_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(score_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"checkpoint score report does not exist: {score_path}"
        ) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"checkpoint score report is invalid JSON: {score_path}"
        ) from error
    if isinstance(data, list):
        data = _legacy_probe_report(data, score_path)
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint score report must be an object: {score_path}")
    return assemble_localization_report(data)


def write_localization_report(
    output_path: Path | str, report: Mapping[str, Any]
) -> None:
    """Atomically write a localization report."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _segments(checkpoints: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[tuple[str, int, int]] = []
    for end_index in range(1, len(checkpoints)):
        start_index = end_index - 1
        start = checkpoints[start_index]
        end = checkpoints[end_index]
        # An unscored point breaks adjacency.  Comparing across it would assign
        # work to an interval whose intermediate verifier observation is unknown.
        if start["reward"] is None or end["reward"] is None:
            continue
        reward_change = end["reward"] - start["reward"]
        if abs(reward_change) <= _SCORE_ABS_TOLERANCE:
            gaps.append(("flat", start_index, end_index))
        elif reward_change < 0.0:
            gaps.append(("regression", start_index, end_index))

    grouped: list[tuple[str, int, int]] = []
    for kind, start_index, end_index in gaps:
        if grouped and grouped[-1][0] == kind and grouped[-1][2] == start_index:
            previous_kind, previous_start, _ = grouped[-1]
            grouped[-1] = (previous_kind, previous_start, end_index)
        else:
            grouped.append((kind, start_index, end_index))

    return [
        _segment(kind, start_index, end_index, checkpoints)
        for kind, start_index, end_index in grouped
    ]


def _segment(
    kind: str,
    start_index: int,
    end_index: int,
    checkpoints: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    start = checkpoints[start_index]
    end = checkpoints[end_index]
    return {
        "type": kind,
        "start": _point_reference(start),
        "end": _point_reference(end),
        "steps": [point["step"] for point in checkpoints[start_index : end_index + 1]],
        "checkpoint_count": end_index - start_index + 1,
        "gap_count": end_index - start_index,
        "start_reward": start["reward"],
        "end_reward": end["reward"],
        "reward_change": end["reward"] - start["reward"],
    }


def _point_reference(point: Mapping[str, Any]) -> dict[str, Any]:
    reference = {
        "index": point["index"],
        "phase": point["phase"],
        "step": point["step"],
    }
    for key in ("candidate_id", "checkpoint_id"):
        if key in point:
            reference[key] = point[key]
    return reference


def _checkpoint(item: Any, index: int, trial_name: str) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(
            f"timeline {trial_name!r} checkpoint {index} must be an object"
        )
    step = item.get("step")
    phase = item.get("phase", 0)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(
            f"timeline {trial_name!r} checkpoint {index} step must be non-negative"
        )
    if isinstance(phase, bool) or not isinstance(phase, int) or phase < 0:
        raise ValueError(
            f"timeline {trial_name!r} checkpoint {index} phase must be non-negative"
        )
    point: dict[str, Any] = {
        "index": index,
        "phase": phase,
        "step": step,
        "reward": _optional_reward(
            item.get("reward"), f"timeline {trial_name!r} checkpoint {index}"
        ),
    }
    for key in ("candidate_id", "checkpoint_id"):
        value = item.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"timeline {trial_name!r} checkpoint {index} {key} must be a string"
                )
            point[key] = value
    return point


def _validate_order(checkpoints: Sequence[Mapping[str, Any]], trial_name: str) -> None:
    for previous, current in pairwise(checkpoints):
        previous_position = (previous["phase"], previous["step"])
        current_position = (current["phase"], current["step"])
        if current_position <= previous_position:
            raise ValueError(
                f"timeline {trial_name!r} checkpoints are not strictly ordered by "
                "phase and step"
            )


def _optional_reward(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} reward must be numeric or null")
    reward = float(value)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"{context} reward is not in [0, 1]")
    return reward


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{context} {key} must be a non-empty string")
    return item


def _legacy_probe_report(records: list[Any], path: Path) -> dict[str, Any]:
    """Normalize the pre-schema round-five probe for reproducibility checks."""

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    final_rewards: dict[str, float | None] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"legacy score record {index} must be an object: {path}")
        task = _required_string(record, "task", f"legacy score record {index}")
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(
                f"legacy score record {index} step must be non-negative: {path}"
            )
        reward = _optional_reward(
            record.get("checkpoint_reward"), f"legacy score record {index} checkpoint"
        )
        final_reward = _optional_reward(
            record.get("final_reward"), f"legacy score record {index} final"
        )
        if task in final_rewards and final_rewards[task] != final_reward:
            raise ValueError(
                f"legacy score records disagree on final reward for {task!r}"
            )
        final_rewards[task] = final_reward
        grouped[task].append({"phase": 0, "step": step, "reward": reward})
    if not grouped:
        raise ValueError(f"legacy checkpoint score report is empty: {path}")

    trials = []
    for task, checkpoints in sorted(grouped.items()):
        checkpoints.sort(key=lambda point: point["step"])
        trials.append(
            {
                "trial_name": task,
                "task_name": task,
                "final_reward": final_rewards[task],
                "checkpoints": checkpoints,
            }
        )
    return {
        "schema_version": 1,
        "mode": "checkpoint-scoring",
        "trials": trials,
    }
