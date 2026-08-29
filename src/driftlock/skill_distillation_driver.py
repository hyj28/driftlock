"""Resumable, dual-arm driving for checkpoint-localized skill distillation.

This module owns no provider transport or credentials.  It consumes an injected
distiller and a monotonic usage reader, writes one completed attempt at a time,
and emits generated candidates in the schema consumed by skill admission.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from driftlock.skill_distillation import (
    SkillDistillationResult,
    SkillDistillationStatus,
    assemble_baseline_evidence,
    assemble_localized_evidence,
    parse_skill,
    serialize_skill,
)
from driftlock.usage import ReplayUsage

DISTILLATION_REPORT_NAME = "skill-candidates.json"
DISTILLATION_ARMS = ("localized", "baseline")


class SkillDistiller(Protocol):
    """The provider-neutral boundary needed by the driver."""

    async def distill(self, evidence: Mapping[str, Any]) -> SkillDistillationResult: ...


@dataclass(frozen=True, slots=True)
class DistillationWorkItem:
    """One paid arm/segment attempt with already-assembled evidence."""

    candidate_id: str
    task_name: str
    trial_name: str
    segment_index: int
    arm: str
    evidence: Mapping[str, Any]

    def identity(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "task_name": self.task_name,
            "trial_name": self.trial_name,
            "segment_index": self.segment_index,
            "arm": self.arm,
        }


@dataclass(frozen=True, slots=True)
class DistillationPlan:
    """Validated free work that precedes any provider call."""

    source_report: Path
    source_report_sha256: str
    source_job_dir: Path
    localized_segment_count: int
    work_items: tuple[DistillationWorkItem, ...]
    refused_tasks: tuple[dict[str, Any], ...]
    evidence_refusals: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_localization_report": str(self.source_report),
            "source_localization_sha256": self.source_report_sha256,
            "source_job_dir": str(self.source_job_dir),
            "localized_segment_count": self.localized_segment_count,
            "callable_segment_count": len(self.work_items) // len(DISTILLATION_ARMS),
            "evidence_refusal_count": len(self.evidence_refusals),
            "evidence_refusals": list(self.evidence_refusals),
            "call_count": len(self.work_items),
            "work_items": [item.identity() for item in self.work_items],
        }


def plan_skill_distillation(localization_report: Path | str) -> DistillationPlan:
    """Validate a localization report and assemble both arms without a model."""

    report_path = Path(localization_report).expanduser().resolve()
    data, serialized = _read_localization_report(report_path)
    source_job_dir = _source_job_directory(data)
    tasks = data["tasks"]
    refused_tasks: list[dict[str, Any]] = []
    evidence_refusals: list[dict[str, Any]] = []
    work_items: list[DistillationWorkItem] = []
    seen_task_names: set[str] = set()
    localized_segment_count = 0

    for task_index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ValueError(f"localization task {task_index} must be an object")
        task_name = _required_text(task, "task_name", f"localization task {task_index}")
        if task_name in seen_task_names:
            raise ValueError(
                f"localization report has multiple records for task {task_name!r}"
            )
        seen_task_names.add(task_name)
        trial_name = _required_text(task, "trial_name", f"task {task_name!r}")
        status = task.get("status")
        if status == "refused":
            refusal = _refusal(task.get("refusal"), f"task {task_name!r}")
            refused_tasks.append(
                {
                    "task_name": task_name,
                    "trial_name": trial_name,
                    "status": "refused",
                    "refusal": refusal,
                }
            )
            continue
        if status != "usable":
            raise ValueError(f"task {task_name!r} status must be 'usable' or 'refused'")
        segments = task.get("segments")
        if not isinstance(segments, list):
            raise ValueError(f"usable task {task_name!r} segments must be a list")
        localized_segment_count += len(segments)
        trial_dir = _trial_directory(source_job_dir, trial_name)
        for segment_index, segment in enumerate(segments):
            if not isinstance(segment, Mapping):
                raise ValueError(
                    f"task {task_name!r} segment {segment_index} must be an object"
                )
            evidence_by_arm = {
                "localized": assemble_localized_evidence(trial_dir, segment),
                "baseline": assemble_baseline_evidence(trial_dir),
            }
            refused = {
                arm: _refusal(evidence.get("refusal"), f"{arm} evidence")
                for arm, evidence in evidence_by_arm.items()
                if evidence.get("status") == "refused"
            }
            malformed = [
                arm
                for arm, evidence in evidence_by_arm.items()
                if evidence.get("status") not in {"usable", "refused"}
            ]
            if malformed:
                raise ValueError(
                    f"task {task_name!r} segment {segment_index} evidence has "
                    f"invalid status for: {', '.join(malformed)}"
                )
            if refused:
                evidence_refusals.append(
                    {
                        "task_name": task_name,
                        "trial_name": trial_name,
                        "segment_index": segment_index,
                        "status": "refused",
                        "refusals_by_arm": refused,
                        "detail": (
                            "both arms were skipped so they remain paired over the "
                            "same localized segments"
                        ),
                    }
                )
                continue
            for arm in DISTILLATION_ARMS:
                work_items.append(
                    DistillationWorkItem(
                        candidate_id=_candidate_id(
                            task_name, trial_name, segment_index, arm, segment
                        ),
                        task_name=task_name,
                        trial_name=trial_name,
                        segment_index=segment_index,
                        arm=arm,
                        evidence=evidence_by_arm[arm],
                    )
                )

    candidate_ids = [item.candidate_id for item in work_items]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("distillation candidate ids are not unique")
    return DistillationPlan(
        source_report=report_path,
        source_report_sha256=hashlib.sha256(serialized).hexdigest(),
        source_job_dir=source_job_dir,
        localized_segment_count=localized_segment_count,
        work_items=tuple(work_items),
        refused_tasks=tuple(refused_tasks),
        evidence_refusals=tuple(evidence_refusals),
    )


async def run_skill_distillation(
    plan: DistillationPlan,
    output_path: Path | str,
    *,
    distiller: SkillDistiller,
    usage_reader: Callable[[], ReplayUsage],
    run_metadata: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run unpaid work sequentially and atomically checkpoint every attempt."""

    destination = Path(output_path).expanduser().resolve()
    previous = _load_previous(destination, plan, run_metadata=run_metadata)
    attempts = [dict(attempt) for attempt in previous]
    latest_by_id = _latest_attempts(attempts)
    reused_ids = {
        candidate_id
        for candidate_id, attempt in latest_by_id.items()
        if attempt["status"] != SkillDistillationStatus.FAILED.value
    }
    for attempt in attempts:
        attempt["reused"] = (
            attempt is latest_by_id[attempt["candidate_id"]]
            and attempt["candidate_id"] in reused_ids
        )
    previous_attempt_count = len(attempts)

    if dry_run:
        return _report(
            plan,
            attempts,
            reused_ids=reused_ids,
            previous_attempt_count=previous_attempt_count,
            run_metadata=run_metadata,
            dry_run=True,
        )

    _write_report(
        destination,
        _report(
            plan,
            attempts,
            reused_ids=reused_ids,
            previous_attempt_count=previous_attempt_count,
            run_metadata=run_metadata,
            dry_run=False,
        ),
    )
    for item in plan.work_items:
        if item.candidate_id in reused_ids:
            continue
        before = _usage_snapshot(usage_reader)
        try:
            result = await distiller.distill(item.evidence)
            if not isinstance(result, SkillDistillationResult):
                raise TypeError("distiller must return SkillDistillationResult")
        except Exception as error:
            result = SkillDistillationResult(
                status=SkillDistillationStatus.FAILED,
                reason=f"skill distiller call failed: {error}",
            )
        after = _usage_snapshot(usage_reader)
        usage = _usage_delta(after, before)
        attempt = {
            **item.identity(),
            "attempt_number": 1
            + sum(
                previous["candidate_id"] == item.candidate_id for previous in attempts
            ),
            "status": result.status.value,
            "reason": result.reason,
            "tokens": result.tokens,
            "usage": usage.as_dict(),
            "reused": False,
        }
        if result.skill is not None:
            attempt["skill"] = serialize_skill(result.skill)
        attempts.append(attempt)
        latest_by_id[item.candidate_id] = attempt
        _write_report(
            destination,
            _report(
                plan,
                attempts,
                reused_ids=reused_ids,
                previous_attempt_count=previous_attempt_count,
                run_metadata=run_metadata,
                dry_run=False,
            ),
        )
    return _report(
        plan,
        attempts,
        reused_ids=reused_ids,
        previous_attempt_count=previous_attempt_count,
        run_metadata=run_metadata,
        dry_run=False,
    )


def _read_localization_report(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        serialized = path.read_bytes()
        data = json.loads(serialized)
    except FileNotFoundError:
        raise FileNotFoundError(f"localization report does not exist: {path}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"localization report is invalid JSON: {path}") from error
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or data.get("mode") != "checkpoint-localization"
    ):
        raise ValueError(
            "distillation input must be a schema-version-1 checkpoint-localization "
            "report"
        )
    if not isinstance(data.get("tasks"), list) or not data["tasks"]:
        raise ValueError("localization report tasks must be a non-empty list")
    return data, serialized


def _source_job_directory(report: Mapping[str, Any]) -> Path:
    raw = report.get("source_job_dir")
    if not isinstance(raw, str) or not raw:
        raise ValueError("localization report has no source_job_dir")
    source = Path(raw).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source job directory does not exist: {source}")
    return source


def _trial_directory(source: Path, trial_name: str) -> Path:
    trial = (source / trial_name).resolve()
    if trial.parent != source or not trial.is_dir() or trial.is_symlink():
        raise ValueError(
            f"localization trial {trial_name!r} is not a direct source-job directory"
        )
    return trial


def _required_text(value: Mapping[str, Any], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{context} {key} must be a non-empty string")
    return item


def _refusal(value: Any, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} refusal must be an object")
    return {
        "reason": _required_text(value, "reason", f"{context} refusal"),
        "detail": _required_text(value, "detail", f"{context} refusal"),
    }


def _candidate_id(
    task_name: str,
    trial_name: str,
    segment_index: int,
    arm: str,
    segment: Mapping[str, Any],
) -> str:
    identity = json.dumps(
        [task_name, trial_name, segment_index, arm, segment],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "skill-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _load_previous(
    path: Path,
    plan: DistillationPlan,
    *,
    run_metadata: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"existing distillation output is invalid JSON: {path}"
        ) from error
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"existing distillation output has an invalid schema: {path}")
    if data.get("mode") != "skill-distillation" or data.get("plan") != plan.as_dict():
        raise ValueError(f"existing distillation output is for a different run: {path}")
    if run_metadata is not None and data.get("model") != dict(run_metadata):
        raise ValueError(
            f"existing distillation output used different model settings: {path}"
        )
    raw_attempts = data.get("attempts")
    if not isinstance(raw_attempts, list):
        raise ValueError(f"existing distillation output has invalid attempts: {path}")
    expected = {item.candidate_id: item for item in plan.work_items}
    attempts: list[dict[str, Any]] = []
    attempt_counts: Counter[str] = Counter()
    terminal_candidates: set[str] = set()
    for raw in raw_attempts:
        if not isinstance(raw, dict):
            raise ValueError(
                f"existing distillation output has invalid attempt: {path}"
            )
        candidate_id = raw.get("candidate_id")
        item = expected.get(candidate_id)
        if item is None:
            raise ValueError(f"existing distillation output has unknown work: {path}")
        if candidate_id in terminal_candidates:
            raise ValueError(
                f"existing distillation output retries terminal work: {path}"
            )
        if any(raw.get(key) != value for key, value in item.identity().items()):
            raise ValueError(
                f"existing distillation output work identity changed: {path}"
            )
        try:
            status = SkillDistillationStatus(raw.get("status"))
            ReplayUsage.from_mapping(raw.get("usage"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"existing distillation output has invalid completed work: {path}"
            ) from error
        if status is SkillDistillationStatus.GENERATED:
            if not isinstance(raw.get("skill"), str):
                raise ValueError(
                    f"existing generated distillation has no skill document: {path}"
                )
            try:
                parse_skill(raw["skill"])
            except ValueError as error:
                raise ValueError(
                    f"existing generated distillation has an invalid skill: {path}"
                ) from error
        if status is not SkillDistillationStatus.GENERATED and "skill" in raw:
            raise ValueError(
                f"existing non-generated distillation carries a skill: {path}"
            )
        reason = raw.get("reason")
        tokens = raw.get("tokens")
        if (
            not isinstance(reason, str)
            or not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or tokens < 0
        ):
            raise ValueError(
                f"existing distillation output has invalid result details: {path}"
            )
        attempt_counts[candidate_id] += 1
        attempt_number = raw.get("attempt_number", attempt_counts[candidate_id])
        if attempt_number != attempt_counts[candidate_id]:
            raise ValueError(
                f"existing distillation output has invalid attempt order: {path}"
            )
        normalized = dict(raw)
        normalized["attempt_number"] = attempt_number
        attempts.append(normalized)
        if status is not SkillDistillationStatus.FAILED:
            terminal_candidates.add(candidate_id)
    return attempts


def _usage_snapshot(reader: Callable[[], ReplayUsage]) -> ReplayUsage:
    usage = reader()
    if not isinstance(usage, ReplayUsage):
        raise TypeError("usage reader must return ReplayUsage")
    return usage


def _usage_delta(after: ReplayUsage, before: ReplayUsage) -> ReplayUsage:
    values = {
        "input_tokens": after.input_tokens - before.input_tokens,
        "cache_tokens": after.cache_tokens - before.cache_tokens,
        "output_tokens": after.output_tokens - before.output_tokens,
        "cost_usd": after.cost_usd - before.cost_usd,
    }
    if any(value < 0 for value in values.values()):
        raise RuntimeError("distillation usage accounting moved backwards")
    return ReplayUsage(**values)


def _sum_usage(attempts: Sequence[Mapping[str, Any]]) -> ReplayUsage:
    totals: dict[str, int | float] = {
        "input_tokens": 0,
        "cache_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    for attempt in attempts:
        usage = ReplayUsage.from_mapping(attempt.get("usage"))
        totals["input_tokens"] += usage.input_tokens
        totals["cache_tokens"] += usage.cache_tokens
        totals["output_tokens"] += usage.output_tokens
        totals["cost_usd"] += usage.cost_usd
    return ReplayUsage(**totals)


def _latest_attempts(
    attempts: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        latest[attempt["candidate_id"]] = attempt
    return latest


def _report(
    plan: DistillationPlan,
    attempts: Sequence[dict[str, Any]],
    *,
    reused_ids: set[str],
    previous_attempt_count: int,
    run_metadata: Mapping[str, Any] | None,
    dry_run: bool,
) -> dict[str, Any]:
    attempts_by_id: dict[str, list[dict[str, Any]]] = {
        item.candidate_id: [] for item in plan.work_items
    }
    for attempt in attempts:
        attempts_by_id[attempt["candidate_id"]].append(attempt)
    ordered_attempts = [
        attempt
        for item in plan.work_items
        for attempt in attempts_by_id[item.candidate_id]
    ]
    latest_by_id = _latest_attempts(ordered_attempts)
    candidates: list[dict[str, Any]] = []
    paired_candidate_segment_count = 0
    unpaired_segment_count = 0
    for item_index in range(0, len(plan.work_items), len(DISTILLATION_ARMS)):
        segment_items = plan.work_items[
            item_index : item_index + len(DISTILLATION_ARMS)
        ]
        segment_attempts = [
            latest_by_id.get(item.candidate_id) for item in segment_items
        ]
        generated = [
            attempt is not None
            and attempt["status"] == SkillDistillationStatus.GENERATED.value
            for attempt in segment_attempts
        ]
        if all(generated):
            paired_candidate_segment_count += 1
            for attempt in segment_attempts:
                assert attempt is not None
                candidates.append(
                    {
                        "candidate_id": attempt["candidate_id"],
                        "arm": attempt["arm"],
                        "skill": attempt["skill"],
                        "paired_deltas": [],
                        "task_name": attempt["task_name"],
                        "trial_name": attempt["trial_name"],
                        "segment_index": attempt["segment_index"],
                        "usage": attempt["usage"],
                    }
                )
        elif any(generated):
            unpaired_segment_count += 1
    usage_by_arm = {
        arm: _sum_usage(
            [attempt for attempt in ordered_attempts if attempt["arm"] == arm]
        ).as_dict()
        for arm in DISTILLATION_ARMS
    }
    status_counts = Counter(attempt["status"] for attempt in ordered_attempts)
    retryable_failure_count = sum(
        attempt["status"] == SkillDistillationStatus.FAILED.value
        for attempt in latest_by_id.values()
    )
    pending = sum(
        item.candidate_id not in latest_by_id
        or latest_by_id[item.candidate_id]["status"]
        == SkillDistillationStatus.FAILED.value
        for item in plan.work_items
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": "skill-distillation",
        "dry_run": dry_run,
        "plan": plan.as_dict(),
        "summary": {
            "planned_call_count": len(plan.work_items),
            "completed_call_count": len(ordered_attempts),
            "pending_call_count": pending,
            "reused_call_count": len(reused_ids),
            "new_call_count": len(ordered_attempts) - previous_attempt_count,
            "candidate_count": len(candidates),
            "paired_candidate_segment_count": paired_candidate_segment_count,
            "unpaired_segment_count": unpaired_segment_count,
            "retryable_failure_count": retryable_failure_count,
            "refused_task_count": len(plan.refused_tasks),
            "evidence_refusal_count": len(plan.evidence_refusals),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "usage_by_arm": usage_by_arm,
        "attempts": ordered_attempts,
        "refused_tasks": list(plan.refused_tasks),
        "evidence_refusals": list(plan.evidence_refusals),
        "candidates": candidates,
    }
    if run_metadata is not None:
        report["model"] = dict(run_metadata)
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
