"""Resumable paired validation of distilled skill candidates.

The driver is provider-neutral.  It owns planning, one-candidate library
isolation, trial-granularity checkpoints, and delta assembly; the CLI injects a
Harbor-backed runner while tests inject a deterministic offline runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from driftlock.skill_admission import (
    DISTILLATION_ARMS,
    VALIDATION_TASK_COUNT,
    SkillLibrary,
)
from driftlock.skill_distillation import parse_skill, serialize_skill

VALIDATION_REPORT_NAME = "validated-skill-candidates.json"
VALIDATION_MODE = "skill-paired-validation"
VALIDATION_PROCEDURE_ID = "shared-single-candidate-paired-v1"

# The most recent paid LHTB measurement supplied for this validation round was
# $0.151 per trial (2026-08-30).  This is an operator-facing budget estimate,
# not a spending limit; expose it as a CLI override when provider pricing changes.
DEFAULT_ESTIMATED_COST_PER_TRIAL_USD = 0.151

# Retrieval fans one result out to both distillation-arm labels, and an empty
# library invokes no embedder and preserves request bytes.  A shared control can
# therefore use either label without changing the no-skill procedure; freezing
# one label makes its config and audit deterministic across every candidate.
CONTROL_DISTILLATION_ARM = "localized"

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ValidationTrialStatus(StrEnum):
    """Whether one paid trial produced a usable reward measurement."""

    MEASURED = "measured"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ValidationCandidate:
    """The immutable candidate identity used while validation is resumed."""

    candidate_id: str
    arm: str
    skill_document: str

    @property
    def skill_sha256(self) -> str:
        return hashlib.sha256(self.skill_document.encode()).hexdigest()

    def identity(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "arm": self.arm,
            "skill_sha256": self.skill_sha256,
        }


@dataclass(frozen=True, slots=True)
class ValidationTrial:
    """One control or treatment trial passed through the shared runner seam."""

    trial_id: str
    task_name: str
    condition: str
    distillation_arm: str
    library_dir: Path
    candidate_id: str | None = None
    expected_injected_candidate_ids: tuple[str, ...] = ()
    procedure_id: str = VALIDATION_PROCEDURE_ID
    attempt_number: int = 1

    @property
    def job_name(self) -> str:
        return f"skill-validation-{self.trial_id}-a{self.attempt_number}"

    def identity(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "task_name": self.task_name,
            "condition": self.condition,
            "candidate_id": self.candidate_id,
            "distillation_arm": self.distillation_arm,
            "expected_injected_candidate_ids": list(
                self.expected_injected_candidate_ids
            ),
            "procedure_id": self.procedure_id,
        }


@dataclass(frozen=True, slots=True)
class ValidationTrialResult:
    """Provider-neutral result returned by a validation trial runner."""

    status: ValidationTrialStatus
    reward: float | None = None
    reason: str = ""
    injected_candidate_ids: tuple[str, ...] | None = None
    audit: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status is ValidationTrialStatus.MEASURED:
            if (
                isinstance(self.reward, bool)
                or not isinstance(self.reward, (int, float))
                or not math.isfinite(float(self.reward))
                or not 0.0 <= float(self.reward) <= 1.0
            ):
                raise ValueError("a measured validation trial needs a reward in [0, 1]")
            if self.injected_candidate_ids is None:
                raise ValueError(
                    "a measured validation trial needs injection provenance"
                )
            object.__setattr__(self, "reward", float(self.reward))
        elif self.reward is not None:
            raise ValueError("a failed validation trial cannot carry a reward")
        if not isinstance(self.reason, str):
            raise TypeError("validation trial reason must be text")
        if self.audit is not None and not isinstance(self.audit, Mapping):
            raise TypeError("validation trial audit must be an object")


class SkillValidationTrialRunner(Protocol):
    """The only paid boundary required by the validation driver."""

    async def run(self, trial: ValidationTrial) -> ValidationTrialResult: ...


@dataclass(frozen=True, slots=True)
class SkillValidationPlan:
    """Validated candidates, held-out tasks, and the exact paid-work plan."""

    source_candidate_file: Path
    source_payload: Mapping[str, Any]
    candidates: tuple[ValidationCandidate, ...]
    tasks: tuple[str, ...]
    estimated_cost_per_trial_usd: float

    @property
    def planned_trial_count(self) -> int:
        return len(self.tasks) * (len(self.candidates) + 1)

    @property
    def estimated_cost_usd(self) -> float:
        return self.planned_trial_count * self.estimated_cost_per_trial_usd

    def as_dict(self) -> dict[str, Any]:
        return {
            "procedure_id": VALIDATION_PROCEDURE_ID,
            "source_candidate_file": str(self.source_candidate_file),
            "candidate_identities": [
                candidate.identity() for candidate in self.candidates
            ],
            "validation_tasks": list(self.tasks),
            "shared_control_trial_count": len(self.tasks),
            "treatment_trial_count": len(self.tasks) * len(self.candidates),
            "planned_trial_count": self.planned_trial_count,
            "estimated_cost_per_trial_usd": self.estimated_cost_per_trial_usd,
            "estimated_cost_usd": self.estimated_cost_usd,
            "work_items": [item.identity() for item in self.work_items(Path("."))],
        }

    def work_items(self, work_dir: Path) -> tuple[ValidationTrial, ...]:
        libraries = work_dir / "libraries"
        controls = tuple(
            _trial(
                task_name=task,
                condition="without_skill",
                candidate=None,
                library_dir=libraries / "empty",
            )
            for task in self.tasks
        )
        treatments = tuple(
            _trial(
                task_name=task,
                condition="with_skill",
                candidate=candidate,
                library_dir=libraries / candidate.candidate_id,
            )
            for candidate in self.candidates
            for task in self.tasks
        )
        return controls + treatments


def plan_skill_validation(
    candidate_file: Path | str,
    validation_tasks_file: Path | str,
    *,
    estimated_cost_per_trial_usd: float = DEFAULT_ESTIMATED_COST_PER_TRIAL_USD,
) -> SkillValidationPlan:
    """Load an admission-ready candidate cohort and a caller-chosen task list."""

    if (
        isinstance(estimated_cost_per_trial_usd, bool)
        or not isinstance(estimated_cost_per_trial_usd, (int, float))
        or not math.isfinite(float(estimated_cost_per_trial_usd))
        or estimated_cost_per_trial_usd < 0
    ):
        raise ValueError("estimated cost per trial must be finite and nonnegative")
    source = Path(candidate_file).expanduser().resolve()
    payload = _read_object(source, "skill candidate input")
    if payload.get("schema_version") != 1:
        raise ValueError("skill candidate input must be a schema-version-1 object")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("skill candidate input candidates must be a non-empty list")

    candidates: list[ValidationCandidate] = []
    candidate_ids: set[str] = set()
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise ValueError(f"skill candidate {index} must be an object")
        candidate_id = raw.get("candidate_id")
        arm = raw.get("arm")
        document = raw.get("skill")
        deltas = raw.get("paired_deltas")
        if (
            not isinstance(candidate_id, str)
            or _SAFE_NAME.fullmatch(candidate_id) is None
        ):
            raise ValueError(f"skill candidate {index} has an unsafe candidate id")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate skill candidate id: {candidate_id}")
        if arm not in DISTILLATION_ARMS:
            raise ValueError(f"skill candidate {candidate_id!r} has an unknown arm")
        if not isinstance(document, str):
            raise ValueError(f"skill candidate {candidate_id!r} needs skill text")
        skill = parse_skill(document)
        canonical_document = serialize_skill(skill)
        if not isinstance(deltas, list):
            raise ValueError(
                f"skill candidate {candidate_id!r} paired_deltas must be a list"
            )
        candidate_ids.add(candidate_id)
        candidates.append(ValidationCandidate(candidate_id, arm, canonical_document))

    tasks_path = Path(validation_tasks_file).expanduser().resolve()
    tasks_payload = _read_json(tasks_path, "validation task list")
    if isinstance(tasks_payload, list):
        raw_tasks = tasks_payload
    elif isinstance(tasks_payload, Mapping):
        raw_tasks = tasks_payload.get("selected_tasks")
    else:
        raw_tasks = None
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(
            "validation task list must be a non-empty array or an object with "
            "selected_tasks"
        )
    tasks: list[str] = []
    for index, task in enumerate(raw_tasks):
        if not isinstance(task, str) or _SAFE_NAME.fullmatch(task) is None:
            raise ValueError(f"validation task {index} has an unsafe name")
        if task in tasks:
            raise ValueError(f"duplicate validation task: {task}")
        tasks.append(task)
    if len(tasks) > VALIDATION_TASK_COUNT:
        raise ValueError(
            f"validation task list has {len(tasks)} tasks; admission accepts at "
            f"most {VALIDATION_TASK_COUNT} paired deltas"
        )

    return SkillValidationPlan(
        source_candidate_file=source,
        source_payload=payload,
        candidates=tuple(candidates),
        tasks=tuple(tasks),
        estimated_cost_per_trial_usd=float(estimated_cost_per_trial_usd),
    )


async def run_skill_validation(
    plan: SkillValidationPlan,
    output_path: Path | str,
    *,
    runner: SkillValidationTrialRunner,
    work_dir: Path | str,
    run_metadata: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run pending trials sequentially and checkpoint after every attempt."""

    destination = Path(output_path).expanduser().resolve()
    workspace = Path(work_dir).expanduser().resolve()
    trials = plan.work_items(workspace)
    previous = _load_previous(destination, plan, trials, run_metadata=run_metadata)
    attempts = [dict(attempt) for attempt in previous]
    latest = _latest_attempts(attempts)
    reused_ids = {
        trial_id
        for trial_id, attempt in latest.items()
        if attempt["status"] == ValidationTrialStatus.MEASURED.value
    }
    for attempt in attempts:
        attempt["reused"] = (
            attempt is latest[attempt["trial_id"]] and attempt["trial_id"] in reused_ids
        )
    previous_attempt_count = len(attempts)

    if dry_run:
        return _report(
            plan,
            trials,
            attempts,
            reused_ids=reused_ids,
            previous_attempt_count=previous_attempt_count,
            run_metadata=run_metadata,
            dry_run=True,
        )

    _stage_libraries(plan, workspace)
    _write_report(
        destination,
        _report(
            plan,
            trials,
            attempts,
            reused_ids=reused_ids,
            previous_attempt_count=previous_attempt_count,
            run_metadata=run_metadata,
            dry_run=False,
        ),
    )
    for trial in trials:
        if trial.trial_id in reused_ids:
            continue
        attempt_number = 1 + sum(
            item["trial_id"] == trial.trial_id for item in attempts
        )
        attempt_trial = replace(trial, attempt_number=attempt_number)
        try:
            result = await runner.run(attempt_trial)
            if not isinstance(result, ValidationTrialResult):
                raise TypeError("validation runner must return ValidationTrialResult")
        except Exception as error:
            result = ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason=(
                    f"validation trial runner failed: {type(error).__name__}: {error}"
                ),
            )
        if (
            result.status is ValidationTrialStatus.MEASURED
            and result.injected_candidate_ids != trial.expected_injected_candidate_ids
        ):
            result = ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason=(
                    "skill injection mismatch: expected "
                    f"{list(trial.expected_injected_candidate_ids)!r}, observed "
                    f"{list(result.injected_candidate_ids or ())!r}"
                ),
                audit=result.audit,
            )
        attempt = {
            **trial.identity(),
            "attempt_number": attempt_number,
            "status": result.status.value,
            "reason": result.reason,
            "reward": result.reward,
            "injected_candidate_ids": (
                list(result.injected_candidate_ids)
                if result.injected_candidate_ids is not None
                else None
            ),
            "audit": dict(result.audit or {}),
            "reused": False,
        }
        attempts.append(attempt)
        _write_report(
            destination,
            _report(
                plan,
                trials,
                attempts,
                reused_ids=reused_ids,
                previous_attempt_count=previous_attempt_count,
                run_metadata=run_metadata,
                dry_run=False,
            ),
        )
    return _report(
        plan,
        trials,
        attempts,
        reused_ids=reused_ids,
        previous_attempt_count=previous_attempt_count,
        run_metadata=run_metadata,
        dry_run=False,
    )


def _trial(
    *,
    task_name: str,
    condition: str,
    candidate: ValidationCandidate | None,
    library_dir: Path,
) -> ValidationTrial:
    candidate_id = candidate.candidate_id if candidate is not None else None
    identity = json.dumps(
        [VALIDATION_PROCEDURE_ID, condition, candidate_id, task_name],
        separators=(",", ":"),
    )
    trial_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return ValidationTrial(
        trial_id=trial_id,
        task_name=task_name,
        condition=condition,
        candidate_id=candidate_id,
        distillation_arm=(
            candidate.arm if candidate is not None else CONTROL_DISTILLATION_ARM
        ),
        library_dir=library_dir,
        expected_injected_candidate_ids=(
            (candidate.candidate_id,) if candidate is not None else ()
        ),
    )


def _stage_libraries(plan: SkillValidationPlan, workspace: Path) -> None:
    libraries = workspace / "libraries"
    empty = SkillLibrary(libraries / "empty")
    if empty.candidate_ids():
        raise ValueError("shared no-skill validation library is not empty")
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    for candidate in plan.candidates:
        library = SkillLibrary(libraries / candidate.candidate_id)
        ids = library.candidate_ids()
        if ids:
            if ids != (candidate.candidate_id,):
                raise ValueError(
                    f"validation library for {candidate.candidate_id!r} is not isolated"
                )
            if serialize_skill(library.read_skill(candidate.candidate_id)) != (
                candidate.skill_document
            ):
                raise ValueError(
                    f"validation library for {candidate.candidate_id!r} changed skill"
                )
            continue
        entry = library.entries / candidate.candidate_id
        temporary = library.entries / f".tmp-{candidate.candidate_id}"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        (temporary / "skill.md").write_text(
            candidate.skill_document + "\n", encoding="utf-8"
        )
        (temporary / "decision.json").write_text(
            json.dumps(
                {
                    "candidate_id": candidate.candidate_id,
                    "arm": candidate.arm,
                    "status": "admitted",
                    "purpose": "isolated_validation_treatment_staging",
                    "skill_sha256": candidate.skill_sha256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(entry)
    if set(by_id) != {
        path.name
        for path in libraries.iterdir()
        if path.is_dir() and path.name != "empty" and not path.name.startswith(".tmp-")
    }:
        raise ValueError("validation workspace contains an unexpected skill library")


def _load_previous(
    path: Path,
    plan: SkillValidationPlan,
    trials: Sequence[ValidationTrial],
    *,
    run_metadata: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = _read_object(path, "existing validation output")
    validation = data.get("validation")
    if validation is None and path == plan.source_candidate_file:
        return []
    if not isinstance(validation, Mapping):
        raise ValueError(f"existing validation output has no validation state: {path}")
    if (
        validation.get("schema_version") != 1
        or validation.get("mode") != VALIDATION_MODE
        or validation.get("plan") != plan.as_dict()
    ):
        raise ValueError(f"existing validation output is for a different run: {path}")
    if run_metadata is not None and validation.get("run") != dict(run_metadata):
        raise ValueError(
            f"existing validation output used different run settings: {path}"
        )
    raw_attempts = validation.get("attempts")
    if not isinstance(raw_attempts, list):
        raise ValueError(f"existing validation output has invalid attempts: {path}")
    expected = {trial.trial_id: trial for trial in trials}
    counts: Counter[str] = Counter()
    completed: set[str] = set()
    attempts: list[dict[str, Any]] = []
    for raw in raw_attempts:
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"existing validation output has an invalid attempt: {path}"
            )
        trial_id = raw.get("trial_id")
        trial = expected.get(trial_id)
        if trial is None or trial_id in completed:
            raise ValueError(
                f"existing validation output has invalid trial work: {path}"
            )
        if any(raw.get(key) != value for key, value in trial.identity().items()):
            raise ValueError(
                f"existing validation output trial identity changed: {path}"
            )
        try:
            status = ValidationTrialStatus(raw.get("status"))
            result = ValidationTrialResult(
                status=status,
                reward=raw.get("reward"),
                reason=raw.get("reason"),
                injected_candidate_ids=(
                    tuple(raw["injected_candidate_ids"])
                    if raw.get("injected_candidate_ids") is not None
                    else None
                ),
                audit=raw.get("audit"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"existing validation output has invalid completed work: {path}"
            ) from error
        if (
            status is ValidationTrialStatus.MEASURED
            and result.injected_candidate_ids != trial.expected_injected_candidate_ids
        ):
            raise ValueError(
                f"existing validation output has invalid injection evidence: {path}"
            )
        counts[trial_id] += 1
        if raw.get("attempt_number") != counts[trial_id]:
            raise ValueError(
                f"existing validation output has invalid attempt order: {path}"
            )
        attempts.append(dict(raw))
        if status is ValidationTrialStatus.MEASURED:
            completed.add(trial_id)
    return attempts


def _latest_attempts(
    attempts: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        latest[attempt["trial_id"]] = attempt
    return latest


def _report(
    plan: SkillValidationPlan,
    trials: Sequence[ValidationTrial],
    attempts: Sequence[dict[str, Any]],
    *,
    reused_ids: set[str],
    previous_attempt_count: int,
    run_metadata: Mapping[str, Any] | None,
    dry_run: bool,
) -> dict[str, Any]:
    attempts_by_id: dict[str, list[dict[str, Any]]] = {
        trial.trial_id: [] for trial in trials
    }
    for attempt in attempts:
        attempts_by_id[attempt["trial_id"]].append(attempt)
    ordered_attempts = [
        attempt for trial in trials for attempt in attempts_by_id[trial.trial_id]
    ]
    latest = _latest_attempts(ordered_attempts)
    measured = {
        trial_id: attempt
        for trial_id, attempt in latest.items()
        if attempt["status"] == ValidationTrialStatus.MEASURED.value
    }
    control_rewards = {
        trial.task_name: measured[trial.trial_id]["reward"]
        for trial in trials
        if trial.condition == "without_skill" and trial.trial_id in measured
    }
    treatment_rewards = {
        (trial.candidate_id, trial.task_name): measured[trial.trial_id]["reward"]
        for trial in trials
        if trial.condition == "with_skill" and trial.trial_id in measured
    }

    report = dict(plan.source_payload)
    raw_by_id = {
        raw["candidate_id"]: dict(raw) for raw in plan.source_payload["candidates"]
    }
    rendered_candidates: list[dict[str, Any]] = []
    paired_measurements: list[dict[str, Any]] = []
    trial_by_key = {(trial.candidate_id, trial.task_name): trial for trial in trials}
    for candidate in plan.candidates:
        raw = raw_by_id[candidate.candidate_id]
        deltas: list[float | None] = []
        for task in plan.tasks:
            baseline = control_rewards.get(task)
            treatment = treatment_rewards.get((candidate.candidate_id, task))
            delta = (
                treatment - baseline
                if treatment is not None and baseline is not None
                else None
            )
            deltas.append(delta)
            control_trial = trial_by_key[(None, task)]
            treatment_trial = trial_by_key[(candidate.candidate_id, task)]
            paired_measurements.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "arm": candidate.arm,
                    "task_name": task,
                    "control_trial_id": control_trial.trial_id,
                    "treatment_trial_id": treatment_trial.trial_id,
                    "control_reward": baseline,
                    "treatment_reward": treatment,
                    "delta": delta,
                    "measured": delta is not None,
                }
            )
        raw["skill"] = candidate.skill_document
        raw["paired_deltas"] = deltas
        rendered_candidates.append(raw)
    report["candidates"] = rendered_candidates
    pending = len(trials) - len(measured)
    status_counts = Counter(attempt["status"] for attempt in ordered_attempts)
    report["validation"] = {
        "schema_version": 1,
        "mode": VALIDATION_MODE,
        "dry_run": dry_run,
        "plan": plan.as_dict(),
        "summary": {
            "planned_trial_count": len(trials),
            "completed_attempt_count": len(ordered_attempts),
            "measured_trial_count": len(measured),
            "pending_trial_count": pending,
            "shared_control_trial_count": len(plan.tasks),
            "treatment_trial_count": len(plan.tasks) * len(plan.candidates),
            "reused_trial_count": len(reused_ids),
            "new_attempt_count": len(ordered_attempts) - previous_attempt_count,
            "failed_attempt_count": status_counts[ValidationTrialStatus.FAILED.value],
            "estimated_cost_per_trial_usd": plan.estimated_cost_per_trial_usd,
            "estimated_planned_cost_usd": plan.estimated_cost_usd,
            "estimated_pending_cost_usd": (pending * plan.estimated_cost_per_trial_usd),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "control_rewards": control_rewards,
        "paired_measurements": paired_measurements,
        "attempts": ordered_attempts,
    }
    if run_metadata is not None:
        report["validation"]["run"] = dict(run_metadata)
    return report


def _read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} does not exist: {path}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is invalid JSON: {path}") from error


def _read_object(path: Path, description: str) -> dict[str, Any]:
    value = _read_json(path, description)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object: {path}")
    return value


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
