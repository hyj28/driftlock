"""Resumable paired validation of distilled skill candidates.

The driver is provider-neutral.  It owns planning, one-candidate library
isolation, trial-granularity checkpoints, and delta assembly; the CLI injects a
Harbor-backed runner while tests inject a deterministic offline runner.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import shutil
import tomllib
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from driftlock.skill_admission import (
    DISTILLATION_ARMS,
    VALIDATION_TASK_COUNT,
    SkillLibrary,
    build_null_channel_summary,
)
from driftlock.skill_distillation import parse_skill, serialize_skill

VALIDATION_REPORT_NAME = "validated-skill-candidates.json"
VALIDATION_MODE = "skill-paired-validation"
VALIDATION_PROCEDURE_ID = "shared-own-task-replicated-single-candidate-paired-v3"

# The most recent paid LHTB measurement supplied for this validation round was
# $0.151 per trial (2026-08-30).  This is an operator-facing budget estimate,
# not a spending limit; expose it as a CLI override when provider pricing changes.
DEFAULT_ESTIMATED_COST_PER_TRIAL_USD = 0.151

# Round five sustained four independent Harbor jobs on the target 8-core / 15 GB
# host, so four is the evidence-backed default rather than an aspirational host
# saturation target.  Keep this operator-controlled: raising it also increases
# simultaneous provider pressure, and upstream 429s under load caused 28 of 32
# trials to fail in round two.
DEFAULT_MAX_CONCURRENT_TRIALS = 4

# Two retries cap a transiently interrupted paid observation at three provider
# calls while giving isolated 429s and transport failures two chances to clear.
DEFAULT_MAX_RETRIES = 2

# One second keeps operator feedback responsive while exponential growth spaces
# consecutive provider failures without creating a long blind wait.
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0

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


class ValidationFailureKind(StrEnum):
    """Machine-readable attribution for an unusable validation attempt."""

    TRIAL_RUNNER = "trial_runner"
    DID_NOT_PRODUCE_RESULT = "did_not_produce_result"
    AMBIGUOUS_RESULT = "ambiguous_result"
    TRANSIENT_INFRASTRUCTURE = "transient_infrastructure"
    NO_REWARD = "no_reward"
    REWARD_EVIDENCE = "reward_evidence"
    SKILL_LAYER_EVIDENCE = "skill_layer_evidence"
    SKILL_INJECTION_MISMATCH = "skill_injection_mismatch"


@dataclass(frozen=True, slots=True)
class ValidationCandidate:
    """The immutable candidate identity used while validation is resumed."""

    candidate_id: str
    arm: str
    skill_document: str
    source_task_name: str
    task_name: str

    @property
    def skill_sha256(self) -> str:
        return hashlib.sha256(self.skill_document.encode()).hexdigest()

    def identity(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "arm": self.arm,
            "skill_sha256": self.skill_sha256,
            "source_task_name": self.source_task_name,
            "validation_task": self.task_name,
        }


@dataclass(frozen=True, slots=True)
class ValidationTrial:
    """One control or treatment trial passed through the shared runner seam."""

    trial_id: str
    task_name: str
    replicate_index: int
    condition: str
    distillation_arm: str
    library_dir: Path
    candidate_id: str | None = None
    available_candidate_ids: tuple[str, ...] = ()
    procedure_id: str = VALIDATION_PROCEDURE_ID
    attempt_number: int = 1

    @property
    def job_name(self) -> str:
        return f"skill-validation-{self.trial_id}-a{self.attempt_number}"

    def identity(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "task_name": self.task_name,
            "replicate_index": self.replicate_index,
            "condition": self.condition,
            "candidate_id": self.candidate_id,
            "distillation_arm": self.distillation_arm,
            "available_candidate_ids": list(self.available_candidate_ids),
            "procedure_id": self.procedure_id,
        }


@dataclass(frozen=True, slots=True)
class ValidationTrialResult:
    """Provider-neutral result returned by a validation trial runner."""

    status: ValidationTrialStatus
    reward: float | None = None
    reason: str = ""
    failure_kind: ValidationFailureKind | None = None
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
        if isinstance(self.failure_kind, str):
            object.__setattr__(
                self, "failure_kind", ValidationFailureKind(self.failure_kind)
            )
        elif self.failure_kind is not None and not isinstance(
            self.failure_kind, ValidationFailureKind
        ):
            raise TypeError("validation trial failure kind must be recognized")
        if (
            self.status is ValidationTrialStatus.MEASURED
            and self.failure_kind is not None
        ):
            raise ValueError("a measured validation trial cannot have a failure kind")
        if not isinstance(self.reason, str):
            raise TypeError("validation trial reason must be text")
        if self.audit is not None and not isinstance(self.audit, Mapping):
            raise TypeError("validation trial audit must be an object")


class SkillValidationTrialRunner(Protocol):
    """The only paid boundary required by the validation driver."""

    async def run(self, trial: ValidationTrial) -> ValidationTrialResult: ...


@dataclass(frozen=True, slots=True)
class SkillValidationPlan:
    """Validated candidates, replicated tasks, and the exact paid-work plan."""

    source_candidate_file: Path
    task_root: Path
    source_payload: Mapping[str, Any]
    candidates: tuple[ValidationCandidate, ...]
    tasks: tuple[str, ...]
    replicate_count: int
    estimated_cost_per_trial_usd: float

    @property
    def observation_count(self) -> int:
        """Number of paired observations produced for each candidate."""

        return self.replicate_count

    @property
    def shared_control_trial_count(self) -> int:
        return len(self.tasks) * self.replicate_count

    @property
    def treatment_trial_count(self) -> int:
        return len(self.candidates) * self.replicate_count

    @property
    def paired_observation_count(self) -> int:
        return self.treatment_trial_count

    @property
    def planned_trial_count(self) -> int:
        return self.shared_control_trial_count + self.treatment_trial_count

    @property
    def estimated_cost_usd(self) -> float:
        return self.planned_trial_count * self.estimated_cost_per_trial_usd

    def as_dict(self) -> dict[str, Any]:
        return {
            "procedure_id": VALIDATION_PROCEDURE_ID,
            "source_candidate_file": str(self.source_candidate_file),
            "task_root": str(self.task_root),
            "candidate_identities": [
                candidate.identity() for candidate in self.candidates
            ],
            "distinct_source_tasks": list(self.tasks),
            "replicate_count": self.replicate_count,
            "observations_per_candidate": self.observation_count,
            "paired_observation_count": self.paired_observation_count,
            "shared_control_trial_count": self.shared_control_trial_count,
            "treatment_trial_count": self.treatment_trial_count,
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
                replicate_index=replicate_index,
                condition="without_skill",
                candidate=None,
                library_dir=libraries / "empty",
            )
            for task in self.tasks
            for replicate_index in range(1, self.replicate_count + 1)
        )
        treatments = tuple(
            _trial(
                task_name=candidate.task_name,
                replicate_index=replicate_index,
                condition="with_skill",
                candidate=candidate,
                library_dir=libraries / candidate.candidate_id,
            )
            for candidate in self.candidates
            for replicate_index in range(1, self.replicate_count + 1)
        )
        return controls + treatments


def plan_skill_validation(
    candidate_file: Path | str,
    lhtb_dir: Path | str,
    *,
    estimated_cost_per_trial_usd: float = DEFAULT_ESTIMATED_COST_PER_TRIAL_USD,
) -> SkillValidationPlan:
    """Plan ten own-task replicates per candidate without provider access."""

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

    root = Path(lhtb_dir).expanduser().resolve()
    task_root = root / "tasks"
    if not task_root.is_dir():
        raise FileNotFoundError(f"LHTB task directory does not exist: {task_root}")

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
        source_task_name, task_name = _resolve_candidate_task(
            task_root, raw.get("task_name"), candidate_id
        )
        candidate_ids.add(candidate_id)
        candidates.append(
            ValidationCandidate(
                candidate_id,
                arm,
                canonical_document,
                source_task_name,
                task_name,
            )
        )

    tasks = tuple(dict.fromkeys(candidate.task_name for candidate in candidates))

    return SkillValidationPlan(
        source_candidate_file=source,
        task_root=task_root,
        source_payload=payload,
        candidates=tuple(candidates),
        tasks=tasks,
        replicate_count=VALIDATION_TASK_COUNT,
        estimated_cost_per_trial_usd=float(estimated_cost_per_trial_usd),
    )


async def run_skill_validation(
    plan: SkillValidationPlan,
    output_path: Path | str,
    *,
    runner: SkillValidationTrialRunner,
    work_dir: Path | str,
    run_metadata: Mapping[str, Any] | None = None,
    max_concurrent_trials: int = DEFAULT_MAX_CONCURRENT_TRIALS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    force_retry_unmeasured: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run bounded pending trials and checkpoint after every attempt."""

    if (
        isinstance(max_concurrent_trials, bool)
        or not isinstance(max_concurrent_trials, int)
        or max_concurrent_trials < 1
    ):
        raise ValueError("max concurrent validation trials must be a positive integer")
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise ValueError("max validation retries must be a nonnegative integer")
    if (
        isinstance(retry_backoff_seconds, bool)
        or not isinstance(retry_backoff_seconds, (int, float))
        or not math.isfinite(float(retry_backoff_seconds))
        or retry_backoff_seconds < 0
    ):
        raise ValueError("validation retry backoff must be finite and nonnegative")
    if not isinstance(force_retry_unmeasured, bool):
        raise TypeError("force retry unmeasured must be a boolean")

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
            max_retries=max_retries,
            force_retry_unmeasured=force_retry_unmeasured,
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
            max_retries=max_retries,
            force_retry_unmeasured=force_retry_unmeasured,
            dry_run=False,
        ),
    )
    maximum_attempts = 1 + max_retries
    attempt_counts = Counter(attempt["trial_id"] for attempt in attempts)

    def should_run(trial: ValidationTrial) -> bool:
        previous_attempt = latest.get(trial.trial_id)
        if previous_attempt is None:
            return True
        if previous_attempt["status"] == ValidationTrialStatus.MEASURED.value:
            return False
        if force_retry_unmeasured:
            return True
        return (
            previous_attempt["failure_kind"]
            == ValidationFailureKind.TRANSIENT_INFRASTRUCTURE.value
            and attempt_counts[trial.trial_id] < maximum_attempts
        )

    async def run_trial(trial: ValidationTrial) -> None:
        completed_attempts = attempt_counts[trial.trial_id]
        forced_reattempt = force_retry_unmeasured and completed_attempts > 0
        allowed_new_attempts = (
            maximum_attempts
            if forced_reattempt
            else maximum_attempts - completed_attempts
        )
        new_attempts = 0
        if completed_attempts and not forced_reattempt:
            await sleep(float(retry_backoff_seconds) * 2 ** (completed_attempts - 1))
        while new_attempts < allowed_new_attempts:
            attempt_number = completed_attempts + 1
            attempt_trial = replace(trial, attempt_number=attempt_number)
            try:
                result = await runner.run(attempt_trial)
                if not isinstance(result, ValidationTrialResult):
                    raise TypeError(
                        "validation runner must return ValidationTrialResult"
                    )
            except Exception as error:
                result = ValidationTrialResult(
                    status=ValidationTrialStatus.FAILED,
                    reason=(
                        "validation trial runner failed: "
                        f"{type(error).__name__}: {error}"
                    ),
                    failure_kind=ValidationFailureKind.TRIAL_RUNNER,
                )
            if (
                result.status is ValidationTrialStatus.MEASURED
                and not _injection_matches_trial(trial, result.injected_candidate_ids)
            ):
                result = ValidationTrialResult(
                    status=ValidationTrialStatus.FAILED,
                    reason=(
                        "skill injection mismatch: available candidate ids were "
                        f"{list(trial.available_candidate_ids)!r}, observed "
                        f"{list(result.injected_candidate_ids or ())!r}"
                    ),
                    failure_kind=ValidationFailureKind.SKILL_INJECTION_MISMATCH,
                    audit=result.audit,
                )
            attempt_audit = dict(result.audit or {})
            attempt_audit["forced_reattempt"] = forced_reattempt
            attempt = {
                **trial.identity(),
                "attempt_number": attempt_number,
                "status": result.status.value,
                "reason": result.reason,
                "failure_kind": (
                    result.failure_kind.value
                    if result.failure_kind is not None
                    else None
                ),
                "reward": result.reward,
                "injected_candidate_ids": (
                    list(result.injected_candidate_ids)
                    if result.injected_candidate_ids is not None
                    else None
                ),
                "skill_injected": (
                    bool(result.injected_candidate_ids)
                    if result.injected_candidate_ids is not None
                    else None
                ),
                "audit": attempt_audit,
                "reused": False,
                "forced_reattempt": forced_reattempt,
            }
            # There is deliberately no await between the runner completing and
            # the atomic replace.  This synchronous critical section serializes
            # writers on the event loop and prevents cancellation from losing a
            # completed attempt while it waits for a checkpoint lock.
            attempts.append(attempt)
            attempt_counts[trial.trial_id] += 1
            completed_attempts += 1
            new_attempts += 1
            _write_report(
                destination,
                _report(
                    plan,
                    trials,
                    attempts,
                    reused_ids=reused_ids,
                    previous_attempt_count=previous_attempt_count,
                    run_metadata=run_metadata,
                    max_retries=max_retries,
                    force_retry_unmeasured=force_retry_unmeasured,
                    dry_run=False,
                ),
            )
            # A measured reward is final regardless of its value.  Retrying is
            # reserved for infrastructure failures that prevented a measurement.
            if (
                result.failure_kind
                is not ValidationFailureKind.TRANSIENT_INFRASTRUCTURE
                or new_attempts >= allowed_new_attempts
            ):
                return
            retry_ordinal = new_attempts if forced_reattempt else completed_attempts
            await sleep(float(retry_backoff_seconds) * 2 ** (retry_ordinal - 1))

    async def run_phase(phase_trials: Sequence[ValidationTrial]) -> None:
        pending = iter(trial for trial in phase_trials if should_run(trial))

        async def worker() -> None:
            while True:
                try:
                    trial = next(pending)
                except StopIteration:
                    return
                await run_trial(trial)

        auxiliary_workers = [
            asyncio.create_task(worker())
            for _ in range(max(0, min(max_concurrent_trials, len(phase_trials)) - 1))
        ]
        try:
            # Keep one worker in the caller task.  In addition to avoiding an
            # unnecessary task at a bound of one, this preserves the sequential
            # runner's BaseException/interrupt behavior on that path.
            await worker()
            await asyncio.gather(*auxiliary_workers)
        except BaseException:
            for worker_task in auxiliary_workers:
                worker_task.cancel()
            await asyncio.gather(*auxiliary_workers, return_exceptions=True)
            raise

    controls = tuple(trial for trial in trials if trial.condition == "without_skill")
    treatments = tuple(trial for trial in trials if trial.condition == "with_skill")
    # Controls are a complete first phase.  A partial paid run therefore cannot
    # consume treatments that have no completed control attempt, and shared
    # controls cannot race between candidates.  Dispatch order does not change a
    # trial or observation, so VALIDATION_PROCEDURE_ID intentionally remains v3
    # and prior measured trials remain reusable.
    await run_phase(controls)
    await run_phase(treatments)
    return _report(
        plan,
        trials,
        attempts,
        reused_ids=reused_ids,
        previous_attempt_count=previous_attempt_count,
        run_metadata=run_metadata,
        max_retries=max_retries,
        force_retry_unmeasured=force_retry_unmeasured,
        dry_run=False,
    )


def _injection_matches_trial(
    trial: ValidationTrial, injected_candidate_ids: tuple[str, ...] | None
) -> bool:
    if injected_candidate_ids is None:
        return False
    if trial.condition == "without_skill":
        return injected_candidate_ids == ()
    return injected_candidate_ids in ((), trial.available_candidate_ids)


def _trial(
    *,
    task_name: str,
    replicate_index: int,
    condition: str,
    candidate: ValidationCandidate | None,
    library_dir: Path,
) -> ValidationTrial:
    candidate_id = candidate.candidate_id if candidate is not None else None
    identity = json.dumps(
        [
            VALIDATION_PROCEDURE_ID,
            condition,
            candidate_id,
            task_name,
            replicate_index,
        ],
        separators=(",", ":"),
    )
    # The replicate index changes the trial ID and therefore the Harbor job name.
    # Harbor creates a separate job/container and the agent is sampled afresh, so
    # this is repeated sampling rather than one run counted twice.  It estimates
    # within-task run variability; it does not create the diversity or independent
    # task effects that sampling another task would provide.
    trial_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return ValidationTrial(
        trial_id=trial_id,
        task_name=task_name,
        replicate_index=replicate_index,
        condition=condition,
        candidate_id=candidate_id,
        distillation_arm=(
            candidate.arm if candidate is not None else CONTROL_DISTILLATION_ARM
        ),
        library_dir=library_dir,
        available_candidate_ids=(
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
    if run_metadata is not None:
        recorded_run = validation.get("run")
        comparable_requested = {
            key: value for key, value in run_metadata.items() if key != "max_retries"
        }
        comparable_recorded = (
            {key: value for key, value in recorded_run.items() if key != "max_retries"}
            if isinstance(recorded_run, Mapping)
            else recorded_run
        )
        # Retry policy changes how an unmeasured trial is resumed, not the trial
        # identity.  Older paid reports also predate this operational field.
        if comparable_recorded != comparable_requested:
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
                failure_kind=raw.get("failure_kind"),
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
        if status is ValidationTrialStatus.MEASURED and not _injection_matches_trial(
            trial, result.injected_candidate_ids
        ):
            raise ValueError(
                f"existing validation output has invalid injection evidence: {path}"
            )
        counts[trial_id] += 1
        if raw.get("attempt_number") != counts[trial_id]:
            raise ValueError(
                f"existing validation output has invalid attempt order: {path}"
            )
        expected_skill_injected = (
            bool(result.injected_candidate_ids)
            if result.injected_candidate_ids is not None
            else None
        )
        if raw.get("skill_injected") != expected_skill_injected:
            raise ValueError(
                f"existing validation output has invalid injection evidence: {path}"
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
    max_retries: int,
    force_retry_unmeasured: bool,
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
        (trial.task_name, trial.replicate_index): measured[trial.trial_id]["reward"]
        for trial in trials
        if trial.condition == "without_skill" and trial.trial_id in measured
    }
    treatment_rewards = {
        (
            trial.candidate_id,
            trial.task_name,
            trial.replicate_index,
        ): measured[trial.trial_id]["reward"]
        for trial in trials
        if trial.condition == "with_skill" and trial.trial_id in measured
    }

    report = dict(plan.source_payload)
    raw_by_id = {
        raw["candidate_id"]: dict(raw) for raw in plan.source_payload["candidates"]
    }
    rendered_candidates: list[dict[str, Any]] = []
    paired_measurements: list[dict[str, Any]] = []
    trial_by_key = {
        (trial.candidate_id, trial.task_name, trial.replicate_index): trial
        for trial in trials
    }
    for candidate in plan.candidates:
        raw = raw_by_id[candidate.candidate_id]
        deltas: list[float | None] = []
        candidate_measurements: list[dict[str, Any]] = []
        task = candidate.task_name
        for replicate_index in range(1, plan.replicate_count + 1):
            observation_key = (task, replicate_index)
            treatment_key = (
                candidate.candidate_id,
                task,
                replicate_index,
            )
            baseline = control_rewards.get(observation_key)
            treatment = treatment_rewards.get(treatment_key)
            delta = (
                treatment - baseline
                if treatment is not None and baseline is not None
                else None
            )
            deltas.append(delta)
            control_trial = trial_by_key[(None, task, replicate_index)]
            treatment_trial = trial_by_key[treatment_key]
            treatment_attempt = measured.get(treatment_trial.trial_id)
            injected_candidate_ids = (
                treatment_attempt["injected_candidate_ids"]
                if treatment_attempt is not None
                else None
            )
            skill_injected = (
                bool(injected_candidate_ids) if treatment_attempt is not None else None
            )
            measurement = {
                "candidate_id": candidate.candidate_id,
                "arm": candidate.arm,
                "source_task_name": candidate.source_task_name,
                "task_name": task,
                "replicate_index": replicate_index,
                "control_trial_id": control_trial.trial_id,
                "treatment_trial_id": treatment_trial.trial_id,
                "control_reward": baseline,
                "treatment_reward": treatment,
                "delta": delta,
                "measured": delta is not None,
                "skill_injected": skill_injected,
                "injected_candidate_ids": injected_candidate_ids,
                "attribution": (
                    "unmeasured"
                    if delta is None
                    else "skill_injected"
                    if skill_injected
                    else "no_skill_injected"
                ),
            }
            paired_measurements.append(measurement)
            candidate_measurements.append(measurement)
        raw["skill"] = candidate.skill_document
        raw["paired_deltas"] = deltas
        raw["injection_flags"] = [
            measurement["skill_injected"] for measurement in candidate_measurements
        ]
        injected_count = sum(
            measurement["skill_injected"] is True
            for measurement in candidate_measurements
        )
        no_injection_count = sum(
            measurement["skill_injected"] is False
            for measurement in candidate_measurements
        )
        unmeasured_pair_count = sum(
            measurement["measured"] is False for measurement in candidate_measurements
        )
        unmeasured_treatment_count = (
            plan.replicate_count - injected_count - no_injection_count
        )
        raw["validation_observation_summary"] = {
            "source_task_name": candidate.source_task_name,
            "task_name": candidate.task_name,
            "expected_observation_count": plan.replicate_count,
            "measured_observation_count": (
                plan.replicate_count - unmeasured_pair_count
            ),
            "unmeasured_observation_count": unmeasured_pair_count,
            "measured_treatment_count": (
                plan.replicate_count - unmeasured_treatment_count
            ),
            "unmeasured_treatment_count": unmeasured_treatment_count,
            "skill_injected_observation_count": injected_count,
            "no_skill_injected_observation_count": no_injection_count,
            "skill_application": (
                "unmeasured"
                if unmeasured_treatment_count == plan.replicate_count
                else "never_injected"
                if injected_count == 0
                else "always_injected"
                if no_injection_count == 0
                else "mixed_injection"
            ),
        }
        rendered_candidates.append(raw)
    report["candidates"] = rendered_candidates
    pending = len(trials) - len(measured)
    status_counts = Counter(attempt["status"] for attempt in ordered_attempts)
    failed_attempts = [
        attempt
        for attempt in ordered_attempts
        if attempt["status"] == ValidationTrialStatus.FAILED.value
    ]
    failure_kind_counts = Counter(
        attempt["failure_kind"] or "unattributed" for attempt in failed_attempts
    )
    exception_name_counts: Counter[str] = Counter()
    for attempt in failed_attempts:
        exception_names = attempt["audit"].get("observed_exception_names", [])
        if isinstance(exception_names, list):
            exception_name_counts.update(
                {
                    exception_name
                    for exception_name in exception_names
                    if isinstance(exception_name, str)
                }
            )
    retried_trial_count = sum(
        len(trial_attempts) > 1 for trial_attempts in attempts_by_id.values()
    )
    forced_reattempt_trial_count = sum(
        any(attempt.get("forced_reattempt") is True for attempt in trial_attempts)
        for trial_attempts in attempts_by_id.values()
    )
    rescued_by_retry_trial_count = sum(
        bool(trial_attempts)
        and trial_attempts[-1]["status"] == ValidationTrialStatus.MEASURED.value
        and any(
            attempt["failure_kind"]
            == ValidationFailureKind.TRANSIENT_INFRASTRUCTURE.value
            for attempt in trial_attempts[:-1]
        )
        for trial_attempts in attempts_by_id.values()
    )
    null_channel = build_null_channel_summary(
        [
            (measurement["delta"], measurement["skill_injected"])
            for measurement in paired_measurements
        ],
        injection_data_available=True,
    )
    summary = {
        "planned_trial_count": len(trials),
        "completed_attempt_count": len(ordered_attempts),
        "measured_trial_count": len(measured),
        "pending_trial_count": pending,
        "distinct_source_task_count": len(plan.tasks),
        "observations_per_candidate": plan.observation_count,
        "paired_observation_count": plan.paired_observation_count,
        "shared_control_trial_count": plan.shared_control_trial_count,
        "treatment_trial_count": plan.treatment_trial_count,
        "reused_trial_count": len(reused_ids),
        "new_attempt_count": len(ordered_attempts) - previous_attempt_count,
        "failed_attempt_count": status_counts[ValidationTrialStatus.FAILED.value],
        "estimated_cost_per_trial_usd": plan.estimated_cost_per_trial_usd,
        "estimated_planned_cost_usd": plan.estimated_cost_usd,
        "estimated_pending_cost_usd": pending * plan.estimated_cost_per_trial_usd,
        "status_counts": dict(sorted(status_counts.items())),
    }
    if not dry_run:
        summary.update(
            {
                "failed_attempt_counts_by_failure_kind": dict(
                    sorted(failure_kind_counts.items())
                ),
                "failed_attempt_counts_by_exception_name": dict(
                    sorted(exception_name_counts.items())
                ),
                "retried_trial_count": retried_trial_count,
                "rescued_by_retry_trial_count": rescued_by_retry_trial_count,
                "configured_max_retries": max_retries,
                "force_retry_unmeasured_requested": force_retry_unmeasured,
                "forced_reattempt_trial_count": forced_reattempt_trial_count,
                "null_channel": null_channel,
            }
        )
    report["validation"] = {
        "schema_version": 1,
        "mode": VALIDATION_MODE,
        "dry_run": dry_run,
        "plan": plan.as_dict(),
        "summary": summary,
        "control_measurements": [
            {
                "task_name": trial.task_name,
                "replicate_index": trial.replicate_index,
                "control_trial_id": trial.trial_id,
                "reward": control_rewards.get((trial.task_name, trial.replicate_index)),
                "measured": (
                    (trial.task_name, trial.replicate_index) in control_rewards
                ),
            }
            for trial in trials
            if trial.condition == "without_skill"
        ],
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


def _resolve_candidate_task(
    task_root: Path, raw_task_name: object, candidate_id: str
) -> tuple[str, str]:
    if not isinstance(raw_task_name, str) or not raw_task_name:
        raise ValueError(
            f"skill candidate {candidate_id!r} refused (missing_task_name): "
            "task_name must be non-empty text"
        )
    parts = raw_task_name.split("/")
    if len(parts) == 1:
        task_name = parts[0]
    elif len(parts) == 2:
        task_name = parts[1]
    else:
        task_name = ""
    if _SAFE_NAME.fullmatch(task_name) is None:
        raise ValueError(
            f"skill candidate {candidate_id!r} refused "
            f"(unresolvable_task_name): unsafe task_name {raw_task_name!r}"
        )
    task_file = task_root / task_name / "task.toml"
    try:
        metadata = tomllib.loads(task_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(
            f"skill candidate {candidate_id!r} refused "
            f"(unresolvable_task_name): no task directory for {raw_task_name!r}"
        ) from None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(
            f"skill candidate {candidate_id!r} refused "
            f"(unresolvable_task_name): invalid task metadata for "
            f"{raw_task_name!r}"
        ) from error
    task = metadata.get("task")
    canonical_name = task.get("name") if isinstance(task, Mapping) else None
    if len(parts) == 2 and canonical_name != raw_task_name:
        raise ValueError(
            f"skill candidate {candidate_id!r} refused "
            f"(unresolvable_task_name): task directory {task_name!r} does not "
            f"declare {raw_task_name!r}"
        )
    if (
        not isinstance(canonical_name, str)
        or canonical_name.split("/")[-1] != task_name
    ):
        raise ValueError(
            f"skill candidate {candidate_id!r} refused "
            f"(unresolvable_task_name): task directory {task_name!r} has "
            "inconsistent metadata"
        )
    return raw_task_name, task_name


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
