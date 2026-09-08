"""Bounded, checkpointable agent-maintained plans."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

# Thirty-two steps are enough to decompose a long task while keeping every
# checkpoint, audit snapshot, and provider rendering at a predictable size.
MAX_PLAN_STEPS = 32

# Two hundred forty characters permit a specific one-line action without
# allowing prose notes to turn the durable plan into another conversation log.
MAX_PLAN_DESCRIPTION_CHARACTERS = 240

# Generated ids only reach ``step-32``. This larger limit specifically guards
# hand-built and deserialized plans without adding another source of unbounded text.
MAX_PLAN_STEP_ID_CHARACTERS = 16

# The plan has its own version because it is nested in versioned checkpoint
# state and may need independent validation if its semantic shape later changes.
PLAN_SCHEMA_VERSION = 1


class PlanStatus(StrEnum):
    """Lifecycle state of one plan step."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ABANDONED = "abandoned"


class PlanOperation(StrEnum):
    """Mutations accepted by the agent's plan tool."""

    CREATE = "create"
    ADD = "add"
    REVISE = "revise"
    SET_STATUS = "set_status"


class PlanUpdateStatus(StrEnum):
    """Whether one requested mutation changed the plan."""

    APPLIED = "applied"
    REJECTED = "rejected"


class PlanError(ValueError):
    """A malformed plan or refused plan mutation."""


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One stable semantic unit of work in an ordered plan."""

    step_id: str
    description: str
    status: PlanStatus = PlanStatus.NOT_STARTED

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id:
            raise PlanError("plan step id must be a non-empty string")
        if len(self.step_id) > MAX_PLAN_STEP_ID_CHARACTERS:
            raise PlanError(
                "plan step id exceeds the "
                f"{MAX_PLAN_STEP_ID_CHARACTERS}-character limit"
            )
        if re.fullmatch(r"[A-Za-z0-9._-]+", self.step_id) is None:
            raise PlanError("plan step id contains unsupported characters")
        if not isinstance(self.description, str):
            raise PlanError("plan step description must be a string")
        if "\n" in self.description or "\r" in self.description:
            raise PlanError("plan step description must be one line")
        if len(self.description) > MAX_PLAN_DESCRIPTION_CHARACTERS:
            raise PlanError(
                "plan step description exceeds the "
                f"{MAX_PLAN_DESCRIPTION_CHARACTERS}-character limit"
            )
        normalized_description = self.description.strip()
        if not normalized_description:
            raise PlanError("plan step description must be a non-empty string")
        object.__setattr__(self, "description", normalized_description)
        if not isinstance(self.status, PlanStatus):
            raise PlanError("plan step status must be a PlanStatus")

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.step_id,
            "description": self.description,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class AgentPlan:
    """An ordered, bounded plan stored alongside checkpointed conversation.

    Stable step identities and explicit current/terminal statuses let future
    rollback localization associate a checkpoint with the semantic plan-step
    boundary it precedes, rather than only with a numeric agent-step interval.
    """

    steps: tuple[PlanStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple) or any(
            not isinstance(step, PlanStep) for step in self.steps
        ):
            raise PlanError("plan steps must be a tuple of PlanStep values")
        if not self.steps:
            raise PlanError("a plan must contain at least one step")
        if len(self.steps) > MAX_PLAN_STEPS:
            raise PlanError(f"plan exceeds the {MAX_PLAN_STEPS}-step limit")
        identifiers = [step.step_id for step in self.steps]
        if len(identifiers) != len(set(identifiers)):
            raise PlanError("plan step ids must be unique")
        if sum(step.status is PlanStatus.IN_PROGRESS for step in self.steps) > 1:
            raise PlanError("a plan can have at most one in-progress step")

    @property
    def current_step(self) -> PlanStep | None:
        """Return the step the agent is currently working in, if any."""

        return next(
            (step for step in self.steps if step.status is PlanStatus.IN_PROGRESS),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, value: object) -> AgentPlan:
        if not isinstance(value, Mapping):
            raise PlanError("checkpointed plan must be an object")
        if set(value) != {"schema_version", "steps"}:
            raise PlanError("checkpointed plan fields are malformed")
        if value.get("schema_version") != PLAN_SCHEMA_VERSION:
            raise PlanError("unsupported checkpointed plan schema version")
        raw_steps = value.get("steps")
        if not isinstance(raw_steps, list):
            raise PlanError("checkpointed plan steps must be a list")
        steps: list[PlanStep] = []
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping) or set(raw_step) != {
                "id",
                "description",
                "status",
            }:
                raise PlanError(f"checkpointed plan step {index} is malformed")
            raw_status = raw_step.get("status")
            try:
                status = PlanStatus(raw_status)
            except (TypeError, ValueError) as error:
                raise PlanError(
                    f"checkpointed plan step {index} has an invalid status"
                ) from error
            try:
                steps.append(
                    PlanStep(
                        step_id=raw_step.get("id"),
                        description=raw_step.get("description"),
                        status=status,
                    )
                )
            except PlanError as error:
                raise PlanError(
                    f"checkpointed plan step {index} is invalid: {error}"
                ) from error
        return cls(tuple(steps))


@dataclass(frozen=True, slots=True)
class PlanMutation:
    """The new plan and exact semantic changes from one accepted operation."""

    plan: AgentPlan
    changes: tuple[Mapping[str, Any], ...]


def apply_plan_operation(
    plan: AgentPlan | None,
    operation: PlanOperation,
    *,
    steps: Sequence[str] | None = None,
    step_id: str | None = None,
    description: str | None = None,
    status: PlanStatus | None = None,
) -> PlanMutation:
    """Validate and apply one atomic plan mutation."""

    if not isinstance(operation, PlanOperation):
        raise PlanError("operation must be a PlanOperation")
    if plan is not None and not isinstance(plan, AgentPlan):
        raise PlanError("plan must be an AgentPlan or None")
    if operation is PlanOperation.CREATE:
        _require_only(
            steps=steps,
            step_id=step_id,
            description=description,
            status=status,
            allowed={"steps"},
        )
        if plan is not None:
            raise PlanError("a plan already exists; revise or add steps instead")
        created = _new_steps((), steps)
        first = replace(created[0], status=PlanStatus.IN_PROGRESS)
        new_plan = AgentPlan((first, *created[1:]))
        return PlanMutation(
            new_plan,
            (
                {
                    "kind": "created",
                    "step_ids": [step.step_id for step in new_plan.steps],
                    "auto_started_step_id": first.step_id,
                },
            ),
        )

    if operation is PlanOperation.ADD:
        _require_only(
            steps=steps,
            step_id=step_id,
            description=description,
            status=status,
            allowed={"steps"},
        )
        existing = _require_plan(plan)
        added = _new_steps(existing.steps, steps)
        combined = (*existing.steps, *added)
        changes: list[Mapping[str, Any]] = [
            {
                "kind": "added",
                "step_ids": [step.step_id for step in added],
            }
        ]
        if existing.current_step is None:
            next_index = next(
                index
                for index, candidate in enumerate(combined)
                if candidate.status is PlanStatus.NOT_STARTED
            )
            current = replace(combined[next_index], status=PlanStatus.IN_PROGRESS)
            combined = _replace_step(combined, next_index, current)
            changes.append(
                {
                    "kind": "status_changed",
                    "step_id": current.step_id,
                    "before": PlanStatus.NOT_STARTED.value,
                    "after": PlanStatus.IN_PROGRESS.value,
                    "reason": "automatic_start_after_add",
                }
            )
        new_plan = AgentPlan(combined)
        return PlanMutation(
            new_plan,
            tuple(changes),
        )
    if operation is PlanOperation.REVISE:
        _require_only(
            steps=steps,
            step_id=step_id,
            description=description,
            status=status,
            allowed={"step_id", "description"},
        )
        existing = _require_plan(plan)
        index = _step_index(existing, step_id)
        target = existing.steps[index]
        if target.status in {PlanStatus.DONE, PlanStatus.ABANDONED}:
            raise PlanError("a terminal plan step cannot be revised")
        revised = PlanStep(
            target.step_id, _validate_description(description), target.status
        )
        if revised.description == target.description:
            raise PlanError("revised plan step description must differ")
        new_plan = AgentPlan(_replace_step(existing.steps, index, revised))
        return PlanMutation(
            new_plan,
            (
                {
                    "kind": "description_changed",
                    "step_id": target.step_id,
                    "before": target.description,
                    "after": revised.description,
                },
            ),
        )

    _require_only(
        steps=steps,
        step_id=step_id,
        description=description,
        status=status,
        allowed={"step_id", "status"},
    )
    existing = _require_plan(plan)
    index = _step_index(existing, step_id)
    target = existing.steps[index]
    if not isinstance(status, PlanStatus):
        raise PlanError("status must be a PlanStatus")
    _validate_status_transition(existing, target, status)
    updated = replace(target, status=status)
    updated_steps = _replace_step(existing.steps, index, updated)
    changes: list[Mapping[str, Any]] = [
        {
            "kind": "status_changed",
            "step_id": target.step_id,
            "before": target.status.value,
            "after": status.value,
        }
    ]
    if target.status is PlanStatus.IN_PROGRESS and status in {
        PlanStatus.DONE,
        PlanStatus.ABANDONED,
    }:
        next_index = next(
            (
                candidate
                for candidate, step in enumerate(updated_steps)
                if step.status is PlanStatus.NOT_STARTED
            ),
            None,
        )
        if next_index is not None:
            next_step = replace(
                updated_steps[next_index], status=PlanStatus.IN_PROGRESS
            )
            updated_steps = _replace_step(updated_steps, next_index, next_step)
            changes.append(
                {
                    "kind": "status_changed",
                    "step_id": next_step.step_id,
                    "before": PlanStatus.NOT_STARTED.value,
                    "after": PlanStatus.IN_PROGRESS.value,
                    "reason": "automatic_advance",
                }
            )
    return PlanMutation(AgentPlan(updated_steps), tuple(changes))


def _require_only(
    *,
    steps: Sequence[str] | None,
    step_id: str | None,
    description: str | None,
    status: PlanStatus | None,
    allowed: set[str],
) -> None:
    supplied = {
        name
        for name, value in (
            ("steps", steps),
            ("step_id", step_id),
            ("description", description),
            ("status", status),
        )
        if value is not None
    }
    missing = allowed - supplied
    unexpected = supplied - allowed
    if missing:
        raise PlanError(f"missing plan argument(s): {', '.join(sorted(missing))}")
    if unexpected:
        raise PlanError(f"unexpected plan argument(s): {', '.join(sorted(unexpected))}")


def _new_steps(
    existing: tuple[PlanStep, ...], steps: Sequence[str] | None
) -> tuple[PlanStep, ...]:
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
        raise PlanError("steps must be a list of description strings")
    if not steps:
        raise PlanError("steps must contain at least one description")
    if len(existing) + len(steps) > MAX_PLAN_STEPS:
        raise PlanError(f"plan exceeds the {MAX_PLAN_STEPS}-step limit")
    used_ids = {step.step_id for step in existing}
    next_ordinal = 1
    created: list[PlanStep] = []
    for value in steps:
        while f"step-{next_ordinal}" in used_ids:
            next_ordinal += 1
        step = PlanStep(f"step-{next_ordinal}", _validate_description(value))
        created.append(step)
        used_ids.add(step.step_id)
        next_ordinal += 1
    return tuple(created)


def _validate_description(value: object) -> str:
    if not isinstance(value, str):
        raise PlanError("plan step description must be a string")
    # Construction owns the remaining validation and supplies one consistent
    # error vocabulary for direct use and checkpoint decoding.
    return PlanStep("step-0", value).description


def _step_index(plan: AgentPlan, step_id: str | None) -> int:
    if not isinstance(step_id, str) or not step_id:
        raise PlanError("step_id must be a non-empty string")
    for index, step in enumerate(plan.steps):
        if step.step_id == step_id:
            return index
    raise PlanError(f"unknown plan step id {step_id!r}")


def _require_plan(plan: AgentPlan | None) -> AgentPlan:
    if plan is None:
        raise PlanError("no plan exists; create one first")
    if not isinstance(plan, AgentPlan):
        raise PlanError("plan must be an AgentPlan or None")
    return plan


def _validate_status_transition(
    plan: AgentPlan, step: PlanStep, status: PlanStatus
) -> None:
    allowed = {
        PlanStatus.NOT_STARTED: {PlanStatus.IN_PROGRESS, PlanStatus.ABANDONED},
        PlanStatus.IN_PROGRESS: {PlanStatus.DONE, PlanStatus.ABANDONED},
        PlanStatus.DONE: set(),
        PlanStatus.ABANDONED: set(),
    }
    if status not in allowed[step.status]:
        raise PlanError(
            f"invalid plan status transition {step.status.value} -> {status.value}"
        )
    if status is PlanStatus.IN_PROGRESS and plan.current_step is not None:
        raise PlanError(
            "cannot start a second step while "
            f"{plan.current_step.step_id!r} is in progress"
        )


def _replace_step(
    steps: tuple[PlanStep, ...], index: int, replacement: PlanStep
) -> tuple[PlanStep, ...]:
    return (*steps[:index], replacement, *steps[index + 1 :])
