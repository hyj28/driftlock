"""Bounded, checkpointable policy around one fresh sub-agent invocation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

DELEGATION_SCHEMA_VERSION = 1
DEFAULT_MAX_DELEGATIONS_PER_TASK = 4
DEFAULT_MAX_DELEGATION_OBJECTIVE_CHARACTERS = 2_000
DEFAULT_MAX_DELEGATION_CONTEXT_CHARACTERS = 4_000
DEFAULT_MAX_DELEGATION_PARENT_GOAL_CHARACTERS = 4_000
DEFAULT_MAX_DELEGATION_RESULT_CHARACTERS = 16_000
DEFAULT_MAX_DELEGATION_STEPS = 8
DEFAULT_MAX_DELEGATION_TOKENS_PER_CALL = 32_000
DEFAULT_MAX_DELEGATION_TOKENS_PER_TASK = 64_000
DEFAULT_DELEGATION_TIMEOUT_SECONDS = 300
DEFAULT_MAX_DELEGATION_ERROR_CHARACTERS = 1_000

# A checkpoint contains at most one compact record per admitted call. Keeping a
# separate bound on its serialized representation also rejects hostile nested
# JSON supplied through a tampered checkpoint before copying it.
_MAX_DELEGATION_RECORD_CHARACTERS = 4_000


class DelegationStatus(StrEnum):
    """Stable outcomes visible to the parent agent and audit log."""

    COMPLETED = "completed"
    FAILED = "failed"
    STEP_LIMIT = "step_limit"
    TOKEN_LIMIT = "token_limit"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"
    RESULT_TOO_LARGE = "result_too_large"


@dataclass(frozen=True, slots=True)
class DelegationConfig:
    """Independent bounds for delegation input, execution, and retained output."""

    max_delegations_per_task: int = DEFAULT_MAX_DELEGATIONS_PER_TASK
    max_objective_characters: int = DEFAULT_MAX_DELEGATION_OBJECTIVE_CHARACTERS
    max_context_characters: int = DEFAULT_MAX_DELEGATION_CONTEXT_CHARACTERS
    max_parent_goal_characters: int = DEFAULT_MAX_DELEGATION_PARENT_GOAL_CHARACTERS
    max_result_characters: int = DEFAULT_MAX_DELEGATION_RESULT_CHARACTERS
    max_steps_per_call: int = DEFAULT_MAX_DELEGATION_STEPS
    max_tokens_per_call: int = DEFAULT_MAX_DELEGATION_TOKENS_PER_CALL
    max_tokens_per_task: int = DEFAULT_MAX_DELEGATION_TOKENS_PER_TASK
    timeout_seconds: float = DEFAULT_DELEGATION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        for name in (
            "max_delegations_per_task",
            "max_objective_characters",
            "max_context_characters",
            "max_parent_goal_characters",
            "max_result_characters",
            "max_steps_per_call",
            "max_tokens_per_call",
            "max_tokens_per_task",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_tokens_per_call > self.max_tokens_per_task:
            raise ValueError("max_tokens_per_call cannot exceed max_tokens_per_task")
        timeout = self.timeout_seconds
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or (isinstance(timeout, float) and not math.isfinite(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class DelegationRequest:
    """Host-facing request for one non-recursive child-agent run."""

    objective: str
    context: str
    parent_goal: str
    sequence: int
    logical_step: int
    attempt: int
    max_steps: int
    max_tokens: int

    def __post_init__(self) -> None:
        for name in ("objective", "context", "parent_goal"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"delegation request {name} must be text")
        if not self.objective:
            raise ValueError("delegation request objective must be non-empty")
        for name in (
            "sequence",
            "logical_step",
            "attempt",
            "max_steps",
            "max_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"delegation request {name} must be a positive integer"
                )


@dataclass(frozen=True, slots=True)
class DelegationExecutionResult:
    """Result returned by a child executor before parent policy is applied."""

    status: DelegationStatus
    output: str
    tokens: int
    steps: int
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, DelegationStatus):
            raise TypeError("delegation status must be a DelegationStatus")
        if self.status not in {
            DelegationStatus.COMPLETED,
            DelegationStatus.FAILED,
            DelegationStatus.STEP_LIMIT,
            DelegationStatus.TOKEN_LIMIT,
        }:
            raise ValueError("executor returned a non-execution delegation status")
        if not isinstance(self.output, str):
            raise TypeError("delegation output must be text")
        for name in ("tokens", "steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"delegation {name} must be a non-negative integer")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("delegation error must be text or None")
        if self.status is DelegationStatus.COMPLETED and self.error is not None:
            raise ValueError("completed delegation cannot contain an error")
        if self.status is not DelegationStatus.COMPLETED and not self.error:
            raise ValueError("incomplete delegation requires an error")


class DelegationExecutor(Protocol):
    async def __call__(
        self, request: DelegationRequest
    ) -> DelegationExecutionResult: ...


@dataclass(frozen=True, slots=True)
class DelegationOutcome:
    """One bounded parent-facing delegation observation."""

    status: DelegationStatus
    objective_sha256: str
    objective_character_count: int
    calls_before: int
    calls_after: int
    tokens_before: int
    tokens_contributed: int
    tokens_after: int
    steps: int = 0
    output: str | None = None
    output_character_count: int = 0
    output_sha256: str | None = None
    error: str | None = None
    executor_invoked: bool = False
    token_accounting_known: bool = True

    def to_report(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": DELEGATION_SCHEMA_VERSION,
            "mode": "sub-agent-delegation",
            "status": self.status.value,
            "objective": {
                "sha256": self.objective_sha256,
                "character_count": self.objective_character_count,
            },
            "calls": {"before": self.calls_before, "after": self.calls_after},
            "tokens": {
                "before": self.tokens_before,
                "contributed": self.tokens_contributed,
                "after": self.tokens_after,
                "accounting_known": self.token_accounting_known,
            },
            "steps": self.steps,
            "executor_invoked": self.executor_invoked,
            "output": {
                "character_count": self.output_character_count,
                "sha256": self.output_sha256,
                "included": self.output is not None,
            },
        }
        if self.output is not None:
            result["output"]["content"] = self.output
        if self.error is not None:
            result["error"] = self.error
        return result

    def to_observation(self, *, max_characters: int | None = None) -> str:
        rendered = json.dumps(
            self.to_report(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        if max_characters is None or len(rendered) <= max_characters:
            return rendered
        if (
            not isinstance(max_characters, int)
            or isinstance(max_characters, bool)
            or max_characters <= 0
        ):
            raise ValueError("max_characters must be a positive integer or None")
        compact = {
            "schema_version": DELEGATION_SCHEMA_VERSION,
            "mode": "sub-agent-delegation",
            "status": self.status.value,
            "objective_sha256": self.objective_sha256,
            "tokens_contributed": self.tokens_contributed,
            "output_sha256": self.output_sha256,
            "details_omitted": {
                "character_count": len(rendered),
                "sha256": _sha256(rendered),
            },
        }
        compact_rendered = json.dumps(
            compact, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        if len(compact_rendered) > max_characters:
            raise ValueError("max_characters is too small for delegation metadata")
        return compact_rendered


DelegationExecutorCallable = Callable[
    [DelegationRequest], Awaitable[DelegationExecutionResult]
]


class DelegationTool:
    """Apply deterministic limits and checkpointed quota to an executor."""

    def __init__(
        self,
        executor: DelegationExecutorCallable,
        *,
        config: DelegationConfig | None = None,
    ) -> None:
        if not callable(executor):
            raise TypeError("delegation executor must be callable")
        self.executor = executor
        self.config = config or DelegationConfig()
        if not isinstance(self.config, DelegationConfig):
            raise TypeError("config must be a DelegationConfig")
        self._calls_used = 0
        self._tokens_used = 0
        self._records: list[dict[str, Any]] = []

    @property
    def calls_used(self) -> int:
        return self._calls_used

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(_copy_records(self._records))

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": DELEGATION_SCHEMA_VERSION,
            "calls_used": self._calls_used,
            "tokens_used": self._tokens_used,
            "records": _copy_records(self._records),
        }

    def restore_checkpoint_state(self, value: object) -> None:
        calls, tokens, records = _decode_state(value, self.config)
        self._calls_used = calls
        self._tokens_used = tokens
        self._records = records

    def validate_checkpoint_state(self, value: object) -> None:
        """Validate a checkpoint without mutating the live delegation ledger."""

        _decode_state(value, self.config)

    async def delegate(
        self,
        objective: object,
        context: object,
        *,
        parent_goal: str,
        sequence: int,
        logical_step: int,
        attempt: int,
        parent_tokens_remaining: int | None,
        max_observation_characters: int,
    ) -> DelegationOutcome:
        before_calls = self._calls_used
        before_tokens = self._tokens_used
        validation_error = _validate_invocation_metadata(
            parent_goal=parent_goal,
            sequence=sequence,
            logical_step=logical_step,
            attempt=attempt,
            parent_tokens_remaining=parent_tokens_remaining,
            max_observation_characters=max_observation_characters,
        )
        if self._calls_used >= self.config.max_delegations_per_task:
            return self._outcome(
                DelegationStatus.REJECTED,
                _digestable_text(objective),
                before_calls,
                before_tokens,
                error="delegation call limit is exhausted",
            )
        # Every admitted delegate_task invocation consumes one call, including a
        # malformed request. This both makes attempts auditable and bounds the
        # checkpoint ledger under repeated bad model output.
        self._calls_used += 1
        normalized_objective, error = _bounded_text(
            objective,
            name="objective",
            limit=self.config.max_objective_characters,
            allow_empty=False,
        )
        normalized_context, context_error = _bounded_text(
            context,
            name="context",
            limit=self.config.max_context_characters,
            allow_empty=True,
        )
        normalized_parent_goal, parent_goal_error = _bounded_text(
            parent_goal,
            name="parent goal",
            limit=self.config.max_parent_goal_characters,
            allow_empty=True,
        )
        error = validation_error or error or context_error or parent_goal_error
        digest_source = (
            normalized_objective
            if isinstance(objective, str)
            else _digestable_text(objective)
        )
        if error is not None:
            outcome = self._outcome(
                DelegationStatus.REJECTED,
                digest_source,
                before_calls,
                before_tokens,
                error=error,
            )
            return self._record(outcome)
        assert isinstance(normalized_objective, str)
        assert isinstance(normalized_context, str)
        assert isinstance(normalized_parent_goal, str)
        remaining = self.config.max_tokens_per_task - self._tokens_used
        if parent_tokens_remaining is not None:
            remaining = min(remaining, parent_tokens_remaining)
        if remaining <= 0:
            outcome = self._outcome(
                DelegationStatus.REJECTED,
                normalized_objective,
                before_calls,
                before_tokens,
                error="delegation token budget is exhausted",
            )
            return self._record(outcome)
        request = DelegationRequest(
            objective=normalized_objective,
            context=normalized_context,
            parent_goal=normalized_parent_goal,
            sequence=sequence,
            logical_step=logical_step,
            attempt=attempt,
            max_steps=self.config.max_steps_per_call,
            max_tokens=min(self.config.max_tokens_per_call, remaining),
        )
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                execution = await self.executor(request)
        except TimeoutError:
            outcome = self._outcome(
                DelegationStatus.TIMED_OUT,
                normalized_objective,
                before_calls,
                before_tokens,
                error=(
                    "delegation timed out after "
                    f"{self.config.timeout_seconds:g} seconds"
                ),
                executor_invoked=True,
                token_accounting_known=False,
            )
        except Exception as exception:
            outcome = self._outcome(
                DelegationStatus.FAILED,
                normalized_objective,
                before_calls,
                before_tokens,
                error=(
                    "delegation executor raised "
                    f"{type(exception).__name__}: {exception}"
                ),
                executor_invoked=True,
                token_accounting_known=False,
            )
        else:
            if not isinstance(execution, DelegationExecutionResult):
                outcome = self._outcome(
                    DelegationStatus.FAILED,
                    normalized_objective,
                    before_calls,
                    before_tokens,
                    error="delegation executor returned an invalid result",
                    executor_invoked=True,
                    token_accounting_known=False,
                )
            else:
                contract_error = _execution_contract_error(execution, request)
                if contract_error is not None:
                    outcome = self._outcome(
                        DelegationStatus.FAILED,
                        normalized_objective,
                        before_calls,
                        before_tokens,
                        error=contract_error,
                        executor_invoked=True,
                        token_accounting_known=False,
                    )
                else:
                    self._tokens_used += execution.tokens
                    output_sha256 = _sha256(execution.output)
                    if len(execution.output) > self.config.max_result_characters:
                        outcome = self._outcome(
                            DelegationStatus.RESULT_TOO_LARGE,
                            normalized_objective,
                            before_calls,
                            before_tokens,
                            tokens=execution.tokens,
                            steps=execution.steps,
                            output_character_count=len(execution.output),
                            output_sha256=output_sha256,
                            error=(
                                "delegation output exceeds the "
                                f"{self.config.max_result_characters}-character limit"
                            ),
                            executor_invoked=True,
                        )
                    else:
                        outcome = self._outcome(
                            execution.status,
                            normalized_objective,
                            before_calls,
                            before_tokens,
                            tokens=execution.tokens,
                            steps=execution.steps,
                            output=execution.output,
                            output_character_count=len(execution.output),
                            output_sha256=output_sha256,
                            error=execution.error,
                            executor_invoked=True,
                        )
        if (
            outcome.output is not None
            and len(outcome.to_observation()) > max_observation_characters
        ):
            outcome = replace(
                outcome,
                status=DelegationStatus.RESULT_TOO_LARGE,
                output=None,
                error=(
                    "delegation result does not fit the parent tool-observation "
                    f"limit ({max_observation_characters})"
                ),
            )
        return self._record(outcome)

    def record_rejected_attempt(
        self, objective: object, error: object
    ) -> DelegationOutcome:
        """Audit malformed tool arguments without invoking the child executor."""

        before_calls = self._calls_used
        before_tokens = self._tokens_used
        if self._calls_used >= self.config.max_delegations_per_task:
            return self._outcome(
                DelegationStatus.REJECTED,
                _digestable_text(objective),
                before_calls,
                before_tokens,
                error="delegation call limit is exhausted",
            )
        self._calls_used += 1
        outcome = self._outcome(
            DelegationStatus.REJECTED,
            _digestable_text(objective),
            before_calls,
            before_tokens,
            error=_bounded_error(error),
        )
        return self._record(outcome)

    def _record(self, outcome: DelegationOutcome) -> DelegationOutcome:
        self._records.append(_record_from_outcome(outcome))
        return outcome

    def _outcome(
        self,
        status: DelegationStatus,
        objective: str,
        calls_before: int,
        tokens_before: int,
        *,
        tokens: int = 0,
        steps: int = 0,
        output: str | None = None,
        output_character_count: int = 0,
        output_sha256: str | None = None,
        error: str | None = None,
        executor_invoked: bool = False,
        token_accounting_known: bool = True,
    ) -> DelegationOutcome:
        return DelegationOutcome(
            status=status,
            objective_sha256=_sha256(objective),
            objective_character_count=len(objective),
            calls_before=calls_before,
            calls_after=self._calls_used,
            tokens_before=tokens_before,
            tokens_contributed=tokens,
            tokens_after=self._tokens_used,
            steps=steps,
            output=output,
            output_character_count=output_character_count,
            output_sha256=output_sha256,
            error=_bounded_error(error) if error is not None else None,
            executor_invoked=executor_invoked,
            token_accounting_known=token_accounting_known,
        )


def _bounded_text(
    value: object, *, name: str, limit: int, allow_empty: bool
) -> tuple[str, str | None]:
    if not isinstance(value, str):
        return "", f"delegation {name} must be text"
    normalized = value.strip()
    if not normalized and not allow_empty:
        return normalized, f"delegation {name} must be non-empty text"
    if len(normalized) > limit:
        return normalized, f"delegation {name} exceeds the {limit}-character limit"
    if "\x00" in normalized:
        return normalized, f"delegation {name} contains a null character"
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError:
        return normalized, f"delegation {name} must be valid UTF-8 text"
    return normalized, None


def _validate_invocation_metadata(
    *,
    parent_goal: object,
    sequence: object,
    logical_step: object,
    attempt: object,
    parent_tokens_remaining: object,
    max_observation_characters: object,
) -> str | None:
    if not isinstance(parent_goal, str):
        return "delegation parent goal must be text"
    for name, value in (
        ("sequence", sequence),
        ("logical_step", logical_step),
        ("attempt", attempt),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return f"delegation {name} must be a positive integer"
    if parent_tokens_remaining is not None and (
        not isinstance(parent_tokens_remaining, int)
        or isinstance(parent_tokens_remaining, bool)
        or parent_tokens_remaining < 0
    ):
        return "delegation parent token budget must be a non-negative integer or None"
    if (
        not isinstance(max_observation_characters, int)
        or isinstance(max_observation_characters, bool)
        or max_observation_characters <= 0
    ):
        return "delegation observation limit must be a positive integer"
    return None


def _execution_contract_error(
    execution: DelegationExecutionResult, request: DelegationRequest
) -> str | None:
    if execution.tokens > request.max_tokens:
        return "delegation executor reported tokens above its assigned limit"
    if execution.steps > request.max_steps:
        return "delegation executor reported steps above its assigned limit"
    return None


def _bounded_error(value: object) -> str:
    rendered = value if isinstance(value, str) else _safe_repr(value)
    if not rendered:
        rendered = "delegation request was rejected"
    if len(rendered) <= DEFAULT_MAX_DELEGATION_ERROR_CHARACTERS:
        return rendered
    marker = (
        f"... [truncated; character_count={len(rendered)},sha256={_sha256(rendered)}]"
    )
    retained = DEFAULT_MAX_DELEGATION_ERROR_CHARACTERS - len(marker)
    return rendered[:retained] + marker


def _digestable_text(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except (RecursionError, TypeError, ValueError):
        return _safe_repr(value)


def _safe_repr(value: object) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__} with unavailable representation>"


def _record_from_outcome(outcome: DelegationOutcome) -> dict[str, Any]:
    report = outcome.to_report()
    report.pop("output", None)
    return report


def _copy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(records, ensure_ascii=True, separators=(",", ":")))


def _decode_state(
    value: object, config: DelegationConfig
) -> tuple[int, int, list[dict[str, Any]]]:
    expected = {"schema_version", "calls_used", "tokens_used", "records"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("delegation checkpoint state fields are malformed")
    if value.get("schema_version") != DELEGATION_SCHEMA_VERSION:
        raise ValueError("unsupported delegation checkpoint schema version")
    calls = value.get("calls_used")
    tokens = value.get("tokens_used")
    records = value.get("records")
    if (
        not isinstance(calls, int)
        or isinstance(calls, bool)
        or not 0 <= calls <= config.max_delegations_per_task
    ):
        raise ValueError("delegation checkpoint call count is invalid")
    if (
        not isinstance(tokens, int)
        or isinstance(tokens, bool)
        or not 0 <= tokens <= config.max_tokens_per_task
    ):
        raise ValueError("delegation checkpoint token count is invalid")
    if not isinstance(records, list) or len(records) != calls:
        raise ValueError("delegation checkpoint records are malformed")
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError("delegation checkpoint record is malformed")
    expected_tokens = 0
    for index, record in enumerate(records, 1):
        _validate_checkpoint_record(record, index=index, config=config)
        token_report = record["tokens"]
        assert isinstance(token_report, Mapping)
        if token_report["before"] != expected_tokens:
            raise ValueError("delegation checkpoint token history is inconsistent")
        expected_tokens = token_report["after"]
    if expected_tokens != tokens:
        raise ValueError("delegation checkpoint token total is inconsistent")
    try:
        serialized = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    except (RecursionError, TypeError, ValueError) as error:
        raise ValueError(
            "delegation checkpoint records must be JSON-compatible"
        ) from error
    if (
        len(serialized)
        > config.max_delegations_per_task * _MAX_DELEGATION_RECORD_CHARACTERS
    ):
        raise ValueError("delegation checkpoint records exceed the serialized limit")
    copied = json.loads(serialized)
    return calls, tokens, copied


def _validate_checkpoint_record(
    record: Mapping[str, Any], *, index: int, config: DelegationConfig
) -> None:
    required = {
        "schema_version",
        "mode",
        "status",
        "objective",
        "calls",
        "tokens",
        "steps",
        "executor_invoked",
    }
    if set(record) not in (required, required | {"error"}):
        raise ValueError("delegation checkpoint record fields are malformed")
    if record.get("schema_version") != DELEGATION_SCHEMA_VERSION:
        raise ValueError("delegation checkpoint record schema is invalid")
    if record.get("mode") != "sub-agent-delegation":
        raise ValueError("delegation checkpoint record mode is invalid")
    try:
        status = DelegationStatus(record.get("status"))
    except (TypeError, ValueError):
        raise ValueError("delegation checkpoint record status is invalid") from None
    objective = record.get("objective")
    if not isinstance(objective, Mapping) or set(objective) != {
        "sha256",
        "character_count",
    }:
        raise ValueError("delegation checkpoint objective is malformed")
    if not _is_sha256(objective.get("sha256")) or not _is_nonnegative_int(
        objective.get("character_count")
    ):
        raise ValueError("delegation checkpoint objective is invalid")
    calls = record.get("calls")
    if (
        not isinstance(calls, Mapping)
        or set(calls) != {"before", "after"}
        or calls.get("before") != index - 1
        or calls.get("after") != index
    ):
        raise ValueError("delegation checkpoint record call counts are invalid")
    token_report = record.get("tokens")
    if not isinstance(token_report, Mapping) or set(token_report) != {
        "before",
        "contributed",
        "after",
        "accounting_known",
    }:
        raise ValueError("delegation checkpoint token report is malformed")
    before = token_report.get("before")
    contributed = token_report.get("contributed")
    after = token_report.get("after")
    if (
        not _is_nonnegative_int(before)
        or not _is_nonnegative_int(contributed)
        or not _is_nonnegative_int(after)
        or after != before + contributed
        or after > config.max_tokens_per_task
        or not isinstance(token_report.get("accounting_known"), bool)
    ):
        raise ValueError("delegation checkpoint token report is invalid")
    steps = record.get("steps")
    if not _is_nonnegative_int(steps) or steps > config.max_steps_per_call:
        raise ValueError("delegation checkpoint step count is invalid")
    if not isinstance(record.get("executor_invoked"), bool):
        raise ValueError("delegation checkpoint executor flag is invalid")
    error = record.get("error")
    if error is not None and (
        not isinstance(error, str)
        or not error
        or len(error) > DEFAULT_MAX_DELEGATION_ERROR_CHARACTERS
    ):
        raise ValueError("delegation checkpoint error is invalid")
    if status is DelegationStatus.COMPLETED and error is not None:
        raise ValueError("completed delegation checkpoint record has an error")
    if status is not DelegationStatus.COMPLETED and not isinstance(error, str):
        raise ValueError("incomplete delegation checkpoint record lacks an error")
    if not token_report["accounting_known"] and contributed != 0:
        raise ValueError("unknown delegation token accounting cannot contribute tokens")


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()
