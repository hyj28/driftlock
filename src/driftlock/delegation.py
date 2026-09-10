"""Bounded, checkpointable policy around one fresh sub-agent invocation."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
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
MAX_DELEGATION_ACCOUNTED_TOKENS = 1_000_000_000_000
_MAX_DELEGATION_LEDGER_TOKENS = 2 * MAX_DELEGATION_ACCOUNTED_TOKENS

# A checkpoint contains at most one compact record per admitted call. Keeping a
# separate bound on its serialized representation also rejects hostile nested
# JSON supplied through a tampered checkpoint before copying it.
_MAX_DELEGATION_RECORD_CHARACTERS = 4_000


@dataclass(slots=True)
class _DelegationUsage:
    tokens: int = 0
    closed: bool = False


_active_usage: ContextVar[_DelegationUsage | None] = ContextVar(
    "delegation_usage", default=None
)


def _report_delegation_tokens(tokens: int) -> None:
    """Retain provider-reported usage before tools or subsequent awaits run."""

    usage = _active_usage.get()
    if usage is not None and not usage.closed:
        usage.tokens += tokens


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
        if self.max_tokens_per_task > MAX_DELEGATION_ACCOUNTED_TOKENS:
            raise ValueError(
                "max_tokens_per_task exceeds the supported accounting limit"
            )
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
        if self.tokens > MAX_DELEGATION_ACCOUNTED_TOKENS:
            raise ValueError("delegation tokens exceed the supported accounting limit")
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
        self._cancelled_tasks: set[asyncio.Task[DelegationExecutionResult]] = set()
        self._delegate_lock = asyncio.Lock()

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
        self._require_idle()
        return {
            "schema_version": DELEGATION_SCHEMA_VERSION,
            "calls_used": self._calls_used,
            "tokens_used": self._tokens_used,
            "records": _copy_records(self._records),
        }

    def restore_checkpoint_state(self, value: object) -> None:
        self._require_idle()
        calls, tokens, records = _decode_state(value, self.config)
        self._calls_used = calls
        self._tokens_used = tokens
        self._records = records

    def validate_checkpoint_state(self, value: object) -> None:
        """Validate a checkpoint without mutating the live delegation ledger."""

        _decode_state(value, self.config)

    def _require_idle(self) -> None:
        if self._delegate_lock.locked():
            raise RuntimeError(
                "delegation is in progress; await it before checkpoint access"
            )

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
        # One ledger represents one sequential task. Serialize public callers so
        # admission, execution accounting, and the appended audit record remain
        # one atomic transition even if a host accidentally submits concurrently.
        async with self._delegate_lock:
            return await self._delegate_serialized(
                objective,
                context,
                parent_goal=parent_goal,
                sequence=sequence,
                logical_step=logical_step,
                attempt=attempt,
                parent_tokens_remaining=parent_tokens_remaining,
                max_observation_characters=max_observation_characters,
            )

    async def _delegate_serialized(
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
        if any(not record["tokens"]["accounting_known"] for record in self._records):
            error = (
                error
                or "delegation is paused because earlier token accounting is incomplete"
            )
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
        usage = _DelegationUsage()

        async def execute() -> DelegationExecutionResult:
            context_token = _active_usage.set(usage)
            try:
                return await self.executor(request)
            finally:
                _active_usage.reset(context_token)

        execution_task = asyncio.create_task(execute())
        try:
            done, _pending = await asyncio.wait(
                {execution_task}, timeout=self.config.timeout_seconds
            )
        except BaseException:
            execution_task.cancel()
            self._retain_cancelled_task(execution_task)
            self._record(
                self._interrupted_outcome(
                    usage,
                    DelegationStatus.FAILED,
                    normalized_objective,
                    before_calls,
                    before_tokens,
                    "delegation was interrupted by its caller",
                )
            )
            raise
        if not done:
            execution_task.cancel()
            self._retain_cancelled_task(execution_task)
            outcome = self._interrupted_outcome(
                usage,
                DelegationStatus.TIMED_OUT,
                normalized_objective,
                before_calls,
                before_tokens,
                error=(
                    "delegation timed out after "
                    f"{self.config.timeout_seconds:g} seconds"
                ),
            )
        else:
            try:
                execution = execution_task.result()
            except asyncio.CancelledError:
                outcome = self._interrupted_outcome(
                    usage,
                    DelegationStatus.FAILED,
                    normalized_objective,
                    before_calls,
                    before_tokens,
                    error="delegation executor cancelled itself",
                )
            except Exception as exception:
                outcome = self._interrupted_outcome(
                    usage,
                    DelegationStatus.FAILED,
                    normalized_objective,
                    before_calls,
                    before_tokens,
                    error=(
                        "delegation executor raised "
                        f"{type(exception).__name__}: {_safe_str(exception)}"
                    ),
                )
            else:
                if not isinstance(execution, DelegationExecutionResult):
                    outcome = self._interrupted_outcome(
                        usage,
                        DelegationStatus.FAILED,
                        normalized_objective,
                        before_calls,
                        before_tokens,
                        error="delegation executor returned an invalid result",
                    )
                else:
                    contract_error = _execution_contract_error(execution, request)
                    self._tokens_used += execution.tokens
                    if contract_error is not None:
                        outcome = self._outcome(
                            DelegationStatus.FAILED,
                            normalized_objective,
                            before_calls,
                            before_tokens,
                            tokens=execution.tokens,
                            error=contract_error,
                            executor_invoked=True,
                        )
                    else:
                        effective_status = execution.status
                        effective_error = execution.error
                        if execution.tokens > request.max_tokens:
                            effective_status = DelegationStatus.TOKEN_LIMIT
                            effective_error = (
                                "delegated child exceeded its assigned token budget"
                            )
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
                                    f"{self.config.max_result_characters}-character "
                                    "limit"
                                ),
                                executor_invoked=True,
                            )
                        else:
                            outcome = self._outcome(
                                effective_status,
                                normalized_objective,
                                before_calls,
                                before_tokens,
                                tokens=execution.tokens,
                                steps=execution.steps,
                                output=execution.output,
                                output_character_count=len(execution.output),
                                output_sha256=output_sha256,
                                error=effective_error,
                                executor_invoked=True,
                            )
        usage.closed = True
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

        self._require_idle()
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

    def _interrupted_outcome(
        self,
        usage: _DelegationUsage,
        status: DelegationStatus,
        objective: str,
        calls_before: int,
        tokens_before: int,
        error: str,
    ) -> DelegationOutcome:
        usage.closed = True
        self._tokens_used += usage.tokens
        return self._outcome(
            status,
            objective,
            calls_before,
            tokens_before,
            tokens=usage.tokens,
            error=error,
            executor_invoked=True,
            token_accounting_known=False,
        )

    def _retain_cancelled_task(
        self, task: asyncio.Task[DelegationExecutionResult]
    ) -> None:
        self._cancelled_tasks.add(task)

        def discard(completed: asyncio.Task[DelegationExecutionResult]) -> None:
            self._cancelled_tasks.discard(completed)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                completed.exception()

        task.add_done_callback(discard)

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


def _safe_str(value: object) -> str:
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__} with unavailable text>"


def _record_from_outcome(outcome: DelegationOutcome) -> dict[str, Any]:
    report = outcome.to_report()
    output = report["output"]
    assert isinstance(output, dict)
    output.pop("content", None)
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
        or not 0 <= tokens <= _MAX_DELEGATION_LEDGER_TOKENS
    ):
        raise ValueError("delegation checkpoint token count is invalid")
    if not isinstance(records, list) or len(records) != calls:
        raise ValueError("delegation checkpoint records are malformed")
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError("delegation checkpoint record is malformed")
    expected_tokens = 0
    accounting_incomplete = False
    for index, record in enumerate(records, 1):
        _validate_checkpoint_record(record, index=index, config=config)
        if accounting_incomplete and record["executor_invoked"]:
            raise ValueError(
                "delegation checkpoint executes after incomplete accounting"
            )
        token_report = record["tokens"]
        assert isinstance(token_report, Mapping)
        if token_report["before"] != expected_tokens:
            raise ValueError("delegation checkpoint token history is inconsistent")
        expected_tokens = token_report["after"]
        accounting_incomplete |= not token_report["accounting_known"]
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
        "output",
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
        or contributed > MAX_DELEGATION_ACCOUNTED_TOKENS
        or after != before + contributed
        or after > _MAX_DELEGATION_LEDGER_TOKENS
        or not isinstance(token_report.get("accounting_known"), bool)
    ):
        raise ValueError("delegation checkpoint token report is invalid")
    steps = record.get("steps")
    if not _is_nonnegative_int(steps) or steps > config.max_steps_per_call:
        raise ValueError("delegation checkpoint step count is invalid")
    if not isinstance(record.get("executor_invoked"), bool):
        raise ValueError("delegation checkpoint executor flag is invalid")
    output = record.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "character_count",
        "sha256",
        "included",
    }:
        raise ValueError("delegation checkpoint output evidence is malformed")
    output_characters = output.get("character_count")
    output_sha256 = output.get("sha256")
    output_included = output.get("included")
    if (
        not _is_nonnegative_int(output_characters)
        or not isinstance(output_included, bool)
        or (output_sha256 is not None and not _is_sha256(output_sha256))
    ):
        raise ValueError("delegation checkpoint output evidence is invalid")
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
    output_absent = (
        not output_included and output_sha256 is None and output_characters == 0
    )
    output_retained = (
        output_included
        and output_sha256 is not None
        and output_characters <= config.max_result_characters
    )
    if status is DelegationStatus.REJECTED:
        if (
            record["executor_invoked"]
            or not token_report["accounting_known"]
            or contributed
            or steps
            or not output_absent
        ):
            raise ValueError("rejected delegation checkpoint record is inconsistent")
    elif not record["executor_invoked"]:
        raise ValueError("executed delegation checkpoint record lacks invocation")
    if record["executor_invoked"] and before >= config.max_tokens_per_task:
        raise ValueError("delegation checkpoint executes after exhausted token budget")
    if status in {DelegationStatus.COMPLETED, DelegationStatus.STEP_LIMIT} and (
        contributed
        > min(config.max_tokens_per_call, config.max_tokens_per_task - before)
    ):
        raise ValueError("delegation checkpoint status contradicts its token budget")
    if status is DelegationStatus.COMPLETED and (
        not token_report["accounting_known"] or not output_retained
    ):
        raise ValueError("completed delegation checkpoint record is inconsistent")
    if status is DelegationStatus.TIMED_OUT and (
        token_report["accounting_known"] or steps or not output_absent
    ):
        raise ValueError("timed-out delegation checkpoint record is inconsistent")
    if status is DelegationStatus.RESULT_TOO_LARGE and (
        not token_report["accounting_known"] or output_included or output_sha256 is None
    ):
        raise ValueError("oversized delegation checkpoint record is inconsistent")
    if status in {DelegationStatus.STEP_LIMIT, DelegationStatus.TOKEN_LIMIT} and (
        not token_report["accounting_known"] or not output_retained
    ):
        raise ValueError("limited delegation checkpoint record is inconsistent")
    if status is DelegationStatus.FAILED:
        known = token_report["accounting_known"]
        if (not known and (steps or not output_absent)) or (
            known and not (output_absent or output_retained)
        ):
            raise ValueError("failed delegation checkpoint record is inconsistent")
    if status is not DelegationStatus.REJECTED and (
        objective["character_count"] > config.max_objective_characters
    ):
        raise ValueError("executed delegation objective exceeds the configured limit")


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
