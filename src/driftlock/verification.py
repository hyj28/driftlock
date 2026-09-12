"""Bounded, replayable records for evidence-based completion verification."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# Three attempts let an agent repair two independently refuted completion claims
# while placing a hard ceiling on retry cost.
DEFAULT_MAX_VERIFICATION_ATTEMPTS = 3

# A small response is enough to select one command or explain why none exists;
# larger generations would spend measured task budget on prose rather than evidence.
DEFAULT_VERIFICATION_MAX_OUTPUT_TOKENS = 256

# This floor avoids issuing provider calls too small to encode a usable tool call.
DEFAULT_VERIFICATION_MIN_OUTPUT_TOKENS = 32

# Two thousand forty-eight tokens cover the conservative request framing plus a
# short check-selection response while fixing the total verification share per run.
DEFAULT_MAX_VERIFICATION_TOKENS = 2_048

# Eight is a defensive public configuration ceiling that also bounds every durable
# per-attempt record without requiring lossy aggregation inside one task.
MAX_VERIFICATION_ATTEMPTS = 8

# Commands longer than this cease to be auditable completion checks and can crowd
# durable state even when the resulting output is short.
MAX_VERIFICATION_COMMAND_CHARACTERS = 2_000

# Four thousand characters retain useful failure diagnostics while preventing test
# output from becoming another unbounded durable log.
MAX_VERIFICATION_EVIDENCE_CHARACTERS = 4_000

# Reasons are decision summaries, so 512 characters preserve actionable context
# without duplicating the separately retained command evidence.
MAX_VERIFICATION_REASON_CHARACTERS = 512

# Version three records both current-workspace executions and restoration failure,
# making interchangeability and infrastructure loss replayable facts.
VERIFICATION_CHECKPOINT_SCHEMA_VERSION = 3


class VerificationStatus(StrEnum):
    """What an evidence-based check established about a completion claim."""

    VERIFIED = "verified"
    REFUTED = "refuted"
    UNVERIFIABLE = "unverifiable"
    TRANSIENT_ERROR = "transient_error"
    RESTORATION_FAILED = "restoration_failed"
    MALFORMED = "malformed"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True, slots=True)
class SelfVerificationConfig:
    """Opt into bounded model-selected, command-decided completion checks.

    The model chooses a falsifiable command, but never chooses whether that command
    passed. Verification requires the same command to pass on two restored current
    workspaces and fail against the initial one. Disagreeing current runs are not
    interchangeable and cannot decide the claim. Exit one on both current runs
    refutes; other execution failures are retryable. An explicitly uncheckable goal
    terminates with an ``UNVERIFIABLE`` record, never a synthetic pass or failure.
    """

    max_attempts: int = DEFAULT_MAX_VERIFICATION_ATTEMPTS
    max_output_tokens: int = DEFAULT_VERIFICATION_MAX_OUTPUT_TOKENS
    min_output_tokens: int = DEFAULT_VERIFICATION_MIN_OUTPUT_TOKENS
    max_tokens: int = DEFAULT_MAX_VERIFICATION_TOKENS

    def __post_init__(self) -> None:
        for name in (
            "max_attempts",
            "max_output_tokens",
            "min_output_tokens",
            "max_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_attempts > MAX_VERIFICATION_ATTEMPTS:
            raise ValueError(f"max_attempts cannot exceed {MAX_VERIFICATION_ATTEMPTS}")
        if self.min_output_tokens > self.max_output_tokens:
            raise ValueError("min_output_tokens cannot exceed max_output_tokens")
        if self.max_output_tokens > self.max_tokens:
            raise ValueError("max_output_tokens cannot exceed max_tokens")


VerificationControl = Callable[
    [Mapping[str, Any], int, Callable[[], Awaitable[Any]]],
    Awaitable[tuple[Any, Any, Any]],
]


class VerificationControlError(RuntimeError):
    """A control-boundary failure with command accounting preserved."""

    def __init__(
        self,
        message: str,
        *,
        commands_run: int,
        commands_failed: int,
    ) -> None:
        super().__init__(message)
        self.commands_run = commands_run
        self.commands_failed = commands_failed


class VerificationRestorationError(VerificationControlError):
    """The runner could not restore the protected current workspace."""


class VerificationCommandError(VerificationControlError):
    """A verification command attempt failed before returning a result."""


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    """One bounded completion-verification attempt and its retained evidence."""

    attempt: int
    status: VerificationStatus
    reason: str
    command: str | None = None
    return_code: int | None = None
    control_return_code: int | None = None
    confirmation_return_code: int | None = None
    evidence: str = ""
    evidence_truncated: bool = False
    tokens: int = 0
    attempt_limit_reached: bool = False
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool):
            raise TypeError("attempt must be an integer")
        if self.attempt <= 0 or self.attempt > MAX_VERIFICATION_ATTEMPTS:
            raise ValueError("attempt is outside the supported bounded range")
        if not isinstance(self.status, VerificationStatus):
            raise TypeError("status must be a VerificationStatus")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if len(self.reason) > MAX_VERIFICATION_REASON_CHARACTERS:
            raise ValueError("reason exceeds its character bound")
        if self.command is not None:
            if not isinstance(self.command, str) or not self.command:
                raise ValueError("command must be a non-empty string or None")
            if len(self.command) > MAX_VERIFICATION_COMMAND_CHARACTERS:
                raise ValueError("command exceeds its character bound")
        if self.return_code is not None and (
            not isinstance(self.return_code, int) or isinstance(self.return_code, bool)
        ):
            raise TypeError("return_code must be an integer or None")
        if self.control_return_code is not None and (
            not isinstance(self.control_return_code, int)
            or isinstance(self.control_return_code, bool)
        ):
            raise TypeError("control_return_code must be an integer or None")
        if self.confirmation_return_code is not None and (
            not isinstance(self.confirmation_return_code, int)
            or isinstance(self.confirmation_return_code, bool)
        ):
            raise TypeError("confirmation_return_code must be an integer or None")
        if not isinstance(self.evidence, str):
            raise TypeError("evidence must be a string")
        if len(self.evidence) > MAX_VERIFICATION_EVIDENCE_CHARACTERS:
            raise ValueError("evidence exceeds its character bound")
        if not isinstance(self.evidence_truncated, bool):
            raise TypeError("evidence_truncated must be a boolean")
        if not isinstance(self.tokens, int) or isinstance(self.tokens, bool):
            raise TypeError("tokens must be an integer")
        if self.tokens < 0:
            raise ValueError("tokens cannot be negative")
        if not isinstance(self.attempt_limit_reached, bool):
            raise TypeError("attempt_limit_reached must be a boolean")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean")
        if self.attempt_limit_reached and not self.retryable:
            raise ValueError("only retryable records may exhaust the attempt limit")
        if (
            self.status
            in {
                VerificationStatus.REFUTED,
                VerificationStatus.TRANSIENT_ERROR,
                VerificationStatus.MALFORMED,
            }
            and not self.retryable
        ):
            raise ValueError(f"{self.status.value} records must be retryable")
        if (
            self.status
            in {
                VerificationStatus.VERIFIED,
                VerificationStatus.RESTORATION_FAILED,
                VerificationStatus.BUDGET_EXHAUSTED,
            }
            and self.retryable
        ):
            raise ValueError(f"{self.status.value} records cannot be retryable")
        if self.status is VerificationStatus.VERIFIED:
            if (
                self.command is None
                or self.return_code != 0
                or self.control_return_code in {None, 0}
                or self.confirmation_return_code != 0
            ):
                raise ValueError(
                    "verified records require a passing command and failing control"
                )
            if self.retryable:
                raise ValueError("verified records cannot be retryable")
        elif self.status is VerificationStatus.REFUTED:
            if (
                self.command is None
                or self.return_code != 1
                or self.confirmation_return_code != 1
            ):
                raise ValueError(
                    "refuted records require two agreeing exit-one current runs"
                )
        elif (
            self.status
            in {
                VerificationStatus.MALFORMED,
                VerificationStatus.BUDGET_EXHAUSTED,
            }
            and self.return_code is not None
        ):
            raise ValueError("malformed or budget records cannot carry an exit code")

    @property
    def allows_completion(self) -> bool:
        """Whether this status authorizes a verified successful completion."""

        return self.status is VerificationStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "attempt": self.attempt,
            "status": self.status.value,
            "reason": self.reason,
            "command": self.command,
            "return_code": self.return_code,
            "control_return_code": self.control_return_code,
            "confirmation_return_code": self.confirmation_return_code,
            "evidence": self.evidence,
            "evidence_truncated": self.evidence_truncated,
            "tokens": self.tokens,
            "attempt_limit_reached": self.attempt_limit_reached,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, value: object) -> VerificationRecord:
        if not isinstance(value, Mapping):
            raise ValueError("verification record must be an object")
        expected = {
            "schema_version",
            "attempt",
            "status",
            "reason",
            "command",
            "return_code",
            "control_return_code",
            "confirmation_return_code",
            "evidence",
            "evidence_truncated",
            "tokens",
            "attempt_limit_reached",
            "retryable",
        }
        if set(value) != expected or value.get("schema_version") != 3:
            raise ValueError("verification record fields are malformed")
        try:
            status = VerificationStatus(value.get("status"))
        except (TypeError, ValueError) as error:
            raise ValueError("verification record status is malformed") from error
        return cls(
            attempt=value.get("attempt"),  # type: ignore[arg-type]
            status=status,
            reason=value.get("reason"),  # type: ignore[arg-type]
            command=value.get("command"),  # type: ignore[arg-type]
            return_code=value.get("return_code"),  # type: ignore[arg-type]
            control_return_code=value.get(  # type: ignore[arg-type]
                "control_return_code"
            ),
            confirmation_return_code=value.get(  # type: ignore[arg-type]
                "confirmation_return_code"
            ),
            evidence=value.get("evidence"),  # type: ignore[arg-type]
            evidence_truncated=value.get("evidence_truncated"),  # type: ignore[arg-type]
            tokens=value.get("tokens"),  # type: ignore[arg-type]
            attempt_limit_reached=value.get(  # type: ignore[arg-type]
                "attempt_limit_reached"
            ),
            retryable=value.get("retryable"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class VerificationCheckpoint:
    """The complete bounded verification ledger stored with an agent checkpoint."""

    records: tuple[VerificationRecord, ...] = ()

    @property
    def attempts_used(self) -> int:
        return len(self.records)

    @property
    def tokens_used(self) -> int:
        return sum(record.tokens for record in self.records)

    def append(
        self, record: VerificationRecord, *, config: SelfVerificationConfig
    ) -> VerificationCheckpoint:
        if record.attempt != self.attempts_used + 1:
            raise ValueError("verification attempts must be contiguous")
        if len(self.records) >= config.max_attempts:
            raise ValueError("verification attempt limit is already exhausted")
        if self.tokens_used + record.tokens > config.max_tokens:
            raise ValueError("verification token allowance would be exceeded")
        return VerificationCheckpoint((*self.records, record))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERIFICATION_CHECKPOINT_SCHEMA_VERSION,
            "attempts_used": self.attempts_used,
            "tokens_used": self.tokens_used,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(
        cls, value: object, *, config: SelfVerificationConfig
    ) -> VerificationCheckpoint:
        if not isinstance(value, Mapping):
            raise ValueError("verification checkpoint must be an object")
        if (
            set(value)
            != {
                "schema_version",
                "attempts_used",
                "tokens_used",
                "records",
            }
            or value.get("schema_version") != VERIFICATION_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("verification checkpoint fields are malformed")
        raw_records = value.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("verification checkpoint records must be a list")
        if len(raw_records) > config.max_attempts:
            raise ValueError("verification checkpoint exceeds the attempt limit")
        records = tuple(VerificationRecord.from_dict(item) for item in raw_records)
        checkpoint = cls(records)
        if [record.attempt for record in records] != list(range(1, len(records) + 1)):
            raise ValueError("verification checkpoint attempts are not contiguous")
        if value.get("attempts_used") != checkpoint.attempts_used:
            raise ValueError("verification checkpoint attempt count is contradictory")
        if value.get("tokens_used") != checkpoint.tokens_used:
            raise ValueError("verification checkpoint token count is contradictory")
        if checkpoint.tokens_used > config.max_tokens:
            raise ValueError("verification checkpoint exceeds the token allowance")
        return checkpoint
