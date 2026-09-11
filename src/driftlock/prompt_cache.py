"""Provider-neutral prompt-cache intent and bounded effectiveness reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# Two hundred fifty-six reports cover the runner's usual long horizon while
# preventing a caller-supplied report collection from becoming another audit log.
DEFAULT_MAX_CACHE_SUMMARY_REPORTS = 256


class PromptCacheReportError(ValueError):
    """Raised when provider cache usage is malformed or contradictory."""


class PromptCacheStatus(StrEnum):
    """What the provider's cache telemetry establishes for one request."""

    HIT = "hit"
    MISS = "miss"
    UNOBSERVABLE = "unobservable"


class PromptCacheInvalidation(StrEnum):
    """A necessary local event that made an earlier prefix unusable."""

    COMPACTION = "compaction"
    ROLLBACK = "rollback"
    COMPACTION_AND_ROLLBACK = "compaction_and_rollback"


class PromptCacheObservability(StrEnum):
    """How completely a summary's steps exposed cache telemetry."""

    OBSERVED = "observed"
    PARTIAL = "partial"
    UNOBSERVABLE = "unobservable"


@dataclass(frozen=True, slots=True)
class PromptCacheConfig:
    """Enable append-only request construction and cache measurement."""


@dataclass(frozen=True, slots=True)
class PromptCacheBreakpoint:
    """Provider-neutral intent to cache a leading number of chat messages."""

    message_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.message_count, int)
            or isinstance(self.message_count, bool)
            or self.message_count <= 0
        ):
            raise ValueError("message_count must be a positive integer")


@dataclass(frozen=True, slots=True)
class PromptCacheReport:
    """One request's observable cache outcome and invalidation cost."""

    status: PromptCacheStatus
    prompt_tokens: int | None
    cached_tokens: int | None
    invalidation: PromptCacheInvalidation | None = None
    reprimed_input_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PromptCacheStatus):
            raise TypeError("status must be a PromptCacheStatus")
        if self.invalidation is not None and not isinstance(
            self.invalidation, PromptCacheInvalidation
        ):
            raise TypeError("invalidation must be a PromptCacheInvalidation or None")
        _validate_optional_token_count(self.prompt_tokens, "prompt_tokens")
        _validate_optional_token_count(self.cached_tokens, "cached_tokens")
        _validate_optional_token_count(
            self.reprimed_input_tokens, "reprimed_input_tokens"
        )
        observed = self.status is not PromptCacheStatus.UNOBSERVABLE
        if observed:
            if self.prompt_tokens is None or self.cached_tokens is None:
                raise ValueError("an observed cache report requires both token counts")
            if self.cached_tokens > self.prompt_tokens:
                raise ValueError("cached_tokens cannot exceed prompt_tokens")
            if (self.status is PromptCacheStatus.HIT) != (self.cached_tokens > 0):
                raise ValueError("cache hit status and cached_tokens disagree")
        elif self.cached_tokens is not None:
            raise ValueError("an unobservable cache report cannot have cached_tokens")
        expected_reprimed = (
            self.prompt_tokens - self.cached_tokens
            if observed and self.invalidation is not None
            else None
        )
        if self.reprimed_input_tokens != expected_reprimed:
            raise ValueError(
                "reprimed_input_tokens must equal observed uncached input after an "
                "invalidation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status.value,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "invalidation": (
                self.invalidation.value if self.invalidation is not None else None
            ),
            "reprimed_input_tokens": self.reprimed_input_tokens,
        }


@dataclass(frozen=True, slots=True)
class PromptCacheSummary:
    """A fixed-size aggregate over a bounded prefix of step reports."""

    observability: PromptCacheObservability
    total_report_count: int
    summarized_report_count: int
    omitted_report_count: int
    hit_steps: int
    miss_steps: int
    unobservable_steps: int
    observed_prompt_tokens: int
    cached_tokens: int
    hit_rate: float | None
    token_hit_rate: float | None
    invalidation_counts: dict[str, int]
    reprimed_input_tokens: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "observability": self.observability.value,
            "total_report_count": self.total_report_count,
            "summarized_report_count": self.summarized_report_count,
            "omitted_report_count": self.omitted_report_count,
            "report_limit_reached": self.omitted_report_count > 0,
            "hit_steps": self.hit_steps,
            "miss_steps": self.miss_steps,
            "unobservable_steps": self.unobservable_steps,
            "observed_prompt_tokens": self.observed_prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "hit_rate": self.hit_rate,
            "token_hit_rate": self.token_hit_rate,
            "invalidation_counts": dict(self.invalidation_counts),
            "reprimed_input_tokens": dict(self.reprimed_input_tokens),
        }


def summarize_prompt_cache(
    reports: Sequence[PromptCacheReport],
) -> PromptCacheSummary:
    """Summarize cache reports without treating blind steps as misses.

    At most :data:`DEFAULT_MAX_CACHE_SUMMARY_REPORTS` reports contribute to the
    aggregate. The number omitted is explicit, so the cap never silently drops
    evidence.
    """

    if isinstance(reports, (str, bytes)) or not isinstance(reports, Sequence):
        raise TypeError("reports must be a sequence of PromptCacheReport values")
    if any(not isinstance(report, PromptCacheReport) for report in reports):
        raise TypeError("reports must contain only PromptCacheReport values")
    retained = reports[:DEFAULT_MAX_CACHE_SUMMARY_REPORTS]
    hit_steps = sum(report.status is PromptCacheStatus.HIT for report in retained)
    miss_steps = sum(report.status is PromptCacheStatus.MISS for report in retained)
    unobservable_steps = sum(
        report.status is PromptCacheStatus.UNOBSERVABLE for report in retained
    )
    observed_steps = hit_steps + miss_steps
    observed_prompt_tokens = sum(
        report.prompt_tokens or 0
        for report in retained
        if report.status is not PromptCacheStatus.UNOBSERVABLE
    )
    cached_tokens = sum(report.cached_tokens or 0 for report in retained)
    all_has_observed = any(
        report.status is not PromptCacheStatus.UNOBSERVABLE for report in reports
    )
    all_has_unobservable = any(
        report.status is PromptCacheStatus.UNOBSERVABLE for report in reports
    )
    if not all_has_observed:
        observability = PromptCacheObservability.UNOBSERVABLE
    elif all_has_unobservable:
        observability = PromptCacheObservability.PARTIAL
    else:
        observability = PromptCacheObservability.OBSERVED
    invalidation_counts = {kind.value: 0 for kind in PromptCacheInvalidation}
    reprimed_input_tokens = {kind.value: 0 for kind in PromptCacheInvalidation}
    for report in retained:
        if report.invalidation is None:
            continue
        key = report.invalidation.value
        invalidation_counts[key] += 1
        if report.reprimed_input_tokens is not None:
            reprimed_input_tokens[key] += report.reprimed_input_tokens
    return PromptCacheSummary(
        observability=observability,
        total_report_count=len(reports),
        summarized_report_count=len(retained),
        omitted_report_count=len(reports) - len(retained),
        hit_steps=hit_steps,
        miss_steps=miss_steps,
        unobservable_steps=unobservable_steps,
        observed_prompt_tokens=observed_prompt_tokens,
        cached_tokens=cached_tokens,
        hit_rate=hit_steps / observed_steps if observed_steps else None,
        token_hit_rate=(
            cached_tokens / observed_prompt_tokens if observed_prompt_tokens else None
        ),
        invalidation_counts=invalidation_counts,
        reprimed_input_tokens=reprimed_input_tokens,
    )


def prompt_cache_report(
    *,
    prompt_tokens: object,
    cached_tokens: object,
    invalidation: PromptCacheInvalidation | None,
) -> PromptCacheReport:
    """Validate provider telemetry and classify its three possible outcomes."""

    if prompt_tokens is not None and (
        not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or prompt_tokens < 0
    ):
        raise PromptCacheReportError(
            "provider prompt_tokens must be a non-negative integer or None"
        )
    if cached_tokens is None:
        return PromptCacheReport(
            PromptCacheStatus.UNOBSERVABLE,
            prompt_tokens,
            None,
            invalidation,
            None,
        )
    if (
        not isinstance(cached_tokens, int)
        or isinstance(cached_tokens, bool)
        or cached_tokens < 0
    ):
        raise PromptCacheReportError(
            "provider cached_tokens must be a non-negative integer or None"
        )
    if prompt_tokens is None:
        raise PromptCacheReportError(
            "provider cached_tokens require reported prompt_tokens"
        )
    if cached_tokens > prompt_tokens:
        raise PromptCacheReportError(
            "provider cached_tokens cannot exceed prompt_tokens"
        )
    status = PromptCacheStatus.HIT if cached_tokens else PromptCacheStatus.MISS
    return PromptCacheReport(
        status,
        prompt_tokens,
        cached_tokens,
        invalidation,
        prompt_tokens - cached_tokens if invalidation is not None else None,
    )


def _validate_optional_token_count(value: object, name: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer or None")
