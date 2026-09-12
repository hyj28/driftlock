"""Provider-neutral prompt-cache intent and observable effectiveness reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# Bounds provider error text while retaining enough context to diagnose bad telemetry.
MAX_PROMPT_CACHE_ERROR_CHARACTERS = 512


class PromptCacheStatus(StrEnum):
    """What a provider's cache telemetry says about one request."""

    HIT = "hit"
    MISS = "miss"
    UNOBSERVABLE = "unobservable"
    MALFORMED = "malformed"


class PromptCacheObservability(StrEnum):
    """How completely a summary's reports could be observed."""

    OBSERVED = "observed"
    PARTIAL = "partial"
    UNOBSERVABLE = "unobservable"
    MALFORMED = "malformed"


class PromptCachePrefixEvent(StrEnum):
    """A detected relationship between consecutive cacheable prefixes."""

    COMPACTION_INVALIDATED = "compaction_invalidated"
    ROLLBACK_PREFIX_RESTORED = "rollback_prefix_restored"
    ROLLBACK_PREFIX_DIVERGED = "rollback_prefix_diverged"
    UNCLASSIFIED_PREFIX_DIVERGENCE = "unclassified_prefix_divergence"


class PromptCacheAttributionStatus(StrEnum):
    """Whether input tokens can be causally attributed to a prefix event."""

    NOT_APPLICABLE = "not_applicable"
    UNATTRIBUTABLE = "unattributable"


class PromptCacheReportError(ValueError):
    """Raised when provider cache telemetry is malformed or contradictory."""


@dataclass(frozen=True, slots=True)
class PromptCacheConfig:
    """Enable provider-neutral prompt-cache management and reporting."""


@dataclass(frozen=True, slots=True)
class PromptCacheBreakpoint:
    """The number of leading messages an adapter should make cacheable."""

    message_count: int

    def __post_init__(self) -> None:
        if isinstance(self.message_count, bool) or not isinstance(
            self.message_count, int
        ):
            raise TypeError("message_count must be an int")
        if self.message_count < 0:
            raise ValueError("message_count must be non-negative")


@dataclass(frozen=True, slots=True)
class PromptCacheReport:
    """One request's bounded cache observation and prefix events."""

    status: PromptCacheStatus
    prompt_tokens: int | None
    cached_tokens: int | None
    prefix_events: tuple[PromptCachePrefixEvent, ...] = ()
    attribution: PromptCacheAttributionStatus = (
        PromptCacheAttributionStatus.NOT_APPLICABLE
    )
    error: str | None = None
    error_truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, PromptCacheStatus):
            raise TypeError("status must be a PromptCacheStatus")
        if not isinstance(self.prefix_events, tuple) or any(
            not isinstance(event, PromptCachePrefixEvent)
            for event in self.prefix_events
        ):
            raise TypeError(
                "prefix_events must be a tuple of PromptCachePrefixEvent values"
            )
        if len(set(self.prefix_events)) != len(self.prefix_events):
            raise ValueError("prefix_events must not contain duplicates")
        if not isinstance(self.attribution, PromptCacheAttributionStatus):
            raise TypeError("attribution must be a PromptCacheAttributionStatus")
        _validate_optional_token_count("prompt_tokens", self.prompt_tokens)
        _validate_optional_token_count("cached_tokens", self.cached_tokens)

        if self.status in (PromptCacheStatus.HIT, PromptCacheStatus.MISS):
            if self.prompt_tokens is None or self.cached_tokens is None:
                raise PromptCacheReportError(
                    "observed cache telemetry requires prompt_tokens and cached_tokens"
                )
            if self.cached_tokens > self.prompt_tokens:
                raise PromptCacheReportError(
                    "cached_tokens cannot exceed prompt_tokens"
                )
            expected = (
                PromptCacheStatus.HIT
                if self.cached_tokens > 0
                else PromptCacheStatus.MISS
            )
            if self.status is not expected:
                raise PromptCacheReportError("cache status contradicts cached_tokens")
            if self.error is not None or self.error_truncated:
                raise PromptCacheReportError("observed telemetry cannot carry an error")
        elif self.status is PromptCacheStatus.UNOBSERVABLE:
            if self.cached_tokens is not None:
                raise PromptCacheReportError(
                    "unobservable cache telemetry cannot carry cached_tokens"
                )
            if self.error is not None or self.error_truncated:
                raise PromptCacheReportError(
                    "unobservable telemetry cannot carry an error"
                )
        else:
            if self.prompt_tokens is not None or self.cached_tokens is not None:
                raise PromptCacheReportError(
                    "malformed telemetry cannot carry token counts"
                )
            if not isinstance(self.error, str) or not self.error:
                raise PromptCacheReportError(
                    "malformed telemetry requires a non-empty error"
                )
            if len(self.error) > MAX_PROMPT_CACHE_ERROR_CHARACTERS:
                raise PromptCacheReportError(
                    "malformed telemetry error exceeds its bound"
                )

        invalidating = any(_event_invalidates(event) for event in self.prefix_events)
        expected_attribution = (
            PromptCacheAttributionStatus.UNATTRIBUTABLE
            if invalidating
            else PromptCacheAttributionStatus.NOT_APPLICABLE
        )
        if self.attribution is not expected_attribution:
            raise PromptCacheReportError(
                "attribution contradicts the recorded prefix events"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "status": self.status.value,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "prefix_events": [event.value for event in self.prefix_events],
            "attribution": self.attribution.value,
            "error": self.error,
            "error_truncated": self.error_truncated,
        }


@dataclass(frozen=True, slots=True)
class PromptCachePrefixEventSummary:
    """Fixed-size aggregate for one prefix-event kind."""

    event: PromptCachePrefixEvent
    event_count: int
    observed_event_count: int
    unobservable_event_count: int
    malformed_event_count: int
    attribution: PromptCacheAttributionStatus

    def __post_init__(self) -> None:
        if not isinstance(self.event, PromptCachePrefixEvent):
            raise TypeError("event must be a PromptCachePrefixEvent")
        if isinstance(self.event_count, bool) or not isinstance(self.event_count, int):
            raise TypeError("event_count must be an int")
        if self.event_count < 0:
            raise ValueError("event_count must be non-negative")
        for name, value in (
            ("observed_event_count", self.observed_event_count),
            ("unobservable_event_count", self.unobservable_event_count),
            ("malformed_event_count", self.malformed_event_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.observed_event_count
            + self.unobservable_event_count
            + self.malformed_event_count
            != self.event_count
        ):
            raise ValueError("event observability counts must sum to event_count")
        expected = (
            PromptCacheAttributionStatus.UNATTRIBUTABLE
            if self.event_count and _event_invalidates(self.event)
            else PromptCacheAttributionStatus.NOT_APPLICABLE
        )
        if self.attribution is not expected:
            raise PromptCacheReportError(
                "event attribution contradicts the event count"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.value,
            "event_count": self.event_count,
            "observed_event_count": self.observed_event_count,
            "unobservable_event_count": self.unobservable_event_count,
            "malformed_event_count": self.malformed_event_count,
            "attribution": self.attribution.value,
        }


@dataclass(frozen=True, slots=True)
class PromptCacheSummary:
    """Fixed-size whole-run aggregate; no per-step cache ledger is retained."""

    observability: PromptCacheObservability
    total_report_count: int
    hit_steps: int
    miss_steps: int
    unobservable_steps: int
    malformed_steps: int
    observed_prompt_tokens: int
    observed_cached_tokens: int
    hit_rate: float | None
    token_hit_rate: float | None
    prefix_events: tuple[PromptCachePrefixEventSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "observability": self.observability.value,
            "total_report_count": self.total_report_count,
            "hit_steps": self.hit_steps,
            "miss_steps": self.miss_steps,
            "unobservable_steps": self.unobservable_steps,
            "malformed_steps": self.malformed_steps,
            "observed_prompt_tokens": self.observed_prompt_tokens,
            "observed_cached_tokens": self.observed_cached_tokens,
            "hit_rate": self.hit_rate,
            "token_hit_rate": self.token_hit_rate,
            "rate_scope": "all_observed_reports",
            "prefix_events": [item.to_dict() for item in self.prefix_events],
        }


def summarize_prompt_cache_reports(
    reports: Sequence[PromptCacheReport],
) -> PromptCacheSummary | None:
    """Summarize every supplied report into a bounded, immutable aggregate."""

    if not reports:
        return None
    statuses = Counter(report.status for report in reports)
    hit_steps = statuses[PromptCacheStatus.HIT]
    miss_steps = statuses[PromptCacheStatus.MISS]
    unobservable_steps = statuses[PromptCacheStatus.UNOBSERVABLE]
    malformed_steps = statuses[PromptCacheStatus.MALFORMED]
    observed_steps = hit_steps + miss_steps
    observed_prompt_tokens = sum(
        report.prompt_tokens or 0
        for report in reports
        if report.status in (PromptCacheStatus.HIT, PromptCacheStatus.MISS)
    )
    observed_cached_tokens = sum(
        report.cached_tokens or 0
        for report in reports
        if report.status in (PromptCacheStatus.HIT, PromptCacheStatus.MISS)
    )
    if observed_steps == len(reports):
        observability = PromptCacheObservability.OBSERVED
    elif unobservable_steps == len(reports):
        observability = PromptCacheObservability.UNOBSERVABLE
    elif malformed_steps == len(reports):
        observability = PromptCacheObservability.MALFORMED
    else:
        observability = PromptCacheObservability.PARTIAL

    event_counts = Counter(
        event for report in reports for event in report.prefix_events
    )
    observed_event_counts = Counter(
        event
        for report in reports
        if report.status in (PromptCacheStatus.HIT, PromptCacheStatus.MISS)
        for event in report.prefix_events
    )
    unobservable_event_counts = Counter(
        event
        for report in reports
        if report.status is PromptCacheStatus.UNOBSERVABLE
        for event in report.prefix_events
    )
    malformed_event_counts = Counter(
        event
        for report in reports
        if report.status is PromptCacheStatus.MALFORMED
        for event in report.prefix_events
    )
    event_summaries = tuple(
        PromptCachePrefixEventSummary(
            event=event,
            event_count=event_counts[event],
            observed_event_count=observed_event_counts[event],
            unobservable_event_count=unobservable_event_counts[event],
            malformed_event_count=malformed_event_counts[event],
            attribution=(
                PromptCacheAttributionStatus.UNATTRIBUTABLE
                if event_counts[event] and _event_invalidates(event)
                else PromptCacheAttributionStatus.NOT_APPLICABLE
            ),
        )
        for event in PromptCachePrefixEvent
    )
    return PromptCacheSummary(
        observability=observability,
        total_report_count=len(reports),
        hit_steps=hit_steps,
        miss_steps=miss_steps,
        unobservable_steps=unobservable_steps,
        malformed_steps=malformed_steps,
        observed_prompt_tokens=observed_prompt_tokens,
        observed_cached_tokens=observed_cached_tokens,
        hit_rate=(hit_steps / observed_steps if observed_steps else None),
        token_hit_rate=(
            observed_cached_tokens / observed_prompt_tokens
            if observed_prompt_tokens
            else None
        ),
        prefix_events=event_summaries,
    )


def prompt_cache_report(
    *,
    prompt_tokens: int | None,
    cached_tokens: int | None,
    prefix_events: tuple[PromptCachePrefixEvent, ...] = (),
) -> PromptCacheReport:
    """Validate provider telemetry and classify one cache observation."""

    _validate_optional_token_count("prompt_tokens", prompt_tokens)
    _validate_optional_token_count("cached_tokens", cached_tokens)
    if cached_tokens is None:
        return _report(
            status=PromptCacheStatus.UNOBSERVABLE,
            prompt_tokens=prompt_tokens,
            cached_tokens=None,
            prefix_events=prefix_events,
        )
    if prompt_tokens is None:
        raise PromptCacheReportError("cached_tokens requires prompt_tokens")
    if cached_tokens > prompt_tokens:
        raise PromptCacheReportError("cached_tokens cannot exceed prompt_tokens")
    return _report(
        status=PromptCacheStatus.HIT if cached_tokens else PromptCacheStatus.MISS,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        prefix_events=prefix_events,
    )


def malformed_prompt_cache_report(
    error: Exception | str,
    *,
    prefix_events: tuple[PromptCachePrefixEvent, ...] = (),
) -> PromptCacheReport:
    """Preserve a malformed observation with bounded error text."""

    text = str(error) or type(error).__name__
    truncated = len(text) > MAX_PROMPT_CACHE_ERROR_CHARACTERS
    bounded = text[:MAX_PROMPT_CACHE_ERROR_CHARACTERS]
    return _report(
        status=PromptCacheStatus.MALFORMED,
        prompt_tokens=None,
        cached_tokens=None,
        prefix_events=prefix_events,
        error=bounded,
        error_truncated=truncated,
    )


def _report(
    *,
    status: PromptCacheStatus,
    prompt_tokens: int | None,
    cached_tokens: int | None,
    prefix_events: tuple[PromptCachePrefixEvent, ...],
    error: str | None = None,
    error_truncated: bool = False,
) -> PromptCacheReport:
    invalidating = any(_event_invalidates(event) for event in prefix_events)
    return PromptCacheReport(
        status=status,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        prefix_events=prefix_events,
        attribution=(
            PromptCacheAttributionStatus.UNATTRIBUTABLE
            if invalidating
            else PromptCacheAttributionStatus.NOT_APPLICABLE
        ),
        error=error,
        error_truncated=error_truncated,
    )


def _event_invalidates(event: PromptCachePrefixEvent) -> bool:
    return event in (
        PromptCachePrefixEvent.COMPACTION_INVALIDATED,
        PromptCachePrefixEvent.ROLLBACK_PREFIX_DIVERGED,
        PromptCachePrefixEvent.UNCLASSIFIED_PREFIX_DIVERGENCE,
    )


def _validate_optional_token_count(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise PromptCacheReportError(f"{name} must be an int or None")
    if value < 0:
        raise PromptCacheReportError(f"{name} must be non-negative")
