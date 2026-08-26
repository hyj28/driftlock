"""Harbor-free validation and reconstruction of trial usage accounting."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Complete round-five trajectories covered 98--100% of their provider-call
# episode directories. Keep a margin for an interrupted final record while
# rejecting summarized trajectory segments such as the observed 11/46 case.
_MIN_TRAJECTORY_EPISODE_COVERAGE = 0.9


class UsageAccountingError(ValueError):
    """Raised when neither reported nor reconstructed usage is trustworthy."""


@dataclass(frozen=True, slots=True)
class ReplayUsage:
    """Conservative source-trial usage attached to every replay candidate."""

    input_tokens: int
    cache_tokens: int
    output_tokens: int
    cost_usd: float

    def __post_init__(self) -> None:
        for name in ("input_tokens", "cache_tokens", "output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.cache_tokens > self.input_tokens:
            raise ValueError("cache_tokens cannot exceed input_tokens")
        if (
            isinstance(self.cost_usd, bool)
            or not isinstance(self.cost_usd, (int, float))
            or not math.isfinite(float(self.cost_usd))
            or self.cost_usd < 0
        ):
            raise ValueError("cost_usd must be finite and nonnegative")

    @classmethod
    def from_mapping(cls, value: object) -> ReplayUsage:
        if not isinstance(value, dict) or set(value) != {
            "input_tokens",
            "cache_tokens",
            "output_tokens",
            "cost_usd",
        }:
            raise ValueError("source usage has unexpected fields")
        return cls(
            input_tokens=value["input_tokens"],
            cache_tokens=value["cache_tokens"],
            output_tokens=value["output_tokens"],
            cost_usd=value["cost_usd"],
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.input_tokens,
            "cache_tokens": self.cache_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": float(self.cost_usd),
        }


@dataclass(frozen=True, slots=True)
class TrialUsage:
    """Validated usage together with its reported or reconstructed provenance."""

    usage: ReplayUsage
    source: Literal["agent_result", "step_results", "trajectory"]
    imputed_cost_steps: int = 0
    imputed_cost_usd: float = 0.0

    @property
    def reconstructed(self) -> bool:
        return self.source == "trajectory"

    def as_analysis_dict(self) -> dict[str, int | float]:
        return {
            **self.usage.as_dict(),
            "imputed_cost_steps": self.imputed_cost_steps,
            "imputed_cost_usd": self.imputed_cost_usd,
        }


@dataclass(frozen=True, slots=True)
class _AgentProviderPricing:
    input_cost_per_token: float
    cache_cost_per_token: float
    output_cost_per_token: float


# Missing costs may only be recovered under the exact provider slug whose rates
# were audited. A future provider therefore fails instead of inheriting stale rates.
_AGENT_PROVIDER_PRICING = {
    "deepinfra/fp8": _AgentProviderPricing(
        input_cost_per_token=0.00000008,
        cache_cost_per_token=0.00000008,
        output_cost_per_token=0.00000018,
    )
}


def load_trial_usage(
    data: dict[str, Any],
    result_file: Path,
    *,
    provider: str | Callable[[], str],
) -> TrialUsage:
    """Load reported usage, falling back to an audited trajectory reconstruction."""
    direct = data.get("agent_result")
    step_results = data.get("step_results")
    if isinstance(direct, dict) and isinstance(step_results, list) and step_results:
        raise UsageAccountingError(
            f"result has both direct and step agent contexts: {result_file}"
        )
    if isinstance(direct, dict):
        return _reported_usage([direct], source="agent_result", result_file=result_file)
    if isinstance(step_results, list):
        contexts = [
            step.get("agent_result")
            for step in step_results
            if isinstance(step, dict) and isinstance(step.get("agent_result"), dict)
        ]
        if contexts:
            return _reported_usage(
                contexts, source="step_results", result_file=result_file
            )
    return reconstruct_trajectory_usage(result_file, provider=provider)


def reconstruct_trajectory_usage(
    result_file: Path, *, provider: str | Callable[[], str]
) -> TrialUsage:
    """Reconstruct usage from a complete-enough Harbor trajectory."""
    trajectory_file = result_file.parent / "agent" / "trajectory.json"
    try:
        data = json.loads(trajectory_file.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise UsageAccountingError(
            "source usage lacks agent_result/step_results accounting; missing "
            f"trajectory needed to reconstruct agent usage: {trajectory_file}"
        ) from error
    except (OSError, UnicodeDecodeError) as error:
        raise UsageAccountingError(
            f"unreadable trajectory needed to reconstruct agent usage: "
            f"{trajectory_file}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise UsageAccountingError(
            f"invalid JSON in trajectory needed to reconstruct agent usage: "
            f"{trajectory_file}: {error}"
        ) from error
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise UsageAccountingError(
            f"malformed trajectory needed to reconstruct agent usage: "
            f"{trajectory_file} must contain a steps array"
        )
    steps = data["steps"]
    if not steps:
        raise UsageAccountingError(
            f"malformed trajectory needed to reconstruct agent usage: "
            f"{trajectory_file} has no steps"
        )
    totals: dict[str, int | float] = {
        "input_tokens": 0,
        "cache_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    imputed_cost_steps = 0
    imputed_cost_usd = 0.0
    fields = {
        "input_tokens": "prompt_tokens",
        "cache_tokens": "cached_tokens",
        "output_tokens": "completion_tokens",
        "cost_usd": "cost_usd",
    }
    agent_step_count = sum(
        isinstance(step, dict) and step.get("source") == "agent" for step in steps
    )
    _validate_trajectory_episode_coverage(
        trajectory_file, agent_step_count=agent_step_count
    )
    for index, step in enumerate(steps):
        context = f"step {index} metrics in {trajectory_file}"
        if not isinstance(step, dict):
            raise UsageAccountingError(
                f"malformed trajectory {context}: step must be an object"
            )
        metrics = step.get("metrics")
        if metrics is None and (
            step.get("source") in {"system", "user"}
            or (step.get("source") == "agent" and step.get("llm_call_count") == 0)
        ):
            continue
        if not isinstance(metrics, dict):
            raise UsageAccountingError(
                f"malformed trajectory {context}: metrics must be an object"
            )
        for output_name, source_name in fields.items():
            raw = metrics.get(source_name)
            if source_name not in metrics and output_name == "cache_tokens":
                value: int | float = 0
            elif source_name not in metrics and output_name == "cost_usd":
                prompt_tokens = _nonnegative_integer(
                    metrics.get("prompt_tokens"), f"prompt_tokens in {context}"
                )
                cache_tokens = _nonnegative_integer(
                    metrics.get("cached_tokens", 0), f"cached_tokens in {context}"
                )
                completion_tokens = _nonnegative_integer(
                    metrics.get("completion_tokens"),
                    f"completion_tokens in {context}",
                )
                value = _imputed_trajectory_cost(
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    cache_tokens=cache_tokens,
                    completion_tokens=completion_tokens,
                    context=context,
                )
                imputed_cost_steps += 1
                imputed_cost_usd += value
            elif output_name == "cost_usd":
                value = _nonnegative_number(raw, f"{source_name} in {context}")
            else:
                value = _nonnegative_integer(raw, f"{source_name} in {context}")
            totals[output_name] += value
    if totals["cache_tokens"] > totals["input_tokens"]:
        raise UsageAccountingError(
            f"trajectory cache tokens exceed input tokens in {trajectory_file}"
        )
    try:
        usage = ReplayUsage.from_mapping(totals)
    except ValueError as error:
        raise UsageAccountingError(
            f"trajectory usage accounting is invalid: {trajectory_file}: {error}"
        ) from error
    return TrialUsage(
        usage=usage,
        source="trajectory",
        imputed_cost_steps=imputed_cost_steps,
        imputed_cost_usd=imputed_cost_usd,
    )


def _reported_usage(
    contexts: list[dict[str, Any]],
    *,
    source: Literal["agent_result", "step_results"],
    result_file: Path,
) -> TrialUsage:
    totals: dict[str, int | float] = {
        "input_tokens": 0,
        "cache_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    fields = {
        "input_tokens": "n_input_tokens",
        "cache_tokens": "n_cache_tokens",
        "output_tokens": "n_output_tokens",
        "cost_usd": "cost_usd",
    }
    for context in contexts:
        for output_name, source_name in fields.items():
            raw = context.get(source_name)
            if output_name == "cost_usd":
                value = _nonnegative_number(raw, f"{source_name} in {result_file}")
            else:
                value = _nonnegative_integer(raw, f"{source_name} in {result_file}")
            totals[output_name] += value
    if totals["cache_tokens"] > totals["input_tokens"]:
        raise UsageAccountingError(f"cache tokens exceed input tokens in {result_file}")
    try:
        usage = ReplayUsage.from_mapping(totals)
    except ValueError as error:
        raise UsageAccountingError(
            f"source usage accounting is invalid: {result_file}: {error}"
        ) from error
    return TrialUsage(usage=usage, source=source)


def _imputed_trajectory_cost(
    *,
    provider: str | Callable[[], str],
    prompt_tokens: int,
    cache_tokens: int,
    completion_tokens: int,
    context: str,
) -> float:
    routed_provider = provider() if callable(provider) else provider
    pricing = _AGENT_PROVIDER_PRICING.get(routed_provider)
    if pricing is None:
        raise UsageAccountingError(
            f"cost_usd is missing in {context}, and agent provider "
            f"{routed_provider!r} has no audited trajectory-imputation pricing"
        )
    if cache_tokens > prompt_tokens:
        raise UsageAccountingError(f"cached_tokens exceed prompt_tokens in {context}")
    uncached_tokens = prompt_tokens - cache_tokens
    return (
        uncached_tokens * pricing.input_cost_per_token
        + cache_tokens * pricing.cache_cost_per_token
        + completion_tokens * pricing.output_cost_per_token
    )


def _validate_trajectory_episode_coverage(
    trajectory_file: Path, *, agent_step_count: int
) -> None:
    episode_count = sum(
        path.is_dir() for path in trajectory_file.parent.glob("episode-*")
    )
    if not episode_count:
        return
    coverage = agent_step_count / episode_count
    if coverage < _MIN_TRAJECTORY_EPISODE_COVERAGE:
        raise UsageAccountingError(
            "trajectory needed to reconstruct agent usage is incomplete: "
            f"{trajectory_file} contains {agent_step_count} agent steps for "
            f"{episode_count} provider-call episode directories "
            f"({coverage:.1%} coverage)"
        )


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UsageAccountingError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise UsageAccountingError(f"{name} must be finite and nonnegative")
    return number


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise UsageAccountingError(f"{name} must be a nonnegative integer")
    return value
