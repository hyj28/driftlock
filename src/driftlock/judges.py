"""Fine semantic judge interfaces and an API-agnostic LLM adapter."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Protocol

from driftlock.models import (
    DriftContext,
    FineJudgeStatus,
    JudgeCompletion,
    JudgeVerdict,
    Verdict,
)

# The pinned reasoning judge exposes an 8,192-token output window. Reasoning and
# the final JSON share that allowance, so use the full declared window: the old
# 512-token cap frequently spent the entire allowance before emitting any JSON.
DEFAULT_JUDGE_MAX_OUTPUT_TOKENS = 8_192
_JUDGE_INPUT_TOKEN_MARGIN = 256


class JudgeTokenBudgetExhausted(RuntimeError):
    """Raised before a judge call when no safe output allowance remains."""


class FineJudge(Protocol):
    """Semantic judge invoked only after a coarse signal fires."""

    async def judge(self, context: DriftContext) -> JudgeVerdict: ...


class CallableLLMJudge:
    """Use any async text-completion function as the fine judge.

    This keeps provider credentials and SDK choices outside the core library. The
    callable receives a complete prompt and must return a JSON object as text.
    """

    def __init__(
        self,
        complete: Callable[[str], Awaitable[str | JudgeCompletion]],
    ) -> None:
        self._complete = complete

    async def judge(self, context: DriftContext) -> JudgeVerdict:
        tokens = 0
        try:
            completion = await self._complete(_build_prompt(context))
            if isinstance(completion, str):
                response = completion
                tokens = 0
            else:
                response = completion.text
                tokens = completion.tokens
            payload = _parse_json_object(response)
            verdict_value = payload["verdict"]
            reason_value = payload["reason"]
            confidence_value = payload.get("confidence", 1.0)
            if not isinstance(verdict_value, str):
                raise TypeError("verdict must be a string")
            if not isinstance(reason_value, str):
                raise TypeError("reason must be a string")
            if isinstance(confidence_value, bool) or not isinstance(
                confidence_value, (int, float, str)
            ):
                raise TypeError("confidence must be numeric")
            return JudgeVerdict(
                verdict=Verdict(verdict_value.lower()),
                reason=reason_value,
                confidence=float(confidence_value),
                tokens=tokens,
            )
        except JudgeTokenBudgetExhausted as error:
            return JudgeVerdict(
                verdict=None,
                reason=f"fine judge has no output-token allowance: {error}",
                confidence=0.0,
                tokens=tokens,
                status=FineJudgeStatus.BUDGET_EXHAUSTED,
            )
        except Exception as error:
            return JudgeVerdict(
                verdict=None,
                reason=f"fine judge failed or returned an invalid response: {error}",
                confidence=0.0,
                tokens=tokens,
                status=FineJudgeStatus.FAILED,
            )


def judge_output_token_limit(
    prompt: str,
    *,
    max_output_tokens: int,
    tokens_remaining: int | None,
) -> int:
    """Return a conservative output ceiling or report preflight exhaustion."""
    if max_output_tokens <= 0:
        raise ValueError("judge output limit must be positive")
    if tokens_remaining is not None and tokens_remaining < 0:
        raise ValueError("tokens remaining cannot be negative")
    # One token per UTF-8 byte plus a fixed margin deliberately overestimates the
    # prompt charge. The shared run budget must never be exceeded by a judge call.
    input_bound = judge_input_token_bound(prompt)
    ceiling = max_output_tokens
    if tokens_remaining is not None:
        ceiling = min(ceiling, tokens_remaining - input_bound)
    if ceiling <= 0:
        raise JudgeTokenBudgetExhausted(
            f"{tokens_remaining} tokens remain but the judge prompt reserves "
            f"{input_bound} input tokens"
        )
    return ceiling


def judge_input_token_bound(prompt: str) -> int:
    """Conservatively bound the prompt charge used by shared-budget accounting."""
    return len(prompt.encode("utf-8")) + _JUDGE_INPUT_TOKEN_MARGIN


def _build_prompt(context: DriftContext) -> str:
    trajectory = [
        {
            "sequence": step.sequence,
            "logical_step": step.logical_step,
            "action": step.outcome.action,
            "summary": step.outcome.summary,
            "error": step.outcome.error,
            "reward": step.outcome.reward,
            "changed_paths": list(step.outcome.changed_paths),
            "workspace_delta_observed": step.outcome.workspace_delta_observed,
            "workspace_observation_error": step.outcome.workspace_observation_error,
        }
        for step in context.recent_steps
    ]
    evidence = {
        "original_goal": context.goal,
        "current_plan": context.plan,
        "checkpoint": {
            "id": context.checkpoint.checkpoint_id,
            "logical_step": context.checkpoint.step,
        },
        "coarse_signals": [
            {
                "kind": signal.kind,
                "detail": signal.detail,
                "lookback": signal.lookback,
            }
            for signal in context.signals
        ],
        "recent_trajectory": trajectory,
        "recent_tool_observations": list(context.tool_observations),
        "latest_diff": context.diff,
        "tokens_remaining": context.tokens_remaining,
    }
    return (
        "You are an independent progress judge for a long-horizon terminal agent. "
        "Decide whether the current workspace remains a sound basis for continuing "
        "toward the original goal. Do not mark ordinary exploration as drift. Mark "
        "a trajectory drifted when continuing from it is less promising than "
        "restoring the stated checkpoint. Return JSON only with keys verdict "
        "(healthy, drifted, or uncertain), reason, and confidence (0 to 1).\n\n"
        + json.dumps(evidence, indent=2, sort_keys=True)
    )


def _parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1:
            raise json.JSONDecodeError("no JSON object found", stripped, 0)
        stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise TypeError("response must be a JSON object")
    return payload
