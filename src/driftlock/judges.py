"""Fine semantic judge interfaces and an API-agnostic LLM adapter."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Protocol

from driftlock.models import DriftContext, JudgeVerdict, Verdict


class FineJudge(Protocol):
    """Semantic judge invoked only after a coarse signal fires."""

    async def judge(self, context: DriftContext) -> JudgeVerdict: ...


class CallableLLMJudge:
    """Use any async text-completion function as the fine judge.

    This keeps provider credentials and SDK choices outside the core library. The
    callable receives a complete prompt and must return a JSON object as text.
    """

    def __init__(self, complete: Callable[[str], Awaitable[str]]) -> None:
        self._complete = complete

    async def judge(self, context: DriftContext) -> JudgeVerdict:
        try:
            response = await self._complete(_build_prompt(context))
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
            )
        except Exception as error:
            return JudgeVerdict(
                verdict=Verdict.UNCERTAIN,
                reason=f"fine judge failed or returned an invalid response: {error}",
                confidence=0.0,
            )


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
            {"kind": signal.kind, "detail": signal.detail} for signal in context.signals
        ],
        "recent_trajectory": trajectory,
        "latest_diff": context.diff,
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
