"""Run the installed CLI with a deterministic offline distillation provider."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import driftlock.lhtb_experiment as experiment
from driftlock.models import JudgeCompletion
from driftlock.skill_distillation import CallableSkillDistiller
from driftlock.usage import ReplayUsage

_SKILL = (
    "## activation\n\nWhen checkpoint evidence shows repeated non-improvement.\n\n"
    "## execution\n\nDo not repeat the stalled action; choose a verified "
    "alternative instead.\n\n"
    "## termination\n\nStop when the alternative produces measured progress."
)


def main() -> None:
    outcomes = iter(sys.argv[1].split(","))
    console_arguments = sys.argv[2:]
    usage = ReplayUsage(0, 0, 0, 0.0)

    async def complete(_prompt: str) -> str | JudgeCompletion:
        nonlocal usage
        usage = ReplayUsage(
            input_tokens=usage.input_tokens + 3,
            cache_tokens=usage.cache_tokens,
            output_tokens=usage.output_tokens + 2,
            cost_usd=usage.cost_usd + 0.005,
        )
        outcome = next(outcomes)
        if outcome == "failed":
            raise RuntimeError("synthetic transient provider failure")
        if outcome == "declined":
            return "DECLINE: synthetic evidence has no preventative lesson"
        if outcome == "malformed":
            return "synthetic malformed response"
        if outcome == "generated":
            return JudgeCompletion(_SKILL, tokens=5)
        raise AssertionError(f"unknown synthetic outcome: {outcome}")

    def usage_reader() -> ReplayUsage:
        return usage

    def build_fake(
        **kwargs: Any,
    ) -> tuple[
        CallableSkillDistiller,
        Callable[[], ReplayUsage],
        dict[str, Any],
    ]:
        return (
            CallableSkillDistiller(complete),
            usage_reader,
            {
                "model": kwargs["model"],
                "provider": kwargs["provider"],
                "api_base": kwargs["api_base"],
                "max_output_tokens": kwargs["max_output_tokens"],
                "timeout_sec": kwargs["timeout_sec"],
            },
        )

    experiment._build_skill_distiller = build_fake
    launcher = Path(sys.executable).with_name("driftlock-lhtb")
    sys.argv = [str(launcher), *console_arguments]
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
