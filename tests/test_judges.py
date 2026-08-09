from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from driftlock.judges import CallableLLMJudge
from driftlock.models import (
    Checkpoint,
    DriftContext,
    DriftSignal,
    JudgeCompletion,
    StepOutcome,
    StepRecord,
    Verdict,
)


def _context() -> DriftContext:
    checkpoint = Checkpoint(
        checkpoint_id="abc",
        step=2,
        created_at=datetime.now(UTC),
        digest="digest",
        path=Path("/tmp/checkpoint"),
    )
    step = StepRecord(
        sequence=3,
        logical_step=3,
        attempt=1,
        outcome=StepOutcome(
            action="rewrite unrelated module",
            state={},
            diff="- goal\n+ distraction",
        ),
    )
    return DriftContext(
        goal="fix the parser",
        plan="add a regression test",
        checkpoint=checkpoint,
        signals=(DriftSignal("no_file_change", "stalled"),),
        recent_steps=(step,),
        diff=step.outcome.diff,
        tokens_remaining=500,
    )


async def test_callable_llm_judge_builds_evidence_and_parses_json() -> None:
    received = ""

    async def complete(prompt: str) -> JudgeCompletion:
        nonlocal received
        received = prompt
        return JudgeCompletion(
            '{"verdict":"drifted","reason":"off goal","confidence":0.9}',
            tokens=17,
        )

    verdict = await CallableLLMJudge(complete).judge(_context())

    assert verdict.verdict is Verdict.DRIFTED
    assert verdict.confidence == 0.9
    assert verdict.tokens == 17
    assert "fix the parser" in received
    assert "rewrite unrelated module" in received


async def test_callable_llm_judge_degrades_invalid_output_to_uncertain() -> None:
    async def complete(_prompt: str) -> str:
        return "not json"

    verdict = await CallableLLMJudge(complete).judge(_context())

    assert verdict.verdict is Verdict.UNCERTAIN
    assert verdict.confidence == 0.0


async def test_callable_llm_judge_degrades_provider_failure_to_uncertain() -> None:
    async def complete(_prompt: str) -> str:
        raise TimeoutError("provider unavailable")

    verdict = await CallableLLMJudge(complete).judge(_context())

    assert verdict.verdict is Verdict.UNCERTAIN
    assert "provider unavailable" in verdict.reason
