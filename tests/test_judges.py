from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from driftlock.judges import (
    DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
    CallableLLMJudge,
    JudgeTokenBudgetExhausted,
    judge_input_token_bound,
    judge_output_token_limit,
)
from driftlock.models import (
    Checkpoint,
    DriftContext,
    DriftSignal,
    FineJudgeStatus,
    JudgeCompletion,
    JudgeVerdict,
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


async def test_callable_llm_judge_marks_invalid_output_as_failed() -> None:
    async def complete(_prompt: str) -> str:
        return "not json"

    verdict = await CallableLLMJudge(complete).judge(_context())

    assert verdict.verdict is None
    assert verdict.status is FineJudgeStatus.FAILED
    assert verdict.confidence == 0.0


async def test_callable_llm_judge_prompt_includes_tool_observations() -> None:
    received = ""

    async def complete(prompt: str) -> str:
        nonlocal received
        received = prompt
        return '{"verdict":"healthy","reason":"failure is changing"}'

    context = replace(
        _context(),
        tool_observations=("step 3:\nrun_shell:\nexit_code: 1\nstderr:\nsyntax",),
    )
    verdict = await CallableLLMJudge(complete).judge(context)

    assert verdict.verdict is Verdict.HEALTHY
    assert '"recent_tool_observations"' in received
    assert "exit_code: 1" in received
    assert "syntax" in received


async def test_callable_llm_judge_marks_provider_failure_as_failed() -> None:
    async def complete(_prompt: str) -> str:
        raise TimeoutError("provider unavailable")

    verdict = await CallableLLMJudge(complete).judge(_context())

    assert verdict.verdict is None
    assert verdict.status is FineJudgeStatus.FAILED
    assert "provider unavailable" in verdict.reason


@pytest.mark.parametrize(
    ("response", "error_text"),
    [
        ("", "Expecting value: line 1 column 1"),
        (
            '{"verdict":"drifted","reason":"unfinished',
            "Unterminated string starting at",
        ),
        ("[]", "response must be a JSON object"),
        ('{"verdict":"healthy"}', "'reason'"),
    ],
)
async def test_callable_llm_judge_marks_unusable_responses_as_failures(
    response: str, error_text: str
) -> None:
    async def complete(_prompt: str) -> JudgeCompletion:
        return JudgeCompletion(response, tokens=23)

    verdict = await CallableLLMJudge(complete).judge(_context())

    assert verdict.status is FineJudgeStatus.FAILED
    assert verdict.verdict is None
    assert verdict.tokens == 23
    assert error_text in verdict.reason


async def test_callable_llm_judge_preserves_model_issued_uncertain_verdict() -> None:
    async def complete(_prompt: str) -> str:
        return '{"verdict":"uncertain","reason":"evidence is mixed","confidence":0.4}'

    verdict = await CallableLLMJudge(complete).judge(_context())

    assert verdict.status is FineJudgeStatus.VERDICT
    assert verdict.verdict is Verdict.UNCERTAIN
    assert verdict.reason == "evidence is mixed"
    assert verdict.confidence == 0.4


async def test_preflight_budget_exhaustion_is_a_distinct_judge_result() -> None:
    async def complete(_prompt: str) -> str:
        raise JudgeTokenBudgetExhausted(
            "300 tokens remain but the judge prompt reserves 301 input tokens"
        )

    verdict = await CallableLLMJudge(complete).judge(_context())

    assert verdict.status is FineJudgeStatus.BUDGET_EXHAUSTED
    assert verdict.verdict is None
    assert "no output-token allowance" in verdict.reason
    assert "300 tokens remain" in verdict.reason


def test_judge_output_budget_uses_full_model_window_without_a_run_limit() -> None:
    assert DEFAULT_JUDGE_MAX_OUTPUT_TOKENS == 8_192
    assert (
        judge_output_token_limit(
            "a 33 KB trajectory",
            max_output_tokens=DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
            tokens_remaining=None,
        )
        == 8_192
    )


def test_judge_output_budget_reserves_utf8_prompt_bytes_and_margin() -> None:
    assert judge_input_token_bound("é") == 258
    assert (
        judge_output_token_limit("é", max_output_tokens=8_192, tokens_remaining=260)
        == 2
    )


def test_judge_output_budget_reports_when_no_output_room_remains() -> None:
    with pytest.raises(
        JudgeTokenBudgetExhausted,
        match="258 tokens remain but the judge prompt reserves 258 input tokens",
    ):
        judge_output_token_limit("é", max_output_tokens=8_192, tokens_remaining=258)


@pytest.mark.parametrize(
    "status",
    [FineJudgeStatus.FAILED, FineJudgeStatus.BUDGET_EXHAUSTED],
)
def test_failed_judge_result_cannot_carry_a_verdict(status: FineJudgeStatus) -> None:
    with pytest.raises(
        ValueError, match="only an adjudicated fine judge may return a verdict"
    ):
        JudgeVerdict(Verdict.UNCERTAIN, "not adjudicated", status=status)


def test_adjudicated_judge_result_requires_a_verdict() -> None:
    with pytest.raises(
        ValueError, match="only an adjudicated fine judge may return a verdict"
    ):
        JudgeVerdict(None, "missing verdict", status=FineJudgeStatus.VERDICT)


def test_judge_result_rejects_a_non_result_status() -> None:
    with pytest.raises(
        ValueError, match="judge result must describe an invoked fine judge"
    ):
        JudgeVerdict(None, "not invoked", status=FineJudgeStatus.NOT_INVOKED)
