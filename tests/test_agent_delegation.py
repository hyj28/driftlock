from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from driftlock.agent import (
    AgentCompletion,
    AgentCompletionRequest,
    AgentConversationCodec,
    AgentProviderError,
    AgentStateError,
    ToolCall,
    ToolCallingAgent,
    ToolCallingSubagentExecutor,
)
from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.delegation import (
    MAX_DELEGATION_ACCOUNTED_TOKENS,
    DelegationConfig,
    DelegationExecutionResult,
    DelegationRequest,
    DelegationStatus,
    DelegationTool,
)
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.local import LocalEnvironment, LocalWorkspaceDeltaObserver
from driftlock.memory import MemoryStore
from driftlock.models import RunStatus, StepContext
from driftlock.runner import DriftlockRunner, RunnerConfig


class ScriptedProvider:
    def __init__(self, *responses: AgentCompletion | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[AgentCompletionRequest] = []

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingExecutor:
    def __init__(self, *results: DelegationExecutionResult | Exception) -> None:
        self.results = list(results)
        self.requests: list[DelegationRequest] = []

    async def __call__(self, request: DelegationRequest) -> DelegationExecutionResult:
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _result(
    output: str = "child finished",
    *,
    tokens: int = 7,
    steps: int = 2,
) -> DelegationExecutionResult:
    return DelegationExecutionResult(DelegationStatus.COMPLETED, output, tokens, steps)


async def _delegate(
    tool: DelegationTool,
    objective: object = "inspect the parser",
    context: object = "focus on error handling",
    *,
    parent_tokens_remaining: int | None = None,
    max_observation_characters: int = 16_000,
):
    return await tool.delegate(
        objective,
        context,
        parent_goal="repair parser",
        sequence=3,
        logical_step=2,
        attempt=1,
        parent_tokens_remaining=parent_tokens_remaining,
        max_observation_characters=max_observation_characters,
    )


def _context(state: dict[str, object], *, tokens: int | None = None) -> StepContext:
    return StepContext(
        goal="repair parser",
        plan="inspect, patch, verify",
        state=state,
        sequence=1,
        logical_step=1,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=tokens,
    )


async def test_delegation_tool_returns_bounded_structured_result_and_checkpoint() -> (
    None
):
    executor = RecordingExecutor(_result("found the bug"))
    tool = DelegationTool(executor)

    outcome = await _delegate(tool)

    assert outcome.status is DelegationStatus.COMPLETED
    assert outcome.output == "found the bug"
    assert outcome.tokens_contributed == 7
    assert outcome.calls_before == 0
    assert outcome.calls_after == 1
    assert outcome.tokens_after == 7
    assert executor.requests == [
        DelegationRequest(
            objective="inspect the parser",
            context="focus on error handling",
            parent_goal="repair parser",
            sequence=3,
            logical_step=2,
            attempt=1,
            max_steps=8,
            max_tokens=32_000,
        )
    ]
    report = json.loads(outcome.to_observation())
    assert report["status"] == "completed"
    assert report["output"]["content"] == "found the bug"
    assert "inspect the parser" not in outcome.to_observation()
    assert len(tool.checkpoint_state()["records"]) == 1


@pytest.mark.parametrize(
    ("objective", "context", "error"),
    [
        ("", "", "non-empty"),
        (3, "", "must be text"),
        ("x" * 2_001, "", "2000-character"),
        ("valid", "x" * 4_001, "4000-character"),
        ("valid\x00", "", "null character"),
    ],
)
async def test_delegation_rejects_malformed_input_without_executor(
    objective: object, context: object, error: str
) -> None:
    executor = RecordingExecutor(_result())
    tool = DelegationTool(executor)

    outcome = await _delegate(tool, objective, context)

    assert outcome.status is DelegationStatus.REJECTED
    assert error in (outcome.error or "")
    assert outcome.calls_after == 1
    assert tool.calls_used == 1
    assert executor.requests == []


async def test_delegation_enforces_call_and_parent_token_quota() -> None:
    executor = RecordingExecutor(_result(tokens=6))
    tool = DelegationTool(
        executor,
        config=DelegationConfig(
            max_delegations_per_task=1,
            max_tokens_per_call=10,
            max_tokens_per_task=10,
        ),
    )

    first = await _delegate(tool, parent_tokens_remaining=6)
    second = await _delegate(tool)

    assert executor.requests[0].max_tokens == 6
    assert first.tokens_contributed == 6
    assert second.status is DelegationStatus.REJECTED
    assert second.calls_before == second.calls_after == 1
    assert len(tool.records) == 1


async def test_delegation_enforces_cumulative_child_token_quota() -> None:
    executor = RecordingExecutor(_result(tokens=6), _result(tokens=4))
    tool = DelegationTool(
        executor,
        config=DelegationConfig(
            max_delegations_per_task=4,
            max_tokens_per_call=6,
            max_tokens_per_task=10,
        ),
    )

    first = await _delegate(tool)
    second = await _delegate(tool)
    third = await _delegate(tool)

    assert [request.max_tokens for request in executor.requests] == [6, 4]
    assert first.tokens_after == 6
    assert second.tokens_after == 10
    assert third.status is DelegationStatus.REJECTED
    assert "token budget" in (third.error or "")
    assert tool.tokens_used == 10


async def test_concurrent_delegate_calls_serialize_shared_quota_and_checkpoint() -> (
    None
):
    calls = 0

    async def yielding_executor(_: DelegationRequest) -> DelegationExecutionResult:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return _result(tokens=1)

    tool = DelegationTool(
        yielding_executor,
        config=DelegationConfig(
            max_delegations_per_task=2,
            max_tokens_per_call=1,
            max_tokens_per_task=1,
        ),
    )

    first, second = await asyncio.gather(_delegate(tool), _delegate(tool))

    assert first.status is DelegationStatus.COMPLETED
    assert second.status is DelegationStatus.REJECTED
    assert calls == 1
    assert tool.tokens_used == 1
    checkpoint = tool.checkpoint_state()
    restored = DelegationTool(RecordingExecutor(_result()), config=tool.config)
    restored.restore_checkpoint_state(checkpoint)
    assert restored.checkpoint_state() == checkpoint


async def test_active_delegation_guards_checkpoint_access_and_survives_cancel() -> None:
    started = asyncio.Event()

    async def blocked(_: DelegationRequest) -> DelegationExecutionResult:
        started.set()
        await asyncio.Event().wait()
        return _result()

    tool = DelegationTool(blocked)
    initial = tool.checkpoint_state()
    task = asyncio.create_task(_delegate(tool))
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        with pytest.raises(RuntimeError, match="in progress"):
            tool.checkpoint_state()
        with pytest.raises(RuntimeError, match="in progress"):
            tool.restore_checkpoint_state(initial)
        with pytest.raises(RuntimeError, match="in progress"):
            tool.record_rejected_attempt("other", "malformed")
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    checkpoint = tool.checkpoint_state()
    assert checkpoint["calls_used"] == 1
    assert len(checkpoint["records"]) == 1
    assert checkpoint["records"][0]["status"] == "failed"
    assert checkpoint["records"][0]["tokens"]["accounting_known"] is False
    restored = DelegationTool(blocked)
    restored.restore_checkpoint_state(checkpoint)
    assert restored.checkpoint_state() == checkpoint
    assert (await _delegate(restored)).status is DelegationStatus.REJECTED


async def test_cancelled_waiter_does_not_consume_quota() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(_: DelegationRequest) -> DelegationExecutionResult:
        started.set()
        await release.wait()
        return _result(tokens=1)

    tool = DelegationTool(blocked)
    active = asyncio.create_task(_delegate(tool))
    await asyncio.wait_for(started.wait(), timeout=1)
    waiter = asyncio.create_task(_delegate(tool))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    assert (await active).status is DelegationStatus.COMPLETED
    assert tool.calls_used == 1
    tool.validate_checkpoint_state(tool.checkpoint_state())


@pytest.mark.parametrize("cancel_outer", [False, True])
async def test_interrupted_child_retains_prior_provider_usage_and_pauses_budget(
    tmp_path: Path,
    cancel_outer: bool,
) -> None:
    second_call = asyncio.Event()
    calls = 0

    async def provider(_: AgentCompletionRequest) -> AgentCompletion:
        nonlocal calls
        calls += 1
        if calls == 1:
            return AgentCompletion(text="first step", tokens=7)
        second_call.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    executor = ToolCallingSubagentExecutor(
        LocalEnvironment(tmp_path),
        LocalWorkspaceDeltaObserver(tmp_path),
        provider,
        min_output_tokens=1,
        prefill_estimator=lambda _: 0,
    )
    config = DelegationConfig(
        max_tokens_per_call=10,
        max_tokens_per_task=10,
        timeout_seconds=30 if cancel_outer else 0.05,
    )
    tool = DelegationTool(executor, config=config)
    task = asyncio.create_task(_delegate(tool))
    await asyncio.wait_for(second_call.wait(), timeout=1)
    if cancel_outer:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        outcome = await asyncio.wait_for(task, timeout=1)
        assert outcome.status is DelegationStatus.TIMED_OUT
        assert outcome.tokens_contributed == 7
    assert tool.tokens_used == 7
    assert tool.records[0]["tokens"]["accounting_known"] is False
    restored = DelegationTool(executor, config=config)
    restored.restore_checkpoint_state(tool.checkpoint_state())
    assert (await _delegate(restored)).status is DelegationStatus.REJECTED
    assert restored.tokens_used == 7
    assert calls == 2


async def test_parent_counts_provider_usage_when_child_tool_times_out(
    tmp_path: Path,
) -> None:
    class BlockingEnvironment:
        async def exec(self, *args, **kwargs):
            await asyncio.Event().wait()

    child_provider = ScriptedProvider(
        AgentCompletion(
            tokens=7,
            tool_calls=(ToolCall("run_shell", {"command": "work"}, "work"),),
        )
    )
    delegation = DelegationTool(
        ToolCallingSubagentExecutor(
            BlockingEnvironment(),
            LocalWorkspaceDeltaObserver(tmp_path),
            child_provider,
            min_output_tokens=1,
            prefill_estimator=lambda _: 0,
        ),
        config=DelegationConfig(timeout_seconds=0.05),
    )
    parent = ToolCallingAgent(
        LocalEnvironment(tmp_path),
        LocalWorkspaceDeltaObserver(tmp_path),
        ScriptedProvider(
            AgentCompletion(
                tokens=2,
                tool_calls=(ToolCall("delegate_task", {"objective": "work"}, "child"),),
            )
        ),
        delegation_tool=delegation,
    )
    outcome = await parent(_context(parent.initial_state()))
    assert outcome.tokens == 9
    assert outcome.tool_audits[0]["result"]["status"] == "timed_out"
    assert delegation.tokens_used == 7
    delegation.validate_checkpoint_state(delegation.checkpoint_state())


@pytest.mark.parametrize(
    ("result", "status", "known", "contributed"),
    [
        (
            RuntimeError("provider secret " + "x" * 2_000),
            DelegationStatus.FAILED,
            False,
            0,
        ),
        (object(), DelegationStatus.FAILED, False, 0),
        (_result(steps=9), DelegationStatus.FAILED, True, 7),
    ],
)
async def test_delegation_contains_executor_failures(
    result: object, status: DelegationStatus, known: bool, contributed: int
) -> None:
    executor = RecordingExecutor(result)  # type: ignore[arg-type]
    tool = DelegationTool(executor)

    outcome = await _delegate(tool)

    assert outcome.status is status
    assert outcome.token_accounting_known is known
    assert outcome.tokens_contributed == contributed
    assert len(outcome.error or "") <= 1_000


async def test_known_token_overshoot_is_counted_and_checkpointed() -> None:
    executor = RecordingExecutor(_result(tokens=10))
    tool = DelegationTool(
        executor,
        config=DelegationConfig(
            max_tokens_per_call=5,
            max_tokens_per_task=5,
        ),
    )

    outcome = await _delegate(tool)

    assert outcome.status is DelegationStatus.TOKEN_LIMIT
    assert outcome.token_accounting_known
    assert outcome.tokens_contributed == 10
    assert tool.tokens_used == 10
    restored = DelegationTool(RecordingExecutor(_result()), config=tool.config)
    restored.restore_checkpoint_state(tool.checkpoint_state())
    assert restored.tokens_used == 10


async def test_checkpoint_accepts_maximum_cumulative_overshoot() -> None:
    executor = RecordingExecutor(
        _result(tokens=1),
        _result(tokens=MAX_DELEGATION_ACCOUNTED_TOKENS),
    )
    tool = DelegationTool(
        executor,
        config=DelegationConfig(
            max_tokens_per_call=MAX_DELEGATION_ACCOUNTED_TOKENS,
            max_tokens_per_task=MAX_DELEGATION_ACCOUNTED_TOKENS,
        ),
    )

    await _delegate(tool)
    outcome = await _delegate(tool)

    assert outcome.status is DelegationStatus.TOKEN_LIMIT
    assert tool.tokens_used == MAX_DELEGATION_ACCOUNTED_TOKENS + 1
    restored = DelegationTool(RecordingExecutor(_result()), config=tool.config)
    restored.restore_checkpoint_state(tool.checkpoint_state())
    assert restored.tokens_used == MAX_DELEGATION_ACCOUNTED_TOKENS + 1


async def test_timeout_cannot_be_suppressed_by_executor_cancellation() -> None:
    returned = asyncio.Event()

    async def suppress_cancel(_: DelegationRequest) -> DelegationExecutionResult:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            returned.set()
            return _result()

    tool = DelegationTool(
        suppress_cancel,
        config=DelegationConfig(timeout_seconds=0.001),
    )

    outcome = await asyncio.wait_for(_delegate(tool), timeout=0.03)

    assert outcome.status is DelegationStatus.TIMED_OUT
    assert not returned.is_set()
    await asyncio.wait_for(returned.wait(), timeout=0.2)


async def test_cancelled_builtin_child_cannot_execute_late_provider_tool(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider_returned = asyncio.Event()

    async def swallowing_provider(_: AgentCompletionRequest) -> AgentCompletion:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_returned.set()
            return AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "write_file",
                        {"path": "late.txt", "content": "late mutation"},
                        "write-late",
                    ),
                )
            )

    environment = LocalEnvironment(workspace)
    executor = ToolCallingSubagentExecutor(
        environment,
        LocalWorkspaceDeltaObserver(workspace),
        swallowing_provider,
        min_output_tokens=1,
        prefill_estimator=lambda _: 0,
    )
    tool = DelegationTool(
        executor,
        config=DelegationConfig(timeout_seconds=0.001),
    )

    outcome = await _delegate(tool)
    await asyncio.wait_for(provider_returned.wait(), timeout=0.2)
    await asyncio.sleep(0.05)

    assert outcome.status is DelegationStatus.TIMED_OUT
    assert not (workspace / "late.txt").exists()


async def test_executor_exception_with_broken_string_is_contained() -> None:
    class BrokenError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("string rendering failed")

    executor = RecordingExecutor(BrokenError())
    tool = DelegationTool(executor)

    outcome = await _delegate(tool)

    assert outcome.status is DelegationStatus.FAILED
    assert "unavailable text" in (outcome.error or "")


async def test_delegation_timeout_is_contained_and_audited() -> None:
    async def never(_: DelegationRequest) -> DelegationExecutionResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    tool = DelegationTool(
        never,
        config=DelegationConfig(timeout_seconds=0.001),
    )

    outcome = await _delegate(tool)

    assert outcome.status is DelegationStatus.TIMED_OUT
    assert outcome.executor_invoked
    assert not outcome.token_accounting_known
    assert tool.calls_used == 1


async def test_large_child_output_is_omitted_but_digest_and_usage_remain() -> None:
    output = "sensitive-result" * 20
    executor = RecordingExecutor(_result(output, tokens=9))
    tool = DelegationTool(
        executor,
        config=DelegationConfig(max_result_characters=20),
    )

    outcome = await _delegate(tool)

    assert outcome.status is DelegationStatus.RESULT_TOO_LARGE
    assert outcome.output is None
    assert outcome.output_character_count == len(output)
    assert len(outcome.output_sha256 or "") == 64
    assert outcome.tokens_contributed == 9
    assert output not in outcome.to_observation()


async def test_long_failure_observation_compacts_with_status_and_digest() -> None:
    executor = RecordingExecutor(RuntimeError("private detail " + "x" * 2_000))
    tool = DelegationTool(executor)

    outcome = await _delegate(tool)
    observation = outcome.to_observation(max_characters=768)
    report = json.loads(observation)

    assert len(observation) <= 768
    assert report["status"] == "failed"
    assert len(report["objective_sha256"]) == 64
    assert len(report["details_omitted"]["sha256"]) == 64
    assert "private detail" not in observation


def test_delegation_checkpoint_round_trip_and_malformed_matrix() -> None:
    tool = DelegationTool(RecordingExecutor(_result()))
    rejected = tool.record_rejected_attempt("objective", "bad request")
    assert rejected.status is DelegationStatus.REJECTED
    checkpoint = tool.checkpoint_state()
    assert checkpoint["records"][0]["output"] == {
        "character_count": 0,
        "sha256": None,
        "included": False,
    }
    restored = DelegationTool(RecordingExecutor(_result()))
    restored.restore_checkpoint_state(checkpoint)
    assert restored.checkpoint_state() == checkpoint
    checkpoint["records"][0]["tokens"]["after"] = 999
    assert restored.tokens_used == 0
    exposed = restored.records
    exposed[0]["tokens"]["after"] = 999
    assert restored.checkpoint_state()["records"][0]["tokens"]["after"] == 0

    malformed: Sequence[object] = (
        None,
        {},
        {**checkpoint, "schema_version": 2},
        {**checkpoint, "calls_used": True},
        {**checkpoint, "tokens_used": -1},
        {**checkpoint, "records": []},
        {**checkpoint, "records": [["not", "an", "object"]]},
        {**checkpoint, "records": [{"x": "y" * 20_000}]},
    )
    for value in malformed:
        with pytest.raises(ValueError):
            restored.restore_checkpoint_state(value)


async def test_checkpoint_retains_output_evidence_and_rejects_impossible_flags() -> (
    None
):
    tool = DelegationTool(RecordingExecutor(_result("evidence")))
    await _delegate(tool)
    checkpoint = tool.checkpoint_state()
    output = checkpoint["records"][0]["output"]
    assert output["character_count"] == 8
    assert len(output["sha256"]) == 64
    assert output["included"] is True
    assert "content" not in output

    impossible = json.loads(json.dumps(checkpoint))
    impossible["records"][0]["executor_invoked"] = False
    with pytest.raises(ValueError, match="lacks invocation"):
        tool.restore_checkpoint_state(impossible)


async def test_checkpoint_rejects_impossible_status_evidence_combinations() -> None:
    limited = DelegationTool(
        RecordingExecutor(_result(tokens=10)),
        config=DelegationConfig(max_tokens_per_call=5, max_tokens_per_task=5),
    )
    await _delegate(limited)
    unknown_limit = limited.checkpoint_state()
    token_report = unknown_limit["records"][0]["tokens"]
    token_report["accounting_known"] = False
    token_report["contributed"] = 0
    token_report["after"] = token_report["before"]
    unknown_limit["tokens_used"] = 0
    with pytest.raises(ValueError, match="limited delegation"):
        limited.restore_checkpoint_state(unknown_limit)

    rejected = DelegationTool(RecordingExecutor(_result()))
    rejected.record_rejected_attempt("objective", "bad request")
    retained_rejection = rejected.checkpoint_state()
    retained_rejection["records"][0]["output"].update(
        {"included": True, "character_count": 999}
    )
    with pytest.raises(ValueError, match="rejected delegation"):
        rejected.restore_checkpoint_state(retained_rejection)


@pytest.mark.parametrize("tokens", [32_001, 65_000])
async def test_checkpoint_rejects_completed_over_configured_quota(tokens: int) -> None:
    tool = DelegationTool(RecordingExecutor(_result(tokens=1)))
    await _delegate(tool)
    checkpoint = tool.checkpoint_state()
    checkpoint["tokens_used"] = tokens
    checkpoint["records"][0]["tokens"].update(contributed=tokens, after=tokens)
    with pytest.raises(ValueError, match="token budget"):
        tool.restore_checkpoint_state(checkpoint)
    assert tool.tokens_used == 1


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1])
def test_delegation_config_rejects_non_finite_or_non_positive_timeout(
    timeout: float,
) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        DelegationConfig(timeout_seconds=timeout)


async def test_agent_exposes_delegation_only_when_configured_and_counts_tokens(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(
                ToolCall(
                    "delegate_task",
                    {"objective": "diagnose failure", "context": "read-only first"},
                    "delegate-1",
                ),
            ),
            tokens=11,
        )
    )
    executor = RecordingExecutor(_result("diagnosis", tokens=7, steps=1))
    delegation = DelegationTool(executor)
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        delegation_tool=delegation,
    )

    outcome = await agent(_context(agent.initial_state(), tokens=100_000))

    assert outcome.action == "Delegate task: diagnose failure"
    assert outcome.tokens == 18
    assert json.loads(outcome.tool_observations[0].split("\n", 1)[1])["status"] == (
        "completed"
    )
    assert outcome.tool_audits[0]["tool_call"]["arguments"] == {
        "objective": {
            "character_count": 16,
            "sha256": outcome.tool_audits[0]["result"]["objective"]["sha256"],
        },
        "context": {
            "character_count": 15,
            "sha256": outcome.tool_audits[0]["tool_call"]["arguments"]["context"][
                "sha256"
            ],
        },
    }
    assert provider.requests[0].tools[-1].name == "delegate_task"
    assert executor.requests[0].max_tokens < 100_000
    state = outcome.state["driftlock_tool_agent"]
    assert state["schema_version"] == 4
    assert state["delegation_checkpoint"]["calls_used"] == 1

    plain_provider = ScriptedProvider(AgentCompletion())
    plain = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        plain_provider,
    )
    assert plain.initial_state() == AgentConversationCodec().initial_state()
    await plain(_context(plain.initial_state()))
    assert "delegate_task" not in {
        tool.name for tool in plain_provider.requests[0].tools
    }
    assert "delegate_task" not in plain_provider.requests[0].messages[0]["content"]


async def test_parent_counts_known_child_token_overshoot_exactly(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(
                ToolCall("delegate_task", {"objective": "inspect"}, "delegate-1"),
            ),
            tokens=1,
        )
    )
    delegation = DelegationTool(
        RecordingExecutor(_result(tokens=10)),
        config=DelegationConfig(
            max_tokens_per_call=5,
            max_tokens_per_task=5,
        ),
    )
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        delegation_tool=delegation,
    )

    outcome = await agent(_context(agent.initial_state(), tokens=100_000))

    assert outcome.tokens == 11
    assert outcome.error == "delegated child exceeded its assigned token budget"
    assert outcome.tool_audits[0]["result"]["status"] == "token_limit"
    assert outcome.tool_audits[0]["result"]["tokens"]["contributed"] == 10


async def test_runner_stops_on_and_reports_known_child_token_overshoot(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(
                ToolCall("delegate_task", {"objective": "inspect"}, "delegate-1"),
            ),
            tokens=1,
        )
    )
    delegation = DelegationTool(
        RecordingExecutor(_result(tokens=10)),
        config=DelegationConfig(max_tokens_per_call=5, max_tokens_per_task=5),
    )
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        delegation_tool=delegation,
        min_output_tokens=1,
        prefill_estimator=lambda _: 0,
    )
    runner = DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(),
        config=RunnerConfig(max_steps=2, max_tokens=6),
    )

    result = await runner.run(
        goal="repair parser", step=agent, initial_state=agent.initial_state()
    )

    assert result.status is RunStatus.TOKEN_LIMIT
    assert result.agent_tokens_used == 11
    assert result.tokens_used == 11


async def test_multiple_delegations_share_parent_step_token_budget(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(
                ToolCall("delegate_task", {"objective": "one"}, "delegate-1"),
                ToolCall("delegate_task", {"objective": "two"}, "delegate-2"),
            ),
            tokens=5,
        )
    )
    executor = RecordingExecutor(_result(tokens=7), _result(tokens=3))
    delegation = DelegationTool(
        executor,
        config=DelegationConfig(
            max_tokens_per_call=100,
            max_tokens_per_task=100,
        ),
    )
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        delegation_tool=delegation,
        min_output_tokens=1,
        prefill_estimator=lambda _: 0,
    )

    outcome = await agent(_context(agent.initial_state(), tokens=20))

    assert [request.max_tokens for request in executor.requests] == [15, 8]
    assert outcome.tokens == 15


async def test_malformed_delegate_call_is_structured_and_never_invokes_executor(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(ToolCall("delegate_task", "not-json", "bad-delegate"),)
        )
    )
    executor = RecordingExecutor(_result())
    delegation = DelegationTool(executor)
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        delegation_tool=delegation,
    )

    outcome = await agent(_context(agent.initial_state()))

    report = outcome.tool_audits[0]["result"]
    assert report["status"] == "rejected"
    assert delegation.calls_used == 1
    assert executor.requests == []


async def test_malformed_delegate_arguments_with_broken_repr_are_contained(
    tmp_path: Path,
) -> None:
    class BrokenRepresentation:
        def __repr__(self) -> str:
            raise RuntimeError("representation failed")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(
                ToolCall("delegate_task", BrokenRepresentation(), "bad-delegate"),
            )
        )
    )
    executor = RecordingExecutor(_result())
    delegation = DelegationTool(executor)
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        delegation_tool=delegation,
    )

    outcome = await agent(_context(agent.initial_state()))

    assert outcome.completed is False
    assert outcome.tool_audits[0]["result"]["status"] == "rejected"
    assert outcome.tool_audits[0]["tool_call"]["arguments"]["argument_type"] == (
        "BrokenRepresentation"
    )
    assert delegation.calls_used == 1
    assert executor.requests == []


async def test_nested_nonrenderable_objective_is_rejected_before_executor(
    tmp_path: Path,
) -> None:
    class BrokenText:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

        def __repr__(self) -> str:
            raise RuntimeError("cannot represent")

    executor = RecordingExecutor(_result())
    tool = DelegationTool(executor)
    agent = ToolCallingAgent(
        LocalEnvironment(tmp_path),
        LocalWorkspaceDeltaObserver(tmp_path),
        ScriptedProvider(
            AgentCompletion(
                tool_calls=(
                    ToolCall("delegate_task", {"objective": BrokenText()}, "malformed"),
                )
            )
        ),
        delegation_tool=tool,
    )
    outcome = await agent(_context(agent.initial_state()))
    assert outcome.action == "Delegate task with malformed objective"
    assert outcome.tool_audits[0]["result"]["status"] == "rejected"
    assert executor.requests == []
    assert tool.calls_used == 1
    tool.validate_checkpoint_state(tool.checkpoint_state())


def test_codec_combines_memory_and_delegation_without_changing_legacy_schema() -> None:
    codec = AgentConversationCodec()
    legacy = codec.initial_state()
    memory = {"schema_version": 1, "entries": []}
    delegation = {
        "schema_version": 1,
        "calls_used": 0,
        "tokens_used": 0,
        "records": [],
    }

    assert legacy[codec.state_key]["schema_version"] == 2
    combined = codec.initial_state(
        memory_checkpoint=memory, delegation_checkpoint=delegation
    )
    decoded = codec.decode_with_extensions(combined)
    assert combined[codec.state_key]["schema_version"] == 4
    assert decoded[3:] == (memory, delegation)

    malformed = json.loads(json.dumps(combined))
    del malformed[codec.state_key]["delegation_checkpoint"]
    with pytest.raises(AgentStateError, match="fields are malformed"):
        codec.decode_with_extensions(malformed)


async def test_real_memory_and_delegation_restore_together(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(
                ToolCall(
                    "manage_memory",
                    {"operation": "record", "content": "temporary child fact"},
                    "memory-1",
                ),
                ToolCall("delegate_task", {"objective": "inspect"}, "delegate-1"),
            )
        )
    )
    memory = MemoryStore(tmp_path / "memory")
    delegation = DelegationTool(RecordingExecutor(_result()))
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        memory_store=memory,
        memory_task_id="task-parser",
        memory_run_id="run-1",
        delegation_tool=delegation,
    )
    initial = agent.initial_state()

    outcome = await agent(_context(initial))

    assert outcome.state["driftlock_tool_agent"]["schema_version"] == 4
    assert len(memory.current_entries()) == 1
    assert delegation.calls_used == 1
    agent.restore_checkpoint_state(initial)
    assert memory.current_entries() == ()
    assert delegation.calls_used == 0


async def test_builtin_subagent_is_fresh_non_recursive_and_can_edit_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(
                ToolCall(
                    "write_file",
                    {"path": "child.txt", "content": "made by child\n"},
                    "write-1",
                ),
            ),
            tokens=3,
        ),
        AgentCompletion(
            tool_calls=(
                ToolCall("complete", {"summary": "child edit complete"}, "done-1"),
            ),
            tokens=4,
        ),
    )
    environment = LocalEnvironment(workspace)
    executor = ToolCallingSubagentExecutor(
        environment,
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        min_output_tokens=1,
        prefill_estimator=lambda _: 0,
    )

    result = await executor(
        DelegationRequest(
            objective="create child.txt",
            context="write one line",
            parent_goal="prepare fixture",
            sequence=1,
            logical_step=1,
            attempt=1,
            max_steps=3,
            max_tokens=100,
        )
    )

    assert result == DelegationExecutionResult(
        DelegationStatus.COMPLETED, "child edit complete", 7, 2
    )
    assert (workspace / "child.txt").read_text(encoding="utf-8") == "made by child\n"
    assert len(provider.requests) == 2
    assert all(
        "delegate_task" not in {tool.name for tool in request.tools}
        for request in provider.requests
    )
    assert (
        "Parent goal:\nprepare fixture" in provider.requests[0].messages[1]["content"]
    )
    assert (
        "Delegated objective:\ncreate child.txt"
        in provider.requests[0].messages[1]["content"]
    )


async def test_builtin_subagent_reports_step_and_token_limits(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    step_provider = ScriptedProvider(AgentCompletion(text="thinking", tokens=2))
    step_executor = ToolCallingSubagentExecutor(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        step_provider,
        min_output_tokens=1,
        prefill_estimator=lambda _: 0,
    )
    request = DelegationRequest("inspect", "", "repair", 1, 1, 1, 1, 10)
    step_result = await step_executor(request)
    assert step_result.status is DelegationStatus.STEP_LIMIT
    assert step_result.tokens == 2

    token_provider = ScriptedProvider(AgentCompletion(text="thinking", tokens=10))
    token_executor = ToolCallingSubagentExecutor(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        token_provider,
        min_output_tokens=1,
        prefill_estimator=lambda _: 0,
    )
    token_result = await token_executor(
        DelegationRequest("inspect", "", "repair", 1, 1, 1, 2, 5)
    )
    assert token_result.status is DelegationStatus.TOKEN_LIMIT
    assert token_result.tokens == 10


async def test_builtin_subagent_reports_provider_failure_with_billed_tokens(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(AgentProviderError("unavailable", tokens=3))
    executor = ToolCallingSubagentExecutor(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        min_output_tokens=1,
        prefill_estimator=lambda _: 0,
    )

    result = await executor(DelegationRequest("inspect", "", "repair", 1, 1, 1, 2, 10))

    assert result.status is DelegationStatus.FAILED
    assert result.tokens == 3
    assert result.steps == 1
    assert "unavailable" in (result.error or "")


async def test_runner_rollback_restores_delegation_quota_even_on_final_step(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        *(
            AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "delegate_task",
                        {"objective": f"attempt {index}"},
                        f"delegate-{index}",
                    ),
                )
            )
            for index in (1, 2)
        )
    )
    executor = RecordingExecutor(_result(), _result())
    delegation = DelegationTool(executor)
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        delegation_tool=delegation,
    )
    runner = DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=2,
                loop_window=2,
                loop_repetitions=2,
                error_window=10,
                reward_stall_steps=10,
                corroborating_signals=frozenset(),
            )
        ),
        config=RunnerConfig(max_steps=2, max_rollbacks=1),
    )

    result = await runner.run(
        goal="repair parser", step=agent, initial_state=agent.initial_state()
    )

    assert result.status is RunStatus.STEP_LIMIT
    assert len(result.rollbacks) == 1
    assert len(executor.requests) == 2
    assert delegation.calls_used == 0
    assert delegation.tokens_used == 0
    assert delegation.records == ()


async def test_retry_can_delegate_again_after_rollback_restores_quota(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(
                ToolCall("delegate_task", {"objective": "first"}, "delegate-1"),
            )
        ),
        AgentCompletion(
            tool_calls=(
                ToolCall("delegate_task", {"objective": "second"}, "delegate-2"),
            )
        ),
        AgentCompletion(
            tool_calls=(
                ToolCall("delegate_task", {"objective": "retry"}, "delegate-3"),
            )
        ),
        AgentCompletion(
            tool_calls=(
                ToolCall("complete", {"summary": "retry finished"}, "complete-1"),
            )
        ),
    )
    executor = RecordingExecutor(_result(), _result(), _result())
    delegation = DelegationTool(
        executor,
        config=DelegationConfig(max_delegations_per_task=1),
    )
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        delegation_tool=delegation,
    )
    runner = DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=2,
                loop_window=2,
                loop_repetitions=2,
                error_window=10,
                reward_stall_steps=10,
                corroborating_signals=frozenset(),
            )
        ),
        config=RunnerConfig(max_steps=4, max_rollbacks=1),
    )

    result = await runner.run(
        goal="repair parser", step=agent, initial_state=agent.initial_state()
    )

    assert result.status is RunStatus.COMPLETED
    assert len(result.rollbacks) == 1
    assert [request.objective for request in executor.requests] == [
        "first",
        "retry",
    ]
    assert delegation.calls_used == 1
    assert delegation.tokens_used == 7
