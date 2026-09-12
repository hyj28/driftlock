from __future__ import annotations

import ast
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from driftlock.agent import (
    AgentCompletion,
    AgentCompletionRequest,
    AgentConversationCodec,
    AgentProviderError,
    ToolCall,
    ToolCallingAgent,
)
from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.lhtb import WorkspaceDelta, WorkspaceSnapshot
from driftlock.local import LocalEnvironment, LocalWorkspaceDeltaObserver
from driftlock.models import RunStatus, StepContext, VerificationRunStatus
from driftlock.runner import DriftlockRunner, RunnerConfig
from driftlock.verification import (
    MAX_VERIFICATION_EVIDENCE_CHARACTERS,
    SelfVerificationConfig,
    VerificationCheckpoint,
    VerificationStatus,
)


class ScriptedProvider:
    def __init__(self, responses: Sequence[AgentCompletion | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[AgentCompletionRequest] = []

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _completion(call_id: str = "complete") -> AgentCompletion:
    return AgentCompletion(
        tool_calls=(ToolCall("complete", {"summary": "work is done"}, call_id),),
        tokens=3,
    )


def _verification(command: str, call_id: str = "verify") -> AgentCompletion:
    return AgentCompletion(
        tool_calls=(ToolCall("run_verification", {"command": command}, call_id),),
        tokens=5,
    )


def _config(*, max_attempts: int = 3) -> SelfVerificationConfig:
    return SelfVerificationConfig(
        max_attempts=max_attempts,
        max_output_tokens=10,
        min_output_tokens=3,
        max_tokens=100,
    )


def _context(
    state: Mapping[str, Any],
    *,
    sequence: int = 1,
    tokens_remaining: int | None = 100,
) -> StepContext:
    return StepContext(
        goal="write answer.txt containing done",
        plan="write then verify",
        state=state,
        sequence=sequence,
        logical_step=sequence,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=tokens_remaining,
    )


def _agent(
    workspace: Path,
    provider: ScriptedProvider,
    *,
    verification: bool = True,
    max_attempts: int = 3,
) -> ToolCallingAgent:
    return ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        min_output_tokens=1,
        prefill_estimator=lambda _request: 2,
        self_verification=_config(max_attempts=max_attempts) if verification else None,
    )


def _assert_no_orphaned_tool_results(messages: Sequence[Mapping[str, Any]]) -> None:
    calls: Counter[str] = Counter()
    results: Counter[str] = Counter()
    for message in messages:
        for call in message.get("tool_calls", []):
            calls[call["id"]] += 1
        if message.get("role") == "tool":
            results[message["tool_call_id"]] += 1
    assert calls == results


async def _bind_local_control(
    agent: ToolCallingAgent,
    workspace: Path,
    store_dir: Path,
    state: Mapping[str, Any],
) -> None:
    store = DirectoryCheckpointStore(workspace, store_dir)
    initial = store.create(state, step=0, label="initial")

    async def control(
        current_state: Mapping[str, Any],
        current_step: int,
        operation: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, Any]:
        scratch = store.create(current_state, step=current_step, label="scratch")
        try:
            store.restore(initial)
            before = await operation()
            store.restore(scratch)
            after = await operation()
            return before, after
        finally:
            store.restore(scratch)
            store.discard(scratch)

    agent.configure_verification_control(control)


@pytest.mark.parametrize(
    ("verifier_response", "expected_status", "expected_completed", "make_artifact"),
    [
        (_verification("test -f answer.txt"), VerificationStatus.VERIFIED, True, True),
        (_verification("exit 1"), VerificationStatus.REFUTED, False, False),
        (
            AgentCompletion(text="nothing objective exists", tokens=5),
            VerificationStatus.UNVERIFIABLE,
            False,
            False,
        ),
        (
            AgentCompletion(
                tool_calls=(ToolCall("unknown_check", {}, "bad"),), tokens=5
            ),
            VerificationStatus.MALFORMED,
            False,
            False,
        ),
    ],
)
@pytest.mark.parametrize("tokens_remaining", [100, 5])
async def test_verification_on_enumerates_outcomes_with_budget_boundary(
    tmp_path: Path,
    verifier_response: AgentCompletion,
    expected_status: VerificationStatus,
    expected_completed: bool,
    make_artifact: bool,
    tokens_remaining: int,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider([_completion(), verifier_response])
    agent = _agent(workspace, provider)

    state = agent.initial_state()
    await _bind_local_control(agent, workspace, tmp_path / "control", state)
    if make_artifact:
        (workspace / "answer.txt").write_text("done\n", encoding="utf-8")
    outcome = await agent(_context(state, tokens_remaining=tokens_remaining))

    assert outcome.verification is not None
    if tokens_remaining == 5:
        assert outcome.verification.status is VerificationStatus.BUDGET_EXHAUSTED
        assert outcome.completed is False
        assert len(provider.requests) == 1
        assert outcome.tokens == 3
    else:
        assert outcome.verification.status is expected_status
        assert outcome.completed is expected_completed
        assert len(provider.requests) == 2
        assert outcome.tokens == 8


@pytest.mark.parametrize("tokens_remaining", [100, 3])
async def test_verification_off_is_identical_at_both_budget_boundaries(
    tmp_path: Path,
    tokens_remaining: int,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider([_completion()])
    agent = _agent(workspace, provider, verification=False)

    outcome = await agent(
        _context(agent.initial_state(), tokens_remaining=tokens_remaining)
    )

    assert outcome.completed is True
    assert outcome.verification is None
    assert outcome.tokens == 3
    assert len(provider.requests) == 1


def test_verification_status_values_are_unique_and_exhaustive() -> None:
    assert [status.value for status in VerificationStatus] == [
        "verified",
        "refuted",
        "unverifiable",
        "transient_error",
        "malformed",
        "budget_exhausted",
    ]
    assert len({status.value for status in VerificationStatus}) == 6
    assert [status.value for status in RunStatus] == [
        "completed",
        "step_limit",
        "token_limit",
        "rollback_limit",
    ]
    assert len({status.value for status in RunStatus}) == 4
    assert [status.value for status in VerificationRunStatus] == [
        "verification_limit",
        "verification_unavailable",
        "verification_budget",
    ]
    assert len({status.value for status in VerificationRunStatus}) == 3


async def test_passing_verification_reaches_completed_runner_status(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [
            AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "write_file",
                        {"path": "answer.txt", "content": "done\n"},
                        "write",
                    ),
                    ToolCall("complete", {"summary": "work is done"}, "complete"),
                ),
                tokens=3,
            ),
            _verification("grep -Fx done answer.txt"),
        ]
    )
    agent = _agent(workspace, provider)

    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(),
        config=RunnerConfig(max_steps=2),
    ).run(
        goal="write answer.txt containing done",
        step=agent,
        initial_state=agent.initial_state(),
    )

    assert result.status is RunStatus.COMPLETED
    assert result.steps[0].outcome.verification is not None
    assert result.steps[0].outcome.verification.status is VerificationStatus.VERIFIED
    assert result.verification_status_counts == {
        "verified": 1,
        "refuted": 0,
        "unverifiable": 0,
        "transient_error": 0,
        "malformed": 0,
        "budget_exhausted": 0,
    }
    assert result.verification_tokens_used == 5


@pytest.mark.parametrize("command", ["true", ":", "exit 0", "echo ok"])
async def test_nondiscriminating_success_cannot_verify(
    tmp_path: Path, command: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider([_completion(), _verification(command)])
    agent = _agent(workspace, provider, max_attempts=1)

    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(),
        config=RunnerConfig(max_steps=4),
    ).run(
        goal="prove the Riemann hypothesis",
        step=agent,
        initial_state=agent.initial_state(),
    )

    record = result.verification_records[0]
    assert result.status is VerificationRunStatus.VERIFICATION_LIMIT
    assert record.status is VerificationStatus.UNVERIFIABLE
    assert record.return_code == 0
    assert record.control_return_code == 0
    assert record.attempt_limit_reached is True


async def test_verification_writes_are_restored_and_commands_are_accounted(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [
            _completion(),
            _verification("rm -rf ./* ; echo wiped > wiped.txt"),
        ]
    )
    agent = _agent(workspace, provider, max_attempts=1)

    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(),
        config=RunnerConfig(max_steps=4),
    ).run(goal="finish", step=agent, initial_state=agent.initial_state())

    outcome = result.steps[0].outcome
    assert result.status is VerificationRunStatus.VERIFICATION_LIMIT
    assert outcome.verification is not None
    assert outcome.verification.status is VerificationStatus.TRANSIENT_ERROR
    assert outcome.commands_run == 2
    assert outcome.commands_failed == 0
    assert outcome.changed_paths == ()
    assert not (workspace / "wiped.txt").exists()


@pytest.mark.parametrize(
    "transient",
    [AgentProviderError("rate limited", tokens=0), _verification("exit 127")],
)
async def test_transient_verification_failure_retries_before_terminating(
    tmp_path: Path,
    transient: AgentProviderError | AgentCompletion,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [
            _completion("c1"),
            transient,
            _completion("c2"),
            AgentCompletion(text="no checkable surface", tokens=5),
        ]
    )
    agent = _agent(workspace, provider, max_attempts=3)

    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=10,
                loop_window=10,
                error_window=10,
                reward_stall_steps=10,
            )
        ),
        config=RunnerConfig(max_steps=10),
    ).run(goal="finish", step=agent, initial_state=agent.initial_state())

    assert result.status is VerificationRunStatus.VERIFICATION_UNAVAILABLE
    assert len(result.steps) == 2
    assert result.verification_records[0].status is VerificationStatus.TRANSIENT_ERROR
    assert result.verification_records[0].attempt_limit_reached is False
    assert result.verification_records[1].status is VerificationStatus.UNVERIFIABLE


async def test_verification_budget_status_does_not_invent_runner_token_limit(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider([_completion()])
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        min_output_tokens=1,
        prefill_estimator=lambda _request: 2,
        self_verification=SelfVerificationConfig(
            max_attempts=3,
            max_output_tokens=4,
            min_output_tokens=3,
            max_tokens=4,
        ),
    )

    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(),
        config=RunnerConfig(max_steps=10, max_tokens=None),
    ).run(goal="finish", step=agent, initial_state=agent.initial_state())

    assert result.status is VerificationRunStatus.VERIFICATION_BUDGET
    assert len(result.steps) == 1
    assert result.verification_records[0].status is VerificationStatus.BUDGET_EXHAUSTED


async def test_refutation_blocks_completion_and_is_actionable_next_step(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [_completion(), _verification("exit 1"), AgentCompletion(text="continue")]
    )
    agent = _agent(workspace, provider)
    state = agent.initial_state()
    await _bind_local_control(agent, workspace, tmp_path / "control", state)
    first = await agent(_context(state))

    second = await agent(_context(first.state, sequence=2))

    assert first.completed is False
    assert first.verification is not None
    assert first.verification.status is VerificationStatus.REFUTED
    assert "refuted" in (first.error or "")
    assert "Repair the work" in json.dumps(provider.requests[2].messages)
    assert second.verification is None


@pytest.mark.parametrize(
    ("response", "expected_status", "expected_reason"),
    [
        (
            AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "report_unverifiable",
                        {"reason": "the goal has no observable workspace surface"},
                        "none",
                    ),
                ),
                tokens=5,
            ),
            VerificationStatus.UNVERIFIABLE,
            "no observable workspace surface",
        ),
        (_verification("exit 2"), VerificationStatus.TRANSIENT_ERROR, "exit code 2"),
    ],
)
async def test_unverifiable_is_distinct_and_visible(
    tmp_path: Path,
    response: AgentCompletion,
    expected_status: VerificationStatus,
    expected_reason: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider([_completion(), response])
    agent = _agent(workspace, provider)
    state = agent.initial_state()
    await _bind_local_control(agent, workspace, tmp_path / "control", state)
    outcome = await agent(_context(state))

    assert outcome.completed is False
    assert outcome.verification is not None
    assert outcome.verification.status is expected_status
    assert expected_reason in outcome.verification.reason
    assert outcome.verification.status is not VerificationStatus.VERIFIED
    assert outcome.verification.status is not VerificationStatus.REFUTED


async def test_unverifiable_terminates_runner_without_claiming_completion(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [_completion(), AgentCompletion(text="no checkable surface", tokens=5)]
    )
    agent = _agent(workspace, provider)
    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(),
        config=RunnerConfig(max_steps=4),
    ).run(
        goal="give subjective advice", step=agent, initial_state=agent.initial_state()
    )

    assert result.status is VerificationRunStatus.VERIFICATION_UNAVAILABLE
    assert result.steps[0].outcome.completed is False
    assert result.verification_records[0].status is VerificationStatus.UNVERIFIABLE


async def test_verification_execution_exception_is_unverifiable_not_a_crash() -> None:
    class RaisingEnvironment:
        async def exec(self, *args: object, **kwargs: object) -> object:
            raise FileNotFoundError("verification executable missing")

    @dataclass(frozen=True, slots=True)
    class EmptyObserver:
        async def canonical_workspace(self) -> str:
            return "/workspace"

        async def snapshot(self) -> WorkspaceSnapshot:
            return WorkspaceSnapshot(files={})

        def compare(
            self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
        ) -> WorkspaceDelta:
            return WorkspaceDelta()

    provider = ScriptedProvider([_completion(), _verification("missing-check")])
    agent = ToolCallingAgent(
        RaisingEnvironment(),
        EmptyObserver(),
        provider,
        min_output_tokens=1,
        prefill_estimator=lambda _request: 2,
        self_verification=_config(),
    )

    async def execute_twice(
        _state: Mapping[str, Any],
        _step: int,
        operation: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, Any]:
        return await operation(), await operation()

    agent.configure_verification_control(execute_twice)

    outcome = await agent(_context(agent.initial_state()))

    assert outcome.completed is False
    assert outcome.verification is not None
    assert outcome.verification.status is VerificationStatus.TRANSIENT_ERROR
    assert "executable missing" in outcome.verification.reason


async def test_repeated_refutation_has_recorded_terminal_bound(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [
            _completion("c1"),
            _verification("exit 1", "v1"),
            _completion("c2"),
            _verification("exit 1", "v2"),
        ]
    )
    agent = _agent(workspace, provider, max_attempts=2)
    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=10,
                loop_window=10,
                error_window=10,
                reward_stall_steps=10,
            )
        ),
        config=RunnerConfig(max_steps=10),
    ).run(goal="finish", step=agent, initial_state=agent.initial_state())

    assert result.status is VerificationRunStatus.VERIFICATION_LIMIT
    assert len(result.steps) == 2
    assert [record.status for record in result.verification_records] == [
        VerificationStatus.REFUTED,
        VerificationStatus.REFUTED,
    ]
    assert result.verification_records[-1].attempt_limit_reached is True


async def test_verification_budget_exhaustion_ends_runner_without_overrun(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider([_completion(), _verification("exit 0")])
    agent = _agent(workspace, provider)
    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(),
        config=RunnerConfig(max_steps=4, max_tokens=5),
    ).run(goal="finish", step=agent, initial_state=agent.initial_state())

    assert result.status is VerificationRunStatus.VERIFICATION_BUDGET
    assert result.tokens_used == 3
    assert result.tokens_used <= 5
    assert len(provider.requests) == 1
    assert result.verification_records[0].status is (
        VerificationStatus.BUDGET_EXHAUSTED
    )


async def test_cumulative_verification_token_share_is_bounded_and_recorded(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [_completion("c1"), _verification("exit 1"), _completion("c2")]
    )
    config = SelfVerificationConfig(
        max_attempts=3,
        max_output_tokens=5,
        min_output_tokens=3,
        max_tokens=8,
    )
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        min_output_tokens=1,
        prefill_estimator=lambda _request: 2,
        self_verification=config,
    )
    state = agent.initial_state()
    await _bind_local_control(agent, workspace, tmp_path / "control", state)
    first = await agent(_context(state))
    second = await agent(_context(first.state, sequence=2))

    assert first.verification is not None
    assert first.verification.status is VerificationStatus.REFUTED
    assert second.verification is not None
    assert second.verification.status is VerificationStatus.BUDGET_EXHAUSTED
    assert first.tokens + second.tokens == 11
    assert (
        sum(
            record["tokens"]
            for record in second.state["driftlock_tool_agent"][
                "verification_checkpoint"
            ]["records"]
        )
        == 5
    )
    assert len(provider.requests) == 3


async def test_verification_evidence_truncation_is_recorded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [
            _completion(),
            _verification("python3 -c \"print('x' * 5000)\"; exit 1"),
        ]
    )
    agent = _agent(workspace, provider)
    state = agent.initial_state()
    await _bind_local_control(agent, workspace, tmp_path / "control", state)
    outcome = await agent(_context(state))

    assert outcome.verification is not None
    assert outcome.verification.status is VerificationStatus.REFUTED
    assert len(outcome.verification.evidence) == MAX_VERIFICATION_EVIDENCE_CHARACTERS
    assert outcome.verification.evidence_truncated is True


async def test_fail_more_work_then_pass_records_both_attempts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [
            _completion("c1"),
            _verification("test -f answer.txt", "v1"),
            AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "write_file",
                        {"path": "answer.txt", "content": "done\n"},
                        "write",
                    ),
                )
            ),
            _completion("c2"),
            _verification("grep -Fx done answer.txt", "v2"),
        ]
    )
    agent = _agent(workspace, provider)
    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(),
        config=RunnerConfig(max_steps=5),
    ).run(
        goal="write answer.txt containing done",
        step=agent,
        initial_state=agent.initial_state(),
    )

    assert result.status is RunStatus.COMPLETED
    assert [record.status for record in result.verification_records] == [
        VerificationStatus.REFUTED,
        VerificationStatus.VERIFIED,
    ]
    assert [record.attempt for record in result.verification_records] == [1, 2]


async def test_verification_checkpoint_survives_runner_rollback_without_orphans(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [
            _completion("c1"),
            _verification("exit 1", "v1"),
            AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "write_file",
                        {"path": "answer.txt", "content": "done\n"},
                        "write",
                    ),
                    ToolCall("complete", {"summary": "c2"}, "c2"),
                )
            ),
            _verification("test -f answer.txt", "v2"),
        ]
    )
    agent = _agent(workspace, provider)
    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "checkpoints"),
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=10,
                loop_window=10,
                loop_repetitions=10,
                error_window=1,
                error_rate=1.0,
                reward_stall_steps=10,
                corroborating_signals=frozenset(),
            )
        ),
        config=RunnerConfig(max_steps=4, max_rollbacks=1),
    ).run(goal="finish", step=agent, initial_state=agent.initial_state())

    assert result.status is RunStatus.COMPLETED
    assert len(result.rollbacks) == 1
    assert [record.attempt for record in result.verification_records] == [1, 2]
    messages, _, _, _, _, raw_verification = (
        AgentConversationCodec().decode_with_verification(result.state)
    )
    checkpoint = VerificationCheckpoint.from_dict(raw_verification, config=_config())
    assert checkpoint.attempts_used == 2
    _assert_no_orphaned_tool_results(messages)
    for step in result.steps:
        step_messages = AgentConversationCodec().decode(step.outcome.state)[0]
        _assert_no_orphaned_tool_results(step_messages)
    _assert_no_orphaned_tool_results(provider.requests[2].messages[2:])


@pytest.mark.parametrize(
    "bad_call",
    [
        ToolCall("run_verification", {}, "missing-command"),
        ToolCall("made_up_verifier", {}, "unknown-tool"),
    ],
)
async def test_malformed_verification_tool_call_is_recorded(
    tmp_path: Path, bad_call: ToolCall
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [_completion(), AgentCompletion(tool_calls=(bad_call,), tokens=5)]
    )
    agent = _agent(workspace, provider)

    outcome = await agent(_context(agent.initial_state()))

    assert outcome.completed is False
    assert outcome.verification is not None
    assert outcome.verification.status is VerificationStatus.MALFORMED
    assert "Malformed verification tool call" in (outcome.error or "")


async def test_malformed_verification_result_is_recorded_not_raised() -> None:
    @dataclass(frozen=True, slots=True)
    class BadResult:
        return_code: str = "zero"
        stdout: str = ""
        stderr: str = ""

    class BadResultEnvironment:
        async def exec(self, *args: object, **kwargs: object) -> BadResult:
            return BadResult()

    @dataclass(frozen=True, slots=True)
    class EmptyObserver:
        async def canonical_workspace(self) -> str:
            return "/workspace"

        async def snapshot(self) -> WorkspaceSnapshot:
            return WorkspaceSnapshot(files={})

        def compare(
            self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
        ) -> WorkspaceDelta:
            return WorkspaceDelta()

    provider = ScriptedProvider([_completion(), _verification("true")])
    agent = ToolCallingAgent(
        BadResultEnvironment(),
        EmptyObserver(),
        provider,
        min_output_tokens=1,
        prefill_estimator=lambda _request: 2,
        self_verification=_config(),
    )

    async def execute_twice(
        _state: Mapping[str, Any],
        _step: int,
        operation: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, Any]:
        return await operation(), await operation()

    agent.configure_verification_control(execute_twice)

    outcome = await agent(_context(agent.initial_state()))

    assert outcome.completed is False
    assert outcome.verification is not None
    assert outcome.verification.status is VerificationStatus.MALFORMED
    assert "return_code" in outcome.verification.reason


async def test_malformed_verification_provider_result_is_recorded(
    tmp_path: Path,
) -> None:
    class MalformedResultProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
            del request
            self.calls += 1
            if self.calls == 1:
                return _completion()
            return {"not": "an AgentCompletion"}  # type: ignore[return-value]

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = MalformedResultProvider()
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        min_output_tokens=1,
        prefill_estimator=lambda _request: 2,
        self_verification=_config(),
    )

    outcome = await agent(_context(agent.initial_state()))

    assert outcome.completed is False
    assert outcome.verification is not None
    assert outcome.verification.status is VerificationStatus.MALFORMED
    assert "malformed result object" in outcome.verification.reason


async def test_verification_is_terminal_barrier_with_paired_tool_results(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        [
            AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "write_file",
                        {"path": "answer.txt", "content": "done\n"},
                        "write",
                    ),
                    ToolCall("complete", {"summary": "done"}, "complete"),
                    ToolCall(
                        "write_file",
                        {"path": "after.txt", "content": "must not exist"},
                        "after",
                    ),
                ),
                tokens=3,
            ),
            _verification("test -f answer.txt"),
        ]
    )
    agent = _agent(workspace, provider)
    state = agent.initial_state()
    await _bind_local_control(agent, workspace, tmp_path / "control", state)
    outcome = await agent(_context(state))

    assert outcome.completed is True
    assert not (workspace / "after.txt").exists()
    messages = AgentConversationCodec().decode(outcome.state)[0]
    _assert_no_orphaned_tool_results(messages)
    assert "verification barrier" in json.dumps(messages)


def test_self_verification_imports_no_hidden_scoring_or_oracle_modules() -> None:
    repository = Path(__file__).parents[1]
    imported: set[str] = set()
    accessed_attributes: set[str] = set()
    for relative in (
        "src/driftlock/agent.py",
        "src/driftlock/verification.py",
        "src/driftlock/runner.py",
        "src/driftlock/models.py",
    ):
        tree = ast.parse((repository / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Attribute):
                accessed_attributes.add(node.attr)

    forbidden = {
        "driftlock.checkpoint_scoring",
        "driftlock.oracle",
        "driftlock.harbor_agent",
        "driftlock.harbor_native_agent",
    }
    assert not any(
        imported_name == forbidden_name
        or imported_name.startswith(f"{forbidden_name}.")
        for imported_name in imported
        for forbidden_name in forbidden
    )
    assert "reward" not in accessed_attributes
