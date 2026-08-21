from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
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
    _truncate,
)
from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.lhtb import WorkspaceDelta, WorkspaceSnapshot
from driftlock.local import LocalEnvironment, LocalWorkspaceDeltaObserver
from driftlock.models import (
    DriftContext,
    JudgeVerdict,
    RunStatus,
    StepContext,
    StepOutcome,
    StepRecord,
    StepTokenBudgetExhausted,
    Verdict,
)
from driftlock.runner import DriftlockRunner, RunnerConfig


class ScriptedCompletion:
    def __init__(self, responses: Sequence[AgentCompletion | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[AgentCompletionRequest] = []

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@dataclass(frozen=True, slots=True)
class FakeExecResult:
    return_code: int
    stdout: str
    stderr: str


class RecordingEnvironment:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int | None, str | int | None]] = []

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> FakeExecResult:
        self.commands.append((command, timeout_sec, user))
        return FakeExecResult(0, "from injected environment", "")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        raise AssertionError("upload_file was not expected")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        raise AssertionError("download_file was not expected")


class LiteralDeltaObserver:
    async def canonical_workspace(self) -> str:
        return "/remote/workspace"

    async def snapshot(self) -> WorkspaceSnapshot:
        return WorkspaceSnapshot(files={})

    def compare(
        self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> WorkspaceDelta:
        return WorkspaceDelta(("model.py",), "literal per-step diff")


class FailingSnapshotObserver:
    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.snapshot_calls = 0

    async def canonical_workspace(self) -> str:
        return "/remote/workspace"

    async def snapshot(self) -> WorkspaceSnapshot:
        self.snapshot_calls += 1
        if self.snapshot_calls == self.fail_on_call:
            raise FileNotFoundError("raced with an editor")
        return WorkspaceSnapshot(files={})

    def compare(
        self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> WorkspaceDelta:
        return WorkspaceDelta()


def _agent(
    workspace: Path,
    provider: ScriptedCompletion,
    *,
    max_output_tokens: int = 100,
    max_tool_output_chars: int = 16_000,
) -> ToolCallingAgent:
    return ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        max_output_tokens=max_output_tokens,
        max_tool_output_chars=max_tool_output_chars,
    )


def _context(
    state: dict[str, object],
    *,
    sequence: int = 1,
    logical_step: int = 1,
    attempt: int = 1,
    rollback_feedback: str | None = None,
    tokens_remaining: int | None = None,
) -> StepContext:
    return StepContext(
        goal="repair broken.py",
        plan="inspect, repair, verify",
        state=state,
        sequence=sequence,
        logical_step=logical_step,
        attempt=attempt,
        rollback_feedback=rollback_feedback,
        tokens_remaining=tokens_remaining,
    )


async def test_normal_four_step_tool_agent_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    provider = ScriptedCompletion(
        [
            AgentCompletion(
                tool_calls=(ToolCall("read_file", {"path": "broken.py"}, "r1"),),
                tokens=11,
            ),
            AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "write_file",
                        {"path": "broken.py", "content": "print('fixed')\n"},
                        "w1",
                    ),
                ),
                tokens=13,
            ),
            AgentCompletion(
                tool_calls=(
                    ToolCall("run_shell", {"command": "python3 broken.py"}, "s1"),
                ),
                tokens=17,
            ),
            AgentCompletion(
                tool_calls=(
                    ToolCall("complete", {"summary": "Syntax repaired and verified."}),
                ),
                tokens=5,
            ),
        ]
    )
    agent = _agent(workspace, provider)
    state = agent.initial_state()
    outcomes = []

    for number in range(1, 5):
        outcome = await agent(_context(state, sequence=number, logical_step=number))
        outcomes.append(outcome)
        state = dict(outcome.state)

    assert len(provider.requests) == 4
    assert [outcome.tokens for outcome in outcomes] == [11, 13, 17, 5]
    assert outcomes[0].changed_paths == ()
    assert outcomes[1].changed_paths == ("broken.py",)
    assert outcomes[2].changed_paths == ()
    assert outcomes[3].changed_paths == ()
    assert [outcome.completed for outcome in outcomes] == [False, False, False, True]
    assert (workspace / "broken.py").read_text(encoding="utf-8") == "print('fixed')\n"
    third_observations = json.dumps(provider.requests[3].messages)
    assert "exit_code: 0" in third_observations
    assert "fixed" in third_observations


async def test_tools_and_delta_use_injected_collaborators() -> None:
    environment = RecordingEnvironment()
    provider = ScriptedCompletion(
        [
            AgentCompletion(
                tool_calls=(
                    ToolCall("run_shell", {"command": "printf ignored"}, "shell"),
                ),
                tokens=14,
            )
        ]
    )
    agent = ToolCallingAgent(environment, LiteralDeltaObserver(), provider)

    outcome = await agent(_context(agent.initial_state()))

    assert len(environment.commands) == 1
    assert environment.commands[0] == (
        "cd -- /remote/workspace && printf ignored",
        60,
        None,
    )
    assert "from injected environment" in json.dumps(outcome.state)
    assert outcome.changed_paths == ("model.py",)
    assert outcome.diff == "literal per-step diff"


async def test_multiple_tools_share_one_provider_step(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "one.txt").write_text("needle\n", encoding="utf-8")
    provider = ScriptedCompletion(
        [
            AgentCompletion(
                tool_calls=(
                    ToolCall("read_file", {"path": "one.txt"}, "a"),
                    ToolCall("search_files", {"query": "needle", "path": "."}, "b"),
                ),
                tokens=19,
            )
        ]
    )
    agent = _agent(workspace, provider)

    outcome = await agent(_context(agent.initial_state()))

    assert len(provider.requests) == 1
    assert outcome.tokens == 19
    assert outcome.changed_paths == ()
    assert outcome.action == "Execute 2 tools: read_file, search_files"
    encoded = json.dumps(outcome.state)
    assert "one.txt:1:needle" in encoded
    assert "needle\\n" in encoded


@pytest.mark.parametrize(
    ("call", "literal_error"),
    [
        (ToolCall("imaginary", {}, "bad-1"), "unknown tool 'imaginary'"),
        (
            ToolCall("write_file", '{"path": 9}', "bad-2"),
            "missing required argument(s): content",
        ),
    ],
)
async def test_bad_tool_call_is_observed_on_next_step(
    tmp_path: Path, call: ToolCall, literal_error: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedCompletion(
        [
            AgentCompletion(tool_calls=(call,), tokens=7),
            AgentCompletion(
                tool_calls=(ToolCall("complete", {"summary": "recovered"}),),
                tokens=3,
            ),
        ]
    )
    agent = _agent(workspace, provider)

    first = await agent(_context(agent.initial_state()))
    second = await agent(_context(dict(first.state), sequence=2, logical_step=2))

    assert first.tokens == 7
    assert literal_error in (first.error or "")
    assert literal_error in json.dumps(provider.requests[1].messages)
    assert second.completed is True
    assert len(provider.requests) == 2


async def test_nonzero_shell_result_is_an_observation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedCompletion(
        [
            AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "run_shell",
                        {"command": "printf 'out'; printf 'bad' >&2; exit 7"},
                        "shell",
                    ),
                ),
                tokens=9,
            ),
            AgentCompletion(),
        ]
    )
    agent = _agent(workspace, provider)

    first = await agent(_context(agent.initial_state()))
    await agent(_context(dict(first.state), sequence=2, logical_step=2))

    observation = json.dumps(provider.requests[1].messages)
    assert "exit_code: 7" in observation
    assert "out" in observation
    assert "bad" in observation
    assert first.tokens == 9


async def test_path_escape_and_symlink_escape_are_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("never reveal this", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)
    provider = ScriptedCompletion(
        [
            AgentCompletion(
                tool_calls=(
                    ToolCall("read_file", {"path": "../../secret.txt"}, "one"),
                    ToolCall("read_file", {"path": "link.txt"}, "two"),
                ),
                tokens=8,
            ),
            AgentCompletion(),
        ]
    )
    agent = _agent(workspace, provider)

    first = await agent(_context(agent.initial_state()))
    await agent(_context(dict(first.state), sequence=2, logical_step=2))

    request_text = json.dumps(provider.requests[1].messages)
    assert request_text.count("path resolves outside the workspace root") == 2
    assert "never reveal this" not in request_text


async def test_search_does_not_follow_symlinked_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("private needle", encoding="utf-8")
    (workspace / "linked.txt").symlink_to(outside)
    provider = ScriptedCompletion(
        [
            AgentCompletion(
                tool_calls=(ToolCall("search_files", {"query": "needle"}),),
                tokens=5,
            )
        ]
    )
    agent = _agent(workspace, provider)

    outcome = await agent(_context(agent.initial_state()))

    assert "private needle" not in json.dumps(outcome.state)
    assert "(no matches)" in json.dumps(outcome.state)


async def test_large_tool_output_has_explicit_truncation_marker(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedCompletion(
        [
            AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "run_shell", {"command": "python3 -c 'print(\"x\"*1000)'"}
                    ),
                ),
                tokens=4,
            ),
            AgentCompletion(),
        ]
    )
    agent = _agent(workspace, provider, max_tool_output_chars=128)

    first = await agent(_context(agent.initial_state()))
    await agent(_context(dict(first.state), sequence=2, logical_step=2))

    request_text = json.dumps(provider.requests[1].messages)
    assert "tool output truncated" in request_text
    assert len(request_text) < 2_000


async def test_prose_only_and_truncated_responses_do_not_complete(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedCompletion(
        [
            AgentCompletion(text="Everything is done.", tokens=2),
            AgentCompletion(text="partial", tokens=6, truncated=True),
        ]
    )
    agent = _agent(workspace, provider)

    first = await agent(_context(agent.initial_state()))
    second = await agent(_context(dict(first.state), sequence=2, logical_step=2))

    assert first.completed is False
    assert first.error is None
    assert second.completed is False
    assert second.tokens == 6
    assert (
        second.error == "Provider response was truncated before it could be acted on."
    )


async def test_token_cap_and_preflight_exhaustion(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedCompletion([AgentCompletion(tokens=3)])
    agent = _agent(workspace, provider, max_output_tokens=80)

    outcome = await agent(_context(agent.initial_state(), tokens_remaining=12))

    assert outcome.tokens == 3
    assert provider.requests[0].max_output_tokens == 12
    with pytest.raises(StepTokenBudgetExhausted):
        await agent(_context(dict(outcome.state), tokens_remaining=0))
    assert len(provider.requests) == 1


async def test_failed_provider_call_is_a_billed_step(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedCompletion(
        [AgentProviderError("upstream disconnected", tokens=23)]
    )
    agent = _agent(workspace, provider)

    outcome = await agent(_context(agent.initial_state()))

    assert len(provider.requests) == 1
    assert outcome.tokens == 23
    assert outcome.action == "Provider call failed"
    assert outcome.error == "Provider call failed: upstream disconnected"
    assert outcome.completed is False


async def test_state_json_round_trip_resumes_identical_conversation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first_response = AgentCompletion(text="thinking", tokens=2)
    control_provider = ScriptedCompletion(
        [
            first_response,
            AgentCompletion(
                tool_calls=(ToolCall("complete", {"summary": "done"}),), tokens=1
            ),
        ]
    )
    control_agent = _agent(workspace, control_provider)
    first = await control_agent(_context(control_agent.initial_state()))
    decoded = json.loads(json.dumps(first.state))

    control_outcome = await control_agent(
        _context(dict(first.state), sequence=2, logical_step=2)
    )
    resumed_provider = ScriptedCompletion(
        [
            AgentCompletion(
                tool_calls=(ToolCall("complete", {"summary": "done"}),), tokens=1
            )
        ]
    )
    resumed_agent = _agent(workspace, resumed_provider)
    resumed_outcome = await resumed_agent(_context(decoded, sequence=2, logical_step=2))

    assert decoded == first.state
    assert resumed_provider.requests[0] == control_provider.requests[1]
    assert resumed_outcome.state == control_outcome.state
    assert resumed_outcome.completed is True


async def test_wrong_state_schema_version_raises(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedCompletion([AgentCompletion()])
    agent = _agent(workspace, provider)
    state = agent.initial_state()
    state["driftlock_tool_agent"]["schema_version"] = 999

    with pytest.raises(AgentStateError, match="unsupported"):
        await agent(_context(state))

    assert provider.requests == []


async def test_runner_token_limit_counts_new_agent_billing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedCompletion(
        [AgentCompletion(tokens=6), AgentCompletion(tokens=4)]
    )
    agent = _agent(workspace, provider)
    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "snapshots"),
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=10,
                loop_window=10,
                loop_repetitions=10,
                error_window=10,
                reward_stall_steps=10,
            )
        ),
        config=RunnerConfig(max_steps=5, max_tokens=10),
    ).run(goal="bounded", step=agent, initial_state=agent.initial_state())

    assert result.status is RunStatus.TOKEN_LIMIT
    assert result.agent_tokens_used == 10
    assert [request.max_output_tokens for request in provider.requests] == [10, 4]
    assert [record.outcome.tokens for record in result.steps] == [6, 4]


async def test_runner_counts_failed_provider_usage(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedCompletion([AgentProviderError("failed", tokens=7)])
    agent = _agent(workspace, provider)
    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "snapshots"),
        HeuristicJudge(),
        config=RunnerConfig(max_steps=3, max_tokens=7),
    ).run(goal="bounded", step=agent, initial_state=agent.initial_state())

    assert result.status is RunStatus.TOKEN_LIMIT
    assert result.agent_tokens_used == 7
    assert len(result.steps) == 1
    assert result.steps[0].outcome.error == "Provider call failed: failed"


async def test_runner_rollback_feedback_is_used_once_and_not_checkpointed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    feedback = "the last branch repeated itself"

    class RollbackAwareCompletion:
        def __init__(self) -> None:
            self.requests: list[AgentCompletionRequest] = []

        async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
            self.requests.append(request)
            if feedback in json.dumps(request.messages):
                return AgentCompletion(
                    tool_calls=(ToolCall("complete", {"summary": "changed course"}),),
                    tokens=3,
                )
            return AgentCompletion(text="still considering", tokens=2)

    class DriftJudge:
        async def judge(self, _context: DriftContext) -> JudgeVerdict:
            return JudgeVerdict(Verdict.DRIFTED, feedback)

    provider = RollbackAwareCompletion()
    agent = ToolCallingAgent(
        LocalEnvironment(workspace), LocalWorkspaceDeltaObserver(workspace), provider
    )
    store = DirectoryCheckpointStore(workspace, tmp_path / "snapshots")
    result = await DriftlockRunner(
        store,
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=2,
                loop_window=2,
                loop_repetitions=2,
                error_window=10,
                reward_stall_steps=10,
            )
        ),
        fine_judge=DriftJudge(),
        config=RunnerConfig(
            max_steps=4,
            max_rollbacks=1,
            checkpoint_interval=10,
            checkpoint_on_exit=True,
        ),
    ).run(goal="change course", step=agent, initial_state=agent.initial_state())

    assert result.status is RunStatus.COMPLETED
    assert [record.attempt for record in result.steps] == [1, 1, 2]
    assert len(result.rollbacks) == 1
    assert feedback in json.dumps(provider.requests[2].messages)
    assert feedback not in json.dumps(result.state)
    terminal_state = store.restore(result.checkpoints[-1])
    assert feedback not in json.dumps(terminal_state)


@pytest.mark.parametrize("fail_on_call", [1, 2])
async def test_workspace_observation_failure_is_explicit_and_not_a_step_error(
    fail_on_call: int,
) -> None:
    environment = RecordingEnvironment()
    observer = FailingSnapshotObserver(fail_on_call)
    provider = ScriptedCompletion([AgentCompletion(text="continue", tokens=2)])
    agent = ToolCallingAgent(environment, observer, provider)

    outcome = await agent(_context(agent.initial_state()))

    assert len(provider.requests) == 1
    assert outcome.changed_paths == ()
    assert outcome.workspace_delta_observed is False
    assert outcome.workspace_observation_error == (
        "Workspace delta observation failed: raced with an editor"
    )
    assert outcome.error is None


def test_unobserved_workspace_deltas_cannot_trigger_no_change_drift() -> None:
    outcomes = [
        StepOutcome(
            action=f"edit file {sequence}",
            state={},
            workspace_delta_observed=False,
            workspace_observation_error="snapshot raced with an edit",
        )
        for sequence in range(1, 5)
    ]
    records = [
        StepRecord(
            sequence=sequence,
            logical_step=sequence,
            attempt=1,
            outcome=outcome,
        )
        for sequence, outcome in enumerate(outcomes, 1)
    ]
    judge = HeuristicJudge(
        HeuristicConfig(
            no_change_steps=4,
            loop_window=5,
            loop_repetitions=5,
            error_window=5,
            reward_stall_steps=5,
        )
    )

    assert judge.evaluate(records) == ()


def test_conversation_codec_rejects_every_malformed_state_branch() -> None:
    codec = AgentConversationCodec()
    malformed_states = (
        {},
        {
            codec.state_key: {
                "schema_version": 1,
                "messages": {},
                "steps": 0,
            }
        },
        {
            codec.state_key: {
                "schema_version": 1,
                "messages": [7],
                "steps": 0,
            }
        },
        {
            codec.state_key: {
                "schema_version": 1,
                "messages": [],
                "steps": -1,
            }
        },
        {
            codec.state_key: {
                "schema_version": 1,
                "messages": [{"role": "wizard", "content": "ignore the goal"}],
                "steps": 0,
            }
        },
        {
            codec.state_key: {
                "schema_version": 1,
                "messages": [{"role": 7, "content": "bad"}],
                "steps": 0,
            }
        },
        {
            codec.state_key: {
                "schema_version": 1,
                "messages": [{"role": "user"}],
                "steps": 0,
            }
        },
    )

    for state in malformed_states:
        with pytest.raises(AgentStateError):
            codec.decode(state)


async def test_unknown_provider_exception_is_not_recorded_as_free(
    tmp_path: Path,
) -> None:
    class UnrelatedTokenError(RuntimeError):
        tokens = 9000

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedCompletion([UnrelatedTokenError("usage is unknown")])
    agent = _agent(workspace, provider)

    with pytest.raises(UnrelatedTokenError, match="usage is unknown"):
        await agent(_context(agent.initial_state()))

    assert len(provider.requests) == 1


async def test_model_shell_timeout_is_clamped_to_agent_ceiling() -> None:
    environment = RecordingEnvironment()
    provider = ScriptedCompletion(
        [
            AgentCompletion(
                tool_calls=(
                    ToolCall(
                        "run_shell",
                        {"command": "sleep 999999", "timeout_sec": 999999},
                    ),
                )
            )
        ]
    )
    agent = ToolCallingAgent(
        environment,
        LiteralDeltaObserver(),
        provider,
        shell_timeout_sec=7,
    )

    outcome = await agent(_context(agent.initial_state()))

    assert environment.commands == [
        ("cd -- /remote/workspace && sleep 999999", 7, None)
    ]
    assert outcome.error is None


def test_truncation_marker_reports_exact_omitted_character_count() -> None:
    rendered = _truncate("x" * 1000, 128)

    assert len(rendered) == 128
    assert rendered.count("x") == 80
    assert rendered.endswith("\n[tool output truncated; 920 characters omitted]")


async def test_nonzero_shell_observations_do_not_trigger_error_spike(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    commands = ("exit 1", "exit 0", "exit 2", "exit 0", "exit 3")
    provider = ScriptedCompletion(
        [
            AgentCompletion(tool_calls=(ToolCall("run_shell", {"command": command}),))
            for command in commands
        ]
    )
    agent = _agent(workspace, provider)
    state = agent.initial_state()
    outcomes = []

    for sequence in range(1, 6):
        outcome = await agent(_context(state, sequence=sequence, logical_step=sequence))
        outcomes.append(outcome)
        state = dict(outcome.state)

    records = [
        StepRecord(
            sequence=sequence,
            logical_step=sequence,
            attempt=1,
            outcome=outcome,
        )
        for sequence, outcome in enumerate(outcomes, 1)
    ]
    judge = HeuristicJudge(
        HeuristicConfig(
            no_change_steps=10,
            loop_window=5,
            loop_repetitions=5,
            error_window=5,
            error_rate=0.6,
            reward_stall_steps=10,
        )
    )

    assert [outcome.error for outcome in outcomes] == [None, None, None, None, None]
    assert all("exit_code:" in json.dumps(outcome.state) for outcome in outcomes)
    assert "error_spike" not in {signal.kind for signal in judge.evaluate(records)}
