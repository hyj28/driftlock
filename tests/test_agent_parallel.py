from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

import pytest

from driftlock import (
    AgentCompletion,
    AgentCompletionRequest,
    AgentConversationCodec,
    LocalEnvironment,
    LocalExecResult,
    LocalWorkspaceDeltaObserver,
    StepContext,
    ToolCall,
    ToolCallingAgent,
    WorkspaceDelta,
    WorkspaceSnapshot,
    conversation_history_characters,
)


class ScriptedCompletion:
    def __init__(self, *responses: AgentCompletion) -> None:
        self.responses = list(responses)
        self.requests: list[AgentCompletionRequest] = []

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
        self.requests.append(request)
        return self.responses.pop(0)


class ReadEnvironment:
    """Public environment seam with deterministic read rendezvous and cleanup."""

    def __init__(self, mode: str = "serial") -> None:
        self.mode = mode
        self.events: list[str] = []
        self.active = 0
        self.peak = 0
        self.started = asyncio.Event()
        self.second_done = asyncio.Event()
        self.release = asyncio.Event()
        self.failure: str | None = None
        self.output: str | None = None

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        args = shlex.split(command)
        if args[:2] != ["python3", "-c"]:
            raise AssertionError(f"unexpected command: {command}")
        path = args[3]
        if "os.path.realpath" in args[2]:
            return LocalExecResult(0, path + "\n", "")
        name = Path(path).name
        self.events.append(f"start {name}")
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.active == 2:
            self.started.set()
        try:
            if self.mode == "overlap" and name == "a.txt":
                await self.second_done.wait()
            elif self.mode == "cancel":
                await self.release.wait()
            else:
                await asyncio.sleep(0)
            if self.failure is not None and name == "a.txt":
                raise RuntimeError(self.failure)
            output = self.output
            if output is None:
                output = "alpha\n" if name == "a.txt" else "beta\n"
                if len(args) == 7:
                    output = f"{name}:1:{output}"
            return LocalExecResult(0, output, "")
        finally:
            self.active -= 1
            self.events.append(f"finish {name}")
            if name == "b.txt":
                self.second_done.set()

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        raise AssertionError("a later write must not start")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        raise AssertionError("unexpected download")


class ReadObserver:
    def __init__(self, environment: ReadEnvironment) -> None:
        self.environment = environment

    async def canonical_workspace(self) -> str:
        return "/workspace"

    async def snapshot(self) -> WorkspaceSnapshot:
        assert self.environment.active == 0
        return WorkspaceSnapshot(files={})

    def compare(
        self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> WorkspaceDelta:
        return WorkspaceDelta()


def context(state: dict, *, sequence: int = 1) -> StepContext:
    return StepContext(
        goal="inspect files",
        plan="read and verify",
        state=state,
        sequence=sequence,
        logical_step=sequence,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=None,
    )


def read(name: str, path: str, call_id: str) -> ToolCall:
    return ToolCall(
        name,
        {"path": path} if name == "read_file" else {"path": path, "query": "a"},
        call_id,
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_parallel_configuration_requires_boolean(value: object) -> None:
    environment = ReadEnvironment()
    with pytest.raises(TypeError, match="parallel_tool_calls must be a boolean"):
        ToolCallingAgent(
            environment,
            ReadObserver(environment),
            ScriptedCompletion(),
            parallel_tool_calls=value,
        )


async def test_default_and_false_preserve_requests_results_and_serial_order() -> None:
    requests = []
    outcomes = []
    for options in ({}, {"parallel_tool_calls": False}):
        environment = ReadEnvironment()
        provider = ScriptedCompletion(
            AgentCompletion(
                tool_calls=(
                    read("read_file", "a.txt", "first"),
                    read("search_files", "b.txt", "second"),
                ),
                tokens=17,
            ),
            AgentCompletion(tool_calls=(ToolCall("complete", {"summary": "done"}),)),
        )
        agent = ToolCallingAgent(
            environment, ReadObserver(environment), provider, **options
        )
        first = await agent(context(agent.initial_state()))
        await agent(context(dict(first.state), sequence=2))
        assert environment.events == [
            "start a.txt",
            "finish a.txt",
            "start b.txt",
            "finish b.txt",
        ]
        assert environment.peak == 1
        assert first.tool_observations == (
            "read_file:\nalpha\n",
            "search_files:\nb.txt:1:beta\n",
        )
        assert first.tokens == 17
        assert first.tool_audits == ()
        requests.append(provider.requests)
        outcomes.append(first)
    assert requests[0] == requests[1]
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("first_name", ["read_file", "search_files"])
@pytest.mark.parametrize("second_name", ["read_file", "search_files"])
async def test_all_eligible_pairs_overlap_and_keep_request_order(
    first_name: str, second_name: str
) -> None:
    environment = ReadEnvironment("overlap")
    provider = ScriptedCompletion(
        AgentCompletion(
            tool_calls=(
                read(first_name, "a.txt", "first"),
                read(second_name, "b.txt", "second"),
            ),
            tokens=23,
        ),
        AgentCompletion(tool_calls=(ToolCall("complete", {"summary": "done"}),)),
    )
    agent = ToolCallingAgent(
        environment, ReadObserver(environment), provider, parallel_tool_calls=True
    )
    outcome = await asyncio.wait_for(agent(context(agent.initial_state())), timeout=5)
    assert environment.peak == 2
    assert environment.events == [
        "start a.txt",
        "start b.txt",
        "finish b.txt",
        "finish a.txt",
    ]
    expected_first = {
        "read_file": "read_file:\nalpha\n",
        "search_files": "search_files:\na.txt:1:alpha\n",
    }
    expected_second = {
        "read_file": "read_file:\nbeta\n",
        "search_files": "search_files:\nb.txt:1:beta\n",
    }
    assert outcome.tool_observations == (
        expected_first[first_name],
        expected_second[second_name],
    )
    assert outcome.tokens == 23
    assert outcome.error is None
    assert outcome.commands_run == 0
    assert outcome.changed_paths == ()
    messages, steps = AgentConversationCodec().decode(outcome.state)
    assert steps == 1
    assert [message["tool_call_id"] for message in messages[1:]] == ["first", "second"]
    await agent(context(dict(outcome.state), sequence=2))
    assert [
        message["tool_call_id"]
        for message in provider.requests[1].messages
        if message["role"] == "tool"
    ] == ["first", "second"]


@pytest.mark.parametrize("barrier", ["write_file", "run_shell"])
async def test_local_mutation_is_a_serial_barrier(tmp_path: Path, barrier: str) -> None:
    (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
    mutation = (
        {"path": "a.txt", "content": "new\n"}
        if barrier == "write_file"
        else {"command": "printf 'new\\n' > a.txt"}
    )
    provider = ScriptedCompletion(
        AgentCompletion(
            tool_calls=(
                read("read_file", "a.txt", "before"),
                ToolCall(barrier, mutation, "mutate"),
                read("read_file", "a.txt", "after"),
            )
        )
    )
    agent = ToolCallingAgent(
        LocalEnvironment(tmp_path),
        LocalWorkspaceDeltaObserver(tmp_path),
        provider,
        parallel_tool_calls=True,
    )
    outcome = await agent(context(agent.initial_state()))
    assert outcome.error is None
    assert outcome.tool_observations[0] == "read_file:\nold\n"
    assert outcome.tool_observations[2] == "read_file:\nnew\n"
    assert outcome.changed_paths == ("a.txt",)
    assert outcome.commands_run == (1 if barrier == "run_shell" else 0)
    messages, _ = AgentConversationCodec().decode(outcome.state)
    assert [message["tool_call_id"] for message in messages[1:]] == [
        "before",
        "mutate",
        "after",
    ]


async def test_plan_barriers_preserve_plan_and_audit_order() -> None:
    environment = ReadEnvironment()
    provider = ScriptedCompletion(
        AgentCompletion(
            tool_calls=(
                read("read_file", "a.txt", "first"),
                ToolCall(
                    "manage_plan",
                    {"operation": "create", "steps": ["Inspect"]},
                    "create",
                ),
                ToolCall(
                    "manage_plan", {"operation": "add", "steps": ["Verify"]}, "add"
                ),
                read("search_files", "b.txt", "second"),
            ),
            tokens=29,
        )
    )
    agent = ToolCallingAgent(
        environment,
        ReadObserver(environment),
        provider,
        parallel_tool_calls=True,
        planning=True,
    )
    outcome = await agent(context(agent.initial_state()))
    assert outcome.error is None
    assert outcome.tokens == 29
    assert environment.peak == 1
    messages, _, plan = AgentConversationCodec().decode_with_plan(outcome.state)
    assert plan is not None
    assert [step.description for step in plan.steps] == ["Inspect", "Verify"]
    assert [message["tool_call_id"] for message in messages[1:]] == [
        "first",
        "create",
        "add",
        "second",
    ]
    assert [audit["tool_call"]["id"] for audit in outcome.tool_audits] == [
        "create",
        "add",
    ]


@pytest.mark.parametrize(
    "barrier",
    [
        "complete",
        "retrieve_context",
        "manage_memory",
        "delegate_task",
        "mcp_unconfigured",
    ],
)
async def test_other_tools_are_serial_barriers(barrier: str) -> None:
    environment = ReadEnvironment()
    provider = ScriptedCompletion(
        AgentCompletion(
            tool_calls=(
                read("read_file", "a.txt", "first"),
                ToolCall(barrier, {"summary": "finished"}, "barrier"),
                read("read_file", "b.txt", "second"),
            )
        )
    )
    agent = ToolCallingAgent(
        environment, ReadObserver(environment), provider, parallel_tool_calls=True
    )
    outcome = await agent(context(agent.initial_state()))
    assert environment.peak == 1
    assert environment.events == [
        "start a.txt",
        "finish a.txt",
        "start b.txt",
        "finish b.txt",
    ]
    messages, _ = AgentConversationCodec().decode(outcome.state)
    assert [message["tool_call_id"] for message in messages[1:]] == [
        "first",
        "barrier",
        "second",
    ]
    assert outcome.completed is (barrier == "complete")
    if barrier == "complete":
        assert outcome.summary == "finished"
        assert outcome.error is None


@pytest.mark.parametrize("arguments", [None, {}, {"path": "missing.txt"}])
async def test_bad_read_preserves_successful_local_sibling(
    tmp_path: Path, arguments: object
) -> None:
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    provider = ScriptedCompletion(
        AgentCompletion(
            tool_calls=(
                ToolCall("read_file", arguments, "bad"),
                read("read_file", "b.txt", "good"),
            )
        )
    )
    agent = ToolCallingAgent(
        LocalEnvironment(tmp_path),
        LocalWorkspaceDeltaObserver(tmp_path),
        provider,
        parallel_tool_calls=True,
        max_tool_output_chars=128,
    )
    outcome = await agent(context(agent.initial_state()))
    assert outcome.error is not None
    assert len(outcome.error) <= 128
    assert outcome.tool_observations[0].startswith("read_file:\nERROR: ")
    assert outcome.tool_observations[1] == "read_file:\nbeta\n"
    messages, _ = AgentConversationCodec().decode(outcome.state)
    assert [message["tool_call_id"] for message in messages[1:]] == ["bad", "good"]
    assert [message["is_error"] for message in messages[1:]] == [True, False]
    assert len(messages[1]["content"]) <= 128


async def test_failing_read_is_bounded_without_losing_overlapping_sibling() -> None:
    environment = ReadEnvironment("overlap")
    environment.failure = "failed " * 1000
    provider = ScriptedCompletion(
        AgentCompletion(
            tool_calls=(
                read("read_file", "a.txt", "bad"),
                read("read_file", "b.txt", "good"),
            )
        )
    )
    agent = ToolCallingAgent(
        environment,
        ReadObserver(environment),
        provider,
        parallel_tool_calls=True,
        max_tool_output_chars=128,
    )
    outcome = await asyncio.wait_for(agent(context(agent.initial_state())), timeout=5)
    assert environment.peak == 2
    assert outcome.error is not None and len(outcome.error) == 128
    assert outcome.error.startswith("read_file failed: failed ")
    assert outcome.tool_observations[1] == "read_file:\nbeta\n"
    messages, _ = AgentConversationCodec().decode(outcome.state)
    assert len(messages[1]["content"]) == 128


@pytest.mark.parametrize(
    ("count", "truncated", "limit"), [(5, False, 4), (2, True, 4), (3, False, 2)]
)
async def test_rejected_response_executes_no_calls(
    count: int, truncated: bool, limit: int
) -> None:
    environment = ReadEnvironment()
    provider = ScriptedCompletion(
        AgentCompletion(
            tool_calls=tuple(read("read_file", "a.txt", str(i)) for i in range(count)),
            truncated=truncated,
            tokens=31,
        )
    )
    agent = ToolCallingAgent(
        environment,
        ReadObserver(environment),
        provider,
        parallel_tool_calls=True,
        max_tool_calls_per_step=limit,
    )
    outcome = await agent(context(agent.initial_state()))
    assert environment.events == []
    assert outcome.tool_observations == ()
    assert outcome.error is not None
    assert outcome.tokens == 31
    messages, _ = AgentConversationCodec().decode(outcome.state)
    assert messages[0]["tool_calls"] == []
    assert [message["role"] for message in messages] == ["assistant", "user"]


async def test_cancellation_joins_both_reads_and_never_starts_later_write() -> None:
    environment = ReadEnvironment("cancel")
    provider = ScriptedCompletion(
        AgentCompletion(
            tool_calls=(
                read("read_file", "a.txt", "first"),
                read("search_files", "b.txt", "second"),
                ToolCall(
                    "write_file", {"path": "later.txt", "content": "later"}, "write"
                ),
            )
        )
    )
    agent = ToolCallingAgent(
        environment, ReadObserver(environment), provider, parallel_tool_calls=True
    )
    task = asyncio.create_task(agent(context(agent.initial_state())))
    try:
        await asyncio.wait_for(environment.started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert environment.active == 0
        assert environment.events == [
            "start a.txt",
            "start b.txt",
            "finish a.txt",
            "finish b.txt",
        ]
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_restoring_initial_checkpoint_has_complete_ordered_fresh_results() -> (
    None
):
    environment = ReadEnvironment("overlap")
    response = AgentCompletion(
        tool_calls=(
            read("read_file", "a.txt", "first"),
            read("search_files", "b.txt", "second"),
        )
    )
    provider = ScriptedCompletion(response, response)
    agent = ToolCallingAgent(
        environment, ReadObserver(environment), provider, parallel_tool_calls=True
    )
    initial = agent.initial_state()
    first = await asyncio.wait_for(agent(context(initial)), timeout=5)
    agent.restore_checkpoint_state(initial)
    environment.second_done.clear()
    environment.output = "fresh\n"
    second = await asyncio.wait_for(agent(context(initial)), timeout=5)
    assert provider.requests[0] == provider.requests[1]
    assert first.tool_observations == (
        "read_file:\nalpha\n",
        "search_files:\nb.txt:1:beta\n",
    )
    assert second.tool_observations == ("read_file:\nfresh\n", "search_files:\nfresh\n")
    messages, steps = AgentConversationCodec().decode(second.state)
    assert steps == 1
    assert [message["role"] for message in messages] == ["assistant", "tool", "tool"]
    assert [call["id"] for call in messages[0]["tool_calls"]] == ["first", "second"]
    assert [message["tool_call_id"] for message in messages[1:]] == ["first", "second"]
    assert AgentConversationCodec().decode(initial) == ([], 0)


async def test_parallel_output_and_next_request_history_remain_bounded() -> None:
    environment = ReadEnvironment("overlap")
    environment.output = "x" * 1000
    provider = ScriptedCompletion(
        AgentCompletion(
            tool_calls=(
                read("read_file", "a.txt", "first"),
                read("search_files", "b.txt", "second"),
            )
        ),
        AgentCompletion(tool_calls=(ToolCall("complete", {"summary": "done"}),)),
    )
    agent = ToolCallingAgent(
        environment,
        ReadObserver(environment),
        provider,
        parallel_tool_calls=True,
        max_tool_output_chars=128,
        max_history_characters=512,
    )
    first = await asyncio.wait_for(agent(context(agent.initial_state())), timeout=5)
    messages, _ = AgentConversationCodec().decode(first.state)
    assert [len(message["content"]) for message in messages[1:]] == [128, 128]
    assert messages[1]["content"].endswith(
        "[tool output truncated; 920 characters omitted]"
    )
    second = await agent(context(dict(first.state), sequence=2))
    assert second.context_compactions
    assert conversation_history_characters(provider.requests[1].messages[2:]) <= 512
    assert second.completed
