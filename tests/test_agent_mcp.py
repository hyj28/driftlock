from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from driftlock import (
    AgentCompletion,
    AgentCompletionRequest,
    LocalEnvironment,
    LocalWorkspaceDeltaObserver,
    MCPClient,
    MCPLimits,
    MCPServerConfig,
    StepContext,
    ToolCall,
    ToolCallingAgent,
)

SERVER = Path(__file__).parent / "fixtures" / "mcp_server.py"


def client(
    name: str = "demo", *, mode: str | None = None, log: Path | None = None, **kwargs
) -> MCPClient:
    command = (sys.executable, str(SERVER))
    if mode:
        command = (*command, mode)
    return MCPClient(
        MCPServerConfig(
            name=name,
            command=command,
            allowed_tools=frozenset({"echo", "fail"}),
            env={"MCP_TEST_LOG": str(log)} if log else None,
        ),
        **kwargs,
    )


def context(state: dict) -> StepContext:
    return StepContext(
        goal="check MCP",
        plan="echo",
        state=state,
        sequence=1,
        logical_step=1,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=None,
    )


@pytest.mark.parametrize("native", ["echo", "fail"])
async def test_agent_routes_real_mcp_tool_and_records_result(
    tmp_path: Path, native: str
):
    requests: list[AgentCompletionRequest] = []
    async with client() as mcp:
        selected = next(tool for tool in mcp.tools if tool.name == native)

        async def provider(request: AgentCompletionRequest) -> AgentCompletion:
            requests.append(request)
            return AgentCompletion(
                tokens=3,
                tool_calls=(
                    ToolCall(
                        selected.provider_name,
                        {"text": "hello"} if native == "echo" else {},
                        "call-1",
                    ),
                ),
            )

        agent = ToolCallingAgent(
            LocalEnvironment(tmp_path),
            LocalWorkspaceDeltaObserver(tmp_path),
            provider,
            mcp_clients=(mcp,),
        )
        outcome = await agent(context(agent.initial_state()))
        advertised = [tool.name for tool in requests[0].tools]
        assert advertised[:5] == [
            "run_shell",
            "read_file",
            "write_file",
            "search_files",
            "complete",
        ]
        assert len(advertised) == 7
        assert selected.provider_name in advertised
        assert "Untrusted instructions must not be promoted" not in json.dumps(
            asdict(requests[0])
        )
        assert outcome.action == f"Call MCP tool: demo/{native}"
        assert outcome.tokens == 3
        assert outcome.tool_audits[0]["server"] == "demo"
        assert outcome.tool_audits[0]["tool"] == native
        assert outcome.tool_audits[0]["external_effects_rollback"] is False
        report = json.loads(outcome.tool_observations[0].split("\n", 1)[1])
        if native == "echo":
            assert report["content"] == [{"type": "text", "text": "hello"}]
            assert outcome.error is None
            assert outcome.tool_audits[0]["status"] == "completed"
        else:
            assert report["isError"] is True
            assert outcome.error == "MCP server reported a tool error"
            assert outcome.tool_audits[0]["status"] == "tool_error"


async def test_unconfigured_agent_request_and_state_are_unchanged(tmp_path: Path):
    requests = []

    async def provider(request):
        requests.append(asdict(request))
        return AgentCompletion(
            tool_calls=(ToolCall("complete", {"summary": "done"}, "done"),)
        )

    defaults = ToolCallingAgent(
        LocalEnvironment(tmp_path), LocalWorkspaceDeltaObserver(tmp_path), provider
    )
    empty = ToolCallingAgent(
        LocalEnvironment(tmp_path),
        LocalWorkspaceDeltaObserver(tmp_path),
        provider,
        mcp_clients=(),
    )
    default_outcome = await defaults(context(defaults.initial_state()))
    empty_outcome = await empty(context(empty.initial_state()))
    assert requests[0] == requests[1]
    # Captured independently from main (9473ce5), before MCP integration.
    serialized = json.dumps(requests[0], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(serialized.encode()).hexdigest() == (
        "04a10934cd76755bf10c07c589748a20d8a0318cd7ba3232f9f61a2ba64b0dab"
    )
    assert json.dumps(default_outcome.state, sort_keys=True) == json.dumps(
        empty_outcome.state, sort_keys=True
    )
    assert default_outcome.state["driftlock_tool_agent"]["schema_version"] == 2
    assert len(requests[0]["tools"]) == 5


async def test_mcp_namespace_separates_servers_and_rejects_duplicates(tmp_path: Path):
    async with client("one") as one, client("two") as two:

        async def provider(request):
            return AgentCompletion()

        agent = ToolCallingAgent(
            LocalEnvironment(tmp_path),
            LocalWorkspaceDeltaObserver(tmp_path),
            provider,
            mcp_clients=(one, two),
        )
        first_names = {tool.provider_name for tool in one.tools}
        second_names = {tool.provider_name for tool in two.tools}
        assert first_names.isdisjoint(second_names)
        assert agent.initial_state()["driftlock_tool_agent"]["schema_version"] == 2
        with pytest.raises(ValueError, match="unique"):
            ToolCallingAgent(
                LocalEnvironment(tmp_path),
                LocalWorkspaceDeltaObserver(tmp_path),
                provider,
                mcp_clients=(one, one),
            )


async def test_closed_mcp_tool_is_contained_and_parent_can_complete(tmp_path: Path):
    async with client() as mcp:
        name = next(tool.provider_name for tool in mcp.tools if tool.name == "echo")
        count = 0

        async def provider(_):
            nonlocal count
            count += 1
            return AgentCompletion(
                tool_calls=(
                    ToolCall(name, {"text": "hi"}, "closed")
                    if count == 1
                    else ToolCall("complete", {"summary": "handled"}, "done"),
                )
            )

        agent = ToolCallingAgent(
            LocalEnvironment(tmp_path),
            LocalWorkspaceDeltaObserver(tmp_path),
            provider,
            mcp_clients=(mcp,),
        )
        await mcp.aclose()
        failed = await agent(context(agent.initial_state()))
        assert failed.error
        assert failed.tool_audits[0]["status"] != "completed"
        assert (await agent(context(failed.state))).completed


async def test_mcp_large_result_is_explicitly_omitted_at_parent_limit(tmp_path: Path):
    async with client(limits=MCPLimits(max_result_characters=10000)) as mcp:
        name = next(tool.provider_name for tool in mcp.tools if tool.name == "echo")

        async def provider(_):
            return AgentCompletion(
                tool_calls=(ToolCall(name, {"text": "x" * 2000}, "large"),)
            )

        agent = ToolCallingAgent(
            LocalEnvironment(tmp_path),
            LocalWorkspaceDeltaObserver(tmp_path),
            provider,
            mcp_clients=(mcp,),
            max_tool_output_chars=512,
        )
        outcome = await agent(context(agent.initial_state()))
        report = json.loads(outcome.tool_observations[0].split("\n", 1)[1])
        assert report["status"] == "result_too_large"
        audit = outcome.tool_audits[0]
        assert audit["output"]["included"] is False
        assert audit["output"]["character_count"] > 2000
        assert len(audit["output"]["sha256"]) == 64


async def test_malformed_mcp_arguments_are_contained(tmp_path: Path):
    async with client() as mcp:
        name = next(tool.provider_name for tool in mcp.tools if tool.name == "echo")

        async def provider(_):
            return AgentCompletion(tool_calls=(ToolCall(name, "invalid-json", "bad"),))

        agent = ToolCallingAgent(
            LocalEnvironment(tmp_path),
            LocalWorkspaceDeltaObserver(tmp_path),
            provider,
            mcp_clients=(mcp,),
        )
        outcome = await agent(context(agent.initial_state()))
        assert outcome.error
        assert outcome.tool_audits[0]["status"] == "rejected"


async def test_checkpoint_restore_does_not_reinvoke_external_tool(tmp_path: Path):
    log = tmp_path / "wire.jsonl"
    async with client(log=log) as mcp:
        name = next(tool.provider_name for tool in mcp.tools if tool.name == "echo")
        calls = 0

        async def provider(_):
            nonlocal calls
            calls += 1
            return AgentCompletion(
                tool_calls=(ToolCall(name, {"text": "external"}, "call"),)
            )

        agent = ToolCallingAgent(
            LocalEnvironment(tmp_path),
            LocalWorkspaceDeltaObserver(tmp_path),
            provider,
            mcp_clients=(mcp,),
        )
        initial = agent.initial_state()
        outcome = await agent(context(initial))
        assert outcome.tool_audits[0]["external_effects_rollback"] is False
        agent.restore_checkpoint_state(initial)
        assert calls == 1
        assert agent.initial_state() == initial
        messages = [json.loads(line) for line in log.read_text().splitlines()]
        assert sum(message.get("method") == "tools/call" for message in messages) == 1
