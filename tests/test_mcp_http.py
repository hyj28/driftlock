"""Real-server seam coverage for Streamable HTTP MCP and host authorization."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from dataclasses import asdict
from pathlib import Path

import pytest
from fixtures.mcp_http_server import running_http_server

from driftlock import (
    AgentCompletion,
    AgentCompletionRequest,
    LocalEnvironment,
    LocalWorkspaceDeltaObserver,
    MCPAuthorizationDiscoveryStatus,
    MCPAuthorizationError,
    MCPAuthorizationStatus,
    MCPClient,
    MCPError,
    MCPLimits,
    MCPServerConfig,
    StepContext,
    ToolCall,
    ToolCallingAgent,
)

STDIO_FIXTURE = Path(__file__).parent / "fixtures" / "mcp_server.py"
FAKE_TOKEN = "fake-token-for-tests-only"


def http_config(url: str, *, allowed: frozenset[str] | None = None) -> MCPServerConfig:
    return MCPServerConfig(
        name="http",
        command=None,
        allowed_tools=allowed or frozenset({"echo"}),
        url=url,
    )


def context(state: dict) -> StepContext:
    return StepContext(
        goal="check HTTP MCP",
        plan="echo",
        state=state,
        sequence=1,
        logical_step=1,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=None,
    )


async def test_http_discovery_call_sse_and_agent_surface(tmp_path: Path):
    with running_http_server() as server:
        async with MCPClient(http_config(server.url)) as client:
            assert client.ready
            assert [tool.name for tool in client.tools] == ["echo"]
            server.mode = "sse"
            result = await client.call_tool("echo", {"text": "over HTTP"})
            assert result == {
                "content": [{"type": "text", "text": "over HTTP"}],
                "isError": False,
            }
            selected = client.tools[0]

            async def provider(_request: AgentCompletionRequest) -> AgentCompletion:
                return AgentCompletion(
                    tool_calls=(
                        ToolCall(selected.provider_name, {"text": "agent"}, "http-1"),
                    )
                )

            agent = ToolCallingAgent(
                LocalEnvironment(tmp_path),
                LocalWorkspaceDeltaObserver(tmp_path),
                provider,
                mcp_clients=(client,),
            )
            outcome = await agent(context(agent.initial_state()))
            assert outcome.action == "Call MCP tool: http/echo"
            assert outcome.tool_audits[0]["status"] == "completed"
            assert json.loads(outcome.tool_observations[0].split("\n", 1)[1]) == {
                "content": [{"text": "agent", "type": "text"}],
                "isError": False,
            }


async def test_supplier_is_only_source_and_is_consulted_per_request(monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "ambient-token-must-not-be-read")
    with running_http_server() as server:
        async with MCPClient(http_config(server.url)) as client:
            await client.call_tool("echo", {"text": "public"})
        assert all(
            "authorization" not in request["headers"] for request in server.requests
        )

    current = ["first-rotated-token"]
    supplier_calls = 0

    def supplier() -> str:
        nonlocal supplier_calls
        supplier_calls += 1
        return current[0]

    with running_http_server() as server:
        server.auth_token = current[0]
        async with MCPClient(
            http_config(server.url), token_supplier=supplier
        ) as client:
            await client.call_tool("echo", {"text": "one"})
            current[0] = "second-rotated-token"
            server.auth_token = current[0]
            await client.call_tool("echo", {"text": "two"})
        posts = [request for request in server.requests if request["method"] == "POST"]
        assert supplier_calls == 5
        assert posts[-2]["headers"]["authorization"] == ("Bearer first-rotated-token")
        assert posts[-1]["headers"]["authorization"] == ("Bearer second-rotated-token")


@pytest.mark.parametrize(
    ("supplier", "required_token", "expected"),
    [
        (None, None, MCPAuthorizationStatus.AUTHORIZED),
        (lambda: FAKE_TOKEN, FAKE_TOKEN, MCPAuthorizationStatus.AUTHORIZED),
        (None, "required", MCPAuthorizationStatus.MISSING_CREDENTIAL),
        (lambda: FAKE_TOKEN, "required", MCPAuthorizationStatus.REJECTED),
    ],
)
async def test_authorization_outcome_matrix(supplier, required_token, expected):
    with running_http_server() as server:
        server.auth_token = required_token
        client = MCPClient(http_config(server.url), token_supplier=supplier)
        if expected is MCPAuthorizationStatus.AUTHORIZED:
            async with client:
                assert client.authorization is not None
                assert client.authorization.status is expected
        else:
            with pytest.raises(MCPAuthorizationError) as caught:
                await client.__aenter__()
            assert caught.value.authorization.status is expected
            assert client.authorization is caught.value.authorization
        if supplier is None:
            assert all(
                "authorization" not in request["headers"] for request in server.requests
            )


@pytest.mark.parametrize("metadata", [True, False])
async def test_401_returns_actionable_typed_discovery_without_retry(metadata):
    with running_http_server() as server:
        server.auth_token = "accepted-token"
        server.metadata = metadata
        client = MCPClient(http_config(server.url), token_supplier=lambda: FAKE_TOKEN)
        with pytest.raises(MCPAuthorizationError) as caught:
            await client.__aenter__()
        result = caught.value.authorization
        assert result.status is MCPAuthorizationStatus.REJECTED
        if metadata:
            assert result.authorization_servers == (
                f"http://127.0.0.1:{server.server_port}/authorize",
            )
            assert result.scopes == ("tools:call", "catalog:read")
            assert result.resource_metadata_url == (
                f"http://127.0.0.1:{server.server_port}/metadata"
            )
            assert result.discovery_status is MCPAuthorizationDiscoveryStatus.DISCOVERED
        else:
            assert result.authorization_servers == ()
            assert result.scopes == ()
            assert result.resource_metadata_url is None
            assert (
                result.discovery_status is MCPAuthorizationDiscoveryStatus.UNAVAILABLE
            )
        assert sum(request["method"] == "POST" for request in server.requests) == 1
        assert all(
            "authorization" not in request["headers"]
            for request in server.requests
            if request["method"] == "GET"
        )


@pytest.mark.parametrize("redirect", [False, True])
@pytest.mark.parametrize("same_origin", [False, True])
async def test_redirect_origin_authorization_matrix(redirect, same_origin):
    with running_http_server() as first, running_http_server() as second:
        target = first if same_origin else second
        target_url = target.url.replace("/mcp", "/target")
        if redirect:
            first.redirect_url = target_url
            endpoint = first.url
        else:
            endpoint = target_url
        async with MCPClient(
            http_config(endpoint), token_supplier=lambda: FAKE_TOKEN
        ) as client:
            assert client.ready
        target_requests = [
            request for request in target.requests if request["path"] == "/target"
        ]
        assert target_requests
        expected = None if redirect and not same_origin else f"Bearer {FAKE_TOKEN}"
        assert {
            request["headers"].get("authorization") for request in target_requests
        } == {expected}
        if redirect:
            assert {
                request["headers"].get("authorization")
                for request in first.requests
                if request["path"] == "/mcp"
            } == {f"Bearer {FAKE_TOKEN}"}


@pytest.mark.parametrize(
    ("mode", "status"),
    [("oversize", "message_too_large"), ("lying_length", "protocol_error")],
)
async def test_http_body_bounds_and_lying_length_are_recorded(mode, status):
    limits = MCPLimits(max_message_bytes=512)
    with running_http_server() as server:
        async with MCPClient(http_config(server.url), limits=limits) as client:
            server.mode = mode
            with pytest.raises(MCPError) as caught:
                await client.call_tool("echo", {"text": "bounded"})
            assert caught.value.status == status
            assert not client.ready
            assert client._http_writers == set()


async def test_http_catalog_bound_is_shared():
    with running_http_server() as server:
        server.mode = "large_catalog"
        client = MCPClient(
            http_config(server.url, allowed=frozenset({"echo", "second"})),
            limits=MCPLimits(max_tools=1),
        )
        with pytest.raises(MCPError) as caught:
            await client.__aenter__()
        assert caught.value.status == "catalog_too_large"
        assert not client.ready


@pytest.mark.parametrize(
    ("mode", "limits", "status"),
    [
        ("pages", MCPLimits(max_pages=1), "catalog_too_large"),
        ("normal", MCPLimits(max_catalog_bytes=100), "catalog_too_large"),
        ("normal", MCPLimits(max_http_header_bytes=100), "message_too_large"),
        ("normal", MCPLimits(max_http_headers=2), "message_too_large"),
        ("large_session", MCPLimits(max_session_id_bytes=5), "protocol_error"),
    ],
)
async def test_http_initialization_limits_are_recorded(mode, limits, status):
    with running_http_server() as server:
        server.mode = mode
        client = MCPClient(http_config(server.url), limits=limits)
        with pytest.raises(MCPError) as caught:
            await client.__aenter__()
        assert caught.value.status == status
        assert not client.ready
        assert client._http_writers == set()


@pytest.mark.parametrize(
    ("mode", "limits", "arguments", "status", "remains_ready"),
    [
        (
            "normal",
            MCPLimits(max_request_bytes=300),
            {"text": "x" * 1_000},
            "request_too_large",
            True,
        ),
        (
            "normal",
            MCPLimits(max_result_characters=50),
            {"text": "result"},
            "result_too_large",
            False,
        ),
        (
            "sse_interleave",
            MCPLimits(max_interleaved_messages=1),
            {},
            "protocol_error",
            False,
        ),
    ],
)
async def test_http_shared_request_result_and_interleave_limits(
    mode, limits, arguments, status, remains_ready
):
    with running_http_server() as server:
        async with MCPClient(http_config(server.url), limits=limits) as client:
            server.mode = mode
            with pytest.raises(MCPError) as caught:
                await client.call_tool("echo", arguments)
            assert caught.value.status == status
            assert client.ready is remains_ready


async def test_redirect_credential_and_authorization_discovery_limits():
    with running_http_server() as server:
        server.redirect_url = server.url
        client = MCPClient(
            http_config(server.url), limits=MCPLimits(max_http_redirects=1)
        )
        with pytest.raises(MCPError) as caught:
            await client.__aenter__()
        assert caught.value.status == "redirect_rejected"
        assert len(server.requests) == 2

    with running_http_server() as server:
        client = MCPClient(
            http_config(server.url),
            limits=MCPLimits(max_credential_bytes=4),
            token_supplier=lambda: FAKE_TOKEN,
        )
        with pytest.raises(MCPError) as caught:
            await client.__aenter__()
        assert caught.value.status == "credential_invalid"
        assert server.requests == []

    with running_http_server() as server:
        server.auth_token = "accepted"
        client = MCPClient(
            http_config(server.url),
            limits=MCPLimits(
                max_authorization_scopes=1,
                max_authorization_scope_characters=4,
            ),
            token_supplier=lambda: FAKE_TOKEN,
        )
        with pytest.raises(MCPAuthorizationError) as caught:
            await client.__aenter__()
        assert (
            caught.value.authorization.discovery_status
            is MCPAuthorizationDiscoveryStatus.LIMIT_EXCEEDED
        )
        assert caught.value.authorization.scopes == ()


async def test_overlong_sse_is_recorded_and_socket_closes():
    with running_http_server() as server:
        async with MCPClient(
            http_config(server.url),
            limits=MCPLimits(max_message_bytes=300, request_timeout_seconds=1),
        ) as client:
            server.mode = "endless_sse"
            with pytest.raises(MCPError) as caught:
                await client.call_tool("echo", {})
            assert caught.value.status == "message_too_large"
            assert client._http_writers == set()


@pytest.mark.parametrize("mode", ["no_response", "endless_sse"])
async def test_nonresponding_server_timeout_leaves_no_client_work(mode):
    limits = MCPLimits(
        request_timeout_seconds=0.12,
        shutdown_timeout_seconds=0.05,
        max_message_bytes=1_000_000,
    )
    with running_http_server() as server:
        async with MCPClient(http_config(server.url), limits=limits) as client:
            server.mode = mode
            current_task = asyncio.current_task()
            before_tasks = {
                task for task in asyncio.all_tasks() if task is not current_task
            }
            before_threads = threading.active_count()
            with pytest.raises(MCPError) as caught:
                await client.call_tool("echo", {})
            assert caught.value.status == "timeout"
            for _ in range(50):
                with server.active_lock:
                    active = server.active_handlers
                if active == 0:
                    break
                await asyncio.sleep(0.01)
            after_tasks = {
                task
                for task in asyncio.all_tasks()
                if task is not current_task and not task.done()
            }
            assert after_tasks == before_tasks
            assert threading.active_count() == before_threads
            assert active == 0
            assert client._http_writers == set()
            assert not client.ready


async def test_http_cancellation_and_external_close_stop_pending_work():
    limits = MCPLimits(
        request_timeout_seconds=2,
        shutdown_timeout_seconds=0.05,
        max_message_bytes=1_000_000,
    )
    for cancel_directly in (True, False):
        with running_http_server() as server:
            async with MCPClient(http_config(server.url), limits=limits) as client:
                server.mode = "endless_sse"
                task = asyncio.create_task(client.call_tool("echo", {}))
                for _ in range(50):
                    with server.active_lock:
                        active = server.active_handlers
                    if active:
                        break
                    await asyncio.sleep(0.01)
                if cancel_directly:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    await client.aclose()
                    with pytest.raises(MCPError) as caught:
                        await task
                    assert caught.value.status == "disconnected"
                for _ in range(50):
                    with server.active_lock:
                        active = server.active_handlers
                    if active == 0:
                        break
                    await asyncio.sleep(0.01)
                assert active == 0
                assert client._http_writers == set()
                assert client._http_tasks == {}
                assert not client.ready


@pytest.mark.parametrize(
    "mode", ["bad_status", "truncated_sse", "non_json", "wrong_content"]
)
async def test_malformed_http_and_sse_are_typed(mode):
    with running_http_server() as server:
        async with MCPClient(http_config(server.url)) as client:
            server.mode = mode
            with pytest.raises(MCPError) as caught:
                await client.call_tool("echo", {})
            assert type(caught.value) is MCPError
            assert caught.value.status == "protocol_error"


async def test_session_and_protocol_headers_and_dropped_session():
    with running_http_server() as server:
        async with MCPClient(http_config(server.url)) as client:
            server.mode = "session_drop"
            with pytest.raises(MCPError) as caught:
                await client.call_tool("echo", {})
            assert caught.value.status == "session_lost"
            assert not client.ready
        posts = [request for request in server.requests if request["method"] == "POST"]
        assert [json.loads(request["body"])["method"] for request in posts] == [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
        ]
        assert "mcp-protocol-version" not in posts[0]["headers"]
        assert {
            request["headers"]["mcp-protocol-version"] for request in posts[1:]
        } == {"2025-11-25"}
        assert {request["headers"]["mcp-session-id"] for request in posts[1:]} == {
            "fixture-session"
        }


async def test_fake_token_absent_from_errors_logs_agent_audit_and_urls(
    tmp_path: Path, caplog
):
    caplog.set_level(logging.DEBUG)
    with running_http_server() as server:
        server.auth_token = FAKE_TOKEN
        async with MCPClient(
            http_config(server.url), token_supplier=lambda: FAKE_TOKEN
        ) as client:
            server.mode = "rpc_echo_secret"
            with pytest.raises(MCPError) as caught:
                await client.call_tool("echo", {})
            assert FAKE_TOKEN not in str(caught.value)
            assert FAKE_TOKEN not in repr(caught.value)

        server.mode = "normal"
        async with MCPClient(
            http_config(server.url), token_supplier=lambda: FAKE_TOKEN
        ) as client:
            selected = client.tools[0]
            server.auth_token = "rotated-away"

            async def provider(_request: AgentCompletionRequest) -> AgentCompletion:
                return AgentCompletion(
                    tool_calls=(ToolCall(selected.provider_name, {}, "auth"),)
                )

            agent = ToolCallingAgent(
                LocalEnvironment(tmp_path),
                LocalWorkspaceDeltaObserver(tmp_path),
                provider,
                mcp_clients=(client,),
            )
            outcome = await agent(context(agent.initial_state()))
            serialized = json.dumps(asdict(outcome), sort_keys=True)
            assert outcome.tool_audits[0]["authorization"]["status"] == "rejected"
            assert FAKE_TOKEN not in serialized
        assert FAKE_TOKEN not in "\n".join(
            record.getMessage() for record in caplog.records
        )
        assert all(FAKE_TOKEN not in request["path"] for request in server.requests)

    def failed_supplier() -> str:
        raise RuntimeError(FAKE_TOKEN)

    with running_http_server() as server:
        client = MCPClient(http_config(server.url), token_supplier=failed_supplier)
        with pytest.raises(MCPError) as caught:
            await client.__aenter__()
        assert caught.value.status == "credential_unavailable"
        assert caught.value.__cause__ is None
        assert FAKE_TOKEN not in str(caught.value)
        assert server.requests == []


async def test_stdio_wire_bytes_remain_literal_identical(tmp_path: Path):
    wire = tmp_path / "stdio-wire.jsonl"
    config = MCPServerConfig(
        name="stdio",
        command=(sys.executable, str(STDIO_FIXTURE), "normal"),
        allowed_tools=frozenset({"echo"}),
        env={"MCP_TEST_RAW_LOG": str(wire)},
    )
    async with MCPClient(config) as client:
        await client.call_tool("echo", {"text": "byte-check"})
    assert wire.read_bytes() == (
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":'
        b'{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":'
        b'{"name":"driftlock","version":"0.1.0"}}}\n'
        b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
        b'{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n'
        b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":'
        b'{"name":"echo","arguments":{"text":"byte-check"}}}\n'
    )


def test_http_config_limits_and_status_enum_are_exhaustive():
    assert [(status.name, status.value) for status in MCPAuthorizationStatus] == [
        ("AUTHORIZED", "authorized"),
        ("REJECTED", "rejected"),
        ("MISSING_CREDENTIAL", "missing_credential"),
    ]
    assert len({status.value for status in MCPAuthorizationStatus}) == 3
    assert [
        (status.name, status.value) for status in MCPAuthorizationDiscoveryStatus
    ] == [
        ("NOT_REQUESTED", "not_requested"),
        ("DISCOVERED", "discovered"),
        ("UNAVAILABLE", "unavailable"),
        ("MALFORMED", "malformed"),
        ("LIMIT_EXCEEDED", "limit_exceeded"),
    ]
    assert len({status.value for status in MCPAuthorizationDiscoveryStatus}) == 5
    for field in (
        "max_http_header_bytes",
        "max_http_headers",
        "max_http_redirects",
        "max_credential_bytes",
        "max_session_id_bytes",
        "max_authorization_servers",
        "max_authorization_scopes",
        "max_authorization_scope_characters",
    ):
        with pytest.raises(ValueError):
            MCPLimits(**{field: 0})
    with pytest.raises(ValueError):
        MCPServerConfig("both", ("true",), frozenset(), url="https://example.com/mcp")
    with pytest.raises(ValueError):
        MCPServerConfig("neither", None, frozenset())
    with pytest.raises(ValueError):
        http_config("http://example.com/mcp")
