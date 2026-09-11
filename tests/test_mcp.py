"""Real subprocess coverage of the bounded, allowlisted stdio MCP client."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from driftlock.mcp import MCPClient, MCPError, MCPLimits, MCPServerConfig

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_server.py"


def config(mode="normal", **kwargs):
    return MCPServerConfig(
        name=kwargs.pop("name", "test"),
        command=(sys.executable, str(FIXTURE), mode),
        allowed_tools=kwargs.pop("allowed_tools", frozenset({"echo", "fail"})),
        **kwargs,
    )


async def test_handshake_allowlist_call_and_snapshot(tmp_path):
    log = tmp_path / "wire.jsonl"
    async with MCPClient(config(env={"MCP_TEST_LOG": str(log)})) as client:
        assert client.ready
        assert [tool.name for tool in client.tools] == ["echo", "fail"]
        schema = client.tools[0].input_schema
        schema["properties"]["text"]["type"] = "number"
        assert client.tools[0].input_schema["properties"]["text"]["type"] == "string"
        result = await client.call_tool("echo", {"text": "hi"})
        assert result["content"] == [{"type": "text", "text": "hi"}]
        assert result["structuredContent"] == {"text": "hi"}
        with pytest.raises(MCPError, match="allowed catalog"):
            await client.call_tool("hidden", {})
    assert not client.ready
    requests = [json.loads(line) for line in log.read_text().splitlines()]
    assert [request["method"] for request in requests] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    assert requests[-1]["params"] == {"name": "echo", "arguments": {"text": "hi"}}
    with pytest.raises(MCPError) as error:
        await client.call_tool("echo", {})
    assert error.value.status == "disconnected"


async def test_empty_allowlist_and_namespaces():
    async with MCPClient(config(allowed_tools=frozenset())) as empty:
        assert empty.tools == ()
    async with (
        MCPClient(config(name="first")) as first,
        MCPClient(config(name="second")) as second,
    ):
        assert first.tools[0].provider_name != second.tools[0].provider_name
        assert first.tools[0].provider_name.startswith("mcp_first_")
        assert len(first.tools[0].provider_name) <= 64


@pytest.mark.parametrize("mode", ["normal", "older", "pagination", "interleave"])
async def test_success_variants(mode):
    async with MCPClient(config(mode)) as client:
        assert (await client.call_tool("echo", {"text": "value"}))["isError"] is False
        assert (await client.call_tool("fail", {}))["isError"] is True


@pytest.mark.parametrize("mode", ["version", "no_tools", "duplicate", "schema", "task"])
async def test_invalid_catalog_or_handshake(mode):
    client = MCPClient(config(mode))
    with pytest.raises(MCPError):
        await client.__aenter__()
    assert not client.ready
    assert client._process is None


@pytest.mark.parametrize(
    ("mode", "status"),
    [
        ("eof", "disconnected"),
        ("malformed", "protocol_error"),
        ("oversize", "message_too_large"),
        ("shape", "protocol_error"),
    ],
)
async def test_broken_transport_closes(mode, status):
    async with MCPClient(
        config(mode), limits=MCPLimits(max_message_bytes=4096)
    ) as client:
        process = client._process
        with pytest.raises(MCPError) as error:
            await client.call_tool("echo", {})
        assert error.value.status == status
        assert not client.ready
        assert process.returncode is not None


async def test_rpc_error_preserves_session():
    async with MCPClient(config("rpc_error")) as client:
        for _ in range(2):
            with pytest.raises(MCPError) as error:
                await client.call_tool("echo", {})
            assert error.value.status == "server_error"
            assert client.ready


async def test_timeout_reaps_stubborn_server():
    limits = MCPLimits(request_timeout_seconds=0.15, shutdown_timeout_seconds=0.05)
    client = MCPClient(config("stubborn"), limits=limits)
    async with client:
        process = client._process
        with pytest.raises(MCPError) as error:
            await client.call_tool("echo", {})
        assert error.value.status == "timeout"
        assert process.returncode is not None
        assert not client.ready
        with pytest.raises(MCPError) as closed:
            await client.call_tool("echo", {})
        assert closed.value.status == "disconnected"


async def test_cancellation_closes_transport():
    async with MCPClient(
        config("timeout"), limits=MCPLimits(shutdown_timeout_seconds=0.05)
    ) as client:
        process = client._process
        task = asyncio.create_task(client.call_tool("echo", {}))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client.ready
        assert process.returncode is not None


async def test_close_during_pending_request_reaps_server():
    async with MCPClient(
        config("timeout"), limits=MCPLimits(shutdown_timeout_seconds=0.05)
    ) as client:
        process = client._process
        task = asyncio.create_task(client.call_tool("echo", {}))
        await asyncio.sleep(0.05)
        await client.aclose()
        with pytest.raises(MCPError) as error:
            await task
        assert error.value.status == "disconnected"
        assert process.returncode is not None


async def test_result_and_request_limits():
    async with MCPClient(
        config(), limits=MCPLimits(max_result_characters=100, max_request_bytes=1000)
    ) as client:
        with pytest.raises(MCPError) as error:
            await client.call_tool("echo", {"text": "x" * 2000})
        assert error.value.status == "request_too_large"
        assert client.ready
        with pytest.raises(MCPError) as error:
            await client.call_tool("echo", {"text": "x" * 100})
        assert error.value.status == "result_too_large"


async def test_catalog_limits():
    for mode, limits in [
        ("normal", MCPLimits(max_tools=2)),
        ("normal", MCPLimits(max_catalog_bytes=100)),
        ("pages", MCPLimits(max_pages=2)),
    ]:
        with pytest.raises(MCPError) as error:
            async with MCPClient(config(mode), limits=limits):
                pytest.fail("catalog should exceed limit")
        assert error.value.status == "catalog_too_large"


async def test_minimal_environment(monkeypatch):
    monkeypatch.setenv("MCP_SECRET", "do-not-inherit")
    async with MCPClient(config("environment", env={"OK": "explicit"})) as client:
        result = await client.call_tool("echo", {})
    assert json.loads(result["content"][0]["text"]) == {
        "MCP_SECRET": None,
        "OK": "explicit",
    }


def test_config_limits_and_error_validation():
    for overrides in (
        {"request_timeout_seconds": float("nan")},
        {"max_tools": 0},
        {"max_pages": True},
        {"max_request_bytes": 1.2},
    ):
        with pytest.raises(ValueError):
            MCPLimits(**overrides)
    with pytest.raises(ValueError):
        config(name="bad name")
    with pytest.raises(ValueError):
        MCPServerConfig("x", "python server.py", frozenset())
    with pytest.raises(ValueError):
        config(allowed_tools=None)
    env = {"OK": "before"}
    cfg = config(env=env)
    env["OK"] = "after"
    assert cfg.env["OK"] == "before"
    error = MCPError("server_error", "\x1b\n" + "x" * 1000)
    assert len(str(error)) == 500
    assert str(error).isprintable()
