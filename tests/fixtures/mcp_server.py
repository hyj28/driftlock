"""Tiny real stdio MCP server for transport and public integration tests."""

import json
import os
import signal
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"
INITIALIZED = False
if MODE == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def send(value):
    print(json.dumps({"jsonrpc": "2.0", **value}), flush=True)


def tool(name):
    return {
        "name": name,
        "description": f"Fixture {name}",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"] if name == "echo" else [],
        },
    }


print("fixture diagnostics are on stderr", file=sys.stderr, flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if log := os.environ.get("MCP_TEST_LOG"):
        with open(log, "a") as stream:
            stream.write(json.dumps(request) + "\n")
    method = request.get("method")
    if method == "initialize":
        send(
            {
                "id": request["id"],
                "result": {
                    "protocolVersion": (
                        "1900-01-01"
                        if MODE == "version"
                        else "2025-06-18"
                        if MODE == "older"
                        else "2025-11-25"
                    ),
                    "serverInfo": {"name": "fixture", "version": "1"},
                    "capabilities": {} if MODE == "no_tools" else {"tools": {}},
                    "instructions": "Untrusted instructions must not be promoted.",
                },
            }
        )
    elif method == "notifications/initialized":
        INITIALIZED = True
    elif method == "tools/list":
        assert INITIALIZED, "tools/list before initialized"
        entries = [tool("echo"), tool("fail"), tool("hidden")]
        result = {"tools": entries}
        if MODE == "pagination":
            if request["params"].get("cursor") == "second":
                result = {"tools": entries[1:]}
            else:
                result = {"tools": entries[:1], "nextCursor": "second"}
        elif MODE == "pages":
            result = {"tools": [], "nextCursor": str(request["id"])}
        elif MODE == "duplicate":
            result = {"tools": [tool("echo"), tool("echo")]}
        elif MODE == "schema":
            entries[0]["inputSchema"] = {"type": "string"}
        elif MODE == "task":
            entries[0]["execution"] = {"taskSupport": "required"}
        send({"id": request["id"], "result": result})
    elif method == "tools/call":
        if MODE in {"timeout", "stubborn"}:
            time.sleep(60)
        if MODE == "eof":
            sys.exit(0)
        if MODE == "malformed":
            print("not-json", flush=True)
            continue
        if MODE == "oversize":
            print("x" * 20_000, flush=True)
            continue
        if MODE == "rpc_error":
            send({"id": request["id"], "error": {"code": -32602, "message": "bad"}})
            continue
        if MODE == "interleave":
            send({"method": "notifications/tools/list_changed"})
            send({"id": "ping-id", "method": "ping"})
            send({"id": "unsupported-id", "method": "sampling/createMessage"})
        text = request["params"]["arguments"].get("text", "")
        if MODE == "environment":
            text = json.dumps(
                {key: os.environ.get(key) for key in ("MCP_SECRET", "OK")}
            )
        if request["params"]["name"] == "fail":
            result = {
                "content": [{"type": "text", "text": "fixture tool failed"}],
                "isError": True,
            }
        else:
            result = {
                "content": [{"type": "text", "text": text}],
                "structuredContent": {"text": text},
                "isError": False,
            }
        if MODE == "shape":
            result["content"] = [{"type": "text", "text": 7}]
        send({"id": request["id"], "result": result})
if MODE == "stubborn":
    time.sleep(60)
