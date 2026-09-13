"""Real local Streamable HTTP server used by MCP transport seam tests."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class MCPHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), MCPHTTPHandler)
        self.requests: list[dict[str, Any]] = []
        self.mode = "normal"
        self.auth_token: str | None = None
        self.metadata = True
        self.include_www_authenticate = True
        self.metadata_servers: list[str] | None = None
        self.metadata_scopes: list[str] | None = None
        self.metadata_body: bytes | None = None
        self.metadata_redirect_url: str | None = None
        self.redirect_url: str | None = None
        self.active_handlers = 0
        self.active_lock = threading.Lock()
        self.stop_stream = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/mcp"


class MCPHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: MCPHTTPServer

    def log_message(self, _format: str, *args: object) -> None:
        pass

    def _record(self, body: bytes = b"") -> None:
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "body": body.decode("utf-8", "replace"),
            }
        )

    def _start(self) -> None:
        with self.server.active_lock:
            self.server.active_handlers += 1

    def _finish(self) -> None:
        with self.server.active_lock:
            self.server.active_handlers -= 1

    def _headers(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str | None = "application/json",
        length: int | None = None,
        extra: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.send_response(status)
        if content_type is not None:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body) if length is None else length))
        self.send_header("Connection", "close")
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._start()
        try:
            self._record()
            if self.server.metadata and (
                self.path == "/metadata"
                or self.path.startswith("/.well-known/oauth-protected-resource")
            ):
                if self.server.metadata_redirect_url is not None:
                    self._headers(
                        307,
                        b"",
                        content_type=None,
                        extra=(("Location", self.server.metadata_redirect_url),),
                    )
                    return
                servers = (
                    [f"http://127.0.0.1:{self.server.server_port}/authorize"]
                    if self.server.metadata_servers is None
                    else self.server.metadata_servers
                )
                scopes = (
                    ["catalog:read", "tools:call"]
                    if self.server.metadata_scopes is None
                    else self.server.metadata_scopes
                )
                body = self.server.metadata_body
                if body is None:
                    body = json.dumps(
                        {
                            "resource": self.server.url,
                            "authorization_servers": servers,
                            "scopes_supported": scopes,
                        },
                        separators=(",", ":"),
                    ).encode()
                self._headers(200, body)
            else:
                self._headers(404, b"", content_type=None)
        finally:
            self._finish()

    def do_POST(self) -> None:
        self._start()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            self._record(body)
            if self.server.redirect_url is not None and self.path == "/mcp":
                self._headers(
                    307,
                    b"",
                    content_type=None,
                    extra=(("Location", self.server.redirect_url),),
                )
                return
            if (
                self.server.auth_token is not None
                and self.headers.get("Authorization")
                != f"Bearer {self.server.auth_token}"
            ):
                extra: tuple[tuple[str, str], ...] = ()
                if self.server.metadata and self.server.include_www_authenticate:
                    extra = (
                        (
                            "WWW-Authenticate",
                            'Bearer resource_metadata="'
                            f"http://127.0.0.1:{self.server.server_port}/metadata"
                            '", scope="tools:call catalog:read"',
                        ),
                    )
                self._headers(401, b"", content_type=None, extra=extra)
                return
            if self.server.mode == "bad_status":
                self.connection.sendall(b"THIS IS NOT HTTP\r\n\r\n")
                self.close_connection = True
                return
            message = json.loads(body)
            method = message.get("method")
            if "id" not in message:
                self._headers(202, b"", content_type=None)
                return
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fixture", "version": "1"},
                }
                self._rpc(message["id"], result, session=True)
            elif method == "tools/list":
                tools = [
                    {
                        "name": "echo",
                        "description": "Echo text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                    }
                ]
                if self.server.mode == "large_catalog":
                    tools.append(
                        {
                            "name": "second",
                            "description": "Second tool",
                            "inputSchema": {"type": "object"},
                        }
                    )
                result = {"tools": tools}
                if self.server.mode == "pages":
                    result = {"tools": [], "nextCursor": str(message["id"])}
                self._rpc(message["id"], result)
            elif method == "tools/call":
                self._call_response(message)
            else:
                self._rpc(message["id"], {})
        finally:
            self._finish()

    def _rpc(
        self, request_id: int, result: dict[str, Any], *, session: bool = False
    ) -> None:
        body = json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "result": result},
            separators=(",", ":"),
        ).encode()
        session_id = (
            "x" * 2_000 if self.server.mode == "large_session" else "fixture-session"
        )
        extra = (("Mcp-Session-Id", session_id),) if session else ()
        self._headers(200, body, extra=extra)

    def _call_response(self, message: dict[str, Any]) -> None:
        mode = self.server.mode
        if mode == "session_drop":
            self._headers(404, b"", content_type=None)
            return
        result = {
            "content": [
                {
                    "type": "text",
                    "text": message["params"]["arguments"].get("text", ""),
                }
            ],
            "isError": False,
        }
        body = json.dumps(
            {"jsonrpc": "2.0", "id": message["id"], "result": result},
            separators=(",", ":"),
        ).encode()
        if mode == "oversize":
            body = b"{" + b"x" * 2_000 + b"}"
            self._headers(200, body)
        elif mode == "lying_length":
            self._headers(200, body, length=len(body) - 1)
        elif mode == "non_json":
            self._headers(200, b"not-json")
        elif mode == "wrong_content":
            self._headers(200, body, content_type="text/plain")
        elif mode == "truncated_sse":
            event = b"data: " + body
            self._headers(200, event, content_type="text/event-stream")
        elif mode == "sse":
            event = b"event: message\r\ndata: " + body + b"\r\n\r\n"
            self._headers(200, event, content_type="text/event-stream")
        elif mode == "sse_interleave":
            notification = (
                b'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
            )
            event = notification * 3 + b"data: " + body + b"\n\n"
            self._headers(200, event, content_type="text/event-stream")
        elif mode == "endless_sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            while not self.server.stop_stream.wait(0.002):
                try:
                    chunk = b": keepalive\n\n"
                    self.wfile.write(f"{len(chunk):x}\r\n".encode())
                    self.wfile.write(chunk + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        elif mode == "no_response":
            self.connection.setblocking(False)
            while not self.server.stop_stream.wait(0.002):
                try:
                    if self.connection.recv(1, socket.MSG_PEEK) == b"":
                        break
                except BlockingIOError:
                    continue
                except OSError:
                    break
        elif mode == "rpc_echo_secret":
            secret = self.headers.get("Authorization", "")
            error = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32000, "message": secret},
                },
                separators=(",", ":"),
            ).encode()
            self._headers(200, error)
        else:
            self._headers(200, body)


@contextmanager
def running_http_server() -> Iterator[MCPHTTPServer]:
    server = MCPHTTPServer()
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01),
        name="mcp-http-fixture",
    )
    thread.start()
    try:
        yield server
    finally:
        server.stop_stream.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError("MCP HTTP fixture thread did not stop")
