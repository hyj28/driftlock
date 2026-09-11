"""Bounded, explicitly authorized MCP tools over a host-owned stdio process.

Server commands run with the host's permissions, not in a security sandbox.
Catalogs are snapshots; transport failures never trigger retries or reconnection.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

_TOOL_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_VERSIONS = {"2025-11-25", "2025-06-18"}
_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SYSTEMROOT")


class MCPError(Exception):
    """A bounded printable diagnostic; server-provided text remains untrusted."""

    def __init__(self, status: str, message: str) -> None:
        self.status = status
        super().__init__("".join(c if c.isprintable() else " " for c in message)[:500])


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: tuple[str, ...]
    allowed_tools: frozenset[str]
    cwd: Path | str | None = None
    env: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,20}", self.name
        ):
            raise ValueError("MCP server name must be 1-20 ASCII letters/digits/_/-")
        if (
            isinstance(self.command, str)
            or not self.command
            or any(
                not isinstance(arg, str) or not arg or "\0" in arg
                for arg in self.command
            )
        ):
            raise ValueError("MCP command must be a nonempty explicit argv sequence")
        if (
            isinstance(self.allowed_tools, str)
            or self.allowed_tools is None
            or any(
                not isinstance(name, str) or not _TOOL_NAME.fullmatch(name)
                for name in self.allowed_tools
            )
        ):
            raise ValueError(
                "MCP allowed_tools must explicitly contain valid tool names"
            )
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "allowed_tools", frozenset(self.allowed_tools))
        if self.cwd is not None and not isinstance(self.cwd, (str, Path)):
            raise ValueError("MCP cwd must be a path")
        if self.env is not None:
            if not isinstance(self.env, Mapping) or any(
                not isinstance(k, str)
                or not k
                or "=" in k
                or "\0" in k
                or not isinstance(v, str)
                or "\0" in v
                for k, v in self.env.items()
            ):
                raise ValueError("MCP env must map valid environment names to strings")
            object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


@dataclass(frozen=True)
class MCPLimits:
    request_timeout_seconds: float = 30
    max_message_bytes: int = 1_048_576
    max_tools: int = 64
    max_pages: int = 16
    max_catalog_bytes: int = 262_144
    max_result_characters: int = 16_000
    max_request_bytes: int = 262_144
    shutdown_timeout_seconds: float = 1
    max_interleaved_messages: int = 128

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            is_timeout = field.name.endswith("_seconds")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float) if is_timeout else int)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field.name} must be positive and finite")


@dataclass(frozen=True, init=False)
class MCPTool:
    name: str
    provider_name: str
    description: str
    _schema_json: str

    def __init__(
        self,
        name: str,
        provider_name: str,
        description: str,
        input_schema: Mapping[str, Any],
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "provider_name", provider_name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "_schema_json", _json(dict(input_schema)))

    @property
    def input_schema(self) -> Mapping[str, Any]:
        """Return an isolated schema, preserving the catalog snapshot."""
        return json.loads(self._schema_json)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


class MCPClient:
    def __init__(
        self, config: MCPServerConfig, *, limits: MCPLimits | None = None
    ) -> None:
        self.config = config
        self.limits = limits or MCPLimits()
        self._process: asyncio.subprocess.Process | None = None
        self._tools: tuple[MCPTool, ...] = ()
        self._ready = False
        self._started = False
        self._request_id = 0
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return (
            self._ready
            and self._process is not None
            and self._process.returncode is None
        )

    @property
    def tools(self) -> tuple[MCPTool, ...]:
        return self._tools

    async def __aenter__(self) -> MCPClient:
        if self._started:
            raise MCPError("rejected", "MCPClient sessions cannot be restarted")
        self._started = True
        env = {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}
        env.update(self.config.env or {})
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.config.command,
                cwd=self.config.cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=self.limits.max_message_bytes,
            )
            result = await self._request(
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "driftlock", "version": "0.1.0"},
                },
            )
            if (
                not isinstance(result.get("protocolVersion"), str)
                or result["protocolVersion"] not in _VERSIONS
            ):
                raise MCPError("protocol_error", "Unsupported MCP protocol version")
            capabilities = result.get("capabilities")
            info = result.get("serverInfo")
            if (
                not isinstance(capabilities, dict)
                or not isinstance(capabilities.get("tools"), dict)
                or not isinstance(info, dict)
                or not isinstance(info.get("name"), str)
                or not isinstance(info.get("version"), str)
            ):
                raise MCPError(
                    "protocol_error", "Invalid MCP initialization/capabilities"
                )
            async with asyncio.timeout(self.limits.request_timeout_seconds):
                await self._send({"method": "notifications/initialized"})
            self._tools = await self._discover()
            self._ready = True
            return self
        except BaseException as exc:
            await self.aclose()
            if isinstance(exc, OSError):
                raise MCPError("disconnected", "Could not start MCP server") from exc
            if isinstance(exc, TimeoutError):
                raise MCPError("timeout", "MCP initialization timed out") from exc
            raise

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _discover(self) -> tuple[MCPTool, ...]:
        tools: list[MCPTool] = []
        seen: set[str] = set()
        cursors: set[str] = set()
        params: dict[str, Any] = {}
        catalog_bytes = 0
        for _ in range(self.limits.max_pages):
            result = await self._request("tools/list", params)
            catalog_bytes += len(_json(result).encode("utf-8"))
            if catalog_bytes > self.limits.max_catalog_bytes:
                raise MCPError("catalog_too_large", "MCP catalog exceeds byte limit")
            entries = result.get("tools")
            if not isinstance(entries, list):
                raise MCPError(
                    "protocol_error", "MCP tools/list must contain tools array"
                )
            for entry in entries:
                if not isinstance(entry, dict):
                    raise MCPError("protocol_error", "Invalid MCP tool entry")
                name = entry.get("name")
                schema = entry.get("inputSchema")
                description = entry.get("description", "")
                if (
                    not isinstance(name, str)
                    or not _TOOL_NAME.fullmatch(name)
                    or name in seen
                    or not isinstance(description, str)
                    or not isinstance(schema, dict)
                    or schema.get("type") != "object"
                ):
                    raise MCPError("protocol_error", "Invalid or duplicate MCP tool")
                seen.add(name)
                if len(seen) > self.limits.max_tools:
                    raise MCPError(
                        "catalog_too_large", "MCP catalog exceeds tool limit"
                    )
                execution = entry.get("execution", {})
                if (
                    not isinstance(execution, dict)
                    or not isinstance(execution.get("taskSupport", "forbidden"), str)
                    or execution.get("taskSupport", "forbidden")
                    not in {"forbidden", "optional", "required"}
                ):
                    raise MCPError(
                        "protocol_error", "Invalid MCP tool execution metadata"
                    )
                if name not in self.config.allowed_tools:
                    continue
                if execution.get("taskSupport") == "required":
                    raise MCPError("rejected", "Task-only MCP tools are unsupported")
                digest = hashlib.sha256(
                    _json([self.config.name, name]).encode()
                ).hexdigest()[:16]
                prefix = f"mcp_{self.config.name}_"
                suffix = re.sub(r"[^A-Za-z0-9_-]", "_", name)
                provider_name = f"{prefix}{suffix[: 47 - len(prefix)]}_{digest}"
                tools.append(MCPTool(name, provider_name, description, schema))
            cursor = result.get("nextCursor")
            if cursor is None:
                return tuple(tools)
            if not isinstance(cursor, str) or not cursor or cursor in cursors:
                raise MCPError(
                    "protocol_error", "Invalid or repeated MCP catalog cursor"
                )
            cursors.add(cursor)
            params = {"cursor": cursor}
        raise MCPError("catalog_too_large", "MCP catalog exceeds page limit")

    async def call_tool(self, native_name: str, arguments: dict[str, Any]) -> dict:
        if not self.ready:
            raise MCPError("disconnected", "MCP client is not ready")
        if native_name not in self.config.allowed_tools or not any(
            tool.name == native_name for tool in self._tools
        ):
            raise MCPError("rejected", "MCP tool is not in the allowed catalog")
        if not isinstance(arguments, dict):
            raise MCPError("rejected", "MCP tool arguments must be an object")
        result = await self._request(
            "tools/call", {"name": native_name, "arguments": arguments}
        )
        try:
            self._validate_result(result)
        except MCPError:
            await self.aclose()
            raise
        return result

    def _validate_result(self, result: dict) -> None:
        if len(_json(result)) > self.limits.max_result_characters:
            raise MCPError(
                "result_too_large", "MCP tool result exceeds character limit"
            )
        content = result.get("content")
        if (
            not isinstance(content, list)
            or ("isError" in result and not isinstance(result["isError"], bool))
            or (
                "structuredContent" in result
                and not isinstance(result["structuredContent"], dict)
            )
        ):
            raise MCPError("protocol_error", "Malformed MCP CallToolResult")
        for item in content:
            if not isinstance(item, dict):
                raise MCPError("protocol_error", "Malformed MCP content block")
            kind = item.get("type")
            if not isinstance(kind, str):
                raise MCPError("protocol_error", "Malformed MCP content type")
            if kind == "text":
                valid = isinstance(item.get("text"), str)
            elif kind in {"image", "audio"}:
                valid = isinstance(item.get("data"), str) and isinstance(
                    item.get("mimeType"), str
                )
            elif kind == "resource_link":
                valid = isinstance(item.get("uri"), str) and isinstance(
                    item.get("name"), str
                )
            elif kind == "resource":
                resource = item.get("resource")
                valid = (
                    isinstance(resource, dict)
                    and isinstance(resource.get("uri"), str)
                    and (
                        isinstance(resource.get("text"), str)
                        or isinstance(resource.get("blob"), str)
                    )
                )
            else:
                valid = False
            if not valid:
                raise MCPError("protocol_error", "Malformed MCP content block")

    async def _send(self, message: dict) -> None:
        try:
            data = (_json({"jsonrpc": "2.0", **message}) + "\n").encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise MCPError("rejected", "MCP request is not valid JSON") from exc
        if len(data) > self.limits.max_request_bytes:
            raise MCPError("request_too_large", "MCP request exceeds byte limit")
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise MCPError("disconnected", "MCP server is disconnected")
        process.stdin.write(data)
        await process.stdin.drain()

    async def _request(self, method: str, params: dict) -> dict:
        async with self._lock:
            self._request_id += 1
            request_id = self._request_id
            try:
                async with asyncio.timeout(self.limits.request_timeout_seconds):
                    await self._send(
                        {"id": request_id, "method": method, "params": params}
                    )
                    return await self._receive(request_id)
            except (TimeoutError, asyncio.CancelledError) as exc:
                # Do not cancel initialize; it has no established session yet.
                if method != "initialize":
                    with contextlib.suppress(Exception):
                        async with asyncio.timeout(
                            self.limits.shutdown_timeout_seconds
                        ):
                            await self._send(
                                {
                                    "method": "notifications/cancelled",
                                    "params": {"requestId": request_id},
                                }
                            )
                await self.aclose()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise MCPError(
                    "timeout", "MCP request timed out; session closed"
                ) from exc
            except MCPError as exc:
                if exc.status not in {"server_error", "rejected", "request_too_large"}:
                    await self.aclose()
                raise
            except (OSError, ValueError, RecursionError) as exc:
                await self.aclose()
                raise MCPError(
                    "protocol_error", "MCP transport or JSON failure"
                ) from exc

    async def _receive(self, request_id: int) -> dict:
        process = self._process
        if process is None or process.stdout is None:
            raise MCPError("disconnected", "MCP server is disconnected")
        for _ in range(self.limits.max_interleaved_messages + 1):
            try:
                raw = await process.stdout.readline()
            except ValueError as exc:
                raise MCPError(
                    "message_too_large", "MCP message exceeds byte limit"
                ) from exc
            if not raw:
                raise MCPError("disconnected", "MCP server closed stdout")
            if len(raw) > self.limits.max_message_bytes:
                raise MCPError("message_too_large", "MCP message exceeds byte limit")
            if not raw.endswith(b"\n"):
                raise MCPError("protocol_error", "Incomplete MCP message")
            message = json.loads(raw.decode("utf-8"), parse_constant=_invalid_constant)
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                raise MCPError("protocol_error", "Invalid MCP JSON-RPC message")
            if "method" in message:
                if (
                    not isinstance(message["method"], str)
                    or "result" in message
                    or "error" in message
                    or ("params" in message and not isinstance(message["params"], dict))
                ):
                    raise MCPError("protocol_error", "Malformed MCP server message")
                if "id" in message:
                    if type(message["id"]) not in (str, int):
                        raise MCPError(
                            "protocol_error", "Invalid MCP server request id"
                        )
                    response = {"id": message["id"]}
                    if message["method"] == "ping":
                        response["result"] = {}
                    else:
                        response["error"] = {
                            "code": -32601,
                            "message": "Client method not supported",
                        }
                    await self._send(response)
                # Notifications are data, not instructions; catalogs remain snapshots.
                continue
            if (
                type(message.get("id")) is not int
                or message["id"] != request_id
                or ("result" in message) == ("error" in message)
            ):
                raise MCPError("protocol_error", "Unexpected MCP response id or shape")
            if "error" in message:
                error = message["error"]
                if (
                    not isinstance(error, dict)
                    or type(error.get("code")) is not int
                    or not isinstance(error.get("message"), str)
                ):
                    raise MCPError("protocol_error", "Malformed MCP JSON-RPC error")
                raise MCPError("server_error", error["message"])
            if not isinstance(message["result"], dict):
                raise MCPError("protocol_error", "MCP result must be an object")
            return message["result"]
        raise MCPError("protocol_error", "Too many interleaved MCP messages")

    async def aclose(self) -> None:
        """Close stdin, then terminate/kill and reap the direct child if necessary."""
        self._ready = False
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()

        async def drain() -> None:
            if process.stdout is not None:
                with contextlib.suppress(ValueError, OSError, RuntimeError):
                    while await process.stdout.read(65536):
                        pass

        drainer = asyncio.create_task(drain())
        try:
            for action in (None, process.terminate, process.kill):
                if action is not None and process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        action()
                try:
                    await asyncio.wait_for(
                        process.wait(), self.limits.shutdown_timeout_seconds
                    )
                    return
                except TimeoutError:
                    continue
            raise MCPError("disconnected", "MCP server did not finish shutdown in time")
        finally:
            drainer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drainer
