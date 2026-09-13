"""Bounded MCP tools over host-owned stdio or Streamable HTTP.

Server commands run with the host's permissions, not in a security sandbox.
Catalogs are snapshots; transport failures never trigger retries or reconnection.
HTTP credentials come only from the injected supplier; this module never acquires one.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
import ssl
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import SplitResult, unquote, urljoin, urlsplit, urlunsplit

# MCP tool names have a protocol-facing length cap and a conservative ASCII alphabet.
_TOOL_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
# Driftlock implements the current stable version and its immediately preceding version.
_VERSIONS = {"2025-11-25", "2025-06-18"}
# Stdio children receive only process basics, preventing ambient secret inheritance.
_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SYSTEMROOT")
# Eight thousand characters accommodate normal endpoint and discovery URLs while
# bounding retained configuration and authorization state.
_MAX_URL_CHARACTERS = 8_000
# HTTP field names use the RFC 9110 token alphabet; rejecting anything else
# prevents header smuggling.
_HEADER_NAME = re.compile(rb"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")
# Only 307 and 308 preserve an MCP POST without rewriting its method or body.
_PRESERVING_REDIRECTS = frozenset({307, 308})
# These statuses exhaust the redirect space without accepting method-changing behavior.
_ALL_REDIRECTS = frozenset({301, 302, 303, 307, 308})
# Three decoding passes catch nested credential reflection without unbounded work.
_MAX_PERCENT_DECODE_PASSES = 3

TokenSupplier = Callable[[], str | Awaitable[str | None] | None]


class MCPError(Exception):
    """A bounded printable diagnostic; server-provided text remains untrusted."""

    def __init__(self, status: str, message: str) -> None:
        self.status = status
        super().__init__("".join(c if c.isprintable() else " " for c in message)[:500])


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: tuple[str, ...] | None
    allowed_tools: frozenset[str]
    cwd: Path | str | None = None
    env: Mapping[str, str] | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,20}", self.name
        ):
            raise ValueError("MCP server name must be 1-20 ASCII letters/digits/_/-")
        if (self.command is None) == (self.url is None):
            raise ValueError("MCP config requires exactly one of command or url")
        if self.command is not None:
            if (
                isinstance(self.command, str)
                or not self.command
                or any(
                    not isinstance(arg, str) or not arg or "\0" in arg
                    for arg in self.command
                )
            ):
                raise ValueError(
                    "MCP command must be a nonempty explicit argv sequence"
                )
            object.__setattr__(self, "command", tuple(self.command))
        else:
            assert self.url is not None
            _validated_url(self.url, endpoint=True)
            if self.cwd is not None or self.env is not None:
                raise ValueError("MCP HTTP configs cannot set cwd or env")
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
    max_http_header_bytes: int = 32_768
    max_http_headers: int = 64
    max_http_redirects: int = 4
    max_credential_bytes: int = 8_192
    max_session_id_bytes: int = 1_024
    max_authorization_servers: int = 8
    max_authorization_scopes: int = 64
    max_authorization_scope_characters: int = 256

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


class MCPAuthorizationStatus(StrEnum):
    """The three mutually exclusive authorization facts for the last request."""

    AUTHORIZED = "authorized"
    REJECTED = "rejected"
    MISSING_CREDENTIAL = "missing_credential"


class MCPAuthorizationDiscoveryStatus(StrEnum):
    """What bounded protected-resource metadata discovery established."""

    NOT_REQUESTED = "not_requested"
    DISCOVERED = "discovered"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"
    LIMIT_EXCEEDED = "limit_exceeded"


@dataclass(frozen=True, slots=True)
class MCPAuthorizationResult:
    """Bounded host action for an HTTP authorization challenge."""

    status: MCPAuthorizationStatus
    authorization_servers: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    resource_metadata_url: str | None = None
    discovery_status: MCPAuthorizationDiscoveryStatus = (
        MCPAuthorizationDiscoveryStatus.NOT_REQUESTED
    )

    def __post_init__(self) -> None:
        if not isinstance(self.status, MCPAuthorizationStatus):
            raise TypeError("status must be an MCPAuthorizationStatus")
        if not isinstance(self.discovery_status, MCPAuthorizationDiscoveryStatus):
            raise TypeError(
                "discovery_status must be an MCPAuthorizationDiscoveryStatus"
            )
        if self.status is MCPAuthorizationStatus.AUTHORIZED:
            if (
                self.authorization_servers
                or self.scopes
                or self.resource_metadata_url is not None
                or self.discovery_status
                is not MCPAuthorizationDiscoveryStatus.NOT_REQUESTED
            ):
                raise ValueError("authorized results cannot carry a challenge")
        elif self.discovery_status is MCPAuthorizationDiscoveryStatus.NOT_REQUESTED:
            raise ValueError("authorization challenges require a discovery outcome")
        if self.discovery_status is MCPAuthorizationDiscoveryStatus.DISCOVERED and (
            self.resource_metadata_url is None
        ):
            raise ValueError("discovered authorization requires metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status.value,
            "authorization_servers": list(self.authorization_servers),
            "scopes": list(self.scopes),
            "resource_metadata_url": self.resource_metadata_url,
            "discovery_status": self.discovery_status.value,
        }


class MCPAuthorizationError(MCPError):
    """An HTTP authorization challenge with a typed, actionable result."""

    def __init__(self, authorization: MCPAuthorizationResult) -> None:
        self.authorization = authorization
        if authorization.status is MCPAuthorizationStatus.MISSING_CREDENTIAL:
            status = "missing_credential"
            message = "MCP server requires a credential but none is available"
        else:
            status = "authorization_rejected"
            message = "MCP server rejected the supplied credential"
        super().__init__(status, message)


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


def _validated_url(value: str, *, endpoint: bool = False) -> SplitResult:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_URL_CHARACTERS
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError("MCP URL must be bounded visible ASCII")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(
            "MCP URL must be an absolute HTTP URL without userinfo or fragment"
        )
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("MCP URL has an invalid port") from None
    del port
    if (
        endpoint
        and parsed.scheme == "http"
        and parsed.hostname
        not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
    ):
        raise ValueError("MCP HTTP endpoints must use HTTPS except on loopback")
    return parsed


def _origin(url: str) -> tuple[str, str, int]:
    parsed = _validated_url(url)
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, parsed.hostname.lower(), parsed.port or default_port


def _request_target(parsed: SplitResult) -> str:
    path = parsed.path or "/"
    return urlunsplit(("", "", path, parsed.query, ""))


def _host_header(parsed: SplitResult) -> str:
    hostname = parsed.hostname or ""
    rendered = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 443 if parsed.scheme == "https" else 80
    return (
        rendered
        if (parsed.port or default_port) == default_port
        else f"{rendered}:{parsed.port}"
    )


@dataclass(frozen=True, slots=True)
class _HTTPResponse:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    url: str

    def values(self, name: str) -> tuple[str, ...]:
        lowered = name.lower()
        return tuple(value for key, value in self.headers if key == lowered)

    def value(self, name: str) -> str | None:
        values = self.values(name)
        return values[0] if len(values) == 1 else None


def _media_type(response: _HTTPResponse) -> str | None:
    value = response.value("content-type")
    return value.split(";", 1)[0].strip().lower() if value is not None else None


def _contains_credential(value: str, token: str | None) -> bool:
    if token is None:
        return False
    decoded = value
    for _ in range(_MAX_PERCENT_DECODE_PASSES):
        if token in decoded:
            return True
        next_value = unquote(decoded)
        if next_value == decoded:
            return False
        decoded = next_value
    return token in decoded


def _parse_bearer_challenge(
    values: tuple[str, ...], token: str | None, limits: MCPLimits
) -> tuple[str | None, tuple[str, ...], bool]:
    joined = ",".join(values)
    if _contains_credential(joined, token):
        return None, (), False
    match = re.search(r"(?:^|,)\s*Bearer(?:\s+|$)(.*)", joined, re.IGNORECASE)
    if match is None:
        return None, (), False
    parameters: dict[str, str] = {}
    pattern = re.compile(
        r'(?:^|,)\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(?:"((?:\\.|[^"\\])*)"|([^,\s]+))'
    )
    for parameter in pattern.finditer(match.group(1)):
        key = parameter.group(1).lower()
        raw = parameter.group(2)
        value = re.sub(r"\\(.)", r"\1", raw) if raw is not None else parameter.group(3)
        if value is not None and key not in parameters:
            parameters[key] = value
    metadata_url = parameters.get("resource_metadata")
    if metadata_url is not None:
        try:
            parsed = _validated_url(metadata_url)
        except ValueError:
            metadata_url = None
        else:
            if parsed.query:
                metadata_url = None
    scope_value = parameters.get("scope")
    scopes = tuple(scope_value.split()) if scope_value else ()
    if len(scopes) > limits.max_authorization_scopes or any(
        not scope
        or len(scope) > limits.max_authorization_scope_characters
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in scope)
        or _contains_credential(scope, token)
        for scope in scopes
    ):
        return metadata_url, (), True
    return metadata_url, scopes, False


def _metadata_urls(endpoint: str | None) -> tuple[str, ...]:
    assert endpoint is not None
    parsed = _validated_url(endpoint)
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    path = parsed.path or "/"
    endpoint_candidate = f"{origin}/.well-known/oauth-protected-resource{path}"
    root_candidate = f"{origin}/.well-known/oauth-protected-resource"
    return (
        (endpoint_candidate, root_candidate)
        if endpoint_candidate != root_candidate
        else (root_candidate,)
    )


class MCPClient:
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        limits: MCPLimits | None = None,
        token_supplier: TokenSupplier | None = None,
    ) -> None:
        self.config = config
        self.limits = limits or MCPLimits()
        if token_supplier is not None and not callable(token_supplier):
            raise TypeError("token_supplier must be callable or None")
        if token_supplier is not None and config.url is None:
            raise ValueError("token_supplier is only valid for MCP HTTP")
        self._token_supplier = token_supplier
        self._process: asyncio.subprocess.Process | None = None
        self._http_open = False
        self._http_writers: set[asyncio.StreamWriter] = set()
        self._http_tasks: dict[asyncio.Task[Any], int] = {}
        self._ssl_context = (
            ssl.create_default_context() if config.url is not None else None
        )
        self._session_id: str | None = None
        self._session_origin: tuple[str, str, int] | None = None
        self._protocol_version: str | None = None
        self._authorization: MCPAuthorizationResult | None = None
        self._tools: tuple[MCPTool, ...] = ()
        self._ready = False
        self._started = False
        self._request_id = 0
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        if self.config.url is not None:
            return self._ready and self._http_open
        return (
            self._ready
            and self._process is not None
            and self._process.returncode is None
        )

    @property
    def tools(self) -> tuple[MCPTool, ...]:
        return self._tools

    @property
    def authorization(self) -> MCPAuthorizationResult | None:
        return self._authorization

    async def __aenter__(self) -> MCPClient:
        if self._started:
            raise MCPError("rejected", "MCPClient sessions cannot be restarted")
        self._started = True
        try:
            if self.config.url is None:
                env = {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}
                env.update(self.config.env or {})
                assert self.config.command is not None
                self._process = await asyncio.create_subprocess_exec(
                    *self.config.command,
                    cwd=self.config.cwd,
                    env=env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    limit=self.limits.max_message_bytes,
                )
            else:
                self._http_open = True
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
            self._protocol_version = result["protocolVersion"]
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

    async def _credential(self) -> str | None:
        supplier = self._token_supplier
        if supplier is None:
            return None
        try:
            supplied = supplier()
            token = await supplied if isinstance(supplied, Awaitable) else supplied
        except asyncio.CancelledError:
            raise
        except BaseException:
            supplier_failed = True
        else:
            supplier_failed = False
        if supplier_failed:
            # Raise after leaving the handler: ``from None`` hides a context from
            # tracebacks but does not clear __context__, which must not retain a
            # supplier exception that may itself contain the credential.
            raise MCPError("credential_unavailable", "MCP credential supplier failed")
        if token is None:
            return None
        if (
            not isinstance(token, str)
            or not token
            or len(token.encode("utf-8")) > self.limits.max_credential_bytes
            or any(
                ord(character) < 0x21 or ord(character) > 0x7E for character in token
            )
        ):
            raise MCPError("credential_invalid", "MCP credential is invalid")
        return token

    async def _http_send(self, message: dict) -> None:
        data = self._http_json(message)
        response = await self._http_exchange(data, request_id=None)
        if response.status != 202 or response.body:
            raise MCPError("protocol_error", "Invalid MCP HTTP notification response")
        self._authorization = MCPAuthorizationResult(MCPAuthorizationStatus.AUTHORIZED)

    async def _http_rpc_request(self, message: dict, request_id: int) -> dict[str, Any]:
        data = self._http_json(message)
        response = await self._http_exchange(data, request_id=request_id)
        if response.status != 200:
            raise MCPError("protocol_error", "Invalid MCP HTTP response status")
        result = await self._http_response_result(response, request_id)
        self._authorization = MCPAuthorizationResult(MCPAuthorizationStatus.AUTHORIZED)
        return result

    def _http_json(self, message: dict) -> bytes:
        try:
            data = _json({"jsonrpc": "2.0", **message}).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise MCPError("rejected", "MCP request is not valid JSON") from exc
        if len(data) > self.limits.max_request_bytes:
            raise MCPError("request_too_large", "MCP request exceeds byte limit")
        return data

    async def _http_exchange(
        self, data: bytes, *, request_id: int | None
    ) -> _HTTPResponse:
        if not self._http_open or self.config.url is None:
            raise MCPError("disconnected", "MCP server is disconnected")
        token = await self._credential()
        if _contains_credential(self.config.url, token):
            raise MCPError("credential_invalid", "MCP credential is invalid")
        response = await self._http_round_trip(
            "POST",
            self.config.url,
            data,
            token=token,
            forbidden_credential=token,
            request_id=request_id,
        )
        if response.status in {401, 403}:
            authorization = await self._authorization_challenge(response, token)
            self._authorization = authorization
            raise MCPAuthorizationError(authorization)
        if response.status == 404 and self._session_id is not None:
            raise MCPError("session_lost", "MCP HTTP session was terminated by server")
        if response.status in _ALL_REDIRECTS:
            raise MCPError("redirect_rejected", "MCP HTTP redirect was refused")
        if response.status not in {200, 202}:
            raise MCPError("http_error", "MCP HTTP server rejected the request")
        self._update_session(response, initialize=request_id == 1)
        return response

    async def _http_round_trip(
        self,
        method: str,
        url: str,
        body: bytes | None,
        *,
        token: str | None,
        forbidden_credential: str | None,
        request_id: int | None,
        metadata: bool = False,
    ) -> _HTTPResponse:
        current = url
        credential_origin = _origin(url)
        for redirect_count in range(self.limits.max_http_redirects + 1):
            try:
                parsed = _validated_url(current, endpoint=True)
            except ValueError:
                raise MCPError(
                    "redirect_rejected", "MCP HTTP redirect was refused"
                ) from None
            if _contains_credential(current, forbidden_credential):
                raise MCPError("credential_invalid", "MCP credential is invalid")
            send_credential = (
                token is not None and _origin(current) == credential_origin
            )
            headers = self._http_request_headers(
                parsed,
                body,
                token if send_credential else None,
                metadata=metadata,
            )
            response = await self._one_http_request(
                method,
                current,
                headers,
                body,
                request_id=None if metadata else request_id,
            )
            if response.status not in _ALL_REDIRECTS:
                return response
            if response.status not in _PRESERVING_REDIRECTS:
                return response
            location = response.value("location")
            if location is None or redirect_count == self.limits.max_http_redirects:
                return response
            try:
                target = urljoin(current, location)
                _validated_url(target, endpoint=True)
                if _contains_credential(target, forbidden_credential):
                    raise ValueError("redirect contains credential")
            except ValueError:
                raise MCPError(
                    "redirect_rejected", "MCP HTTP redirect was refused"
                ) from None
            # Sensitive headers are rebuilt for every hop. Authorization is scoped
            # to the configured origin, and a session ID to the origin that issued
            # it, so urllib-style cross-origin header replay is impossible.
            current = target
        raise MCPError("redirect_rejected", "MCP HTTP redirect was refused")

    def _http_request_headers(
        self,
        parsed: SplitResult,
        body: bytes | None,
        token: str | None,
        *,
        metadata: bool,
    ) -> tuple[tuple[str, str], ...]:
        headers = [
            ("Host", _host_header(parsed)),
            ("Connection", "close"),
            (
                "Accept",
                "application/json"
                if metadata
                else "application/json, text/event-stream",
            ),
        ]
        if body is not None:
            headers.extend(
                (
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(body))),
                )
            )
        if self._protocol_version is not None:
            headers.append(("MCP-Protocol-Version", self._protocol_version))
        current_origin = (
            parsed.scheme,
            (parsed.hostname or "").lower(),
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
        if (
            not metadata
            and self._session_id is not None
            and current_origin == self._session_origin
        ):
            headers.append(("Mcp-Session-Id", self._session_id))
        if token is not None:
            headers.append(("Authorization", f"Bearer {token}"))
        return tuple(headers)

    async def _one_http_request(
        self,
        method: str,
        url: str,
        headers: tuple[tuple[str, str], ...],
        body: bytes | None,
        *,
        request_id: int | None,
    ) -> _HTTPResponse:
        parsed = _validated_url(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        task = asyncio.current_task()
        if task is not None:
            self._http_tasks[task] = self._http_tasks.get(task, 0) + 1
        writer: asyncio.StreamWriter | None = None
        try:
            # HTTP/1.1 is parsed directly over asyncio streams. Unlike to_thread
            # around a blocking stdlib client, every socket wait belongs to this
            # cancellable task and aclose can abort its live transport immediately.
            reader, writer = await asyncio.open_connection(
                parsed.hostname,
                port,
                ssl=self._ssl_context if parsed.scheme == "https" else None,
                server_hostname=parsed.hostname if parsed.scheme == "https" else None,
                limit=max(
                    self.limits.max_http_header_bytes,
                    self.limits.max_message_bytes,
                )
                + 1,
            )
            if not self._http_open:
                raise MCPError("disconnected", "MCP server is disconnected")
            self._http_writers.add(writer)
            head = [f"{method} {_request_target(parsed)} HTTP/1.1\r\n"]
            head.extend(f"{name}: {value}\r\n" for name, value in headers)
            request = "".join(head).encode("ascii") + b"\r\n" + (body or b"")
            writer.write(request)
            await writer.drain()
            return await self._read_http_response(reader, url, request_id)
        except asyncio.CancelledError:
            if not self._http_open:
                raise MCPError("disconnected", "MCP server is disconnected") from None
            raise
        except MCPError:
            # aclose owns the transport shutdown, not the task that happened to
            # call us.  Its abort can surface as a parser error first; once the
            # client is closed, expose the stable disconnected outcome instead.
            if not self._http_open:
                raise MCPError("disconnected", "MCP server is disconnected") from None
            raise
        except (OSError, UnicodeError, ValueError, asyncio.IncompleteReadError):
            if not self._http_open:
                raise MCPError("disconnected", "MCP server is disconnected") from None
            raise MCPError("protocol_error", "Malformed MCP HTTP response") from None
        finally:
            if writer is not None:
                self._http_writers.discard(writer)
                writer.transport.abort()
                with contextlib.suppress(OSError, RuntimeError):
                    await writer.wait_closed()
            if task is not None:
                remaining = self._http_tasks.get(task, 1) - 1
                if remaining:
                    self._http_tasks[task] = remaining
                else:
                    self._http_tasks.pop(task, None)

    async def _read_http_response(
        self,
        reader: asyncio.StreamReader,
        url: str,
        request_id: int | None,
    ) -> _HTTPResponse:
        try:
            raw_head = await reader.readuntil(b"\r\n\r\n")
        except asyncio.LimitOverrunError:
            raise MCPError(
                "message_too_large", "MCP HTTP headers exceed byte limit"
            ) from None
        except asyncio.IncompleteReadError:
            raise MCPError("protocol_error", "Malformed MCP HTTP response") from None
        if len(raw_head) > self.limits.max_http_header_bytes:
            raise MCPError("message_too_large", "MCP HTTP headers exceed byte limit")
        lines = raw_head[:-4].split(b"\r\n")
        if not lines or not re.fullmatch(
            rb"HTTP/1\.[01] [0-9]{3}(?: [\x20-\x7e]*)?", lines[0]
        ):
            raise MCPError("protocol_error", "Malformed MCP HTTP response")
        status = int(lines[0].split(b" ", 2)[1])
        if len(lines) - 1 > self.limits.max_http_headers:
            raise MCPError("message_too_large", "MCP HTTP headers exceed count limit")
        parsed_headers: list[tuple[str, str]] = []
        for line in lines[1:]:
            if b":" not in line:
                raise MCPError("protocol_error", "Malformed MCP HTTP response")
            name, value = line.split(b":", 1)
            if not _HEADER_NAME.fullmatch(name) or any(
                byte < 0x20 and byte != 0x09 for byte in value
            ):
                raise MCPError("protocol_error", "Malformed MCP HTTP response")
            parsed_headers.append(
                (name.decode("ascii").lower(), value.strip().decode("latin-1"))
            )
        headers_tuple = tuple(parsed_headers)
        provisional = _HTTPResponse(status, headers_tuple, b"", url)
        if status in _ALL_REDIRECTS or status in {401, 403, 404}:
            body = await self._read_bounded_http_body(reader, provisional)
            return _HTTPResponse(status, headers_tuple, body, url)
        if request_id is not None and _media_type(provisional) == "text/event-stream":
            message = await self._read_sse_response(reader, provisional, request_id)
            return _HTTPResponse(status, headers_tuple, _json(message).encode(), url)
        body = await self._read_bounded_http_body(reader, provisional)
        return _HTTPResponse(status, headers_tuple, body, url)

    async def _read_bounded_http_body(
        self, reader: asyncio.StreamReader, response: _HTTPResponse
    ) -> bytes:
        body = bytearray()
        async for chunk in self._http_body_chunks(reader, response):
            body.extend(chunk)
            if len(body) > self.limits.max_message_bytes:
                raise MCPError("message_too_large", "MCP HTTP body exceeds byte limit")
        return bytes(body)

    async def _http_body_chunks(
        self, reader: asyncio.StreamReader, response: _HTTPResponse
    ):
        transfer_values = response.values("transfer-encoding")
        length_values = response.values("content-length")
        if transfer_values and length_values:
            raise MCPError("protocol_error", "Malformed MCP HTTP response")
        if transfer_values:
            if len(transfer_values) != 1 or transfer_values[0].lower() != "chunked":
                raise MCPError("protocol_error", "Unsupported MCP HTTP encoding")
            trailer_bytes = 0
            trailer_count = 0
            while True:
                line = await reader.readline()
                if (
                    not line
                    or len(line) > self.limits.max_http_header_bytes
                    or not line.endswith(b"\r\n")
                ):
                    raise MCPError("protocol_error", "Malformed MCP HTTP response")
                try:
                    size = int(line[:-2].split(b";", 1)[0], 16)
                except ValueError:
                    raise MCPError(
                        "protocol_error", "Malformed MCP HTTP response"
                    ) from None
                if size < 0 or size > self.limits.max_message_bytes:
                    raise MCPError(
                        "message_too_large", "MCP HTTP body exceeds byte limit"
                    )
                if size == 0:
                    while True:
                        trailer = await reader.readline()
                        trailer_bytes += len(trailer)
                        if (
                            not trailer
                            or trailer_bytes > self.limits.max_http_header_bytes
                            or not trailer.endswith(b"\r\n")
                        ):
                            raise MCPError(
                                "protocol_error", "Malformed MCP HTTP response"
                            )
                        if trailer == b"\r\n":
                            return
                        trailer_count += 1
                        if trailer_count > self.limits.max_http_headers:
                            raise MCPError(
                                "message_too_large",
                                "MCP HTTP headers exceed count limit",
                            )
                try:
                    data = await reader.readexactly(size)
                    ending = await reader.readexactly(2)
                except asyncio.IncompleteReadError:
                    raise MCPError(
                        "protocol_error", "Malformed MCP HTTP response"
                    ) from None
                if ending != b"\r\n":
                    raise MCPError("protocol_error", "Malformed MCP HTTP response")
                yield data
        elif length_values:
            if len(set(length_values)) != 1:
                raise MCPError("protocol_error", "Malformed MCP HTTP response")
            try:
                length = int(length_values[0])
            except ValueError:
                raise MCPError(
                    "protocol_error", "Malformed MCP HTTP response"
                ) from None
            if length < 0:
                raise MCPError("protocol_error", "Malformed MCP HTTP response")
            if length > self.limits.max_message_bytes:
                raise MCPError("message_too_large", "MCP HTTP body exceeds byte limit")
            try:
                if length:
                    yield await reader.readexactly(length)
            except asyncio.IncompleteReadError:
                raise MCPError(
                    "protocol_error", "Malformed MCP HTTP response"
                ) from None
            # Connection: close gives an unambiguous end marker. Reading through it
            # catches a server that understates Content-Length instead of silently
            # leaving attacker-controlled bytes buffered on an abandoned socket.
            extra = await reader.read(self.limits.max_message_bytes + 1)
            if extra:
                raise MCPError("protocol_error", "MCP HTTP Content-Length is incorrect")
        else:
            while True:
                chunk = await reader.read(
                    min(65_536, self.limits.max_message_bytes + 1)
                )
                if not chunk:
                    return
                yield chunk

    async def _read_sse_response(
        self,
        reader: asyncio.StreamReader,
        response: _HTTPResponse,
        request_id: int,
    ) -> dict[str, Any]:
        buffered = bytearray()
        data_lines: list[bytes] = []
        total = 0
        messages = 0
        async for chunk in self._http_body_chunks(reader, response):
            total += len(chunk)
            if total > self.limits.max_message_bytes:
                raise MCPError("message_too_large", "MCP SSE stream exceeds byte limit")
            buffered.extend(chunk)
            while b"\n" in buffered:
                raw_line, _, remainder = buffered.partition(b"\n")
                buffered = bytearray(remainder)
                line = raw_line[:-1] if raw_line.endswith(b"\r") else raw_line
                if len(line) > self.limits.max_message_bytes:
                    raise MCPError(
                        "message_too_large", "MCP SSE event exceeds byte limit"
                    )
                if not line:
                    if not data_lines:
                        continue
                    messages += 1
                    if messages > self.limits.max_interleaved_messages + 1:
                        raise MCPError(
                            "protocol_error", "Too many interleaved MCP messages"
                        )
                    payload = b"\n".join(data_lines)
                    data_lines.clear()
                    try:
                        message = json.loads(
                            payload.decode("utf-8"), parse_constant=_invalid_constant
                        )
                    except (UnicodeError, ValueError, RecursionError):
                        raise MCPError(
                            "protocol_error", "Invalid MCP SSE event"
                        ) from None
                    result = await self._http_message_result(message, request_id)
                    if result is not None:
                        return message
                elif line.startswith(b":"):
                    # SSE comments carry no MCP message but still consume byte budget.
                    continue
                else:
                    field, separator, value = line.partition(b":")
                    if separator and value.startswith(b" "):
                        value = value[1:]
                    if field == b"data":
                        data_lines.append(value)
                    # Event IDs are deliberately neither retained nor replayed: a
                    # broken stream is terminal under driftlock's no-resume policy.
                    # SSE requires unknown fields to be ignored. All their bytes
                    # still count toward the stream and line caps above.
        if buffered or data_lines:
            raise MCPError("protocol_error", "Incomplete MCP SSE event")
        raise MCPError("disconnected", "MCP SSE stream ended before response")

    async def _http_response_result(
        self, response: _HTTPResponse, request_id: int
    ) -> dict[str, Any]:
        if _media_type(response) not in {"application/json", "text/event-stream"}:
            raise MCPError("protocol_error", "Invalid MCP HTTP content type")
        try:
            message = json.loads(
                response.body.decode("utf-8"), parse_constant=_invalid_constant
            )
        except (UnicodeError, ValueError, RecursionError):
            raise MCPError("protocol_error", "Invalid MCP HTTP JSON response") from None
        result = await self._http_message_result(message, request_id)
        if result is None:
            raise MCPError("protocol_error", "MCP HTTP response omitted request result")
        return result

    async def _http_message_result(
        self, message: Any, request_id: int
    ) -> dict[str, Any] | None:
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
                    raise MCPError("protocol_error", "Invalid MCP server request id")
                reply: dict[str, Any] = {"id": message["id"]}
                if message["method"] == "ping":
                    reply["result"] = {}
                else:
                    reply["error"] = {
                        "code": -32601,
                        "message": "Client method not supported",
                    }
                await self._http_send(reply)
            return None
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
            # Unlike the legacy stdio behavior, HTTP never places server text in an
            # exception. A resource server has seen the bearer token and could echo
            # it; keeping the diagnostic fixed closes that exfiltration channel.
            raise MCPError("server_error", "MCP HTTP server returned a JSON-RPC error")
        if not isinstance(message["result"], dict):
            raise MCPError("protocol_error", "MCP result must be an object")
        return message["result"]

    def _update_session(self, response: _HTTPResponse, *, initialize: bool) -> None:
        values = response.values("mcp-session-id")
        if len(values) > 1:
            raise MCPError("protocol_error", "Invalid MCP HTTP session header")
        if initialize:
            if not values:
                return
            session_id = values[0]
            if (
                not session_id
                or len(session_id.encode("latin-1")) > self.limits.max_session_id_bytes
                or any(
                    ord(character) < 0x21 or ord(character) > 0x7E
                    for character in session_id
                )
            ):
                raise MCPError("protocol_error", "Invalid MCP HTTP session header")
            self._session_id = session_id
            self._session_origin = _origin(response.url)
        elif values and values[0] != self._session_id:
            raise MCPError("session_lost", "MCP HTTP session changed unexpectedly")

    async def _authorization_challenge(
        self, response: _HTTPResponse, token: str | None
    ) -> MCPAuthorizationResult:
        challenge_url, challenge_scopes, challenge_limited = _parse_bearer_challenge(
            response.values("www-authenticate"), token, self.limits
        )
        metadata_urls = (
            (challenge_url,)
            if challenge_url is not None
            else _metadata_urls(self.config.url)
        )
        metadata_url: str | None = None
        servers: tuple[str, ...] = ()
        metadata_scopes: tuple[str, ...] = ()
        discovery_status = (
            MCPAuthorizationDiscoveryStatus.LIMIT_EXCEEDED
            if challenge_limited
            else MCPAuthorizationDiscoveryStatus.UNAVAILABLE
        )
        # Discovery is informative and never retries the rejected MCP request. Each
        # candidate receives a fractional budget so a dead metadata host cannot
        # consume the request's entire timeout and erase the typed auth outcome.
        discovery_timeout = min(
            5.0, self.limits.request_timeout_seconds / (len(metadata_urls) + 1)
        )
        for candidate in metadata_urls:
            try:
                async with asyncio.timeout(discovery_timeout):
                    discovered = await self._http_round_trip(
                        "GET",
                        candidate,
                        None,
                        token=None,
                        forbidden_credential=token,
                        request_id=None,
                        metadata=True,
                    )
                if (
                    discovered.status != 200
                    or _media_type(discovered) != "application/json"
                ):
                    continue
                if _contains_credential(discovered.url, token):
                    discovery_status = MCPAuthorizationDiscoveryStatus.MALFORMED
                    continue
                parsed_servers, parsed_scopes, parsed_status = (
                    self._parse_resource_metadata(discovered.body, token)
                )
                if parsed_status is MCPAuthorizationDiscoveryStatus.DISCOVERED:
                    metadata_url = discovered.url
                    servers = parsed_servers
                    metadata_scopes = parsed_scopes
                    discovery_status = (
                        MCPAuthorizationDiscoveryStatus.LIMIT_EXCEEDED
                        if challenge_limited
                        else parsed_status
                    )
                    break
                discovery_status = parsed_status
            except asyncio.CancelledError:
                raise
            except MCPError as exc:
                if exc.status in {"message_too_large", "catalog_too_large"}:
                    discovery_status = MCPAuthorizationDiscoveryStatus.LIMIT_EXCEEDED
                elif exc.status in {"credential_invalid", "redirect_rejected"}:
                    discovery_status = MCPAuthorizationDiscoveryStatus.MALFORMED
                continue
            except TimeoutError:
                continue
        status = (
            MCPAuthorizationStatus.MISSING_CREDENTIAL
            if token is None
            else MCPAuthorizationStatus.REJECTED
        )
        return MCPAuthorizationResult(
            status=status,
            authorization_servers=servers,
            scopes=() if challenge_limited else (challenge_scopes or metadata_scopes),
            resource_metadata_url=metadata_url,
            discovery_status=discovery_status,
        )

    def _parse_resource_metadata(
        self, body: bytes, token: str | None
    ) -> tuple[tuple[str, ...], tuple[str, ...], MCPAuthorizationDiscoveryStatus]:
        try:
            value = json.loads(body.decode("utf-8"), parse_constant=_invalid_constant)
        except (UnicodeError, ValueError, RecursionError):
            return (), (), MCPAuthorizationDiscoveryStatus.MALFORMED
        if not isinstance(value, dict):
            return (), (), MCPAuthorizationDiscoveryStatus.MALFORMED
        servers_value = value.get("authorization_servers", [])
        scopes_value = value.get("scopes_supported", [])
        if not isinstance(servers_value, list) or not isinstance(scopes_value, list):
            return (), (), MCPAuthorizationDiscoveryStatus.MALFORMED
        if (
            len(servers_value) > self.limits.max_authorization_servers
            or len(scopes_value) > self.limits.max_authorization_scopes
        ):
            return (), (), MCPAuthorizationDiscoveryStatus.LIMIT_EXCEEDED
        servers: list[str] = []
        for server in servers_value:
            if not isinstance(server, str) or _contains_credential(server, token):
                return (), (), MCPAuthorizationDiscoveryStatus.MALFORMED
            try:
                parsed = _validated_url(server)
            except ValueError:
                return (), (), MCPAuthorizationDiscoveryStatus.MALFORMED
            if parsed.query:
                return (), (), MCPAuthorizationDiscoveryStatus.MALFORMED
            servers.append(server)
        scopes: list[str] = []
        for scope in scopes_value:
            if (
                not isinstance(scope, str)
                or not scope
                or any(
                    ord(character) < 0x21 or ord(character) > 0x7E
                    for character in scope
                )
                or _contains_credential(scope, token)
            ):
                return (), (), MCPAuthorizationDiscoveryStatus.MALFORMED
            if len(scope) > self.limits.max_authorization_scope_characters:
                return (), (), MCPAuthorizationDiscoveryStatus.LIMIT_EXCEEDED
            scopes.append(scope)
        return (
            tuple(servers),
            tuple(scopes),
            MCPAuthorizationDiscoveryStatus.DISCOVERED,
        )

    async def _send(self, message: dict) -> None:
        if self.config.url is not None:
            await self._http_send(message)
            return
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
                    message = {"id": request_id, "method": method, "params": params}
                    if self.config.url is not None:
                        return await self._http_rpc_request(message, request_id)
                    await self._send(message)
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
                if exc.status not in {
                    "server_error",
                    "rejected",
                    "request_too_large",
                    "missing_credential",
                    "authorization_rejected",
                }:
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
        """Close HTTP sockets, or close and reap the direct stdio child."""
        self._ready = False
        if self.config.url is not None:
            self._http_open = False
            writers = tuple(self._http_writers)
            self._http_writers.clear()
            for writer in writers:
                writer.transport.abort()
            for writer in writers:
                with contextlib.suppress(OSError, RuntimeError):
                    await writer.wait_closed()
            self._session_id = None
            self._session_origin = None
            return
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
