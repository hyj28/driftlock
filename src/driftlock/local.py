"""Subprocess-backed local implementations for laptop agent development."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import difflib
import hashlib
import os
import shutil
import signal
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from driftlock.lhtb import WorkspaceDelta, WorkspaceSnapshot

_PROCESS_CLEANUP_TIMEOUT_SEC = 1.0


@dataclass(frozen=True, slots=True)
class LocalExecResult:
    """Captured result of a local subprocess command."""

    return_code: int
    stdout: str
    stderr: str


class LocalEnvironment:
    """Execute the remote-environment protocol within a configured local root."""

    def __init__(
        self,
        root: Path | str,
        *,
        default_timeout_sec: int = 60,
        max_output_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        resolved = Path(root).resolve()
        if not resolved.is_dir():
            raise ValueError("local environment root must be an existing directory")
        if default_timeout_sec <= 0:
            raise ValueError("default_timeout_sec must be positive")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.root = resolved
        self.default_timeout_sec = default_timeout_sec
        self.max_output_bytes = max_output_bytes
        self._runtime = tempfile.TemporaryDirectory(prefix="driftlock-local-")
        self._runtime_root = Path(self._runtime.name)
        self._home = self._runtime_root / "home"
        self._temporary = self._runtime_root / "tmp"
        self._home.mkdir()
        self._temporary.mkdir()

    def close(self) -> None:
        """Remove the environment-owned runtime directory."""

        self._runtime.cleanup()

    def __enter__(self) -> LocalEnvironment:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if not isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty string")
        if user is not None:
            raise ValueError("LocalEnvironment does not permit user switching")
        timeout = self.default_timeout_sec if timeout_sec is None else timeout_sec
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout_sec must be a positive integer or None")
        if not self._runtime_root.is_dir():
            raise RuntimeError("LocalEnvironment is closed")
        environment = {
            "HOME": str(self._home),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "TMPDIR": str(self._temporary),
        }
        return await self._exec_process(command, timeout, environment)

    async def _exec_process(
        self, command: str, timeout: int, environment: dict[str, str]
    ) -> LocalExecResult:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.root,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("subprocess output pipes were not created")
        stdout_task = asyncio.create_task(
            _read_capped(process.stdout, self.max_output_bytes)
        )
        stderr_task = asyncio.create_task(
            _read_capped(process.stderr, self.max_output_bytes)
        )
        wait_task = asyncio.create_task(process.wait())
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=timeout)
            return_code = process.returncode if process.returncode is not None else 1
        except asyncio.CancelledError:
            # Delegation and other outer deadlines may cancel this coroutine
            # before its own command timeout. Stop the group before enumerating
            # descendants, closing the race where a process forks and detaches
            # between the snapshot and the kill.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGSTOP)
            descendants = await asyncio.to_thread(_descendant_process_ids, process.pid)
            _kill_processes(descendants)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            _close_process_pipes(process)
            stdout_task.cancel()
            stderr_task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            try:
                await asyncio.wait_for(
                    asyncio.shield(wait_task), timeout=_PROCESS_CLEANUP_TIMEOUT_SEC
                )
            except TimeoutError:
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)
            raise
        except TimeoutError:
            timed_out = True
            known_return_code = process.returncode
            process_group_terminated = False
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGSTOP)
            descendants = await asyncio.to_thread(_descendant_process_ids, process.pid)
            _kill_processes(descendants)
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process_group_terminated = True
            except ProcessLookupError:
                pass
            _close_process_pipes(process)
            stdout_task.cancel()
            stderr_task.cancel()
            return_code = (
                known_return_code
                if known_return_code is not None and process_group_terminated
                else 124
            )
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        if timed_out:
            try:
                await asyncio.wait_for(
                    asyncio.shield(wait_task),
                    timeout=_PROCESS_CLEANUP_TIMEOUT_SEC,
                )
            except TimeoutError:
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)
        if timed_out and return_code == 124:
            stderr += f"\ncommand timed out after {timeout} seconds".encode()
        return LocalExecResult(
            return_code=return_code,
            stdout=_decode_capped(stdout, self.max_output_bytes),
            stderr=_decode_capped(stderr, self.max_output_bytes),
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"upload source is not a file: {source}")
        target = self._workspace_path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, source, target)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = self._workspace_path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"download source is not a file: {source}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, source, target)

    def _workspace_path(self, value: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("workspace path must be a non-empty string")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise ValueError("path resolves outside the local environment root")
        return resolved


class LocalWorkspaceDeltaObserver:
    """Observe per-step file changes in a local environment root."""

    def __init__(self, root: Path | str) -> None:
        resolved = Path(root).resolve()
        if not resolved.is_dir():
            raise ValueError("workspace root must be an existing directory")
        self.root = resolved

    async def canonical_workspace(self) -> str:
        return self.root.as_posix()

    async def snapshot(self) -> WorkspaceSnapshot:
        files: dict[str, str] = {}
        rendered: list[str] = []
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            if ".git" in relative.parts or not (path.is_file() or path.is_symlink()):
                continue
            name = relative.as_posix()
            if path.is_symlink():
                content = f"symlink:{os.readlink(path)}"
                data = content.encode()
            else:
                data = await asyncio.to_thread(path.read_bytes)
                try:
                    content = data.decode("utf-8")
                except UnicodeDecodeError:
                    content = f"binary sha256:{hashlib.sha256(data).hexdigest()}"
            files[name] = hashlib.sha256(data).hexdigest()
            rendered.extend((f"--- {name}", content))
        return WorkspaceSnapshot(files=files, git_view="\n".join(rendered))

    def compare(
        self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> WorkspaceDelta:
        paths = sorted(set(before.files) | set(after.files))
        changed_paths = tuple(
            path for path in paths if before.files.get(path) != after.files.get(path)
        )
        diff = "\n".join(
            difflib.unified_diff(
                before.git_view.splitlines(),
                after.git_view.splitlines(),
                fromfile="workspace-before",
                tofile="workspace-after",
                lineterm="",
            )
        )
        return WorkspaceDelta(changed_paths=changed_paths, diff=diff)


def _decode_capped(value: bytes, limit: int) -> str:
    if len(value) <= limit:
        return value.decode("utf-8", errors="replace")
    marker = f"\n[process output truncated after {limit} bytes]".encode()
    retained = max(0, limit - len(marker))
    return (value[:retained] + marker).decode("utf-8", errors="replace")


def _descendant_process_ids(root_pid: int) -> tuple[int, ...]:
    """Snapshot descendants deepest-first, including children in new sessions."""

    children: dict[int, list[int]] = {}
    for pid, parent in _process_parent_pairs():
        children.setdefault(parent, []).append(pid)
    discovered: list[tuple[int, int]] = []
    pending = [(root_pid, 0)]
    seen = {root_pid}
    while pending:
        parent, depth = pending.pop()
        for pid in children.get(parent, ()):
            if pid in seen:
                continue
            seen.add(pid)
            discovered.append((depth + 1, pid))
            pending.append((pid, depth + 1))
    discovered.sort(reverse=True)
    return tuple(pid for _depth, pid in discovered)


def _process_parent_pairs() -> tuple[tuple[int, int], ...]:
    if sys.platform == "darwin":
        return _darwin_process_parent_pairs()
    proc = Path("/proc")
    if not proc.is_dir():
        return ()
    pairs: list[tuple[int, int]] = []
    for stat_path in proc.glob("[0-9]*/stat"):
        try:
            pid = int(stat_path.parent.name)
            fields = stat_path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            parent = int(fields[1])
        except (IndexError, OSError, ValueError):
            continue
        pairs.append((pid, parent))
    return tuple(pairs)


def _darwin_process_parent_pairs() -> tuple[tuple[int, int], ...]:
    """Use libproc because sandboxed macOS processes may not execute ``ps``."""

    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        libproc.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libproc.proc_listallpids.restype = ctypes.c_int
        capacity = max(1, libproc.proc_listallpids(None, 0) * 2)
        pid_buffer = (ctypes.c_int * capacity)()
        count = libproc.proc_listallpids(pid_buffer, ctypes.sizeof(pid_buffer))
        libproc.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        libproc.proc_pidinfo.restype = ctypes.c_int
    except (AttributeError, OSError):
        return ()
    pairs: list[tuple[int, int]] = []
    for pid in pid_buffer[: max(0, count)]:
        info = ctypes.create_string_buffer(256)
        try:
            copied = libproc.proc_pidinfo(pid, 3, 0, info, len(info))
        except (OSError, ValueError):
            continue
        if copied < 20:
            continue
        recorded_pid = struct.unpack_from("=I", info.raw, 12)[0]
        parent = struct.unpack_from("=I", info.raw, 16)[0]
        if recorded_pid == pid:
            pairs.append((pid, parent))
    return tuple(pairs)


def _kill_processes(process_ids: tuple[int, ...]) -> None:
    for pid in process_ids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


async def _read_capped(stream: asyncio.StreamReader, limit: int) -> bytes:
    retained = bytearray()
    try:
        while chunk := await stream.read(64 * 1024):
            if len(retained) <= limit:
                retained.extend(chunk[: limit + 1 - len(retained)])
    except asyncio.CancelledError:
        pass
    return bytes(retained)


def _close_process_pipes(process: asyncio.subprocess.Process) -> None:
    transport = getattr(process, "_transport", None)
    if transport is None:
        return
    for descriptor in (1, 2):
        pipe = transport.get_pipe_transport(descriptor)
        if pipe is not None:
            pipe.close()
