"""Subprocess-backed local implementations for laptop agent development."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import os
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path

from driftlock.lhtb import WorkspaceDelta, WorkspaceSnapshot


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
        environment = {
            "HOME": str(self.root),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "TMPDIR": str(self.root),
        }
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
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
            return_code = process.returncode if process.returncode is not None else 1
        except TimeoutError:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            return_code = 124
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        if timed_out:
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


async def _read_capped(stream: asyncio.StreamReader, limit: int) -> bytes:
    retained = bytearray()
    while chunk := await stream.read(64 * 1024):
        if len(retained) <= limit:
            retained.extend(chunk[: limit + 1 - len(retained)])
    return bytes(retained)
