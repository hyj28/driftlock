"""Host-side checkpoints for a workspace accessed through a remote environment."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import uuid
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from driftlock.checkpoints import SnapshotIntegrityError
from driftlock.models import Checkpoint


class RemoteCheckpointError(RuntimeError):
    """Raised when a remote archive or restore command fails."""


class ExecResultLike(Protocol):
    return_code: int
    stdout: str
    stderr: str


class RemoteEnvironment(Protocol):
    """The subset of Harbor's environment API needed by the checkpoint store."""

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResultLike: ...

    async def upload_file(self, source_path: Path | str, target_path: str) -> Any: ...

    async def download_file(self, source_path: str, target_path: Path | str) -> Any: ...


class RemoteArchiveCheckpointStore:
    """Persist remote workspace snapshots in a host directory.

    The implementation uses only ``exec``, ``upload_file``, and ``download_file``,
    making it compatible with Harbor Docker, Daytona, E2B, Modal, and similar
    environments. Restores replace the *contents* of the workspace while preserving
    its directory inode, so a paused shell whose cwd is the workspace stays usable.
    Keep ``store_dir`` outside any directory mounted into the agent environment.
    """

    def __init__(
        self,
        environment: RemoteEnvironment,
        remote_workspace: str,
        store_dir: Path | str,
        *,
        remote_tmp_dir: str = "/tmp",
        user: str | int | None = None,
        timeout_sec: int = 300,
    ) -> None:
        self.environment = environment
        self.remote_workspace = _validated_remote_path(
            remote_workspace, name="remote_workspace", allow_root=False
        )
        self.remote_tmp_dir = _validated_remote_path(
            remote_tmp_dir, name="remote_tmp_dir", allow_root=False
        )
        if _is_relative_to(self.remote_tmp_dir, self.remote_workspace):
            raise ValueError("remote_tmp_dir must be outside remote_workspace")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        self.store_dir = Path(store_dir).expanduser().resolve()
        self.user = user
        self.timeout_sec = timeout_sec
        (self.store_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    @property
    def checkpoints_dir(self) -> Path:
        return self.store_dir / "checkpoints"

    async def create(
        self,
        state: Mapping[str, Any],
        *,
        step: int,
        parent_id: str | None = None,
        label: str | None = None,
    ) -> Checkpoint:
        if step < 0:
            raise ValueError("step cannot be negative")
        state_json = json.dumps(state, sort_keys=True, separators=(",", ":"))
        checkpoint_id = uuid.uuid4().hex
        temporary = self.checkpoints_dir / f".tmp-{checkpoint_id}"
        final = self.checkpoints_dir / checkpoint_id
        local_archive = temporary / "workspace.tar.gz"
        remote_archive = self._remote_temp_path(checkpoint_id, "snapshot.tar.gz")
        temporary.mkdir()
        try:
            command = " ".join(
                [
                    "tar -czf",
                    shlex.quote(remote_archive),
                    "-C",
                    shlex.quote(self.remote_workspace),
                    ".",
                ]
            )
            await self._checked_exec(command, operation="create remote archive")
            await self.environment.download_file(remote_archive, local_archive)
            if not local_archive.is_file():
                raise RemoteCheckpointError(
                    "environment did not download the checkpoint archive"
                )
            created_at = datetime.now(UTC)
            (temporary / "state.json").write_text(state_json, encoding="utf-8")
            digest = _archive_digest(local_archive, state_json)
            manifest = {
                "checkpoint_id": checkpoint_id,
                "step": step,
                "created_at": created_at.isoformat(),
                "digest": digest,
                "parent_id": parent_id,
                "label": label,
                "remote_workspace": self.remote_workspace,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            temporary.rename(final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            await self._best_effort_remove(remote_archive)
        return Checkpoint(
            checkpoint_id=checkpoint_id,
            step=step,
            created_at=created_at,
            digest=digest,
            path=final,
            parent_id=parent_id,
            label=label,
        )

    async def restore(self, checkpoint: Checkpoint) -> dict[str, Any]:
        checkpoint_dir = checkpoint.path.resolve()
        if checkpoint_dir.parent != self.checkpoints_dir.resolve():
            raise ValueError("checkpoint does not belong to this store")
        archive = checkpoint_dir / "workspace.tar.gz"
        state_path = checkpoint_dir / "state.json"
        if not archive.is_file():
            raise SnapshotIntegrityError("checkpoint workspace archive is missing")
        try:
            state_json = state_path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise SnapshotIntegrityError("checkpoint agent state is missing") from error
        if _archive_digest(archive, state_json) != checkpoint.digest:
            raise SnapshotIntegrityError(
                f"checkpoint {checkpoint.checkpoint_id} failed integrity verification"
            )
        try:
            state = json.loads(state_json)
        except json.JSONDecodeError as error:
            raise SnapshotIntegrityError(
                "checkpoint agent state is invalid JSON"
            ) from error
        if not isinstance(state, dict):
            raise SnapshotIntegrityError("checkpoint agent state must be a JSON object")

        restore_id = uuid.uuid4().hex
        remote_archive = self._remote_temp_path(restore_id, "restore.tar.gz")
        remote_backup = self._remote_temp_path(restore_id, "backup.tar.gz")
        remote_staging = self._remote_temp_path(restore_id, "staging")
        try:
            await self.environment.upload_file(archive, remote_archive)
            await self._checked_exec(
                self._restore_script(
                    archive=remote_archive,
                    backup=remote_backup,
                    staging=remote_staging,
                ),
                operation="restore remote checkpoint",
            )
        finally:
            await self._best_effort_remove(
                remote_archive, remote_backup, remote_staging
            )
        return state

    def _remote_temp_path(self, identifier: str, suffix: str) -> str:
        return str(
            PurePosixPath(self.remote_tmp_dir) / f"driftlock-{identifier}-{suffix}"
        )

    def _restore_script(self, *, archive: str, backup: str, staging: str) -> str:
        workspace = shlex.quote(self.remote_workspace)
        archive_q = shlex.quote(archive)
        backup_q = shlex.quote(backup)
        staging_q = shlex.quote(staging)
        clear_workspace = (
            f"find {workspace} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +"
        )
        script = f"""set -eu
rm -rf -- {staging_q}
rm -f -- {backup_q}
mkdir -p -- {staging_q}
tar -xzf {archive_q} -C {staging_q}
tar -czf {backup_q} -C {workspace} .
if {clear_workspace} && cp -a {staging_q}/. {workspace}/; then
    exit 0
fi
{clear_workspace}
tar -xzf {backup_q} -C {workspace}
exit 1
"""
        return "sh -ceu " + shlex.quote(script)

    async def _checked_exec(self, command: str, *, operation: str) -> ExecResultLike:
        result = await self.environment.exec(
            command,
            timeout_sec=self.timeout_sec,
            user=self.user,
        )
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "no output").strip()
            raise RemoteCheckpointError(
                f"{operation} failed with exit code {result.return_code}: {detail}"
            )
        return result

    async def _best_effort_remove(self, *paths: str) -> None:
        if not paths:
            return
        command = "rm -rf -- " + " ".join(shlex.quote(path) for path in paths)
        with suppress(Exception):
            await self.environment.exec(
                command,
                timeout_sec=min(self.timeout_sec, 30),
                user=self.user,
            )


def _validated_remote_path(value: str, *, name: str, allow_root: bool) -> str:
    if not value or "\0" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be an absolute normalized POSIX path")
    normalized = str(path)
    if normalized == "/" and not allow_root:
        raise ValueError(f"{name} cannot be the filesystem root")
    return normalized


def _is_relative_to(path: str, parent: str) -> bool:
    path_parts = PurePosixPath(path).parts
    parent_parts = PurePosixPath(parent).parts
    return path_parts[: len(parent_parts)] == parent_parts


def _archive_digest(archive: Path, state_json: str) -> str:
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    digest.update(b"\0state\0")
    digest.update(state_json.encode())
    return digest.hexdigest()
