"""Host-side checkpoints for a workspace accessed through a remote environment."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import uuid
import warnings
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from driftlock.checkpoints import SnapshotIntegrityError
from driftlock.models import Checkpoint


class RemoteCheckpointError(RuntimeError):
    """Raised when a remote archive or restore command fails."""


class ExecResultLike(Protocol):
    return_code: int
    stdout: str | None
    stderr: str | None


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

    The implementation uses only ``exec``, ``upload_file``, and ``download_file``
    plus standard Linux userland tools. It works with POSIX Harbor Docker, Daytona,
    E2B, Modal, and similar environments. Keep ``store_dir`` outside any directory
    mounted into the agent environment.
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
        before_restore: Callable[[str], Awaitable[None]] | None = None,
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
        self.before_restore = before_restore
        self._canonical_workspace: str | None = None
        self._canonical_tmp_dir: str | None = None
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
        await self._ensure_remote_paths_are_safe()
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
                    shlex.quote(self._workspace_path),
                    ".",
                ]
            )
            unstable_paths = await self._create_remote_archive(command)
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
            if unstable_paths:
                manifest["unstable_paths"] = list(unstable_paths)
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
            unstable_paths=unstable_paths,
        )

    async def restore(self, checkpoint: Checkpoint) -> dict[str, Any]:
        await self._ensure_remote_paths_are_safe()
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
        remote_host_backup = self._remote_temp_path(restore_id, "host-backup.tar.gz")
        remote_staging = self._remote_temp_path(restore_id, "staging")
        remote_recovery_staging = self._remote_temp_path(restore_id, "recovery-staging")
        recovery_dir = self.store_dir / "recovery"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        local_recovery = recovery_dir / f"restore-{restore_id}.tar.gz"

        try:
            await self.environment.upload_file(archive, remote_archive)
            prepare_result = await self._checked_exec(
                self._prepare_restore_script(
                    archive=remote_archive,
                    backup=remote_backup,
                    staging=remote_staging,
                ),
                operation="prepare remote checkpoint restore",
            )
            remote_backup_digest = _parse_sha256(prepare_result.stdout)
            await self.environment.download_file(remote_backup, local_recovery)
            if not local_recovery.is_file():
                raise RemoteCheckpointError(
                    "environment did not download the pre-restore recovery archive"
                )
            if _file_digest(local_recovery) != remote_backup_digest:
                raise RemoteCheckpointError(
                    "downloaded recovery archive does not match the remote SHA-256"
                )
        except BaseException:
            await self._best_effort_remove(
                remote_archive, remote_backup, remote_staging
            )
            raise

        try:
            if self.before_restore is not None:
                await self.before_restore(self._workspace_path)
        except BaseException:
            await self._best_effort_remove(
                remote_archive,
                remote_backup,
                remote_staging,
                remote_host_backup,
                remote_recovery_staging,
            )
            local_recovery.unlink(missing_ok=True)
            raise

        try:
            await self._checked_exec(
                self._apply_restore_script(staging=remote_staging),
                operation="apply remote checkpoint restore",
            )
        except Exception as error:
            recovery_detail = await self._attempt_recovery(
                local_recovery=local_recovery,
                remote_backup=remote_backup,
                remote_backup_digest=remote_backup_digest,
                remote_host_backup=remote_host_backup,
                recovery_staging=remote_recovery_staging,
            )
            await self._best_effort_remove(
                remote_archive,
                remote_staging,
                remote_recovery_staging,
            )
            raise RemoteCheckpointError(
                f"remote restore failed: {error}; {recovery_detail}; "
                f"host recovery archive retained at {local_recovery}; "
                "remote recovery archives retained when present at "
                f"{remote_backup} and {remote_host_backup}"
            ) from error

        await self._best_effort_remove(
            remote_archive,
            remote_backup,
            remote_staging,
            remote_host_backup,
            remote_recovery_staging,
        )
        local_recovery.unlink(missing_ok=True)
        return state

    def _remote_temp_path(self, identifier: str, suffix: str) -> str:
        return str(PurePosixPath(self._tmp_path) / f"driftlock-{identifier}-{suffix}")

    @property
    def _workspace_path(self) -> str:
        return self._canonical_workspace or self.remote_workspace

    @property
    def _tmp_path(self) -> str:
        return self._canonical_tmp_dir or self.remote_tmp_dir

    def _prepare_restore_script(
        self, *, archive: str, backup: str, staging: str
    ) -> str:
        workspace = shlex.quote(self._workspace_path)
        archive_q = shlex.quote(archive)
        backup_q = shlex.quote(backup)
        staging_q = shlex.quote(staging)
        script = f"""set -eu
rm -rf -- {staging_q}
rm -f -- {backup_q}
mkdir -p -- {staging_q}
tar -xzf {archive_q} -C {staging_q}
tar -czf {backup_q} -C {workspace} .
sha256sum < {backup_q}
"""
        return "sh -ceu " + shlex.quote(script)

    def _apply_restore_script(self, *, staging: str) -> str:
        workspace = shlex.quote(self._workspace_path)
        staging_q = shlex.quote(staging)
        clear_workspace = (
            f"find {workspace} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +"
        )
        script = f"""set -eu
{clear_workspace}
cp -a {staging_q}/. {workspace}/
"""
        return "sh -ceu " + shlex.quote(script)

    async def _attempt_recovery(
        self,
        *,
        local_recovery: Path,
        remote_backup: str,
        remote_backup_digest: str,
        remote_host_backup: str,
        recovery_staging: str,
    ) -> str:
        recovery_errors: list[str] = []
        try:
            await self._checked_exec(
                self._exact_recovery_script(
                    archive=remote_backup,
                    staging=recovery_staging,
                    expected_digest=remote_backup_digest,
                ),
                operation="recover failed restore from remote backup",
            )
            return "the exact pre-restore workspace was recovered"
        except Exception as recovery_error:
            recovery_errors.append(f"remote backup: {recovery_error}")

        try:
            await self.environment.upload_file(local_recovery, remote_host_backup)
            await self._checked_exec(
                self._exact_recovery_script(
                    archive=remote_host_backup,
                    staging=recovery_staging,
                    expected_digest=remote_backup_digest,
                ),
                operation="recover failed restore from host backup",
            )
            return "the exact pre-restore workspace was recovered from the host copy"
        except Exception as recovery_error:
            recovery_errors.append(f"host backup: {recovery_error}")
        return "automatic exact recovery failed: " + "; ".join(recovery_errors)

    def _exact_recovery_script(
        self,
        *,
        archive: str,
        staging: str,
        expected_digest: str,
    ) -> str:
        workspace = shlex.quote(self._workspace_path)
        archive_q = shlex.quote(archive)
        staging_q = shlex.quote(staging)
        fifo_q = shlex.quote(staging + ".sha256-fifo")
        digest_file_q = shlex.quote(staging + ".sha256-result")
        expected_q = shlex.quote(expected_digest)
        clear_workspace = (
            f"find {workspace} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +"
        )
        script = f"""set -eu
rm -rf -- {staging_q}
rm -f -- {fifo_q} {digest_file_q}
cleanup_digest_stream() {{ rm -f -- {fifo_q} {digest_file_q}; }}
trap cleanup_digest_stream EXIT
trap 'exit 1' HUP INT TERM
mkdir -p -- {staging_q}
exec 3< {archive_q}
mkfifo {fifo_q}
sha256sum < {fifo_q} > {digest_file_q} &
hash_pid=$!
if tee {fifo_q} <&3 | tar -xzf - -C {staging_q}; then
    wait "$hash_pid"
else
    wait "$hash_pid" || true
    exit 1
fi
read -r actual_digest _ < {digest_file_q}
test "$actual_digest" = {expected_q}
{clear_workspace}
cp -a {staging_q}/. {workspace}/
"""
        return "sh -ceu " + shlex.quote(script)

    async def _ensure_remote_paths_are_safe(self) -> None:
        if self._canonical_workspace is not None:
            return
        workspace = shlex.quote(self.remote_workspace)
        tmp_dir = shlex.quote(self.remote_tmp_dir)
        script = f"""set -eu
for tool in sh tar find rm cp realpath sha256sum mkfifo tee
do
    command -v "$tool" >/dev/null
done
test -d {workspace}
test -d {tmp_dir}
workspace_real=$(realpath -- {workspace})
tmp_real=$(realpath -- {tmp_dir})
printf '%s\n' "$workspace_real" "$tmp_real"
find "$workspace_real" -type d -samefile "$tmp_real" -print -quit
"""
        result = await self._checked_exec(
            "sh -ceu " + shlex.quote(script),
            operation="validate remote checkpoint paths and tools",
        )
        lines = (result.stdout or "").splitlines()
        if len(lines) < 2:
            raise RemoteCheckpointError(
                "remote path validation did not return canonical paths"
            )
        canonical_workspace = _validated_remote_path(
            lines[0], name="canonical remote_workspace", allow_root=False
        )
        canonical_tmp = _validated_remote_path(
            lines[1], name="canonical remote_tmp_dir", allow_root=False
        )
        alias_inside_workspace = lines[2] if len(lines) > 2 else ""
        if (
            _is_relative_to(canonical_tmp, canonical_workspace)
            or alias_inside_workspace
        ):
            raise ValueError(
                "remote_tmp_dir resolves to or aliases a directory inside "
                "remote_workspace"
            )
        self._canonical_workspace = canonical_workspace
        self._canonical_tmp_dir = canonical_tmp

    async def _checked_exec(self, command: str, *, operation: str) -> ExecResultLike:
        result = await self.environment.exec(
            command,
            timeout_sec=self.timeout_sec,
            user=self.user,
        )
        _require_success(result, operation=operation)
        return result

    async def _create_remote_archive(self, command: str) -> tuple[str, ...]:
        result = await self.environment.exec(
            command,
            timeout_sec=self.timeout_sec,
            user=self.user,
        )
        if result.return_code == 0:
            return ()
        unstable_paths = _parse_file_changed_warnings(result)
        if unstable_paths is None:
            _require_success(result, operation="create remote archive")

        retry = await self.environment.exec(
            command,
            timeout_sec=self.timeout_sec,
            user=self.user,
        )
        if retry.return_code == 0:
            return ()
        retry_unstable_paths = _parse_file_changed_warnings(retry)
        if retry_unstable_paths is None:
            _require_success(retry, operation="create remote archive")
        return retry_unstable_paths

    async def _best_effort_remove(self, *paths: str) -> None:
        if not paths:
            return
        command = "rm -rf -- " + " ".join(shlex.quote(path) for path in paths)
        try:
            result = await self.environment.exec(
                command,
                timeout_sec=min(self.timeout_sec, 30),
                user=self.user,
            )
        except Exception as error:
            warnings.warn(
                f"failed to clean remote checkpoint artifacts: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "no output").strip()
            warnings.warn(
                "failed to clean remote checkpoint artifacts: "
                f"exit code {result.return_code}: {detail}",
                RuntimeWarning,
                stacklevel=2,
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


def _parse_file_changed_warnings(result: ExecResultLike) -> tuple[str, ...] | None:
    if result.return_code != 1 or not result.stderr:
        return None
    prefix = "tar: "
    suffix = ": file changed as we read it"
    paths: list[str] = []
    for line in result.stderr.splitlines():
        if not line.startswith(prefix) or not line.endswith(suffix):
            return None
        path = line[len(prefix) : -len(suffix)]
        if not path:
            return None
        if path not in paths:
            paths.append(path)
    return tuple(paths) if paths else None


def _require_success(result: ExecResultLike, *, operation: str) -> None:
    if result.return_code != 0:
        detail = (result.stderr or result.stdout or "no output").strip()
        raise RemoteCheckpointError(
            f"{operation} failed with exit code {result.return_code}: {detail}"
        )


def _archive_digest(archive: Path, state_json: str) -> str:
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    digest.update(b"\0state\0")
    digest.update(state_json.encode())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sha256(output: str | None) -> str:
    value = (output or "").strip().split(maxsplit=1)[0] if output else ""
    value = value.removeprefix("\\")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RemoteCheckpointError("remote command did not return a valid SHA-256")
    return value
