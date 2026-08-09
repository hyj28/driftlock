"""Checkpoint storage interfaces and a local directory implementation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from collections.abc import Awaitable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from driftlock.models import Checkpoint


class SnapshotIntegrityError(RuntimeError):
    """Raised when stored checkpoint content no longer matches its manifest."""


class CheckpointStore(Protocol):
    """Persistence boundary used by :class:`DriftlockRunner`."""

    def create(
        self,
        state: Mapping[str, Any],
        *,
        step: int,
        parent_id: str | None = None,
        label: str | None = None,
    ) -> Checkpoint | Awaitable[Checkpoint]: ...

    def restore(
        self, checkpoint: Checkpoint
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...


class DirectoryCheckpointStore:
    """Copy-on-checkpoint storage for a local workspace.

    The checkpoint store must live outside the workspace. Restores are performed by
    copying into a sibling staging directory and swapping directories, so a failed
    restore leaves either the old or restored workspace available.

    By default snapshots are exact. Custom ``ignore`` patterns intentionally omit
    matching paths, which means those paths will not exist after a restore.
    """

    def __init__(
        self,
        workspace: Path | str,
        store_dir: Path | str,
        *,
        ignore: tuple[str, ...] = (),
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.store_dir = Path(store_dir).expanduser().resolve()
        self.ignore = ignore
        self._validate_paths()
        (self.store_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _validate_paths(self) -> None:
        if not self.workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {self.workspace}")
        if self.workspace.parent == self.workspace:
            raise ValueError("filesystem root cannot be used as a workspace")
        if self.store_dir == self.workspace or self.workspace in self.store_dir.parents:
            raise ValueError("store_dir must be outside the workspace")
        git_metadata = self.workspace / ".git"
        if git_metadata.is_file() or git_metadata.is_symlink():
            raise ValueError(
                "linked Git worktrees and submodules are not supported because their "
                "repository state lives outside the workspace"
            )

    @property
    def checkpoints_dir(self) -> Path:
        return self.store_dir / "checkpoints"

    def create(
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
        snapshot = temporary / "workspace"
        try:
            shutil.copytree(
                self.workspace,
                snapshot,
                symlinks=True,
                ignore=shutil.ignore_patterns(*self.ignore),
            )
            created_at = datetime.now(UTC)
            (temporary / "state.json").write_text(state_json, encoding="utf-8")
            digest = _checkpoint_digest(snapshot, state_json)
            manifest = {
                "checkpoint_id": checkpoint_id,
                "step": step,
                "created_at": created_at.isoformat(),
                "digest": digest,
                "parent_id": parent_id,
                "label": label,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            temporary.rename(final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return Checkpoint(
            checkpoint_id=checkpoint_id,
            step=step,
            created_at=created_at,
            digest=digest,
            path=final,
            parent_id=parent_id,
            label=label,
        )

    def restore(self, checkpoint: Checkpoint) -> dict[str, Any]:
        checkpoint_dir = checkpoint.path.resolve()
        if checkpoint_dir.parent != self.checkpoints_dir.resolve():
            raise ValueError("checkpoint does not belong to this store")
        snapshot = checkpoint_dir / "workspace"
        if not snapshot.is_dir():
            raise FileNotFoundError(f"checkpoint snapshot is missing: {snapshot}")
        state_path = checkpoint_dir / "state.json"
        try:
            state_json = state_path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise SnapshotIntegrityError("checkpoint agent state is missing") from error
        actual_digest = _checkpoint_digest(snapshot, state_json)
        if actual_digest != checkpoint.digest:
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

        suffix = uuid.uuid4().hex
        staging = self.workspace.parent / f".{self.workspace.name}.restore-{suffix}"
        backup = self.workspace.parent / f".{self.workspace.name}.backup-{suffix}"
        original_cwd = Path.cwd().resolve()
        try:
            relative_cwd: Path | None = original_cwd.relative_to(self.workspace)
        except ValueError:
            relative_cwd = None
        try:
            shutil.copytree(snapshot, staging, symlinks=True)
            self.workspace.rename(backup)
            try:
                staging.rename(self.workspace)
            except BaseException:
                backup.rename(self.workspace)
                raise
            if relative_cwd is not None:
                restored_cwd = self.workspace / relative_cwd
                os.chdir(restored_cwd if restored_cwd.is_dir() else self.workspace)
            shutil.rmtree(backup)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return state


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(stat.S_IMODE(mode)).encode())
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"link\0")
            digest.update(os.readlink(path).encode())
        elif path.is_dir():
            digest.update(b"dir\0")
        elif path.is_file():
            digest.update(b"file\0")
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        else:
            digest.update(b"other\0")
    return digest.hexdigest()


def _checkpoint_digest(snapshot: Path, state_json: str) -> str:
    digest = hashlib.sha256()
    digest.update(_tree_digest(snapshot).encode())
    digest.update(b"\0state\0")
    digest.update(state_json.encode())
    return digest.hexdigest()
