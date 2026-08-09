from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from driftlock.checkpoints import SnapshotIntegrityError
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.models import RunStatus, StepContext, StepOutcome
from driftlock.remote import RemoteArchiveCheckpointStore, RemoteCheckpointError
from driftlock.runner import DriftlockRunner, RunnerConfig


@dataclass
class LocalExecResult:
    return_code: int
    stdout: str | None
    stderr: str | None


class LocalRemoteEnvironment:
    """Exercise remote commands locally without depending on Harbor."""

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_sec
        )
        return LocalExecResult(
            return_code=process.returncode or 0,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        shutil.copy2(source_path, target_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        shutil.copy2(source_path, target_path)


class PartialApplyFailureEnvironment(LocalRemoteEnvironment):
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if "cp -a" in command:
            (self.workspace / "important.txt").unlink(missing_ok=True)
            return LocalExecResult(1, "", "injected copy failure")
        return await super().exec(
            command,
            timeout_sec=timeout_sec,
            user=user,
        )


class CleanupFailureEnvironment(LocalRemoteEnvironment):
    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if command.startswith("rm -rf --") and "driftlock-" in command:
            return LocalExecResult(1, "", "injected cleanup failure")
        return await super().exec(
            command,
            timeout_sec=timeout_sec,
            user=user,
        )


class CancelApplyEnvironment(LocalRemoteEnvironment):
    def __init__(self) -> None:
        self.apply_started = asyncio.Event()

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if "cp -a" in command:
            self.apply_started.set()
            await asyncio.Future()
        return await super().exec(
            command,
            timeout_sec=timeout_sec,
            user=user,
        )


def _remote_store(
    tmp_path: Path,
) -> tuple[Path, RemoteArchiveCheckpointStore]:
    workspace = tmp_path / "remote workspace"
    remote_tmp = tmp_path / "remote tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    store = RemoteArchiveCheckpointStore(
        LocalRemoteEnvironment(),
        str(workspace),
        tmp_path / "host checkpoints",
        remote_tmp_dir=str(remote_tmp),
    )
    return workspace, store


async def test_remote_checkpoint_round_trip_preserves_workspace_inode(
    tmp_path: Path,
) -> None:
    workspace, store = _remote_store(tmp_path)
    (workspace / ".hidden").write_text("healthy", encoding="utf-8")
    (workspace / "nested").mkdir()
    (workspace / "nested" / "data.txt").write_text("one", encoding="utf-8")
    inode = workspace.stat().st_ino

    checkpoint = await store.create({"turn": 4}, step=4)
    (workspace / ".hidden").write_text("drifted", encoding="utf-8")
    (workspace / "nested" / "data.txt").unlink()
    (workspace / "new.txt").write_text("delete", encoding="utf-8")

    state = await store.restore(checkpoint)

    assert state == {"turn": 4}
    assert workspace.stat().st_ino == inode
    assert (workspace / ".hidden").read_text(encoding="utf-8") == "healthy"
    assert (workspace / "nested" / "data.txt").read_text(encoding="utf-8") == "one"
    assert not (workspace / "new.txt").exists()
    assert not any((tmp_path / "remote tmp").iterdir())


async def test_remote_checkpoint_detects_archive_tampering(tmp_path: Path) -> None:
    workspace, store = _remote_store(tmp_path)
    (workspace / "file.txt").write_text("healthy", encoding="utf-8")
    checkpoint = await store.create({}, step=0)
    archive = checkpoint.path / "workspace.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"tampered")

    with pytest.raises(SnapshotIntegrityError):
        await store.restore(checkpoint)


async def test_remote_store_reports_archive_command_failure(tmp_path: Path) -> None:
    workspace, store = _remote_store(tmp_path)
    workspace.rmdir()

    with pytest.raises(RemoteCheckpointError, match="validate remote checkpoint"):
        await store.create({}, step=0)


async def test_failed_restore_recovers_and_retains_host_backup(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote-tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    (workspace / "important.txt").write_text("checkpoint", encoding="utf-8")
    environment = PartialApplyFailureEnvironment(workspace)
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )
    checkpoint = await store.create({}, step=0)
    (workspace / "important.txt").write_text("before-restore", encoding="utf-8")

    with pytest.raises(RemoteCheckpointError, match="recovery archive retained"):
        await store.restore(checkpoint)

    assert (workspace / "important.txt").read_text(encoding="utf-8") == "before-restore"
    assert list((tmp_path / "host" / "recovery").glob("*.tar.gz"))
    assert list(remote_tmp.glob("*backup.tar.gz"))


async def test_remote_path_validation_rejects_symlink_alias_inside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    inside_tmp = workspace / "inside-tmp"
    workspace.mkdir()
    inside_tmp.mkdir()
    tmp_alias = tmp_path / "tmp-alias"
    tmp_alias.symlink_to(inside_tmp, target_is_directory=True)
    store = RemoteArchiveCheckpointStore(
        LocalRemoteEnvironment(),
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(tmp_alias),
    )

    with pytest.raises(ValueError, match="aliases"):
        await store.create({}, step=0)


async def test_cancelled_restore_retains_host_and_remote_recovery(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote-tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    (workspace / "important.txt").write_text("checkpoint", encoding="utf-8")
    environment = CancelApplyEnvironment()
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )
    checkpoint = await store.create({}, step=0)
    (workspace / "important.txt").write_text("before-restore", encoding="utf-8")

    restore_task = asyncio.create_task(store.restore(checkpoint))
    await environment.apply_started.wait()
    restore_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await restore_task

    assert list((tmp_path / "host" / "recovery").glob("*.tar.gz"))
    assert list(remote_tmp.glob("*backup.tar.gz"))


async def test_symlinked_workspace_is_restored_through_canonical_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "real-workspace"
    workspace.mkdir()
    workspace_alias = tmp_path / "workspace-alias"
    workspace_alias.symlink_to(workspace, target_is_directory=True)
    remote_tmp = tmp_path / "remote-tmp"
    remote_tmp.mkdir()
    store = RemoteArchiveCheckpointStore(
        LocalRemoteEnvironment(),
        str(workspace_alias),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )
    (workspace / "keep.txt").write_text("checkpoint", encoding="utf-8")
    checkpoint = await store.create({}, step=0)
    (workspace / "stale.txt").write_text("remove", encoding="utf-8")

    await store.restore(checkpoint)

    assert not (workspace / "stale.txt").exists()
    assert (workspace / "keep.txt").read_text(encoding="utf-8") == "checkpoint"


async def test_cleanup_failure_is_surfaced_as_warning(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote-tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    store = RemoteArchiveCheckpointStore(
        CleanupFailureEnvironment(),
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )

    with pytest.warns(RuntimeWarning, match="failed to clean"):
        checkpoint = await store.create({}, step=0)

    assert checkpoint.path.is_dir()


async def test_before_restore_hook_receives_canonical_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    remote_tmp = tmp_path / "remote-tmp"
    remote_tmp.mkdir()
    called_with: list[str] = []

    async def before_restore(canonical_workspace: str) -> None:
        called_with.append(canonical_workspace)

    store = RemoteArchiveCheckpointStore(
        LocalRemoteEnvironment(),
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
        before_restore=before_restore,
    )
    checkpoint = await store.create({}, step=0)

    await store.restore(checkpoint)

    assert called_with == [str(workspace.resolve())]


def test_remote_temp_directory_must_be_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside"):
        RemoteArchiveCheckpointStore(
            LocalRemoteEnvironment(),
            str(workspace),
            tmp_path / "host",
            remote_tmp_dir=str(workspace / "tmp"),
        )


async def test_runner_accepts_async_remote_checkpoint_store(tmp_path: Path) -> None:
    workspace, store = _remote_store(tmp_path)
    (workspace / "answer.txt").write_text("healthy", encoding="utf-8")
    coarse = HeuristicJudge(
        HeuristicConfig(
            no_change_steps=2,
            loop_window=2,
            loop_repetitions=2,
            error_window=2,
            reward_stall_steps=2,
        )
    )

    async def agent_step(context: StepContext) -> StepOutcome:
        if context.attempt == 1:
            (workspace / "answer.txt").write_text("drifted", encoding="utf-8")
            return StepOutcome(
                action=f"wander {context.logical_step}",
                state={"status": "drifted"},
            )
        assert (workspace / "answer.txt").read_text(encoding="utf-8") == "healthy"
        return StepOutcome(
            action="finish",
            state={"status": "solved"},
            completed=True,
        )

    result = await DriftlockRunner(
        store,
        coarse,
        config=RunnerConfig(max_steps=4, max_rollbacks=1),
    ).run(goal="restore remotely", step=agent_step, initial_state={})

    assert result.status is RunStatus.COMPLETED
    assert len(result.rollbacks) == 1
