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
    stdout: str
    stderr: str


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

    with pytest.raises(RemoteCheckpointError, match="create remote archive"):
        await store.create({}, step=0)


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
