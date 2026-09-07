from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import tarfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from driftlock.checkpoints import SnapshotIntegrityError
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.models import (
    Checkpoint,
    CheckpointRestoreStatus,
    RunStatus,
    StepContext,
    StepOutcome,
)
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


class ArchiveResultEnvironment(LocalRemoteEnvironment):
    def __init__(self, archive_results: list[LocalExecResult]) -> None:
        self.archive_results = list(archive_results)
        self.archive_attempts = 0

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        result = await super().exec(
            command,
            timeout_sec=timeout_sec,
            user=user,
        )
        if not command.startswith("tar -czf"):
            return result
        assert result.return_code == 0
        self.archive_attempts += 1
        return self.archive_results.pop(0)


class MissingArchiveDownloadEnvironment(ArchiveResultEnvironment):
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        raise FileNotFoundError(source_path)


class PathWalkResultEnvironment(ArchiveResultEnvironment):
    def __init__(
        self,
        archive_results: list[LocalExecResult],
        path_walk_result: LocalExecResult,
    ) -> None:
        super().__init__(archive_results)
        self.path_walk_result = path_walk_result

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if command.startswith("find "):
            return self.path_walk_result
        return await super().exec(command, timeout_sec=timeout_sec, user=user)


class PartialApplyFailureEnvironment(LocalRemoteEnvironment):
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.failed_once = False

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if "cp -a" in command and not self.failed_once:
            self.failed_once = True
            (self.workspace / "important.txt").unlink(missing_ok=True)
            (self.workspace / "checkpoint-only.txt").write_text(
                "partial", encoding="utf-8"
            )
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


class TruncatedRecoveryDownloadEnvironment(LocalRemoteEnvironment):
    def __init__(self) -> None:
        self.download_count = 0

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        self.download_count += 1
        if self.download_count == 2:
            Path(target_path).write_bytes(b"")
            return
        await super().download_file(source_path, target_path)


class CorruptRemoteBackupFailureEnvironment(LocalRemoteEnvironment):
    def __init__(self, workspace: Path, remote_tmp: Path) -> None:
        self.workspace = workspace
        self.remote_tmp = remote_tmp
        self.failed_once = False

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if "cp -a" in command and not self.failed_once:
            self.failed_once = True
            (self.workspace / "important.txt").unlink(missing_ok=True)
            for backup in self.remote_tmp.glob("*backup.tar.gz"):
                backup.write_bytes(b"corrupt")
            return LocalExecResult(1, "", "injected apply and backup failure")
        return await super().exec(
            command,
            timeout_sec=timeout_sec,
            user=user,
        )


class MissingRemoteBackupFailureEnvironment(PartialApplyFailureEnvironment):
    def __init__(self, workspace: Path, remote_tmp: Path) -> None:
        super().__init__(workspace)
        self.remote_tmp = remote_tmp

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if "cp -a" in command and not self.failed_once:
            for backup in self.remote_tmp.glob("*backup.tar.gz"):
                backup.unlink()
        return await super().exec(
            command,
            timeout_sec=timeout_sec,
            user=user,
        )


class FailedHostFallbackEnvironment(CorruptRemoteBackupFailureEnvironment):
    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if "host-backup.tar.gz" in command and "cp -a" in command:
            return LocalExecResult(1, "", "injected host fallback failure")
        return await super().exec(
            command,
            timeout_sec=timeout_sec,
            user=user,
        )


class ArchiveSwapAfterChecksumEnvironment(PartialApplyFailureEnvironment):
    def __init__(
        self,
        workspace: Path,
        remote_tmp: Path,
        alternate_archive: Path,
    ) -> None:
        super().__init__(workspace)
        self.remote_tmp = remote_tmp
        self.alternate_archive = alternate_archive
        self.swap_fired = False

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        arguments = shlex.split(command)
        streamed_recovery = (
            arguments[:2] == ["sh", "-ceu"]
            and "recovery-staging" in arguments[2]
            and "host-backup.tar.gz" not in arguments[2]
        )
        if streamed_recovery and not self.swap_fired:
            remote_backups = [
                path
                for path in self.remote_tmp.glob("*backup.tar.gz")
                if not path.name.endswith("host-backup.tar.gz")
            ]
            assert len(remote_backups) == 1
            shutil.copy2(self.alternate_archive, remote_backups[0])
            self.swap_fired = True

        result = await super().exec(
            command,
            timeout_sec=timeout_sec,
            user=user,
        )
        if (
            arguments[:1] == ["sha256sum"]
            and arguments[1].endswith("backup.tar.gz")
            and not self.swap_fired
        ):
            shutil.copy2(self.alternate_archive, arguments[1])
            self.swap_fired = True
        return result


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


def _remote_store_with_archive_results(
    tmp_path: Path, archive_results: list[LocalExecResult]
) -> tuple[Path, RemoteArchiveCheckpointStore, ArchiveResultEnvironment]:
    workspace = tmp_path / "remote workspace"
    remote_tmp = tmp_path / "remote tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    environment = ArchiveResultEnvironment(archive_results)
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host checkpoints",
        remote_tmp_dir=str(remote_tmp),
    )
    return workspace, store, environment


async def test_clean_remote_archive_is_not_retried_or_annotated(
    tmp_path: Path,
) -> None:
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path, [LocalExecResult(0, "", "")]
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({"turn": 1}, step=1)

    manifest = json.loads((checkpoint.path / "manifest.json").read_text())
    assert environment.archive_attempts == 1
    assert checkpoint.unstable_paths == ()
    assert "unstable_paths" not in manifest


async def test_transient_file_changed_warning_retries_to_clean_checkpoint(
    tmp_path: Path,
) -> None:
    warning = (
        "tar: ./output/workspace/tests/retire_trace.log: file changed as we read it"
    )
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(0, "", "")],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    manifest = json.loads((checkpoint.path / "manifest.json").read_text())
    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == ()
    assert "unstable_paths" not in manifest


async def test_persistent_file_changed_warning_records_path_and_restores(
    tmp_path: Path,
) -> None:
    warning = (
        "tar: ./output/workspace/tests/retire_trace.log: file changed as we read it"
    )
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)],
    )
    affected = workspace / "output" / "workspace" / "tests" / "retire_trace.log"
    affected.parent.mkdir(parents=True)
    affected.write_text("checkpoint", encoding="utf-8")

    checkpoint = await store.create({"turn": 2}, step=2)
    affected.write_text("changed", encoding="utf-8")
    state = await store.restore(checkpoint)

    manifest = json.loads((checkpoint.path / "manifest.json").read_text())
    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == ("./output/workspace/tests/retire_trace.log",)
    assert manifest["unstable_paths"] == ["./output/workspace/tests/retire_trace.log"]
    assert state == {"turn": 2}
    assert affected.read_text(encoding="utf-8") == "checkpoint"


async def test_changed_and_removed_warnings_are_one_concurrent_modification_class(
    tmp_path: Path,
) -> None:
    warning = (
        "tar: ./output/workspace/tests/retire_trace.log: file changed as we read it\n"
        "tar: ./output/workspace/tests/sw.mem: File removed before we read it"
    )
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)],
    )
    (workspace / "answer.txt").write_text("checkpoint", encoding="utf-8")

    checkpoint = await store.create({"turn": 4}, step=4)
    state = await store.restore(checkpoint)

    manifest = json.loads((checkpoint.path / "manifest.json").read_text())
    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == (
        "./output/workspace/tests/retire_trace.log",
        "./output/workspace/tests/sw.mem",
    )
    assert checkpoint.restorable is True
    assert checkpoint.restore_status is CheckpointRestoreStatus.ELIGIBLE
    assert checkpoint.unaccounted_archive_output is None
    assert manifest["unstable_paths"] == [
        "./output/workspace/tests/retire_trace.log",
        "./output/workspace/tests/sw.mem",
    ]
    assert manifest["restore_status"] == "eligible"
    assert state == {"turn": 4}


async def test_changed_and_removed_warnings_work_in_harbors_merged_stream(
    tmp_path: Path,
) -> None:
    warning = (
        "tar: ./output/workspace/tests/retire_trace.log: file changed as we read it\n"
        "tar: ./output/workspace/tests/sw.mem: File removed before we read it"
    )
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, warning, None), LocalExecResult(1, warning, None)],
    )
    (workspace / "answer.txt").write_text("checkpoint", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == (
        "./output/workspace/tests/retire_trace.log",
        "./output/workspace/tests/sw.mem",
    )
    assert checkpoint.restorable is True
    assert checkpoint.unaccounted_archive_output is None


async def test_file_shrank_warning_records_the_unstable_path(tmp_path: Path) -> None:
    warning = "tar: ./output/live.mem: File shrank by 4096 bytes; padding with zeros"
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)],
    )
    (workspace / "answer.txt").write_text("checkpoint", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == ("./output/live.mem",)
    assert checkpoint.restorable is True


async def test_benign_exit_one_lines_do_not_block_restore(tmp_path: Path) -> None:
    diagnostics = (
        "\n tar: Removing leading `/' from member names   \n"
        "tar: Removing leading `/' from hard link targets\n"
        "tar: Exiting with failure status due to previous errors  \n"
    )
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [
            LocalExecResult(1, "", diagnostics),
            LocalExecResult(1, "", diagnostics),
        ],
    )
    (workspace / "answer.txt").write_text("checkpoint", encoding="utf-8")

    checkpoint = await store.create({}, step=0)
    await store.restore(checkpoint)

    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == ()
    assert checkpoint.restorable is True
    assert checkpoint.unaccounted_archive_output is None


async def test_harbor_merged_stream_file_changed_warning_records_path(
    tmp_path: Path,
) -> None:
    warning = "tar: ./output/live.log: file changed as we read it"
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, warning, None), LocalExecResult(1, warning, None)],
    )
    affected = workspace / "output" / "live.log"
    affected.parent.mkdir(parents=True)
    affected.write_text("busy", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    manifest = json.loads((checkpoint.path / "manifest.json").read_text())
    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == ("./output/live.log",)
    assert manifest["unstable_paths"] == ["./output/live.log"]


async def test_persistent_file_changed_warnings_record_every_path(
    tmp_path: Path,
) -> None:
    first_warning = "tar: ./first.log: file changed as we read it"
    both_warnings = "\n".join(
        [
            "tar: ./first.log: file changed as we read it",
            "tar: ./second.log: file changed as we read it",
        ]
    )
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [
            LocalExecResult(1, "", first_warning),
            LocalExecResult(1, "", both_warnings),
        ],
    )
    (workspace / "first.log").write_text("one", encoding="utf-8")
    (workspace / "second.log").write_text("two", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    manifest = json.loads((checkpoint.path / "manifest.json").read_text())
    assert environment.archive_attempts == 2
    assert manifest["unstable_paths"] == ["./first.log", "./second.log"]


async def test_unrelated_exit_one_archive_warning_is_not_silently_benign(
    tmp_path: Path,
) -> None:
    warning = "tar: ./ignored.sock: socket ignored"
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    manifest = json.loads((checkpoint.path / "manifest.json").read_text())
    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == ()
    assert checkpoint.restorable is False
    assert checkpoint.restore_status is CheckpointRestoreStatus.INELIGIBLE
    assert checkpoint.unaccounted_archive_output == warning
    assert manifest["restore_status"] == "ineligible"
    assert manifest["unaccounted_archive_output"] == warning
    (workspace / "answer.txt").write_text("live", encoding="utf-8")
    with pytest.raises(RemoteCheckpointError, match="not eligible for restore"):
        await store.restore(checkpoint)
    assert (workspace / "answer.txt").read_text(encoding="utf-8") == "live"


@pytest.mark.parametrize(
    ("stderr", "expected_unstable"),
    [
        (
            "tar: ./ws/retire_trace.log: file changed as we read it\n",
            ("./ws/retire_trace.log",),
        ),
        ("tar: Exiting with failure status due to previous errors\n", ()),
    ],
)
async def test_split_tar_streams_are_both_classified(
    tmp_path: Path, stderr: str, expected_unstable: tuple[str, ...]
) -> None:
    stdout = "tar: ./ws/secret.key: Cannot open: Permission denied\n"
    result = LocalExecResult(1, stdout, stderr)
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path, [result, result]
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == expected_unstable
    assert checkpoint.restore_status is CheckpointRestoreStatus.INELIGIBLE
    assert checkpoint.unaccounted_archive_output == (
        "tar: ./ws/secret.key: Cannot open: Permission denied"
    )
    with pytest.raises(RemoteCheckpointError, match="not eligible for restore"):
        await store.restore(checkpoint)


async def test_unaccounted_archive_output_is_visibly_bounded(tmp_path: Path) -> None:
    warning = "\n".join(
        f"tar: ./secret-{index:04d}.key: Cannot open: Permission denied"
        for index in range(2_000)
    )
    workspace, store, _environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    manifest = json.loads((checkpoint.path / "manifest.json").read_text())
    assert checkpoint.restorable is False
    assert checkpoint.unaccounted_archive_output is not None
    assert len(checkpoint.unaccounted_archive_output) == 8_192
    assert checkpoint.unaccounted_archive_output.startswith(
        "tar: ./secret-0000.key: Cannot open: Permission denied"
    )
    assert checkpoint.unaccounted_archive_output.endswith(
        "\n[unaccounted archive output truncated]"
    )
    assert "secret-1999.key" not in checkpoint.unaccounted_archive_output
    assert manifest["unaccounted_archive_output"] == (
        checkpoint.unaccounted_archive_output
    )


async def test_garbled_error_suffix_is_not_absorbed_into_an_unstable_path(
    tmp_path: Path,
) -> None:
    warning = "tar: ./x: Cannot open: Permission denied: file changed as we read it"
    workspace, store, _environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert checkpoint.unstable_paths == ()
    assert checkpoint.restorable is False
    assert checkpoint.unaccounted_archive_output == warning


@pytest.mark.parametrize(
    "warning",
    [
        (
            "tar: ./build/priv: Cannot savedir: Permission denied: "
            "file changed as we read it"
        ),
        (
            "tar: ./x: Cannot add file: No space left on device: "
            "file changed as we read it"
        ),
        (
            "tar: /tmp/snap.tar.gz: Wrote only 4096 of 10240 bytes: "
            "file changed as we read it"
        ),
        (
            "tar: ./a: Cannot savedir: Permission deniedtar: ./b: "
            "file changed as we read it"
        ),
    ],
)
async def test_only_positive_archive_member_shapes_become_unstable_paths(
    tmp_path: Path, warning: str
) -> None:
    workspace, store, _environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, warning, None), LocalExecResult(1, warning, None)],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert checkpoint.unstable_paths == ()
    assert checkpoint.restore_status is CheckpointRestoreStatus.INELIGIBLE
    assert checkpoint.unaccounted_archive_output == warning


async def test_exit_one_without_diagnostics_is_not_restorable(tmp_path: Path) -> None:
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", None), LocalExecResult(1, "", None)],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == ()
    assert checkpoint.restore_status is CheckpointRestoreStatus.INELIGIBLE
    assert checkpoint.unaccounted_archive_output == (
        "tar exited with code 1 without diagnostic output"
    )


@pytest.mark.parametrize(
    "warning",
    [
        "tar: ./x: File shrank unexpectedly",
        "tar: ./x: file changed as we read itX",
    ],
)
async def test_concurrency_warning_matches_must_be_complete_suffixes(
    tmp_path: Path, warning: str
) -> None:
    workspace, store, _environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert checkpoint.unstable_paths == ()
    assert checkpoint.restorable is False
    assert checkpoint.unaccounted_archive_output == warning


@pytest.mark.parametrize(
    ("output", "expected_unstable", "expected_unaccounted"),
    [
        (
            "tar: ./out/x.log: Cannot open: Permission denied",
            (),
            "tar: ./out/x.log: Cannot open: Permission denied",
        ),
        (
            "tar: ./out/x.log: file changed as we read it\n"
            "tar: ./out/y.log: Cannot open: Permission denied",
            ("./out/x.log",),
            "tar: ./out/y.log: Cannot open: Permission denied",
        ),
        (
            "tar: : file changed as we read it",
            (),
            "tar: : file changed as we read it",
        ),
        (
            "tar:    : file changed as we read it",
            (),
            "tar:    : file changed as we read it",
        ),
    ],
)
async def test_harbor_merged_stream_unrecognised_exit_one_is_not_restorable(
    tmp_path: Path,
    output: str,
    expected_unstable: tuple[str, ...],
    expected_unaccounted: str,
) -> None:
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, output, None), LocalExecResult(1, output, None)],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == expected_unstable
    assert checkpoint.restorable is False
    assert checkpoint.unaccounted_archive_output == expected_unaccounted
    with pytest.raises(RemoteCheckpointError, match="not eligible for restore"):
        await store.restore(checkpoint)


async def test_fatal_archive_exit_still_raises_without_retry(tmp_path: Path) -> None:
    error = "tar: Cowardly refusing to create an empty archive"
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path, [LocalExecResult(2, "", error)]
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    with pytest.raises(RemoteCheckpointError) as raised:
        await store.create({}, step=0)

    assert environment.archive_attempts == 1
    assert str(raised.value) == (
        "create remote archive failed with exit code 2: "
        "tar: Cowardly refusing to create an empty archive"
    )


async def test_unparseable_file_changed_warning_is_not_restorable(
    tmp_path: Path,
) -> None:
    warning = "tar: : file changed as we read it"
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == ()
    assert checkpoint.restorable is False
    assert checkpoint.unaccounted_archive_output == warning
    with pytest.raises(RemoteCheckpointError, match="not eligible for restore"):
        await store.restore(checkpoint)


async def test_missing_exit_one_archive_raises_the_library_error(
    tmp_path: Path,
) -> None:
    warning = "tar: ./output/live.log: file changed as we read it"
    workspace = tmp_path / "remote workspace"
    remote_tmp = tmp_path / "remote tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    environment = MissingArchiveDownloadEnvironment(
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)]
    )
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host checkpoints",
        remote_tmp_dir=str(remote_tmp),
    )

    with pytest.raises(RemoteCheckpointError) as raised:
        await store.create({}, step=0)

    assert environment.archive_attempts == 2
    assert str(raised.value) == "environment did not provide the checkpoint archive"


async def test_restore_reloads_ineligible_status_from_the_manifest(
    tmp_path: Path,
) -> None:
    warning = "tar: ./ignored.sock: socket ignored"
    workspace, store, _environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)],
    )
    (workspace / "answer.txt").write_text("checkpoint", encoding="utf-8")
    checkpoint = await store.create({}, step=0)
    reconstructed = Checkpoint(
        checkpoint_id=checkpoint.checkpoint_id,
        step=checkpoint.step,
        created_at=checkpoint.created_at,
        digest=checkpoint.digest,
        path=checkpoint.path,
        parent_id=checkpoint.parent_id,
        label=checkpoint.label,
        unstable_paths=checkpoint.unstable_paths,
    )
    assert reconstructed.restorable is True
    (workspace / "answer.txt").write_text("live", encoding="utf-8")
    before_restore_calls: list[str] = []

    async def before_restore(path: str) -> None:
        before_restore_calls.append(path)

    reopened_store = RemoteArchiveCheckpointStore(
        LocalRemoteEnvironment(),
        str(workspace),
        store.store_dir,
        remote_tmp_dir=str(tmp_path / "remote tmp"),
        before_restore=before_restore,
    )

    with pytest.raises(RemoteCheckpointError) as raised:
        await reopened_store.restore(reconstructed)

    assert "not eligible for restore" in str(raised.value)
    assert warning in str(raised.value)
    assert (workspace / "answer.txt").read_text(encoding="utf-8") == "live"
    assert before_restore_calls == []
    with pytest.raises(ValueError, match="restorable checkpoint"):
        replace(checkpoint, restore_status=CheckpointRestoreStatus.ELIGIBLE)


async def test_restore_rejects_in_memory_metadata_divergence(tmp_path: Path) -> None:
    workspace, store = _remote_store(tmp_path)
    (workspace / "answer.txt").write_text("checkpoint", encoding="utf-8")
    checkpoint = await store.create({}, step=0)
    divergent = replace(
        checkpoint,
        restore_status=CheckpointRestoreStatus.INELIGIBLE,
        unaccounted_archive_output="caller supplied an unrecorded warning",
    )

    with pytest.raises(SnapshotIntegrityError) as raised:
        await store.restore(divergent)

    assert str(raised.value) == (
        f"checkpoint {checkpoint.checkpoint_id} restore metadata differs from its "
        "manifest"
    )


async def test_restore_rejects_manifest_checkpoint_identity_mismatch(
    tmp_path: Path,
) -> None:
    workspace, store = _remote_store(tmp_path)
    (workspace / "answer.txt").write_text("checkpoint", encoding="utf-8")
    checkpoint = await store.create({}, step=0)
    manifest_path = checkpoint.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_id"] = "f" * 32
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError) as raised:
        await store.restore(checkpoint)

    assert str(raised.value) == (
        f"checkpoint {checkpoint.checkpoint_id} identity differs from its manifest"
    )


async def test_restore_rejects_manifest_digest_identity_mismatch(
    tmp_path: Path,
) -> None:
    workspace, store = _remote_store(tmp_path)
    (workspace / "answer.txt").write_text("checkpoint", encoding="utf-8")
    checkpoint = await store.create({}, step=0)
    manifest_path = checkpoint.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["digest"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError) as raised:
        await store.restore(checkpoint)

    assert str(raised.value) == (
        f"checkpoint {checkpoint.checkpoint_id} identity differs from its manifest"
    )


async def test_restore_rejects_eligible_manifest_with_unaccounted_output(
    tmp_path: Path,
) -> None:
    workspace, store = _remote_store(tmp_path)
    (workspace / "answer.txt").write_text("checkpoint", encoding="utf-8")
    checkpoint = await store.create({}, step=0)
    manifest_path = checkpoint.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unaccounted_archive_output"] = "tar: unexplained warning"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError) as raised:
        await store.restore(checkpoint)

    assert str(raised.value) == (
        "checkpoint restore eligibility metadata is inconsistent"
    )


async def test_legacy_manifest_without_restore_status_remains_restorable(
    tmp_path: Path,
) -> None:
    workspace, store = _remote_store(tmp_path)
    (workspace / "answer.txt").write_text("checkpoint", encoding="utf-8")
    checkpoint = await store.create({"legacy": True}, step=0)
    manifest_path = checkpoint.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["restore_status"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (workspace / "answer.txt").write_text("live", encoding="utf-8")

    state = await store.restore(checkpoint)

    assert state == {"legacy": True}
    assert (workspace / "answer.txt").read_text(encoding="utf-8") == "checkpoint"


def test_non_restorable_checkpoint_requires_unaccounted_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as raised:
        Checkpoint(
            checkpoint_id="unsafe",
            step=0,
            created_at=datetime(2026, 9, 6, tzinfo=UTC),
            digest="digest",
            path=tmp_path / "unsafe",
            restore_status=CheckpointRestoreStatus.INELIGIBLE,
        )

    assert str(raised.value) == (
        "a non-restorable checkpoint needs unaccounted archive output"
    )


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
    assert not (workspace / "checkpoint-only.txt").exists()
    assert list((tmp_path / "host" / "recovery").glob("*.tar.gz"))
    remaining = list(remote_tmp.iterdir())
    assert len(remaining) == 1
    assert remaining[0].name.endswith("backup.tar.gz")


async def test_truncated_recovery_download_is_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote-tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    (workspace / "important.txt").write_text("checkpoint", encoding="utf-8")
    environment = TruncatedRecoveryDownloadEnvironment()
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )
    checkpoint = await store.create({}, step=0)
    (workspace / "important.txt").write_text("before-restore", encoding="utf-8")

    with pytest.raises(RemoteCheckpointError, match="does not match"):
        await store.restore(checkpoint)

    assert (workspace / "important.txt").read_text(encoding="utf-8") == "before-restore"
    assert not any(remote_tmp.iterdir())


async def test_validated_host_copy_recovers_when_remote_backup_is_corrupt(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote-tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    (workspace / "important.txt").write_text("checkpoint", encoding="utf-8")
    environment = CorruptRemoteBackupFailureEnvironment(workspace, remote_tmp)
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )
    checkpoint = await store.create({}, step=0)
    (workspace / "important.txt").write_text("before-restore", encoding="utf-8")

    with pytest.raises(RemoteCheckpointError, match="host copy"):
        await store.restore(checkpoint)

    assert (workspace / "important.txt").read_text(encoding="utf-8") == "before-restore"
    remaining = list(remote_tmp.iterdir())
    assert len(remaining) == 2
    assert (
        sum(
            path.name.endswith("backup.tar.gz")
            and not path.name.endswith("host-backup.tar.gz")
            for path in remaining
        )
        == 1
    )
    assert sum(path.name.endswith("host-backup.tar.gz") for path in remaining) == 1


async def test_missing_remote_backup_falls_back_without_fifo_deadlock(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote-tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    (workspace / "important.txt").write_text("checkpoint", encoding="utf-8")
    environment = MissingRemoteBackupFailureEnvironment(workspace, remote_tmp)
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
        timeout_sec=30,
    )
    checkpoint = await store.create({}, step=0)
    (workspace / "important.txt").write_text("before-restore", encoding="utf-8")

    with pytest.raises(RemoteCheckpointError, match="host copy"):
        await asyncio.wait_for(store.restore(checkpoint), timeout=2)

    assert (workspace / "important.txt").read_text(encoding="utf-8") == "before-restore"


async def test_failed_host_fallback_retains_uploaded_verified_archive(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote-tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    (workspace / "important.txt").write_text("checkpoint", encoding="utf-8")
    environment = FailedHostFallbackEnvironment(workspace, remote_tmp)
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )
    checkpoint = await store.create({}, step=0)
    (workspace / "important.txt").write_text("before-restore", encoding="utf-8")

    with pytest.raises(RemoteCheckpointError, match="automatic exact recovery failed"):
        await store.restore(checkpoint)

    retained_host_backups = list(remote_tmp.glob("*host-backup.tar.gz"))
    assert len(retained_host_backups) == 1
    assert retained_host_backups[0].stat().st_size > 0
    assert not list(remote_tmp.glob("*recovery-staging*"))


async def test_recovery_hashes_the_same_archive_stream_that_it_extracts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote-tmp"
    alternate = tmp_path / "alternate"
    workspace.mkdir()
    remote_tmp.mkdir()
    alternate.mkdir()
    (alternate / "attacker.txt").write_text("swapped", encoding="utf-8")
    alternate_archive = tmp_path / "alternate.tar.gz"
    with tarfile.open(alternate_archive, "w:gz") as archive:
        archive.add(alternate, arcname=".")
    (workspace / "important.txt").write_text("checkpoint", encoding="utf-8")
    environment = ArchiveSwapAfterChecksumEnvironment(
        workspace,
        remote_tmp,
        alternate_archive,
    )
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )
    checkpoint = await store.create({}, step=0)
    (workspace / "important.txt").write_text("before-restore", encoding="utf-8")

    with pytest.raises(RemoteCheckpointError, match="exact pre-restore workspace"):
        await store.restore(checkpoint)

    assert (workspace / "important.txt").read_text(encoding="utf-8") == "before-restore"
    assert not (workspace / "attacker.txt").exists()
    assert environment.swap_fired


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


async def test_remote_path_alias_walk_tolerates_only_vanished_entries(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote-tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    environment = PathWalkResultEnvironment(
        [LocalExecResult(0, "", "")],
        LocalExecResult(
            1,
            "",
            f"find: '{workspace}/short-lived': No such file or directory",
        ),
    )
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )

    checkpoint = await store.create({}, step=0)

    assert checkpoint.restorable is True
    assert environment.archive_attempts == 1


@pytest.mark.parametrize(
    ("path_walk_result", "expected_message"),
    [
        (
            LocalExecResult(1, "", "find: './private': Permission denied"),
            "validate remote checkpoint path aliases failed with exit code 1: "
            "find: './private': Permission denied",
        ),
        (
            LocalExecResult(0, "/workspace/aliased-tmp\0", ""),
            "remote_tmp_dir resolves to or aliases a directory inside remote_workspace",
        ),
    ],
)
async def test_remote_path_alias_walk_keeps_safety_failures_hard(
    tmp_path: Path,
    path_walk_result: LocalExecResult,
    expected_message: str,
) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote-tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    environment = PathWalkResultEnvironment([], path_walk_result)
    store = RemoteArchiveCheckpointStore(
        environment,
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )

    with pytest.raises((RemoteCheckpointError, ValueError)) as raised:
        await store.create({}, step=0)

    assert str(raised.value) == expected_message
    assert environment.archive_attempts == 0


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


async def test_before_restore_failure_does_not_rebuild_workspace_children(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    child = workspace / "child"
    child.mkdir(parents=True)
    remote_tmp = tmp_path / "remote-tmp"
    remote_tmp.mkdir()

    async def before_restore(_canonical_workspace: str) -> None:
        raise RuntimeError("tmux evacuation failed")

    store = RemoteArchiveCheckpointStore(
        LocalRemoteEnvironment(),
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
        before_restore=before_restore,
    )
    (child / "state.txt").write_text("checkpoint", encoding="utf-8")
    checkpoint = await store.create({}, step=0)
    (child / "state.txt").write_text("live", encoding="utf-8")
    child_inode = child.stat().st_ino

    with pytest.raises(RuntimeError, match="tmux evacuation failed"):
        await store.restore(checkpoint)

    assert child.stat().st_ino == child_inode
    assert (child / "state.txt").read_text(encoding="utf-8") == "live"
    assert not any(remote_tmp.iterdir())
    assert not list((tmp_path / "host" / "recovery").glob("*.tar.gz"))


async def test_restore_supports_backslash_in_remote_temp_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    remote_tmp = tmp_path / "remote\\tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    store = RemoteArchiveCheckpointStore(
        LocalRemoteEnvironment(),
        str(workspace),
        tmp_path / "host",
        remote_tmp_dir=str(remote_tmp),
    )
    (workspace / "state.txt").write_text("checkpoint", encoding="utf-8")
    checkpoint = await store.create({}, step=0)
    (workspace / "state.txt").write_text("live", encoding="utf-8")

    await store.restore(checkpoint)

    assert (workspace / "state.txt").read_text(encoding="utf-8") == "checkpoint"
    assert not any(remote_tmp.iterdir())


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
            # Empty so this fixture's only live detector (no_file_change) can
            # still reach the fine judge; the gate is tested in test_heuristics.
            corroborating_signals=frozenset(),
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


async def test_runner_continues_after_an_unaccounted_exit_one(tmp_path: Path) -> None:
    warning = "tar: ./ignored.sock: socket ignored"
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", warning), LocalExecResult(1, "", warning)],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    async def agent_step(_context: StepContext) -> StepOutcome:
        return StepOutcome(action="finish", state={"done": True}, completed=True)

    result = await DriftlockRunner(
        store,
        HeuristicJudge(),
        config=RunnerConfig(max_steps=1, checkpoint_on_exit=False),
    ).run(goal="finish despite suspect checkpoint", step=agent_step, initial_state={})

    assert environment.archive_attempts == 2
    assert result.status is RunStatus.COMPLETED
    assert len(result.checkpoints) == 1
    assert result.checkpoints[0].unstable_paths == ()
    assert result.checkpoints[0].restorable is False
    assert result.checkpoints[0].unaccounted_archive_output == warning


@pytest.mark.parametrize(
    ("label", "stderr"),
    [
        (
            "leading blank line",
            "\ntar: ./out/x.log: file changed as we read it",
        ),
        (
            "trailing whitespace",
            "tar: ./out/x.log: file changed as we read it  \n",
        ),
        (
            "removed warning whitespace",
            "\n  tar: ./out/x.log: File removed before we read it   \n",
        ),
        (
            "shrank warning whitespace",
            "  tar: ./out/x.log: File shrank by 7 bytes; padding with zeros  \n\n",
        ),
        (
            "tar's exit summary",
            "tar: ./out/x.log: file changed as we read it\n"
            "tar: Exiting with failure status due to previous errors\n",
        ),
        (
            "leading-slash notice",
            "tar: Removing leading `/' from member names\n"
            "tar: ./out/x.log: file changed as we read it\n",
        ),
        (
            "blank lines between warnings",
            "tar: ./out/x.log: file changed as we read it\n\n"
            "tar: ./out/y.log: file changed as we read it\n",
        ),
    ],
)
async def test_a_concurrent_write_is_recognised_whatever_tar_pads_it_with(
    tmp_path: Path, label: str, stderr: str
) -> None:
    # Every shape here is an ordinary concurrent write. Treating any of them as
    # an archive failure kills the trial, and because the error text is stripped
    # for display the message looks identical to the handled case.
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", stderr), LocalExecResult(1, "", stderr)],
    )
    (workspace / "out").mkdir()
    (workspace / "out" / "x.log").write_text("busy", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert environment.archive_attempts == 2
    assert "./out/x.log" in checkpoint.unstable_paths


@pytest.mark.parametrize(
    ("label", "stderr", "expected_unstable", "expected_unaccounted"),
    [
        (
            "unrelated error",
            "tar: ./out/x.log: Cannot open: Permission denied",
            (),
            "tar: ./out/x.log: Cannot open: Permission denied",
        ),
        (
            "mixed with a real error",
            "tar: ./out/x.log: file changed as we read it\n"
            "tar: ./out/y.log: Cannot open: Permission denied\n",
            ("./out/x.log",),
            "tar: ./out/y.log: Cannot open: Permission denied",
        ),
        (
            "empty path",
            "tar: : file changed as we read it",
            (),
            "tar: : file changed as we read it",
        ),
        (
            "whitespace path",
            "tar:    : file changed as we read it",
            (),
            "tar:    : file changed as we read it",
        ),
    ],
)
async def test_an_unrecognised_tar_line_marks_the_checkpoint_not_restorable(
    tmp_path: Path,
    label: str,
    stderr: str,
    expected_unstable: tuple[str, ...],
    expected_unaccounted: str,
) -> None:
    workspace, store, environment = _remote_store_with_archive_results(
        tmp_path,
        [LocalExecResult(1, "", stderr), LocalExecResult(1, "", stderr)],
    )
    (workspace / "answer.txt").write_text("stable", encoding="utf-8")

    checkpoint = await store.create({}, step=0)

    assert environment.archive_attempts == 2
    assert checkpoint.unstable_paths == expected_unstable
    assert checkpoint.restorable is False
    assert checkpoint.unaccounted_archive_output == expected_unaccounted
    with pytest.raises(RemoteCheckpointError, match="not eligible for restore"):
        await store.restore(checkpoint)


def test_checkpoint_restore_eligibility_defaults_keep_existing_readers_working(
    tmp_path: Path,
) -> None:
    checkpoint = Checkpoint(
        checkpoint_id="legacy",
        step=0,
        created_at=datetime.now(UTC),
        digest="digest",
        path=tmp_path / "legacy",
    )

    assert checkpoint.restore_status is CheckpointRestoreStatus.ELIGIBLE
    assert checkpoint.restorable is True
    assert checkpoint.unaccounted_archive_output is None
    assert {status.value for status in CheckpointRestoreStatus} == {
        "eligible",
        "ineligible",
    }
