from __future__ import annotations

import asyncio
import json
import os
import shlex
import stat
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from driftlock.agent import (
    MAX_EDIT_FILE_BYTES,
    MAX_EDIT_MATCH_CHARACTERS,
    MAX_EDIT_REPLACEMENT_CHARACTERS,
    MAX_EDIT_STALE_FILES,
    AgentCompletion,
    AgentCompletionRequest,
    FileEditResult,
    FileEditStatus,
    ToolCall,
    ToolCallingAgent,
)
from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.local import (
    LocalEnvironment,
    LocalExecResult,
    LocalWorkspaceDeltaObserver,
)
from driftlock.models import DriftContext, JudgeVerdict, RunStatus, StepContext, Verdict
from driftlock.runner import DriftlockRunner, RunnerConfig


class _ScriptedProvider:
    def __init__(self, responses: Sequence[AgentCompletion]) -> None:
        self.responses = list(responses)
        self.requests: list[AgentCompletionRequest] = []

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
        self.requests.append(request)
        return self.responses.pop(0)


class _ConcurrentEnvironment:
    """Run the real protocol while changing the target after staging begins."""

    def __init__(self, workspace: Path, target: Path, content: bytes) -> None:
        self.environment = LocalEnvironment(workspace)
        self.workspace = workspace
        self.target = target
        self.content = content
        self.writer_ran = False

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ):
        if ".driftlock-edit-" not in command:
            return await self.environment.exec(
                command, timeout_sec=timeout_sec, user=user
            )

        async def write_when_staged() -> None:
            while not tuple(self.workspace.glob(".driftlock-edit-*")):
                await asyncio.sleep(0)
            self.target.write_bytes(self.content)
            self.writer_ran = True

        writer = asyncio.create_task(write_when_staged())
        try:
            return await self.environment.exec(
                command, timeout_sec=timeout_sec, user=user
            )
        finally:
            await asyncio.wait_for(writer, timeout=2)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self.environment.upload_file(source_path, target_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.environment.download_file(source_path, target_path)


class _ScriptTransformEnvironment:
    """Execute the real remote program after injecting one deterministic fault."""

    def __init__(
        self,
        workspace: Path,
        transform: Callable[[str], str],
        *,
        decorate_stdout: bool = False,
        fail_cleanup: bool = False,
    ) -> None:
        self.environment = LocalEnvironment(workspace)
        self.transform = transform
        self.decorate_stdout = decorate_stdout
        self.fail_cleanup = fail_cleanup

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ):
        if (
            self.fail_cleanup
            and "DRIFTLOCK_EDIT_RESULT=" not in command
            and ".driftlock-edit-stage-" in command
        ):
            return LocalExecResult(7, "", "forced cleanup failure")
        if "DRIFTLOCK_EDIT_RESULT=" in command:
            arguments = shlex.split(command)
            arguments[2] = self.transform(arguments[2])
            command = " ".join(shlex.quote(argument) for argument in arguments)
        result = await self.environment.exec(
            command, timeout_sec=timeout_sec, user=user
        )
        if self.decorate_stdout and "DRIFTLOCK_EDIT_RESULT=" in command:
            result = replace(
                result,
                stdout=(
                    "DeprecationWarning: merged before result\n"
                    f"{result.stdout}"
                    "DeprecationWarning: merged after result\n"
                ),
            )
        return result

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self.environment.upload_file(source_path, target_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.environment.download_file(source_path, target_path)


class _SymlinkSwapEnvironment:
    def __init__(self, workspace: Path, target: Path, destination: Path) -> None:
        self.environment = LocalEnvironment(workspace)
        self.target = target
        self.destination = destination
        self.swapped = False

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ):
        result = await self.environment.exec(
            command, timeout_sec=timeout_sec, user=user
        )
        if "os.path.realpath" in command and not self.swapped:
            self.target.unlink()
            self.target.symlink_to(self.destination.name)
            self.swapped = True
        return result

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self.environment.upload_file(source_path, target_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.environment.download_file(source_path, target_path)


def _context(state: dict[str, object], *, sequence: int = 1) -> StepContext:
    return StepContext(
        goal="make one bounded edit",
        plan="edit, verify",
        state=state,
        sequence=sequence,
        logical_step=sequence,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=None,
    )


def _edit_call(
    old_text: object = "OLD",
    new_text: object = "NEW",
    *,
    path: object = "target.txt",
    call_id: str = "edit-1",
) -> ToolCall:
    return ToolCall(
        "edit_file",
        {"path": path, "old_text": old_text, "new_text": new_text},
        call_id,
    )


async def _run_edit(
    workspace: Path,
    call: ToolCall,
    *,
    max_tool_output_chars: int = 16_000,
    environment: object | None = None,
):
    provider = _ScriptedProvider([AgentCompletion(tool_calls=(call,), tokens=3)])
    agent = ToolCallingAgent(
        environment or LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        edit_file=True,
        max_tool_output_chars=max_tool_output_chars,
    )
    return await agent(_context(agent.initial_state()))


def _result(outcome) -> dict[str, object]:
    prefix = "edit_file:\n"
    assert outcome.tool_observations[0].startswith(prefix)
    return json.loads(outcome.tool_observations[0][len(prefix) :])


def _write_displaced_inode(script: str) -> str:
    needle = (
        "    atomic_exchange(temporary, target)\n"
        "    exchange_holds_displaced_file = True\n"
    )
    replacement = needle + '    temporary.write_bytes(b"WRITER")\n'
    assert script.count(needle) == 1
    return script.replace(needle, replacement)


def _write_installed_inode(script: str) -> str:
    needle = "    installed = bounded_read(target, limit)\n"
    replacement = '    target.write_bytes(b"WRITER")\n' + needle
    assert script.count(needle) == 1
    return script.replace(needle, replacement)


def _fail_second_exchange(script: str) -> str:
    transformed = _write_displaced_inode(script)
    needle = "def atomic_exchange(left, right):\n    libc ="
    replacement = (
        "atomic_exchange_calls = 0\n\n\n"
        "def atomic_exchange(left, right):\n"
        "    global atomic_exchange_calls\n"
        "    atomic_exchange_calls += 1\n"
        "    if atomic_exchange_calls == 2:\n"
        '        raise OSError(errno.ENOSPC, "forced swap-back failure")\n'
        "    libc ="
    )
    assert transformed.count(needle) == 1
    return transformed.replace(needle, replacement)


def _fail_all_restore_operations(script: str) -> str:
    transformed = _fail_second_exchange(script)
    needle = "                os.replace(temporary, target)\n"
    replacement = (
        '                raise OSError(errno.EIO, "forced fallback failure")\n'
    )
    assert transformed.count(needle) == 1
    return transformed.replace(needle, replacement)


def _unavailable_exchange(script: str) -> str:
    needle = "def atomic_exchange(left, right):\n    libc ="
    replacement = (
        "def atomic_exchange(left, right):\n"
        '    raise AtomicExchangeUnavailable(errno.ENOSYS, "forced")\n'
        "    libc ="
    )
    assert script.count(needle) == 1
    return script.replace(needle, replacement)


def _read_only_failure(script: str) -> str:
    needle = "    descriptor, temporary_name = tempfile.mkstemp(\n"
    replacement = (
        '    raise PermissionError(errno.EROFS, "forced read-only parent")\n' + needle
    )
    assert script.count(needle) == 1
    return script.replace(needle, replacement)


def _kill_after_staging(script: str) -> str:
    needle = "    copy_metadata(target, temporary, first_stat)\n"
    replacement = needle + "    os._exit(9)\n"
    assert script.count(needle) == 1
    return script.replace(needle, replacement)


def _fail_after_exchange(script: str) -> str:
    needle = "    exchange_holds_displaced_file = True\n"
    replacement = needle + '    raise OSError(errno.EIO, "forced after exchange")\n'
    assert script.count(needle) == 1
    return script.replace(needle, replacement)


def _fail_recovery_rename(script: str) -> str:
    transformed = _fail_after_exchange(script)
    needle = "        temporary.rename(recovery)\n"
    replacement = '        raise OSError(errno.ENOSPC, "forced recovery rename")\n'
    assert transformed.count(needle) == 1
    return transformed.replace(needle, replacement)


def _change_restored_file(script: str) -> str:
    transformed = _write_displaced_inode(script)
    needle = (
        "            raise SystemExit\n        restored = bounded_read(target, limit)\n"
    )
    replacement = (
        "            raise SystemExit\n"
        '        target.write_bytes(b"LATER")\n'
        "        restored = bounded_read(target, limit)\n"
    )
    assert transformed.count(needle) == 1
    return transformed.replace(needle, replacement)


def _make_fallback_restore_unreadable(script: str) -> str:
    transformed = _fail_second_exchange(script)
    needle = (
        "                try:\n"
        "                    restored = bounded_read(target, limit)\n"
    )
    replacement = (
        "                try:\n"
        '                    raise OSError(errno.EIO, "forced unreadable restore")\n'
        "                    restored = bounded_read(target, limit)\n"
    )
    assert transformed.count(needle) == 1
    return transformed.replace(needle, replacement)


def _install_symlink_with_edited_bytes(script: str) -> str:
    needle = "    installed = bounded_read(target, limit)\n"
    replacement = (
        '    link_target = target.parent / ".driftlock-installed-target"\n'
        "    link_target.write_bytes(edited)\n"
        "    target.unlink()\n"
        "    target.symlink_to(link_target.name)\n" + needle
    )
    assert script.count(needle) == 1
    return script.replace(needle, replacement)


def _kill_after_staging_with_verbose_stderr(script: str) -> str:
    needle = "    copy_metadata(target, temporary, first_stat)\n"
    replacement = needle + '    os.write(2, b"x" * 4096)\n' + "    os._exit(9)\n"
    assert script.count(needle) == 1
    return script.replace(needle, replacement)


def _suppress_remote_result(script: str) -> str:
    needle = (
        "def emit(status, reason, *, count=None, before=None, after=None, "
        "recovery=None):\n"
    )
    replacement = needle + "    return\n"
    assert script.count(needle) == 1
    return script.replace(needle, replacement)


def test_file_edit_status_values_are_complete_and_distinct() -> None:
    assert [(item.name, item.value) for item in FileEditStatus] == [
        ("APPLIED", "applied"),
        ("REFUSED", "refused"),
        ("COULD_NOT_DETERMINE", "could_not_determine"),
    ]


@pytest.mark.parametrize(
    ("content", "status", "reason", "count", "expected"),
    [
        (b"zero\n", "refused", "match_not_found", 0, b"zero\n"),
        (
            b"before OLD after\n",
            "applied",
            "unique_match_replaced",
            1,
            b"before NEW after\n",
        ),
        (
            b"OLD one OLD two OLD\n",
            "refused",
            "match_not_unique",
            3,
            b"OLD one OLD two OLD\n",
        ),
    ],
)
async def test_zero_one_many_match_matrix(
    tmp_path: Path,
    content: bytes,
    status: str,
    reason: str,
    count: int,
    expected: bytes,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(content)

    outcome = await _run_edit(workspace, _edit_call())

    assert target.read_bytes() == expected
    assert _result(outcome)["status"] == status
    assert _result(outcome)["reason"] == reason
    assert _result(outcome)["match_count"] == count
    assert (outcome.error is None) is (status == "applied")


async def test_unique_edit_preserves_every_other_byte_and_reports_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    before_lines = [f"line {number:03d}: untouched\n" for number in range(320)]
    before_lines[173] = "line 173: OLD payload\n"
    target.write_text("".join(before_lines), encoding="utf-8")
    expected_lines = list(before_lines)
    expected_lines[173] = "line 173: NEW payload\n"

    outcome = await _run_edit(workspace, _edit_call())

    assert target.read_text(encoding="utf-8") == "".join(expected_lines)
    result = _result(outcome)
    assert result["status"] == "applied"
    assert result["reason"] == "unique_match_replaced"
    assert result["file_bytes_before"] == 6402
    assert result["file_bytes_after"] == 6402
    assert result["before_sha256"] == (
        "66c5935909d0f23408afb126bfdf12e021e0fc8d864f974a7aced4620833f8eb"
    )
    assert result["after_sha256"] == (
        "0966781f49f7edb444b81054e2b26d3165eae0ed5de304537f055968564c74c6"
    )
    assert outcome.changed_paths == ("target.txt",)


async def test_configured_visibility_limit_admits_files_above_legacy_16k(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    content = b"x" * 21_000 + b"OLD\n"
    target.write_bytes(content)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        max_tool_output_chars=64_000,
    )

    assert target.read_bytes() == b"x" * 21_000 + b"NEW\n"
    assert _result(outcome)["status"] == "applied"
    assert outcome.tool_audits[0]["limits"]["file_bytes"] == 64_000


def test_applied_result_requires_complete_evidence() -> None:
    with pytest.raises(
        ValueError,
        match="an applied edit requires complete before/after evidence",
    ):
        FileEditResult(
            FileEditStatus.APPLIED,
            "unique_match_replaced",
            "target.txt",
            match_count=1,
        )


def test_applied_result_rejects_a_recovery_path() -> None:
    with pytest.raises(
        ValueError,
        match="an applied edit requires complete before/after evidence",
    ):
        FileEditResult(
            FileEditStatus.APPLIED,
            "unique_match_replaced",
            "target.txt",
            match_count=1,
            file_bytes_before=3,
            file_bytes_after=3,
            before_sha256=(
                "099d90cbee62f89e6478e153eb3240efcbe4ac2231bedc3e84549bbeaaba87e8"
            ),
            after_sha256=(
                "a253ff09c5a8678e1fd1962b2c329245e139e45f9cc6ced4e5d7ad42c4108fc0"
            ),
            recovery_path=".driftlock-edit-recovery-1-literal",
        )


async def test_mode_preservation_is_load_bearing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD\n")
    target.chmod(0o754)

    outcome = await _run_edit(workspace, _edit_call())

    assert _result(outcome)["status"] == "applied"
    assert stat.S_IMODE(target.stat().st_mode) == 0o754


async def test_xattrs_are_preserved_when_supported(tmp_path: Path) -> None:
    if not all(hasattr(os, name) for name in ("setxattr", "getxattr")):
        pytest.skip("platform has no xattr API")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD\n")
    try:
        os.setxattr(target, "user.note", b"literal metadata")
    except OSError as error:
        pytest.skip(f"test filesystem does not permit user xattrs: {error}")

    outcome = await _run_edit(workspace, _edit_call())

    assert _result(outcome)["status"] == "applied"
    assert os.getxattr(target, "user.note") == b"literal metadata"


async def test_group_is_preserved_when_an_alternate_group_is_available(
    tmp_path: Path,
) -> None:
    alternate_groups = [group for group in os.getgroups() if group != os.getegid()]
    if not alternate_groups:
        pytest.skip("process has no alternate group for a preservation probe")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD\n")
    try:
        os.chown(target, -1, alternate_groups[0])
    except PermissionError as error:
        pytest.skip(f"process cannot assign an alternate group: {error}")

    outcome = await _run_edit(workspace, _edit_call())

    assert _result(outcome)["status"] == "applied"
    assert target.stat().st_gid == alternate_groups[0]


async def test_identical_replacement_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"before OLD after\n")

    outcome = await _run_edit(workspace, _edit_call(new_text="OLD"))

    assert target.read_bytes() == b"before OLD after\n"
    assert _result(outcome)["status"] == "refused"
    assert _result(outcome)["reason"] == "replacement_is_identical"
    assert _result(outcome)["match_count"] == 1


async def test_concurrent_writer_is_not_overwritten_or_mixed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD" + b"x" * 15_000)
    concurrent = b"concurrent writer owns this content\n"
    environment = _ConcurrentEnvironment(workspace, target, concurrent)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    assert environment.writer_ran is True
    assert target.read_bytes() == concurrent
    assert _result(outcome)["status"] == "could_not_determine"
    assert _result(outcome)["reason"] in {
        "file_changed_before_atomic_replace",
        "file_changed_after_atomic_replace",
    }


async def test_displaced_inode_guard_alone_restores_writer_bytes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _write_displaced_inode)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    assert target.read_bytes() == b"WRITER"
    assert _result(outcome)["status"] == "could_not_determine"
    assert _result(outcome)["reason"] == "file_changed_at_atomic_exchange"
    assert not tuple(workspace.glob(".driftlock-edit-stage-*"))


async def test_post_exchange_guard_alone_detects_installed_file_change(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _write_installed_inode)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    assert target.read_bytes() == b"WRITER"
    result = _result(outcome)
    assert result["status"] == "could_not_determine"
    assert result["reason"] == "file_changed_after_atomic_replace"
    assert result["after_sha256"] == (
        "8808adca7c367de1dabbd54e69b19db6da655fc127b72efdf04701bdf0cfca5d"
    )
    recovery = result["recovery_path"]
    assert isinstance(recovery, str)
    assert recovery.startswith(".driftlock-edit-recovery-")
    assert (workspace / recovery).read_bytes() == b"OLD"


async def test_non_regular_installed_target_is_detected_independently(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(
        workspace,
        _install_symlink_with_edited_bytes,
    )

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    result = _result(outcome)
    assert target.is_symlink()
    assert target.read_bytes() == b"NEW"
    assert result["status"] == "could_not_determine"
    assert result["reason"] == "file_changed_after_atomic_replace"
    recovery = result["recovery_path"]
    assert isinstance(recovery, str)
    assert recovery.startswith(".driftlock-edit-recovery-")
    assert (workspace / recovery).read_bytes() == b"OLD"


async def test_failed_swap_back_restores_displaced_bytes_with_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _fail_second_exchange)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    result = _result(outcome)
    assert target.read_bytes() == b"WRITER"
    assert result["status"] == "could_not_determine"
    assert str(result["reason"]).startswith(
        "atomic_swap_back_failed_restored_with_replace_errno_"
    )
    assert result["before_sha256"] == (
        "099d90cbee62f89e6478e153eb3240efcbe4ac2231bedc3e84549bbeaaba87e8"
    )
    assert result["after_sha256"] == (
        "8808adca7c367de1dabbd54e69b19db6da655fc127b72efdf04701bdf0cfca5d"
    )
    assert result["recovery_path"] is None
    assert not tuple(workspace.glob(".driftlock-edit-*"))


async def test_failed_swap_back_and_fallback_preserve_displaced_recovery(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _fail_all_restore_operations)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    result = _result(outcome)
    assert target.read_bytes() == b"NEW"
    assert str(result["reason"]).startswith("atomic_restore_failed_errno_")
    assert result["before_sha256"] == (
        "099d90cbee62f89e6478e153eb3240efcbe4ac2231bedc3e84549bbeaaba87e8"
    )
    assert result["after_sha256"] == (
        "a253ff09c5a8678e1fd1962b2c329245e139e45f9cc6ced4e5d7ad42c4108fc0"
    )
    recovery = result["recovery_path"]
    assert isinstance(recovery, str)
    assert (workspace / recovery).read_bytes() == b"WRITER"


async def test_post_exchange_oserror_preserves_named_recovery_across_later_edits(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _fail_after_exchange)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    result = _result(outcome)
    assert target.read_bytes() == b"NEW"
    assert result["reason"] == "remote_file_operation_failed_errno_5"
    assert result["before_sha256"] == (
        "099d90cbee62f89e6478e153eb3240efcbe4ac2231bedc3e84549bbeaaba87e8"
    )
    assert result["after_sha256"] == (
        "a253ff09c5a8678e1fd1962b2c329245e139e45f9cc6ced4e5d7ad42c4108fc0"
    )
    recovery = result["recovery_path"]
    assert isinstance(recovery, str)
    assert recovery.startswith(".driftlock-edit-recovery-")
    recovery_file = workspace / recovery
    assert recovery_file.read_bytes() == b"OLD"

    later = await _run_edit(workspace, _edit_call("NEW", "NEXT"))

    assert _result(later)["status"] == "applied"
    assert target.read_bytes() == b"NEXT"
    assert recovery_file.read_bytes() == b"OLD"


async def test_failed_recovery_rename_never_advertises_a_sweep_owned_name(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _fail_recovery_rename)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    result = _result(outcome)
    stages = tuple(workspace.glob(".driftlock-edit-stage-*"))
    assert target.read_bytes() == b"NEW"
    assert result["reason"] == "remote_file_operation_failed_errno_5"
    assert result["recovery_path"] is None
    assert len(stages) == 1
    assert stages[0].read_bytes() == b"OLD"

    later = await _run_edit(workspace, _edit_call("NEW", "NEXT"))

    assert _result(later)["status"] == "applied"
    assert stages[0].exists() is False


async def test_concurrent_write_during_swap_back_has_its_own_reason(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _change_restored_file)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    result = _result(outcome)
    assert target.read_bytes() == b"LATER"
    assert result["reason"] == "file_changed_while_restoring_concurrent_write"
    assert result["after_sha256"] == (
        "8a3e05d0736c69c89307714151e94b211f071b402c3ae5f4ef6dab9267930ae0"
    )


async def test_swap_back_fallback_unreadable_restore_has_its_own_reason(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(
        workspace,
        _make_fallback_restore_unreadable,
    )

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    result = _result(outcome)
    assert target.read_bytes() == b"WRITER"
    assert result["reason"] == "atomic_swap_back_failed_restore_unreadable_errno_5"
    assert result["before_sha256"] == (
        "099d90cbee62f89e6478e153eb3240efcbe4ac2231bedc3e84549bbeaaba87e8"
    )
    assert result["after_sha256"] is None


async def test_file_beyond_read_limit_is_not_edited(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    content = b"a" * 128 + b"OLD"
    target.write_bytes(content)

    outcome = await _run_edit(workspace, _edit_call(), max_tool_output_chars=128)

    assert target.read_bytes() == content
    assert _result(outcome)["status"] == "could_not_determine"
    assert _result(outcome)["reason"] == "file_byte_limit_exceeded"
    assert outcome.tool_audits[0]["limits"]["file_bytes"] == 128


async def test_absolute_file_memory_cap_is_recorded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    content = b"a" * MAX_EDIT_FILE_BYTES + b"OLD"
    target.write_bytes(content)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        max_tool_output_chars=MAX_EDIT_FILE_BYTES * 2,
    )

    assert target.read_bytes() == content
    assert _result(outcome)["reason"] == "file_byte_limit_exceeded"
    assert outcome.tool_audits[0]["limits"]["file_bytes"] == MAX_EDIT_FILE_BYTES


async def test_result_file_size_cap_is_not_crossed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    content = b"OLD" + b"a" * 125
    target.write_bytes(content)

    outcome = await _run_edit(
        workspace,
        _edit_call(new_text="NEWER"),
        max_tool_output_chars=128,
    )

    assert target.read_bytes() == content
    assert _result(outcome)["status"] == "refused"
    assert _result(outcome)["reason"] == "result_file_byte_limit_exceeded"


async def test_non_utf8_file_is_not_edited(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    content = b"prefix\xffOLD suffix"
    target.write_bytes(content)

    outcome = await _run_edit(workspace, _edit_call())

    assert target.read_bytes() == content
    assert _result(outcome)["status"] == "could_not_determine"
    assert _result(outcome)["reason"] == "file_is_not_strict_utf8"


async def test_missing_target_has_distinct_errno_reason(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outcome = await _run_edit(workspace, _edit_call())

    assert _result(outcome)["status"] == "could_not_determine"
    assert str(_result(outcome)["reason"]).startswith("target_not_found_errno_")


async def test_unavailable_atomic_exchange_has_distinct_errno_reason(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _unavailable_exchange)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    assert target.read_bytes() == b"OLD"
    assert str(_result(outcome)["reason"]).startswith(
        "atomic_exchange_unavailable_errno_"
    )
    assert _result(outcome)["before_sha256"] == (
        "099d90cbee62f89e6478e153eb3240efcbe4ac2231bedc3e84549bbeaaba87e8"
    )
    assert not tuple(workspace.glob(".driftlock-edit-*"))


async def test_read_only_parent_has_distinct_errno_reason(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _read_only_failure)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    assert target.read_bytes() == b"OLD"
    assert str(_result(outcome)["reason"]).startswith("target_not_writable_errno_")


async def test_final_component_symlink_swap_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD target")
    destination = workspace / "real.txt"
    destination.write_bytes(b"OLD real")
    environment = _SymlinkSwapEnvironment(workspace, target, destination)

    outcome = await _run_edit(
        workspace,
        _edit_call(old_text="OLD target"),
        environment=environment,
    )

    assert environment.swapped is True
    assert target.is_symlink()
    assert destination.read_bytes() == b"OLD real"
    assert _result(outcome)["reason"] == "target_is_not_a_regular_file"


async def test_stale_stage_from_killed_process_is_swept(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    stale = workspace / ".driftlock-edit-stage-999999999-literal"
    stale.write_bytes(b"abandoned")

    outcome = await _run_edit(workspace, _edit_call())

    assert _result(outcome)["status"] == "applied"
    assert stale.exists() is False


async def test_failed_remote_process_sweeps_its_killed_stage(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _kill_after_staging)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    assert target.read_bytes() == b"OLD"
    assert _result(outcome)["reason"] == "remote_edit_process_failed"
    assert not tuple(workspace.glob(".driftlock-edit-stage-*"))


async def test_cleanup_failure_precedes_verbose_remote_diagnostic(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(
        workspace,
        _kill_after_staging_with_verbose_stderr,
        fail_cleanup=True,
    )

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    result = _result(outcome)
    assert result["reason"] == "remote_edit_process_failed"
    assert str(result["detail"]).startswith("stale edit-stage cleanup failed:")
    assert "forced cleanup failure" in str(result["detail"])
    assert len(str(result["detail"])) == 1_000
    assert len(tuple(workspace.glob(".driftlock-edit-stage-*"))) == 1


async def test_empty_success_output_is_a_recorded_malformed_result(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(workspace, _suppress_remote_result)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    result = _result(outcome)
    assert target.read_bytes() == b"NEW"
    assert result["status"] == "could_not_determine"
    assert result["reason"] == "malformed_remote_edit_result"
    assert result["detail"] == (
        "ValueError: remote edit output must contain exactly one result record"
    )


async def test_stale_stage_scan_cap_is_recorded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    for index in range(MAX_EDIT_STALE_FILES + 1):
        (workspace / f".driftlock-edit-stage-999999999-{index}").write_bytes(b"x")

    outcome = await _run_edit(workspace, _edit_call())

    assert target.read_bytes() == b"OLD"
    assert str(_result(outcome)["reason"]).startswith(
        "stale_stage_limit_exceeded_errno_"
    )


async def test_merged_warning_output_does_not_hide_applied_result(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")
    environment = _ScriptTransformEnvironment(
        workspace,
        lambda script: script,
        decorate_stdout=True,
    )

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    assert target.read_bytes() == b"NEW"
    assert _result(outcome)["status"] == "applied"


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (
            _edit_call(old_text="x" * (MAX_EDIT_MATCH_CHARACTERS + 1)),
            "match_character_limit_exceeded",
        ),
        (
            _edit_call(new_text="x" * (MAX_EDIT_REPLACEMENT_CHARACTERS + 1)),
            "replacement_character_limit_exceeded",
        ),
    ],
)
async def test_text_caps_are_recorded(
    tmp_path: Path, call: ToolCall, reason: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")

    outcome = await _run_edit(workspace, call)

    assert target.read_bytes() == b"OLD"
    assert _result(outcome)["status"] == "refused"
    assert _result(outcome)["reason"] == reason
    assert outcome.tool_audits[0]["result"]["reason"] == reason


@pytest.mark.parametrize(
    ("arguments", "literal_error"),
    [
        (
            {"path": "target.txt", "old_text": "OLD"},
            "missing required argument(s): new_text",
        ),
        (
            {"path": "target.txt", "old_text": 4, "new_text": "NEW"},
            "old_text must be a non-empty string",
        ),
        (
            {"path": "target.txt", "old_text": "", "new_text": "NEW"},
            "old_text must be a non-empty string",
        ),
    ],
)
async def test_malformed_edit_is_a_recorded_tool_error(
    tmp_path: Path, arguments: dict[str, object], literal_error: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")

    outcome = await _run_edit(workspace, ToolCall("edit_file", arguments, "malformed"))

    assert target.read_bytes() == b"OLD"
    assert literal_error in (outcome.error or "")
    assert outcome.tool_observations == (
        f"edit_file:\nERROR: malformed arguments for edit_file: {literal_error}",
    )


async def test_runner_rollback_restores_content_from_before_edit(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD\n")
    provider = _ScriptedProvider(
        [
            AgentCompletion(tool_calls=(_edit_call(call_id="one"),), tokens=2),
            AgentCompletion(tool_calls=(_edit_call(call_id="two"),), tokens=2),
            AgentCompletion(
                tool_calls=(
                    ToolCall("complete", {"summary": "rolled back safely"}, "done"),
                ),
                tokens=2,
            ),
        ]
    )
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        edit_file=True,
    )

    class _DriftJudge:
        async def judge(self, _context: DriftContext) -> JudgeVerdict:
            return JudgeVerdict(Verdict.DRIFTED, "repeat edit should roll back")

    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "snapshots"),
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=10,
                loop_window=2,
                loop_repetitions=2,
                error_window=10,
                command_failure_window=10,
                reward_stall_steps=10,
            )
        ),
        fine_judge=_DriftJudge(),
        config=RunnerConfig(max_steps=3, max_rollbacks=1, checkpoint_interval=10),
    ).run(
        goal="edit and exercise rollback",
        step=agent,
        initial_state=agent.initial_state(),
    )

    assert result.status is RunStatus.COMPLETED
    assert result.steps[0].outcome.changed_paths == ("target.txt",)
    assert len(result.rollbacks) == 1
    assert target.read_bytes() == b"OLD\n"
