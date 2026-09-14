from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import driftlock.native_lhtb as native_lhtb
from driftlock.local import LocalEnvironment
from driftlock.native_lhtb import NativeProcessQuiescer
from driftlock.processes import (
    PROCESS_TABLE_KERNEL_PREFIX,
    PROCESS_TABLE_SNAPSHOT_PREFIX,
    ProcessIdentitySnapshot,
    ProcessTableStatus,
    parse_process_identity_snapshot,
)


@pytest.mark.parametrize(
    ("status", "identities", "accepted"),
    [
        ("observed", (), True),
        ("observed", ("2:100",), True),
        ("absent", (), True),
        ("absent", ("2:100",), False),
        ("unreadable", (), True),
        ("unreadable", ("2:100",), False),
    ],
)
def test_process_snapshot_status_identity_space(
    status: str, identities: tuple[str, ...], accepted: bool
) -> None:
    output = "\n".join((f"{PROCESS_TABLE_SNAPSHOT_PREFIX}{status}", *identities))

    if not accepted:
        with pytest.raises(
            ValueError, match="unavailable process snapshot cannot contain identities"
        ):
            parse_process_identity_snapshot(output)
        return

    assert parse_process_identity_snapshot(output) == ProcessIdentitySnapshot(
        ProcessTableStatus(status), identities
    )


def test_headerless_empty_snapshot_is_unobservable_not_observed_empty() -> None:
    assert parse_process_identity_snapshot("") == ProcessIdentitySnapshot(
        ProcessTableStatus.UNREADABLE, ()
    )


async def test_prepare_observes_empty_readable_process_table(tmp_path: Path) -> None:
    process_root = tmp_path / "proc"
    process_root.mkdir()

    with LocalEnvironment(tmp_path) as environment:
        result = await NativeProcessQuiescer(
            environment,
            user=None,
            process_table_root=str(process_root),
        ).prepare()

    assert result.status is ProcessTableStatus.OBSERVED
    assert result.identities == ()
    assert result.kernel in {"Darwin", "Linux"}


async def test_pid_one_emitted_by_snapshot_is_validated_as_baseline(
    tmp_path: Path,
) -> None:
    process_root = tmp_path / "proc"
    pid_root = process_root / "1"
    pid_root.mkdir(parents=True)
    (pid_root / "stat").write_text(
        "1 (init) S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12345 0\n",
        encoding="utf-8",
    )

    with LocalEnvironment(tmp_path) as environment:
        result = await NativeProcessQuiescer(
            environment,
            user=None,
            process_table_root=str(process_root),
        ).prepare()

    assert result.status is ProcessTableStatus.OBSERVED
    assert result.identities == ("1:12345",)
    assert result.kernel in {"Darwin", "Linux"}
    assert native_lhtb._valid_process_identity("1:12345") is True


async def test_prepare_distinguishes_absent_process_table(tmp_path: Path) -> None:
    with LocalEnvironment(tmp_path) as environment:
        result = await NativeProcessQuiescer(
            environment,
            user=None,
            process_table_root=str(tmp_path / "missing-proc"),
        ).prepare()

    assert result.status is ProcessTableStatus.ABSENT
    assert result.identities == ()
    assert result.kernel in {"Darwin", "Linux"}


async def test_prepare_distinguishes_unreadable_process_table(tmp_path: Path) -> None:
    process_root = tmp_path / "proc"
    process_root.mkdir()
    process_root.chmod(0)
    try:
        with LocalEnvironment(tmp_path) as environment:
            result = await NativeProcessQuiescer(
                environment,
                user=None,
                process_table_root=str(process_root),
            ).prepare()
    finally:
        process_root.chmod(0o700)

    assert result.status is ProcessTableStatus.UNREADABLE
    assert result.identities == ()
    assert result.kernel in {"Darwin", "Linux"}


class _SnapshotEnvironment:
    def __init__(self, *, status: str, kernel: str) -> None:
        self.output = (
            f"{PROCESS_TABLE_SNAPSHOT_PREFIX}{status}\n"
            f"{PROCESS_TABLE_KERNEL_PREFIX}{kernel}\n"
        )
        self.calls = 0

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> object:
        del command, timeout_sec, user
        self.calls += 1
        if self.calls > 1:
            return type(
                "CleanupResult",
                (),
                {"return_code": 0, "stdout": "", "stderr": ""},
            )()
        return type(
            "SnapshotResult",
            (),
            {"return_code": 0, "stdout": self.output, "stderr": ""},
        )()


async def test_before_restore_records_darwin_absence_and_proceeds() -> None:
    environment = _SnapshotEnvironment(status="absent", kernel="Darwin")
    quiescer = NativeProcessQuiescer(environment, user=None)

    await quiescer.prepare()
    await quiescer.before_restore("/workspace")

    assert environment.calls == 2
    assert quiescer.observation_report() == {
        "status": "absent",
        "kernel": "Darwin",
        "baseline_process_count": 0,
        "observation_degraded": True,
        "rollback_process_cleanup": "unavailable_recorded",
    }


async def test_before_restore_refuses_unexpected_linux_absence() -> None:
    environment = _SnapshotEnvironment(status="absent", kernel="Linux")
    quiescer = NativeProcessQuiescer(environment, user=None)

    await quiescer.prepare()

    with pytest.raises(RuntimeError, match="process table was absent"):
        await quiescer.before_restore("/workspace")


def test_process_snapshot_script_is_valid_shell_for_quoted_root(tmp_path: Path) -> None:
    process_root = tmp_path / "proc root'quoted"
    script = native_lhtb._process_identity_snapshot_script(str(process_root))

    result = subprocess.run(
        ["sh", "-n"], input=script, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
