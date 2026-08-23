from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
from pathlib import Path

import pytest

from driftlock.local import LocalEnvironment, LocalWorkspaceDeltaObserver


async def test_local_environment_exec_captures_exit_and_streams(
    tmp_path: Path,
) -> None:
    environment = LocalEnvironment(tmp_path)

    result = await environment.exec("printf 'hello'; printf 'problem' >&2; exit 6")

    assert result.return_code == 6
    assert result.stdout == "hello"
    assert result.stderr == "problem"


async def test_local_environment_times_out_endless_output(tmp_path: Path) -> None:
    environment = LocalEnvironment(tmp_path, default_timeout_sec=1)

    result = await environment.exec(
        "python3 -c 'import time; "
        'exec("while True:\\n print(123, flush=True)\\n time.sleep(0.01)")\''
    )

    assert result.return_code == 124
    assert "123" in result.stdout
    assert "command timed out after 1 seconds" in result.stderr


async def test_local_environment_preserves_exit_of_pipe_holding_child(
    tmp_path: Path,
) -> None:
    script = """\
import subprocess

subprocess.Popen(["sleep", "30"])
print("started", flush=True)
"""
    environment = LocalEnvironment(tmp_path)

    result = await asyncio.wait_for(
        environment.exec(f"python3 -c {shlex.quote(script)}", timeout_sec=1),
        timeout=4,
    )

    assert result.return_code == 0
    assert result.stdout == "started\n"
    assert "timed out" not in result.stderr


async def test_local_environment_home_persists_until_close_and_stays_isolated(
    tmp_path: Path,
) -> None:
    environment = LocalEnvironment(tmp_path)
    write_script = """\
import pathlib

target = pathlib.Path.home() / ".config" / "identity"
target.parent.mkdir(parents=True)
target.write_text("a@b.c", encoding="utf-8")
print(target)
"""

    written = await environment.exec(f"python3 -c {shlex.quote(write_script)}")
    read_back = await environment.exec(
        "python3 -c 'import pathlib; "
        'print((pathlib.Path.home() / ".config" / "identity").read_text())\''
    )
    runtime_file = Path(written.stdout.strip())

    assert written.return_code == 0
    assert read_back.return_code == 0
    assert read_back.stdout == "a@b.c\n"
    assert tuple(tmp_path.iterdir()) == ()
    assert not runtime_file.is_relative_to(tmp_path)

    environment.close()

    assert not runtime_file.exists()
    with pytest.raises(RuntimeError, match="closed"):
        await environment.exec("true")


async def test_local_environment_upload_download_and_path_refusal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")
    destination = tmp_path / "downloaded.txt"
    environment = LocalEnvironment(root)

    await environment.upload_file(source, "nested/remote.txt")
    await environment.download_file("nested/remote.txt", destination)

    assert (root / "nested" / "remote.txt").read_text(encoding="utf-8") == "payload"
    assert destination.read_text(encoding="utf-8") == "payload"
    with pytest.raises(ValueError, match="outside"):
        await environment.upload_file(source, "../escaped.txt")
    with pytest.raises(ValueError, match="outside"):
        await environment.download_file("../source.txt", destination)


async def test_local_environment_refuses_symlink_escape_and_user_switch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")
    environment = LocalEnvironment(root)

    with pytest.raises(ValueError, match="outside"):
        await environment.upload_file(source, "escape/copied.txt")
    with pytest.raises(ValueError, match="user switching"):
        await environment.exec("id", user="root")


async def test_local_environment_timeout_closes_daemon_held_pipes(
    tmp_path: Path,
) -> None:
    script = """\
import os
import pathlib
import time

child = os.fork()
if child == 0:
    os.setsid()
    pathlib.Path("daemon.pid").write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(30)
else:
    time.sleep(30)
"""
    environment = LocalEnvironment(tmp_path)
    pid_path = tmp_path / "daemon.pid"

    try:
        result = await asyncio.wait_for(
            environment.exec(f"python3 -c {shlex.quote(script)}", timeout_sec=1),
            timeout=4,
        )
    finally:
        if pid_path.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_path.read_text(encoding="utf-8")), signal.SIGKILL)

    assert result.return_code == 124
    assert "command timed out after 1 seconds" in result.stderr


async def test_local_environment_timeout_ignores_reaped_process_group(
    tmp_path: Path,
) -> None:
    script = """\
import os
import pathlib
import time

child = os.fork()
if child == 0:
    os.setsid()
    pathlib.Path("daemon.pid").write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(30)
"""
    environment = LocalEnvironment(tmp_path)
    pid_path = tmp_path / "daemon.pid"

    try:
        result = await asyncio.wait_for(
            environment.exec(f"python3 -c {shlex.quote(script)}", timeout_sec=1),
            timeout=4,
        )
    finally:
        if pid_path.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_path.read_text(encoding="utf-8")), signal.SIGKILL)

    assert result.return_code == 124
    assert "command timed out after 1 seconds" in result.stderr


async def test_local_environment_runtime_files_are_outside_observed_workspace(
    tmp_path: Path,
) -> None:
    script = """\
import pathlib
import tempfile

temporary = tempfile.NamedTemporaryFile(delete=False)
temporary.close()
cache = pathlib.Path.home() / ".cache" / "probe"
cache.parent.mkdir(parents=True)
cache.write_text("cache", encoding="utf-8")
print(temporary.name)
print(cache)
"""
    environment = LocalEnvironment(tmp_path)
    observer = LocalWorkspaceDeltaObserver(tmp_path)
    before = await observer.snapshot()

    result = await environment.exec(f"python3 -c {shlex.quote(script)}")
    after = await observer.snapshot()

    assert result.return_code == 0
    assert observer.compare(before, after).changed_paths == ()
    assert tuple(tmp_path.iterdir()) == ()
    for rendered_path in result.stdout.splitlines():
        assert not Path(rendered_path).resolve().is_relative_to(tmp_path.resolve())
