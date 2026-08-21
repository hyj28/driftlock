from __future__ import annotations

from pathlib import Path

import pytest

from driftlock.local import LocalEnvironment


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
