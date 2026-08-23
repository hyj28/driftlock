from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from driftlock.oracle import (
    OracleCheckpointError,
    ReplayUsage,
    load_remote_checkpoint_bundle,
)


def _checkpoint(root: Path, *, workspace: str = "/app") -> Path:
    checkpoint_id = "a" * 32
    directory = root / "checkpoints" / checkpoint_id
    directory.mkdir(parents=True)
    archive = b"retained remote tar bytes"
    state_text = json.dumps({"messages": ["one"]}, separators=(",", ":"))
    digest = hashlib.sha256(archive)
    digest.update(b"\0state\0")
    digest.update(state_text.encode())
    (directory / "workspace.tar.gz").write_bytes(archive)
    (directory / "state.json").write_text(state_text, encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "step": 7,
                "created_at": datetime.now(UTC).isoformat(),
                "digest": digest.hexdigest(),
                "parent_id": None,
                "label": "step-7",
                "remote_workspace": workspace,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_load_remote_checkpoint_bundle_binds_archive_state_and_workspace(
    tmp_path: Path,
) -> None:
    directory = _checkpoint(tmp_path)
    expected = json.loads((directory / "manifest.json").read_text())["digest"]

    bundle = load_remote_checkpoint_bundle(
        directory, expected_digest=expected, expected_workspace="/app"
    )

    assert bundle.checkpoint.step == 7
    assert bundle.state == {"messages": ["one"]}
    assert bundle.remote_workspace == "/app"
    assert len(bundle.archive_sha256) == 64
    assert len(bundle.state_sha256) == 64


def test_load_remote_checkpoint_bundle_accepts_recorded_unstable_paths(
    tmp_path: Path,
) -> None:
    directory = _checkpoint(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unstable_paths"] = ["./output/live.log"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    bundle = load_remote_checkpoint_bundle(directory)

    assert bundle.checkpoint.unstable_paths == ("./output/live.log",)


@pytest.mark.parametrize("name", ["workspace.tar.gz", "state.json", "manifest.json"])
def test_load_remote_checkpoint_bundle_rejects_symlinked_inputs(
    tmp_path: Path, name: str
) -> None:
    directory = _checkpoint(tmp_path)
    target = tmp_path / f"saved-{name}"
    (directory / name).rename(target)
    (directory / name).symlink_to(target)

    with pytest.raises(OracleCheckpointError, match="symlink"):
        load_remote_checkpoint_bundle(directory)


def test_load_remote_checkpoint_bundle_rejects_tampering(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path)
    (directory / "state.json").write_text("{}", encoding="utf-8")

    with pytest.raises(OracleCheckpointError, match="integrity"):
        load_remote_checkpoint_bundle(directory)


@pytest.mark.parametrize("workspace", ["/", "relative", "/app/../other", "/app/"])
def test_load_remote_checkpoint_bundle_rejects_unsafe_workspace(
    tmp_path: Path, workspace: str
) -> None:
    directory = _checkpoint(tmp_path, workspace=workspace)

    with pytest.raises(OracleCheckpointError, match="workspace"):
        load_remote_checkpoint_bundle(directory)


def test_replay_usage_is_strict_and_conservative() -> None:
    usage = ReplayUsage.from_mapping(
        {
            "input_tokens": 100,
            "cache_tokens": 20,
            "output_tokens": 5,
            "cost_usd": 1.25,
        }
    )

    assert usage.as_dict()["input_tokens"] == 100
    with pytest.raises(ValueError, match="cache_tokens"):
        ReplayUsage(1, 2, 3, 0.0)
    with pytest.raises(ValueError, match="input_tokens"):
        ReplayUsage(True, 0, 0, 0.0)
