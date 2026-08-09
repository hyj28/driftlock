from __future__ import annotations

from pathlib import Path

import pytest

from driftlock.checkpoints import DirectoryCheckpointStore, SnapshotIntegrityError


def test_checkpoint_restores_workspace_and_agent_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("healthy", encoding="utf-8")
    (workspace / "nested").mkdir()
    (workspace / "nested" / "data.txt").write_text("one", encoding="utf-8")
    store = DirectoryCheckpointStore(workspace, tmp_path / "snapshots")

    checkpoint = store.create({"turn": 2, "messages": ["a"]}, step=2)
    (workspace / "answer.txt").write_text("drifted", encoding="utf-8")
    (workspace / "nested" / "data.txt").unlink()
    (workspace / "new.txt").write_text("remove me", encoding="utf-8")

    state = store.restore(checkpoint)

    assert state == {"turn": 2, "messages": ["a"]}
    assert (workspace / "answer.txt").read_text(encoding="utf-8") == "healthy"
    assert (workspace / "nested" / "data.txt").read_text(encoding="utf-8") == "one"
    assert not (workspace / "new.txt").exists()


def test_checkpoint_detects_snapshot_tampering(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("healthy", encoding="utf-8")
    store = DirectoryCheckpointStore(workspace, tmp_path / "snapshots")
    checkpoint = store.create({}, step=0)
    (checkpoint.path / "workspace" / "answer.txt").write_text(
        "tampered", encoding="utf-8"
    )

    with pytest.raises(SnapshotIntegrityError):
        store.restore(checkpoint)


def test_checkpoint_detects_agent_state_tampering(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DirectoryCheckpointStore(workspace, tmp_path / "snapshots")
    checkpoint = store.create({"turn": 1}, step=0)
    (checkpoint.path / "state.json").write_text('{"turn": 2}', encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError):
        store.restore(checkpoint)


def test_store_must_be_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside"):
        DirectoryCheckpointStore(workspace, workspace / ".driftlock")


def test_checkpoint_includes_repository_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / ".git" / "index").write_text("metadata", encoding="utf-8")
    (workspace / "tracked.txt").write_text("content", encoding="utf-8")
    store = DirectoryCheckpointStore(workspace, tmp_path / "snapshots")

    checkpoint = store.create({}, step=0)
    (workspace / ".git" / "index").write_text("changed", encoding="utf-8")
    store.restore(checkpoint)

    assert (checkpoint.path / "workspace" / ".git" / "index").exists()
    assert (workspace / ".git" / "index").read_text(encoding="utf-8") == "metadata"
    assert (workspace / "tracked.txt").exists()
