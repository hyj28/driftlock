"""Audited inputs for isolated hindsight-oracle checkpoint replay."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from driftlock.checkpoints import SnapshotIntegrityError
from driftlock.lhtb import openrouter_provider_from_call_kwargs
from driftlock.models import Checkpoint
from driftlock.usage import ReplayUsage, UsageAccountingError, load_trial_usage

_HEX_32 = re.compile(r"[0-9a-f]{32}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_FIELDS = {
    "checkpoint_id",
    "step",
    "created_at",
    "digest",
    "parent_id",
    "label",
    "remote_workspace",
}
_OPTIONAL_MANIFEST_FIELDS = {"unstable_paths"}


class OracleCheckpointError(SnapshotIntegrityError):
    """Raised when a retained checkpoint is unsafe or lacks audit provenance."""


@dataclass(frozen=True, slots=True)
class RemoteCheckpointBundle:
    """A verified retained remote checkpoint plus its serialized agent state."""

    checkpoint: Checkpoint
    state: dict[str, Any]
    remote_workspace: str
    archive_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class SourceTrialProvenance:
    """Identity and accounting re-derived from a hashed Harbor source result."""

    trial_id: str
    task_name: str
    model_name: str
    usage: ReplayUsage
    usage_source: str
    result_path: Path
    result_sha256: str
    data: dict[str, Any]


def file_sha256(path: Path | str) -> str:
    """Return a streaming SHA-256 for an ordinary, non-symlink file."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise OracleCheckpointError(f"expected an ordinary file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_trial_provenance(
    result_path: Path | str, *, expected_sha256: str | None = None
) -> SourceTrialProvenance:
    """Load source identity and usage from the result artifact that binds them."""
    path = Path(result_path).expanduser()
    if path.parent.is_symlink():
        raise OracleCheckpointError("source trial directory cannot be a symlink")
    path = path.resolve()
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise OracleCheckpointError("source result differs from replay provenance")
    data = _read_object(path, "source trial result")
    trial_id = data.get("id")
    try:
        parsed_trial_id = UUID(trial_id)
    except (TypeError, ValueError) as error:
        raise OracleCheckpointError("source trial id is invalid") from error
    if str(parsed_trial_id) != trial_id:
        raise OracleCheckpointError("source trial id is not canonical")
    task_name = data.get("task_name")
    if not isinstance(task_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
        task_name,
    ):
        raise OracleCheckpointError("source task name is invalid")
    config = data.get("config")
    agent = config.get("agent") if isinstance(config, dict) else None
    kwargs = agent.get("kwargs") if isinstance(agent, dict) else None
    model_name = agent.get("model_name") if isinstance(agent, dict) else None
    if (
        not isinstance(kwargs, dict)
        or agent.get("import_path") != "driftlock.harbor_agent:LHTBDriftlockAgent"
        or not isinstance(model_name, str)
        or not model_name
    ):
        raise OracleCheckpointError("source result is not a driftlock trial")
    info = data.get("agent_info")
    model_info = info.get("model_info") if isinstance(info, dict) else None
    provider, separator, name = model_name.partition("/")
    if not separator:
        provider, name = None, provider
    if (
        not isinstance(info, dict)
        or info.get("name") != "driftlock-terminus-2"
        or not isinstance(info.get("version"), str)
        or model_info != {"provider": provider, "name": name}
    ):
        raise OracleCheckpointError("source agent identity is inconsistent")
    try:
        trial_usage = load_trial_usage(
            data,
            path,
            provider=lambda: _source_agent_provider(agent, path),
        )
    except UsageAccountingError as error:
        raise OracleCheckpointError(str(error)) from error
    return SourceTrialProvenance(
        trial_id=trial_id,
        task_name=task_name,
        model_name=model_name,
        usage=trial_usage.usage,
        usage_source=trial_usage.source,
        result_path=path,
        result_sha256=digest,
        data=data,
    )


def validate_checkpoint_source_audit(
    checkpoint_dir: Path | str,
    *,
    source_result: Path | str,
    source_audit: Path | str,
    expected_audit_sha256: str,
) -> dict[str, Any]:
    """Bind a candidate directory to its source trial and retained phase audit."""
    result_path = Path(source_result).expanduser().resolve()
    trial_dir = result_path.parent
    checkpoint = Path(checkpoint_dir).expanduser()
    if checkpoint.is_symlink():
        raise OracleCheckpointError("checkpoint directory cannot be a symlink")
    checkpoint = checkpoint.resolve()
    phase_dir = checkpoint.parent.parent
    expected_root = trial_dir / ".driftlock-checkpoints"
    if (
        checkpoint.parent.name != "checkpoints"
        or phase_dir.parent != expected_root
        or not re.fullmatch(r"phase-[0-9]+", phase_dir.name)
    ):
        raise OracleCheckpointError(
            "checkpoint is outside the source trial retained tree"
        )
    audit_path = Path(source_audit).expanduser()
    if audit_path.is_symlink() or audit_path.resolve() != trial_dir / "agent" / (
        "driftlock-result.json"
    ):
        raise OracleCheckpointError("source audit path is not canonical")
    audit_path = audit_path.resolve()
    if file_sha256(audit_path) != expected_audit_sha256:
        raise OracleCheckpointError("source audit differs from replay provenance")
    audit = _read_object(audit_path, "source checkpoint audit")
    phases = audit.get("phases")
    if not isinstance(phases, list):
        raise OracleCheckpointError("source checkpoint audit lacks phases")
    matches = [
        phase
        for phase in phases
        if isinstance(phase, dict)
        and isinstance(phase.get("checkpoint_dir"), str)
        and Path(phase["checkpoint_dir"]).expanduser().resolve() == phase_dir
    ]
    if len(matches) != 1:
        raise OracleCheckpointError("checkpoint phase is absent from source audit")
    phase = matches[0]
    actual_count = sum(
        1
        for candidate in checkpoint.parent.iterdir()
        if not candidate.is_symlink() and candidate.is_dir()
    )
    if (
        phase.get("checkpoints_retained") is not True
        or phase.get("checkpoint_count") != actual_count
        or actual_count <= 0
    ):
        raise OracleCheckpointError("source checkpoint audit count is inconsistent")
    return phase


def load_remote_checkpoint_bundle(
    checkpoint_dir: Path | str,
    *,
    expected_digest: str | None = None,
    expected_workspace: str | None = None,
) -> RemoteCheckpointBundle:
    """Strictly validate a retained checkpoint before any remote mutation."""
    directory = Path(checkpoint_dir).expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise OracleCheckpointError("checkpoint directory is missing or a symlink")
    directory = directory.resolve()
    manifest_path = _direct_file(directory, "manifest.json")
    state_path = _direct_file(directory, "state.json")
    archive_path = _direct_file(directory, "workspace.tar.gz")
    manifest = _read_object(manifest_path, "checkpoint manifest")
    if (
        not _MANIFEST_FIELDS.issubset(manifest)
        or set(manifest) - _MANIFEST_FIELDS - _OPTIONAL_MANIFEST_FIELDS
    ):
        raise OracleCheckpointError("checkpoint manifest has unexpected fields")

    checkpoint_id = manifest["checkpoint_id"]
    if (
        not isinstance(checkpoint_id, str)
        or not _HEX_32.fullmatch(checkpoint_id)
        or checkpoint_id != directory.name
    ):
        raise OracleCheckpointError("checkpoint id does not match its directory")
    step = manifest["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise OracleCheckpointError("checkpoint step must be a nonnegative integer")
    created_at_raw = manifest["created_at"]
    try:
        created_at = datetime.fromisoformat(created_at_raw)
    except (TypeError, ValueError) as error:
        raise OracleCheckpointError("checkpoint creation time is invalid") from error
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise OracleCheckpointError("checkpoint creation time must include a timezone")

    digest = manifest["digest"]
    if not isinstance(digest, str) or not _HEX_64.fullmatch(digest):
        raise OracleCheckpointError("checkpoint digest is invalid")
    if expected_digest is not None and digest != expected_digest:
        raise OracleCheckpointError("checkpoint digest differs from replay provenance")
    parent_id = manifest["parent_id"]
    if parent_id is not None and (
        not isinstance(parent_id, str) or not _HEX_32.fullmatch(parent_id)
    ):
        raise OracleCheckpointError("checkpoint parent id is invalid")
    label = manifest["label"]
    if label is not None and not isinstance(label, str):
        raise OracleCheckpointError("checkpoint label is invalid")
    workspace = _canonical_workspace(manifest["remote_workspace"])
    if expected_workspace is not None and workspace != _canonical_workspace(
        expected_workspace
    ):
        raise OracleCheckpointError("checkpoint workspace differs from replay target")
    unstable_paths = manifest.get("unstable_paths", [])
    if not isinstance(unstable_paths, list) or any(
        not isinstance(path, str) or not path for path in unstable_paths
    ):
        raise OracleCheckpointError("checkpoint unstable paths are invalid")

    try:
        state_text = state_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise OracleCheckpointError("checkpoint state cannot be read") from error
    try:
        state = json.loads(state_text)
    except json.JSONDecodeError as error:
        raise OracleCheckpointError("checkpoint state is invalid JSON") from error
    if not isinstance(state, dict):
        raise OracleCheckpointError("checkpoint state must be a JSON object")

    archive_sha256, actual_digest = _archive_hashes(archive_path)
    actual_digest.update(b"\0state\0")
    actual_digest.update(state_text.encode())
    if actual_digest.hexdigest() != digest:
        raise OracleCheckpointError(
            "checkpoint archive or state failed integrity check"
        )

    return RemoteCheckpointBundle(
        checkpoint=Checkpoint(
            checkpoint_id=checkpoint_id,
            step=step,
            created_at=created_at,
            digest=digest,
            path=directory,
            parent_id=parent_id,
            label=label,
            unstable_paths=tuple(unstable_paths),
        ),
        state=state,
        remote_workspace=workspace,
        archive_sha256=archive_sha256,
        state_sha256=hashlib.sha256(state_text.encode()).hexdigest(),
    )


def _direct_file(directory: Path, name: str) -> Path:
    candidate = directory / name
    if candidate.is_symlink() or not candidate.is_file():
        raise OracleCheckpointError(f"checkpoint {name} is missing or a symlink")
    if candidate.resolve().parent != directory:
        raise OracleCheckpointError(f"checkpoint {name} escapes its directory")
    return candidate


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OracleCheckpointError(f"{description} is invalid JSON") from error
    if not isinstance(value, dict):
        raise OracleCheckpointError(f"{description} must be a JSON object")
    return value


def _canonical_workspace(value: object) -> str:
    if not isinstance(value, str) or not value or not value.startswith("/"):
        raise OracleCheckpointError("checkpoint workspace must be absolute")
    path = PurePosixPath(value)
    if path == PurePosixPath("/") or ".." in path.parts or str(path) != value:
        raise OracleCheckpointError(
            "checkpoint workspace must be canonical and non-root"
        )
    return value


def _archive_hashes(path: Path) -> tuple[str, Any]:
    plain = hashlib.sha256()
    combined = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            plain.update(chunk)
            combined.update(chunk)
    return plain.hexdigest(), combined


def _source_agent_provider(agent: dict[str, Any], path: Path) -> str:
    kwargs = agent.get("kwargs")
    call_kwargs = kwargs.get("llm_call_kwargs") if isinstance(kwargs, dict) else None
    if not isinstance(call_kwargs, dict):
        raise UsageAccountingError(
            f"source trajectory reconstruction lacks llm_call_kwargs: {path}"
        )
    try:
        return openrouter_provider_from_call_kwargs(
            call_kwargs, source=f"llm_call_kwargs in {path}"
        )
    except ValueError as error:
        raise UsageAccountingError(str(error)) from error
