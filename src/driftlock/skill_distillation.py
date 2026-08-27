"""Harbor-free skill distillation inputs and callable model seam.

Both experiment arms use the same prompt and skill schema.  The localized arm
supplies a checkpoint-bounded trajectory slice and workspace diff; the baseline
arm supplies the complete trajectory.  No provider SDK or credential is used in
this module.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import tarfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from driftlock.models import JudgeCompletion
from driftlock.oracle import (
    OracleCheckpointError,
    RemoteCheckpointBundle,
    load_remote_checkpoint_bundle,
)

SKILL_SECTIONS = ("activation", "execution", "termination")
EVIDENCE_START = "<evidence>"
EVIDENCE_END = "</evidence>"
_SECTION_HEADING = re.compile(
    r"^#{1,6}[ \t]+(activation|execution|termination)[ \t]*$",
    re.MULTILINE,
)
_DECLINE = re.compile(r"^DECLINE:[ \t]*(.+)$", re.IGNORECASE | re.DOTALL)
_MAX_TEXT_FILE_BYTES = 256 * 1024
_MAX_DIFF_CHARACTERS = 100_000


class SkillValidationError(ValueError):
    """A skill document does not satisfy the shared schema."""


@dataclass(frozen=True, slots=True)
class Skill:
    """A preventative procedural memory shared by both distillation arms."""

    activation: str
    execution: str
    termination: str

    def __post_init__(self) -> None:
        for section in SKILL_SECTIONS:
            value = getattr(self, section)
            if not isinstance(value, str) or not value.strip():
                raise SkillValidationError(
                    f"skill section {section!r} must be non-empty"
                )

    @classmethod
    def parse(cls, document: str) -> Skill:
        return parse_skill(document)

    def to_markdown(self) -> str:
        return serialize_skill(self)


def parse_skill(document: str) -> Skill:
    """Parse a structured-markdown skill and validate all required sections."""

    if not isinstance(document, str):
        raise SkillValidationError("skill document must be text")
    text = _unwrap_markdown_fence(document.strip())
    matches = list(_SECTION_HEADING.finditer(text))
    present = [match.group(1) for match in matches]
    for section in SKILL_SECTIONS:
        if section not in present:
            raise SkillValidationError(f"skill is missing required section: {section}")
    duplicates = sorted(
        section for section in SKILL_SECTIONS if present.count(section) > 1
    )
    if duplicates:
        raise SkillValidationError(f"skill repeats section: {', '.join(duplicates)}")
    if tuple(present) != SKILL_SECTIONS:
        raise SkillValidationError(
            "skill sections must appear in activation, execution, termination order"
        )
    if text[: matches[0].start()].strip():
        raise SkillValidationError("skill document has content before activation")

    values: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        values[match.group(1)] = text[match.end() : end].strip()
    return Skill(**values)


def validate_skill(skill: Skill) -> None:
    """Validate a constructed skill using the same rules as parsed documents."""

    if not isinstance(skill, Skill):
        raise SkillValidationError("skill must be a Skill instance")
    for section in SKILL_SECTIONS:
        value = getattr(skill, section)
        if not value.strip():
            raise SkillValidationError(f"skill section {section!r} must be non-empty")


def serialize_skill(skill: Skill) -> str:
    """Serialize a validated skill to the canonical structured markdown."""

    validate_skill(skill)
    return "\n\n".join(
        f"## {section}\n\n{getattr(skill, section).strip()}"
        for section in SKILL_SECTIONS
    )


class EvidenceAssemblyError(ValueError):
    """Internal typed refusal while grounding distillation evidence."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def assemble_localized_evidence(
    trial_dir: Path | str, segment: Mapping[str, Any]
) -> dict[str, Any]:
    """Build checkpoint-localized evidence or return an explicit refusal."""

    try:
        trial = _trial_directory(trial_dir)
        bounds = _segment_bounds(segment)
        trajectory = _load_trajectory(trial)
        phase_offset = _phase_offset(trial, bounds["phase"])
        start_position = phase_offset + bounds["start_step"]
        end_position = phase_offset + bounds["end_step"]
        steps = _localized_trajectory_steps(
            trajectory,
            start_position=start_position,
            end_position=end_position,
        )
        start_bundle = _checkpoint_bundle(
            trial,
            phase=bounds["phase"],
            step=bounds["start_step"],
            checkpoint_id=bounds["start_checkpoint_id"],
        )
        end_bundle = _checkpoint_bundle(
            trial,
            phase=bounds["phase"],
            step=bounds["end_step"],
            checkpoint_id=bounds["end_checkpoint_id"],
        )
        workspace_diff = _workspace_diff(
            start_bundle.checkpoint.path / "workspace.tar.gz",
            end_bundle.checkpoint.path / "workspace.tar.gz",
        )
        normalized_segment = _json_value(segment, "localized segment")
    except EvidenceAssemblyError as error:
        return _refusal("localized", error.reason, error.detail)

    return {
        "status": "usable",
        "mode": "localized",
        "evidence": {
            "trial_name": trial.name,
            "segment": normalized_segment,
            "trajectory": {
                "phase": bounds["phase"],
                "phase_offset": phase_offset,
                "start_step": start_position,
                "end_step": end_position,
                "steps": steps,
            },
            "workspace_diff": workspace_diff,
        },
    }


def assemble_baseline_evidence(trial_dir: Path | str) -> dict[str, Any]:
    """Build whole-trajectory evidence or return an explicit refusal."""

    try:
        trial = _trial_directory(trial_dir)
        trajectory = _load_trajectory(trial)
        steps = _evidence_steps(trajectory, range(len(trajectory)))
    except EvidenceAssemblyError as error:
        return _refusal("baseline", error.reason, error.detail)
    return {
        "status": "usable",
        "mode": "baseline",
        "evidence": {
            "trial_name": trial.name,
            "trajectory": {"steps": steps},
        },
    }


def build_distillation_prompt(evidence: Mapping[str, Any]) -> str:
    """Insert either arm's evidence into the single shared prompt template."""

    if not isinstance(evidence, Mapping):
        raise ValueError("distillation evidence must be an object")
    if evidence.get("status") == "usable" and isinstance(
        evidence.get("evidence"), Mapping
    ):
        evidence_block: Mapping[str, Any] = evidence["evidence"]
    elif "status" in evidence:
        raise ValueError("cannot build a prompt from refused evidence")
    else:
        evidence_block = evidence
    serialized = json.dumps(evidence_block, indent=2, sort_keys=True)
    return (
        "You distill preventative procedural memory from a coding-agent record.\n"
        "Use only the supplied evidence; do not infer missing events or outcomes.\n"
        "Return exactly one structured-markdown skill with these headings in order:\n"
        "## activation — when the skill applies\n"
        "## execution — what to do, including what failed action to avoid and what "
        "to do instead\n"
        "## termination — when the skill is finished or no longer applies\n"
        "Write preventative guidance: when X appears, do not do Y; do Z instead.\n"
        "If the evidence cannot support such a skill, return `DECLINE: <reason>`.\n\n"
        f"{EVIDENCE_START}\n{serialized}\n{EVIDENCE_END}\n\n"
        "Return only the skill document or the explicit decline."
    )


class SkillDistillationStatus(StrEnum):
    """Disposition of one callable distillation attempt."""

    GENERATED = "generated"
    DECLINED = "declined"
    MALFORMED = "malformed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SkillDistillationResult:
    """Parsed output without conflating a decline with an invalid response."""

    status: SkillDistillationStatus
    skill: Skill | None = None
    reason: str = ""
    tokens: int = 0

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens cannot be negative")
        if (self.skill is not None) != (
            self.status is SkillDistillationStatus.GENERATED
        ):
            raise ValueError("only generated distillation may carry a skill")


class CallableSkillDistiller:
    """Distill through an injected async completion callable, with no SDK coupling."""

    def __init__(
        self,
        complete: Callable[[str], Awaitable[str | JudgeCompletion]],
    ) -> None:
        self._complete = complete

    async def distill(self, evidence: Mapping[str, Any]) -> SkillDistillationResult:
        tokens = 0
        try:
            completion = await self._complete(build_distillation_prompt(evidence))
            if isinstance(completion, str):
                response = completion
            else:
                response = completion.text
                tokens = completion.tokens
        except Exception as error:
            return SkillDistillationResult(
                status=SkillDistillationStatus.FAILED,
                reason=f"skill distiller call failed: {error}",
                tokens=tokens,
            )

        decline = _DECLINE.fullmatch(response.strip())
        if decline is not None:
            return SkillDistillationResult(
                status=SkillDistillationStatus.DECLINED,
                reason=decline.group(1).strip(),
                tokens=tokens,
            )
        try:
            skill = parse_skill(response)
        except SkillValidationError as error:
            return SkillDistillationResult(
                status=SkillDistillationStatus.MALFORMED,
                reason=f"skill distiller returned a malformed response: {error}",
                tokens=tokens,
            )
        return SkillDistillationResult(
            status=SkillDistillationStatus.GENERATED,
            skill=skill,
            tokens=tokens,
        )


def _unwrap_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().lower() in {"```", "```markdown", "```md"}:
        if lines[-1].strip() != "```":
            raise SkillValidationError("skill markdown fence is not closed")
        return "\n".join(lines[1:-1]).strip()
    return text


def _trial_directory(path: Path | str) -> Path:
    trial = Path(path).expanduser().resolve()
    if not trial.is_dir():
        raise EvidenceAssemblyError(
            "missing_trial", f"trial directory does not exist: {trial}"
        )
    return trial


def _segment_bounds(segment: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(segment, Mapping):
        raise EvidenceAssemblyError(
            "malformed_segment", "localized segment must be an object"
        )
    start = segment.get("start")
    end = segment.get("end")
    if not isinstance(start, Mapping) or not isinstance(end, Mapping):
        raise EvidenceAssemblyError(
            "malformed_segment", "localized segment needs start and end objects"
        )
    start_phase = _nonnegative_integer(start.get("phase"), "segment start phase")
    end_phase = _nonnegative_integer(end.get("phase"), "segment end phase")
    if start_phase != end_phase:
        raise EvidenceAssemblyError(
            "cross_phase_segment",
            "localized segment endpoints belong to different phases",
        )
    start_step = _nonnegative_integer(start.get("step"), "segment start step")
    end_step = _nonnegative_integer(end.get("step"), "segment end step")
    if end_step <= start_step:
        raise EvidenceAssemblyError(
            "malformed_segment", "localized segment end must follow its start"
        )
    return {
        "phase": start_phase,
        "start_step": start_step,
        "end_step": end_step,
        "start_checkpoint_id": _checkpoint_id(start, "start"),
        "end_checkpoint_id": _checkpoint_id(end, "end"),
    }


def _checkpoint_id(point: Mapping[str, Any], endpoint: str) -> str:
    checkpoint_id = point.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", checkpoint_id
    ):
        raise EvidenceAssemblyError(
            "missing_checkpoint_id",
            f"localized segment {endpoint} has no valid checkpoint id",
        )
    return checkpoint_id


def _nonnegative_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceAssemblyError(
            "malformed_segment", f"{description} must be a non-negative integer"
        )
    return value


def _load_trajectory(trial: Path) -> list[dict[str, Any]]:
    path = trial / "agent" / "trajectory.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvidenceAssemblyError(
            "missing_trajectory", f"trial trajectory does not exist: {path}"
        ) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceAssemblyError(
            "malformed_trajectory", f"trial trajectory is unreadable: {path}: {error}"
        ) from error
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list) or not steps:
        raise EvidenceAssemblyError(
            "malformed_trajectory", f"trial trajectory has no steps array: {path}"
        )
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise EvidenceAssemblyError(
                "malformed_trajectory",
                f"trajectory step {index} must be an object: {path}",
            )
        if step.get("source") not in {"agent", "user"}:
            raise EvidenceAssemblyError(
                "malformed_trajectory",
                f"trajectory step {index} source must be agent or user: {path}",
            )
        if not isinstance(step.get("message"), str):
            raise EvidenceAssemblyError(
                "malformed_trajectory",
                f"trajectory step {index} message must be text: {path}",
            )
    return steps


def _phase_offset(trial: Path, target_phase: int) -> int:
    path = trial / "agent" / "driftlock-result.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvidenceAssemblyError(
            "missing_phase_audit", f"trial phase audit does not exist: {path}"
        ) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceAssemblyError(
            "malformed_phase_audit", f"trial phase audit is unreadable: {path}: {error}"
        ) from error
    phases = payload.get("phases") if isinstance(payload, dict) else None
    if not isinstance(phases, list):
        raise EvidenceAssemblyError(
            "malformed_phase_audit", f"trial phase audit has no phases array: {path}"
        )
    by_phase: dict[int, Mapping[str, Any]] = {}
    for index, phase in enumerate(phases):
        number = phase.get("phase") if isinstance(phase, Mapping) else None
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise EvidenceAssemblyError(
                "malformed_phase_audit",
                f"phase audit entry {index} has an invalid phase number: {path}",
            )
        if number in by_phase:
            raise EvidenceAssemblyError(
                "malformed_phase_audit", f"phase audit repeats phase {number}: {path}"
            )
        by_phase[number] = phase
    if target_phase not in by_phase:
        raise EvidenceAssemblyError(
            "phase_not_found",
            f"localized phase {target_phase} is absent from trial phase audit: {path}",
        )
    offset = 0
    for number in range(target_phase):
        phase = by_phase.get(number)
        if phase is None:
            raise EvidenceAssemblyError(
                "unknown_phase_offset",
                f"cannot locate phase {target_phase}: prior phase {number} is absent",
            )
        count = phase.get("steps")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise EvidenceAssemblyError(
                "unknown_phase_offset",
                f"cannot locate phase {target_phase}: prior phase {number} has "
                "unknown step count",
            )
        offset += count
    return offset


def _localized_trajectory_steps(
    trajectory: Sequence[dict[str, Any]], *, start_position: int, end_position: int
) -> list[dict[str, Any]]:
    agent_indices = [
        index for index, step in enumerate(trajectory) if step["source"] == "agent"
    ]
    if end_position > len(agent_indices):
        raise EvidenceAssemblyError(
            "trajectory_slice_unavailable",
            f"localized segment ends at cumulative agent step {end_position}, but "
            f"the trajectory contains only {len(agent_indices)} agent steps",
        )
    first_index = 0 if start_position == 0 else agent_indices[start_position - 1]
    last_index = agent_indices[end_position - 1]
    return _evidence_steps(trajectory, range(first_index, last_index + 1))


def _evidence_steps(
    trajectory: Sequence[dict[str, Any]], indices: Sequence[int] | range
) -> list[dict[str, Any]]:
    agent_position = 0
    positions: dict[int, int] = {}
    for index, step in enumerate(trajectory):
        if step["source"] == "agent":
            agent_position += 1
            positions[index] = agent_position
    result: list[dict[str, Any]] = []
    for index in indices:
        step = trajectory[index]
        item: dict[str, Any] = {
            "trajectory_index": index,
            "source": step["source"],
            "message": step["message"],
            "terminal_output": _terminal_output(step, index),
        }
        if index in positions:
            item["agent_step"] = positions[index]
        step_id = step.get("step_id")
        if isinstance(step_id, int) and not isinstance(step_id, bool):
            item["step_id"] = step_id
        result.append(item)
    return result


def _terminal_output(step: Mapping[str, Any], index: int) -> str | None:
    observation = step.get("observation")
    if observation is None:
        return None
    results = observation.get("results") if isinstance(observation, Mapping) else None
    if not isinstance(results, list) or not results:
        raise EvidenceAssemblyError(
            "malformed_trajectory",
            f"trajectory step {index} observation has no results",
        )
    first = results[0]
    content = first.get("content") if isinstance(first, Mapping) else None
    if not isinstance(content, str):
        raise EvidenceAssemblyError(
            "malformed_trajectory",
            f"trajectory step {index} first observation result has no text content",
        )
    return content


def _checkpoint_bundle(
    trial: Path, *, phase: int, step: int, checkpoint_id: str
) -> RemoteCheckpointBundle:
    directory = (
        trial
        / ".driftlock-checkpoints"
        / f"phase-{phase}"
        / "checkpoints"
        / checkpoint_id
    )
    try:
        bundle = load_remote_checkpoint_bundle(directory)
    except OracleCheckpointError as error:
        raise EvidenceAssemblyError(
            "checkpoint_archive_unavailable",
            f"cannot load checkpoint archive for phase {phase} step {step}: {error}",
        ) from error
    if bundle.checkpoint.step != step:
        raise EvidenceAssemblyError(
            "checkpoint_step_mismatch",
            f"checkpoint {checkpoint_id} records step {bundle.checkpoint.step}, "
            f"not localized step {step}",
        )
    return bundle


@dataclass(frozen=True, slots=True)
class _ArchiveEntry:
    kind: str
    mode: int
    size: int
    digest: str
    content: bytes | None
    link_target: str | None = None


def _workspace_diff(before_path: Path, after_path: Path) -> dict[str, Any]:
    try:
        before = _archive_entries(before_path)
        after = _archive_entries(after_path)
    except (OSError, tarfile.TarError, ValueError) as error:
        raise EvidenceAssemblyError(
            "checkpoint_archive_unavailable",
            f"cannot compare checkpoint workspace archives: {error}",
        ) from error
    changes: list[dict[str, Any]] = []
    remaining_diff_characters = _MAX_DIFF_CHARACTERS
    for path in sorted(set(before) | set(after)):
        old = before.get(path)
        new = after.get(path)
        if old == new:
            continue
        change = "added" if old is None else "removed" if new is None else "modified"
        item: dict[str, Any] = {"path": path, "change": change}
        if old is not None:
            item["before"] = _entry_metadata(old)
        if new is not None:
            item["after"] = _entry_metadata(new)
        text_diff = _entry_text_diff(path, old, new)
        if text_diff is not None:
            if len(text_diff) <= remaining_diff_characters:
                item["diff"] = text_diff
                remaining_diff_characters -= len(text_diff)
            else:
                item["diff_omitted"] = "workspace diff text limit reached"
        changes.append(item)
    counts = {
        kind: sum(change["change"] == kind for change in changes)
        for kind in ("added", "removed", "modified")
    }
    return {"summary": counts, "files": changes}


def _archive_entries(path: Path) -> dict[str, _ArchiveEntry]:
    entries: dict[str, _ArchiveEntry] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if not name:
                continue
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.isdir():
                continue
            if name in entries:
                raise ValueError(f"duplicate archive member: {name}")
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read archive member: {name}")
                digest = hashlib.sha256()
                chunks: list[bytes] | None = (
                    [] if member.size <= _MAX_TEXT_FILE_BYTES else None
                )
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    if chunks is not None:
                        chunks.append(chunk)
                content = b"".join(chunks) if chunks is not None else None
                entries[name] = _ArchiveEntry(
                    kind="file",
                    mode=member.mode,
                    size=member.size,
                    digest=digest.hexdigest(),
                    content=content,
                )
            elif member.issym() or member.islnk():
                target = member.linkname
                entries[name] = _ArchiveEntry(
                    kind="symlink" if member.issym() else "hardlink",
                    mode=member.mode,
                    size=0,
                    digest=hashlib.sha256(target.encode()).hexdigest(),
                    content=None,
                    link_target=target,
                )
            else:
                raise ValueError(f"unsupported archive member type: {name}")
    return entries


def _entry_metadata(entry: _ArchiveEntry) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "kind": entry.kind,
        "mode": f"{entry.mode:04o}",
        "size": entry.size,
        "sha256": entry.digest,
    }
    if entry.link_target is not None:
        metadata["link_target"] = entry.link_target
    return metadata


def _entry_text_diff(
    path: str, old: _ArchiveEntry | None, new: _ArchiveEntry | None
) -> str | None:
    if old is not None and (old.kind != "file" or old.content is None):
        return None
    if new is not None and (new.kind != "file" or new.content is None):
        return None
    try:
        before = [] if old is None else old.content.decode("utf-8").splitlines(True)
        after = [] if new is None else new.content.decode("utf-8").splitlines(True)
    except UnicodeDecodeError:
        return None
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=f"a/{path}" if old is not None else "/dev/null",
            tofile=f"b/{path}" if new is not None else "/dev/null",
        )
    )


def _json_value(value: object, description: str) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as error:
        raise EvidenceAssemblyError(
            "malformed_segment", f"{description} is not JSON-compatible"
        ) from error


def _refusal(mode: str, reason: str, detail: str) -> dict[str, Any]:
    return {
        "status": "refused",
        "mode": mode,
        "refusal": {"reason": reason, "detail": detail},
    }
