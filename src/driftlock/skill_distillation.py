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
# A gloss is heading metadata, not skill content: accept only conventional,
# explicit separators after the complete section name, then discard that bounded
# same-line label.  Requiring a whitespace-delimited dash, colon, or one non-nested
# parenthetical keeps phrases such as "activation and termination notes" from
# silently becoming activation, while still admitting the labels models commonly
# add.  Procedural meaning belongs in the required non-empty body, where it
# survives serialization.
_SECTION_HEADING = re.compile(
    r"^#{1,6}[ \t]+(activation|execution|termination)"
    r"(?:[ \t]+[-\N{EN DASH}—][ \t]+[^\r\n]{1,120}"
    r"|[ \t]*:[ \t]+[^\r\n]{1,120}"
    r"|[ \t]+\([^()\r\n]{1,120}\))?"
    r"[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_DECLINE = re.compile(r"^DECLINE:[ \t]*(.+)$", re.IGNORECASE | re.DOTALL)
_MAX_TEXT_FILE_BYTES = 256 * 1024
_MAX_DIFF_CHARACTERS = 100_000

# This is 90% of the provider's reported 1,000,000 input-length limit; the error
# does not establish whether that unit means tokens or characters.  Evidence is
# JSON-serialized with ASCII escaping, so its character count is also its UTF-8
# byte count and therefore a conservative token upper bound.  The remaining 10%
# pays for the shared prompt and avoids relying on a tokenizer-specific average.
# In the first paid run all seven successful localized inputs were <=161,291
# characters (the largest baseline was 350,483), while the rejected checkout diff
# was 2,552,299.  This bound therefore removes that failure without changing the
# observed viable cohort.  A tokenizer-backed count for the pinned dated model,
# validated against the provider's accepted envelope, would settle a tighter bound.
EVIDENCE_CHARACTER_LIMIT = 900_000

# Only declined and malformed completions retain model text: generated text is
# already represented by its parsed Skill, while a failed call has no response.
# A 16 Ki-character head/tail excerpt is enough to expose wrappers, headings,
# and trailing prose without letting repeated failures dominate the report.
# Counts make every omission explicit.  The report writer's ASCII JSON escaping
# safely represents controls and lone surrogates without changing this excerpt.
# Decline reasons get a smaller cap because they otherwise duplicate that text.
DISTILLATION_RESPONSE_CHARACTER_LIMIT = 16_384
_DISTILLATION_REASON_CHARACTER_LIMIT = 1_024


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
    present = [match.group(1).lower() for match in matches]
    if not present:
        if not text:
            raise SkillValidationError(
                "skill document is empty; no recognizable required section headings"
            )
        mentioned = [
            section
            for section in SKILL_SECTIONS
            if re.search(rf"\b{section}\b", text, re.IGNORECASE)
        ]
        if mentioned:
            raise SkillValidationError(
                "skill has no recognizable required section headings; section names "
                "appear without the required Markdown heading syntax: "
                f"{', '.join(mentioned)}"
            )
        raise SkillValidationError(
            "skill has no recognizable required section headings"
        )
    duplicates = sorted(
        section for section in SKILL_SECTIONS if present.count(section) > 1
    )
    if duplicates:
        raise SkillValidationError(
            f"skill has duplicate section: {', '.join(duplicates)}"
        )
    missing = [section for section in SKILL_SECTIONS if section not in present]
    if len(missing) == 1:
        raise SkillValidationError(f"skill is missing required section: {missing[0]}")
    if missing:
        raise SkillValidationError(
            f"skill is missing required sections: {', '.join(missing)}"
        )
    if tuple(present) != SKILL_SECTIONS:
        raise SkillValidationError(
            "skill sections must appear in activation, execution, termination order"
        )
    if text[: matches[0].start()].strip():
        raise SkillValidationError("skill document has content before activation")

    values: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        values[match.group(1).lower()] = text[match.end() : end].strip()
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

    evidence = {
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
    }
    return _bounded_evidence("localized", evidence)


def assemble_baseline_evidence(trial_dir: Path | str) -> dict[str, Any]:
    """Build whole-trajectory evidence or return an explicit refusal."""

    try:
        trial = _trial_directory(trial_dir)
        trajectory = _load_trajectory(trial)
        steps = _evidence_steps(trajectory, range(len(trajectory)))
    except EvidenceAssemblyError as error:
        return _refusal("baseline", error.reason, error.detail)
    return _bounded_evidence(
        "baseline",
        {"trial_name": trial.name, "trajectory": {"steps": steps}},
    )


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
    serialized = _serialize_evidence(evidence_block)
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


def _response_omission_marker(dropped_characters: int) -> str:
    return (
        "\n\n[... "
        f"{dropped_characters} response characters omitted between excerpt halves"
        " ...]\n\n"
    )


@dataclass(frozen=True, slots=True)
class SkillDistillationResponse:
    """Bounded diagnostic evidence from a non-generated model response."""

    excerpt: str
    original_characters: int
    retained_characters: int
    dropped_characters: int
    truncated: bool

    def __post_init__(self) -> None:
        counts = (
            self.original_characters,
            self.retained_characters,
            self.dropped_characters,
        )
        if (
            not isinstance(self.excerpt, str)
            or not isinstance(self.truncated, bool)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts
            )
            or self.retained_characters > DISTILLATION_RESPONSE_CHARACTER_LIMIT
        ):
            raise ValueError("distillation response evidence has invalid values")
        if self.original_characters != (
            self.retained_characters + self.dropped_characters
        ) or self.truncated != (self.dropped_characters > 0):
            raise ValueError("distillation response evidence has inconsistent counts")
        if self.truncated:
            marker = _response_omission_marker(self.dropped_characters)
            marker_start = DISTILLATION_RESPONSE_CHARACTER_LIMIT // 2
            if (
                self.retained_characters != DISTILLATION_RESPONSE_CHARACTER_LIMIT
                or len(self.excerpt)
                != DISTILLATION_RESPONSE_CHARACTER_LIMIT + len(marker)
                or self.excerpt[marker_start : marker_start + len(marker)] != marker
            ):
                raise ValueError("distillation response excerpt is not bounded")
        elif len(self.excerpt) != self.retained_characters:
            raise ValueError("complete distillation response count does not match text")

    def as_dict(self) -> dict[str, Any]:
        return {
            "excerpt": self.excerpt,
            "original_characters": self.original_characters,
            "retained_characters": self.retained_characters,
            "retained_character_limit": DISTILLATION_RESPONSE_CHARACTER_LIMIT,
            "dropped_characters": self.dropped_characters,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class SkillDistillationResult:
    """Parsed output without conflating a decline with an invalid response."""

    status: SkillDistillationStatus
    skill: Skill | None = None
    reason: str = ""
    tokens: int = 0
    response: SkillDistillationResponse | None = None

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens cannot be negative")
        if (self.skill is not None) != (
            self.status is SkillDistillationStatus.GENERATED
        ):
            raise ValueError("only generated distillation may carry a skill")
        retains_response = self.status in {
            SkillDistillationStatus.DECLINED,
            SkillDistillationStatus.MALFORMED,
        }
        if (self.response is not None) != retains_response:
            raise ValueError(
                "only declined and malformed distillation must carry response evidence"
            )


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
                reason=_bounded_reason(decline.group(1).strip()),
                tokens=tokens,
                response=_response_excerpt(response),
            )
        try:
            skill = parse_skill(response)
        except SkillValidationError as error:
            return SkillDistillationResult(
                status=SkillDistillationStatus.MALFORMED,
                reason=f"skill distiller returned a malformed response: {error}",
                tokens=tokens,
                response=_response_excerpt(response),
            )
        return SkillDistillationResult(
            status=SkillDistillationStatus.GENERATED,
            skill=skill,
            tokens=tokens,
        )


def _response_excerpt(response: str) -> SkillDistillationResponse:
    original_characters = len(response)
    if original_characters <= DISTILLATION_RESPONSE_CHARACTER_LIMIT:
        return SkillDistillationResponse(
            excerpt=response,
            original_characters=original_characters,
            retained_characters=original_characters,
            dropped_characters=0,
            truncated=False,
        )
    head_characters = DISTILLATION_RESPONSE_CHARACTER_LIMIT // 2
    tail_characters = DISTILLATION_RESPONSE_CHARACTER_LIMIT - head_characters
    dropped_characters = original_characters - DISTILLATION_RESPONSE_CHARACTER_LIMIT
    marker = _response_omission_marker(dropped_characters)
    return SkillDistillationResponse(
        excerpt=response[:head_characters] + marker + response[-tail_characters:],
        original_characters=original_characters,
        retained_characters=DISTILLATION_RESPONSE_CHARACTER_LIMIT,
        dropped_characters=dropped_characters,
        truncated=True,
    )


def _bounded_reason(reason: str) -> str:
    if len(reason) <= _DISTILLATION_REASON_CHARACTER_LIMIT:
        return reason
    dropped = len(reason) - _DISTILLATION_REASON_CHARACTER_LIMIT
    return (
        reason[:_DISTILLATION_REASON_CHARACTER_LIMIT]
        + f" [... {dropped} response characters omitted; see response excerpt]"
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


def _serialize_evidence(evidence: Mapping[str, Any]) -> str:
    return json.dumps(evidence, indent=2, sort_keys=True)


def _bounded_evidence(mode: str, evidence: dict[str, Any]) -> dict[str, Any]:
    characters = len(_serialize_evidence(evidence))
    if characters >= EVIDENCE_CHARACTER_LIMIT:
        return _refusal(
            mode,
            "evidence_size_limit_exceeded",
            f"{mode} evidence is {characters} characters, at or over the "
            f"{EVIDENCE_CHARACTER_LIMIT}-character input bound",
            evidence_characters=characters,
        )
    return {
        "status": "usable",
        "mode": mode,
        "evidence_characters": characters,
        "evidence_character_limit": EVIDENCE_CHARACTER_LIMIT,
        "evidence": evidence,
    }


def _refusal(
    mode: str,
    reason: str,
    detail: str,
    *,
    evidence_characters: int | None = None,
) -> dict[str, Any]:
    refusal = {
        "status": "refused",
        "mode": mode,
        "refusal": {"reason": reason, "detail": detail},
    }
    if evidence_characters is not None:
        refusal["evidence_characters"] = evidence_characters
        refusal["evidence_character_limit"] = EVIDENCE_CHARACTER_LIMIT
    return refusal
