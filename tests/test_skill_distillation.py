from __future__ import annotations

import hashlib
import io
import json
import tarfile
from enum import StrEnum
from pathlib import Path

import pytest

from driftlock.models import (
    DriftTriggerOutcome,
    FineJudgeStatus,
    JudgeCompletion,
    JudgeReliabilityStatus,
    RunStatus,
)
from driftlock.skill_distillation import (
    DISTILLATION_RESPONSE_CHARACTER_LIMIT,
    EVIDENCE_CHARACTER_LIMIT,
    EVIDENCE_END,
    EVIDENCE_START,
    CallableSkillDistiller,
    Skill,
    SkillDistillationStatus,
    SkillValidationError,
    assemble_baseline_evidence,
    assemble_localized_evidence,
    build_distillation_prompt,
    parse_skill,
    serialize_skill,
)


def _archive(files: dict[str, str]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, text in files.items():
            content = text.encode()
            member = tarfile.TarInfo(f"./{name}")
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


def _checkpoint(
    trial: Path,
    *,
    phase: int,
    step: int,
    checkpoint_id: str,
    files: dict[str, str],
) -> None:
    directory = (
        trial
        / ".driftlock-checkpoints"
        / f"phase-{phase}"
        / "checkpoints"
        / checkpoint_id
    )
    directory.mkdir(parents=True)
    archive = _archive(files)
    state_text = json.dumps({"phase": phase, "step": step}, separators=(",", ":"))
    digest = hashlib.sha256(archive)
    digest.update(b"\0state\0")
    digest.update(state_text.encode())
    (directory / "workspace.tar.gz").write_bytes(archive)
    (directory / "state.json").write_text(state_text, encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "step": step,
                "created_at": "2026-08-26T12:00:00+00:00",
                "digest": digest.hexdigest(),
                "parent_id": None,
                "label": f"step-{step}",
                "remote_workspace": "/app",
            }
        ),
        encoding="utf-8",
    )


def _replace_checkpoint_archive(
    trial: Path, *, phase: int, checkpoint_id: str, files: dict[str, str]
) -> None:
    directory = (
        trial
        / ".driftlock-checkpoints"
        / f"phase-{phase}"
        / "checkpoints"
        / checkpoint_id
    )
    archive = _archive(files)
    state = (directory / "state.json").read_bytes()
    digest = hashlib.sha256(archive)
    digest.update(b"\0state\0")
    digest.update(state)
    (directory / "workspace.tar.gz").write_bytes(archive)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["digest"] = digest.hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _trial(tmp_path: Path, *, prior_steps: int | None = 3) -> Path:
    trial = tmp_path / "synthetic__trial"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    trajectory = [{"step_id": 1, "source": "user", "message": "repair the parser"}]
    for number in range(1, 12):
        trajectory.append(
            {
                "step_id": number + 1,
                "source": "agent",
                "message": f"agent-{number}",
                "observation": {"results": [{"content": f"terminal-{number}"}]},
            }
        )
    (agent / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.0", "steps": trajectory}),
        encoding="utf-8",
    )
    (agent / "driftlock-result.json").write_text(
        json.dumps(
            {
                "phases": [
                    {"phase": 0, "steps": 2, "status": "completed"},
                    {"phase": 1, "steps": prior_steps, "status": "completed"},
                    {"phase": 2, "steps": None, "status": "exception"},
                ]
            }
        ),
        encoding="utf-8",
    )
    _checkpoint(
        trial,
        phase=2,
        step=2,
        checkpoint_id="a" * 32,
        files={"src/parser.py": "mode = 'old'\n", "README.md": "same\n"},
    )
    _checkpoint(
        trial,
        phase=2,
        step=4,
        checkpoint_id="b" * 32,
        files={"src/parser.py": "mode = 'new'\n", "README.md": "same\n"},
    )
    return trial


def _segment() -> dict[str, object]:
    return {
        "type": "regression",
        "steps": [2, 4],
        "start": {"phase": 2, "step": 2, "checkpoint_id": "a" * 32},
        "end": {"phase": 2, "step": 4, "checkpoint_id": "b" * 32},
        "start_reward": 0.8,
        "end_reward": 0.6,
        "reward_change": -0.2,
    }


def test_skill_schema_parses_validates_and_round_trips() -> None:
    document = (
        "## activation\n\n"
        "When a phase-relative index must address a cumulative log.\n\n"
        "## execution\n\nDo not slice by the raw index; add verified prior counts "
        "instead.\n\n"
        "## termination\n\nStop when both checkpoint boundaries map to recorded steps."
    )

    skill = parse_skill(document)

    assert skill == Skill(
        activation="When a phase-relative index must address a cumulative log.",
        execution=("Do not slice by the raw index; add verified prior counts instead."),
        termination="Stop when both checkpoint boundaries map to recorded steps.",
    )
    assert parse_skill(serialize_skill(skill)) == skill


@pytest.mark.parametrize("section", ["activation", "execution", "termination"])
def test_skill_schema_names_each_missing_section(section: str) -> None:
    parts = {
        "activation": "## activation\n\nWhen parsing a log.",
        "execution": "## execution\n\nDo not guess; inspect it instead.",
        "termination": "## termination\n\nStop after validation.",
    }
    document = "\n\n".join(value for key, value in parts.items() if key != section)

    with pytest.raises(
        SkillValidationError, match=f"missing required section: {section}"
    ):
        parse_skill(document)


@pytest.mark.parametrize(
    ("document", "expected_reason"),
    [
        (
            "The evidence is inconclusive, so no skill was produced.",
            "skill has no recognizable required section headings",
        ),
        (
            "**activation**\nWhen parsing.\n\n**execution**\nInspect first.",
            (
                "skill has no recognizable required section headings; section names "
                "appear without the required Markdown heading syntax: activation, "
                "execution"
            ),
        ),
        (
            "## activation\n\nWhen parsing.\n\n## execution\n\nInspect first.",
            "skill is missing required section: termination",
        ),
        (
            "## execution\n\nInspect first.\n\n## activation\n\nWhen parsing.\n\n"
            "## termination\n\nStop after validation.",
            ("skill sections must appear in activation, execution, termination order"),
        ),
        (
            "## activation\n\nWhen parsing.\n\n## activation\n\nStill parsing.\n\n"
            "## execution\n\nInspect first.\n\n## termination\n\n"
            "Stop after validation.",
            "skill has duplicate section: activation",
        ),
    ],
)
def test_skill_schema_classifies_malformed_document_shape(
    document: str, expected_reason: str
) -> None:
    with pytest.raises(SkillValidationError) as raised:
        parse_skill(document)

    assert str(raised.value) == expected_reason


def test_localized_evidence_uses_nonzero_phase_offset_and_real_workspace_diff(
    tmp_path: Path,
) -> None:
    result = assemble_localized_evidence(_trial(tmp_path), _segment())

    assert result["status"] == "usable"
    trajectory = result["evidence"]["trajectory"]
    assert trajectory["phase_offset"] == 5
    assert trajectory["start_step"] == 7
    assert trajectory["end_step"] == 9
    assert [step["agent_step"] for step in trajectory["steps"]] == [7, 8, 9]
    assert [step["message"] for step in trajectory["steps"]] == [
        "agent-7",
        "agent-8",
        "agent-9",
    ]
    assert [step["terminal_output"] for step in trajectory["steps"]] == [
        "terminal-7",
        "terminal-8",
        "terminal-9",
    ]
    workspace = result["evidence"]["workspace_diff"]
    assert workspace["summary"] == {"added": 0, "removed": 0, "modified": 1}
    assert workspace["files"][0]["path"] == "src/parser.py"
    assert "-mode = 'old'" in workspace["files"][0]["diff"]
    assert "+mode = 'new'" in workspace["files"][0]["diff"]


def test_localized_evidence_refuses_an_oversized_checkout_diff(tmp_path: Path) -> None:
    trial = _trial(tmp_path)
    files = {f"vendor/{index:04d}-{'x' * 340}.py": "x\n" for index in range(2_500)}
    _replace_checkpoint_archive(trial, phase=2, checkpoint_id="b" * 32, files=files)

    result = assemble_localized_evidence(trial, _segment())

    assert result["status"] == "refused"
    assert result["refusal"]["reason"] == "evidence_size_limit_exceeded"
    assert result["evidence_character_limit"] == 900_000
    assert result["evidence_characters"] >= 900_000
    assert "at or over the 900000-character input bound" in result["refusal"]["detail"]


@pytest.mark.parametrize(
    "measured_characters",
    [146_276, 28_586, 134_003, 94_231, 161_291, 51_528, 15_905],
)
def test_evidence_bound_keeps_each_observed_viable_localized_segment(
    measured_characters: int,
) -> None:
    assert measured_characters < EVIDENCE_CHARACTER_LIMIT


def test_evidence_bound_refuses_the_observed_checkout_segment_size() -> None:
    assert EVIDENCE_CHARACTER_LIMIT <= 2_552_299


def test_localized_evidence_maps_an_offset_zero_phase_directly(tmp_path: Path) -> None:
    trial = _trial(tmp_path)
    _checkpoint(
        trial,
        phase=0,
        step=2,
        checkpoint_id="c" * 32,
        files={"src/parser.py": "mode = 'old'\n"},
    )
    _checkpoint(
        trial,
        phase=0,
        step=4,
        checkpoint_id="d" * 32,
        files={"src/parser.py": "mode = 'new'\n"},
    )
    segment = {
        "type": "flat",
        "steps": [2, 4],
        "start": {"phase": 0, "step": 2, "checkpoint_id": "c" * 32},
        "end": {"phase": 0, "step": 4, "checkpoint_id": "d" * 32},
        "start_reward": 0.5,
        "end_reward": 0.5,
        "reward_change": 0.0,
    }

    result = assemble_localized_evidence(trial, segment)

    trajectory = result["evidence"]["trajectory"]
    assert trajectory["phase_offset"] == 0
    assert trajectory["start_step"] == 2
    assert trajectory["end_step"] == 4
    assert [step["message"] for step in trajectory["steps"]] == [
        "agent-2",
        "agent-3",
        "agent-4",
    ]


def test_baseline_evidence_contains_the_whole_trajectory(tmp_path: Path) -> None:
    result = assemble_baseline_evidence(_trial(tmp_path))

    assert result["status"] == "usable"
    steps = result["evidence"]["trajectory"]["steps"]
    assert len(steps) == 12
    assert steps[0] == {
        "trajectory_index": 0,
        "source": "user",
        "message": "repair the parser",
        "terminal_output": None,
        "step_id": 1,
    }
    assert steps[-1]["message"] == "agent-11"


def test_both_arms_use_identical_prompt_text_outside_evidence(tmp_path: Path) -> None:
    trial = _trial(tmp_path)
    localized = build_distillation_prompt(
        assemble_localized_evidence(trial, _segment())
    )
    baseline = build_distillation_prompt(assemble_baseline_evidence(trial))

    def outside(prompt: str) -> tuple[str, str]:
        before, remainder = prompt.split(EVIDENCE_START, 1)
        _, after = remainder.split(EVIDENCE_END, 1)
        return before, after

    assert outside(localized) == outside(baseline)
    assert localized != baseline


def test_unknown_prior_phase_count_refuses_instead_of_guessing(tmp_path: Path) -> None:
    result = assemble_localized_evidence(_trial(tmp_path, prior_steps=None), _segment())

    assert result["status"] == "refused"
    assert result["refusal"]["reason"] == "unknown_phase_offset"
    assert "prior phase 1 has unknown step count" in result["refusal"]["detail"]


def test_missing_checkpoint_id_refuses_with_stable_reason(tmp_path: Path) -> None:
    segment = _segment()
    segment["start"] = {"phase": 2, "step": 2}

    result = assemble_localized_evidence(_trial(tmp_path), segment)

    assert result["status"] == "refused"
    assert result["refusal"]["reason"] == "missing_checkpoint_id"
    assert "segment start has no valid checkpoint id" in result["refusal"]["detail"]


def test_segment_beyond_recorded_trajectory_refuses_instead_of_empty_slice(
    tmp_path: Path,
) -> None:
    segment = _segment()
    segment["end"] = {
        "phase": 2,
        "step": 20,
        "checkpoint_id": "b" * 32,
    }

    result = assemble_localized_evidence(_trial(tmp_path), segment)

    assert result["status"] == "refused"
    assert result["refusal"]["reason"] == "trajectory_slice_unavailable"
    assert "ends at cumulative agent step 25" in result["refusal"]["detail"]
    assert "only 11 agent steps" in result["refusal"]["detail"]


def test_checkpoint_manifest_step_mismatch_refuses_with_both_steps(
    tmp_path: Path,
) -> None:
    trial = _trial(tmp_path)
    manifest_path = (
        trial
        / ".driftlock-checkpoints/phase-2/checkpoints"
        / ("b" * 32)
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["step"] = 20
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = assemble_localized_evidence(trial, _segment())

    assert result["status"] == "refused"
    assert result["refusal"]["reason"] == "checkpoint_step_mismatch"
    assert result["refusal"]["detail"] == (
        f"checkpoint {'b' * 32} records step 20, not localized step 4"
    )


def test_missing_checkpoint_archive_refuses_with_reason(tmp_path: Path) -> None:
    trial = _trial(tmp_path)
    (
        trial
        / ".driftlock-checkpoints/phase-2/checkpoints"
        / ("b" * 32)
        / "workspace.tar.gz"
    ).unlink()

    result = assemble_localized_evidence(trial, _segment())

    assert result["status"] == "refused"
    assert result["refusal"]["reason"] == "checkpoint_archive_unavailable"
    assert "workspace.tar.gz is missing" in result["refusal"]["detail"]


async def test_callable_distiller_parses_valid_skill_and_preserves_tokens() -> None:
    async def complete(_prompt: str) -> JudgeCompletion:
        return JudgeCompletion(
            "## activation\n\nWhen offsets differ.\n\n"
            "## execution\n\nDo not guess; verify counts instead.\n\n"
            "## termination\n\nStop after both bounds resolve.",
            tokens=19,
        )

    result = await CallableSkillDistiller(complete).distill({"trajectory": []})

    assert result.status is SkillDistillationStatus.GENERATED
    assert result.skill is not None
    assert result.skill.activation == "When offsets differ."
    assert result.tokens == 19


async def test_callable_distiller_distinguishes_malformed_from_declined() -> None:
    responses = iter(
        [
            "## activation\n\nWhen offsets differ.\n\n## execution\n\nVerify counts.",
            "DECLINE: the evidence shows no repeatable failure",
        ]
    )

    async def complete(_prompt: str) -> str:
        return next(responses)

    distiller = CallableSkillDistiller(complete)
    malformed = await distiller.distill({"trajectory": []})
    declined = await distiller.distill({"trajectory": []})

    assert malformed.status is SkillDistillationStatus.MALFORMED
    assert malformed.status.value == "malformed"
    assert "missing required section: termination" in malformed.reason
    assert malformed.response is not None
    assert malformed.response.as_dict() == {
        "excerpt": (
            "## activation\n\nWhen offsets differ.\n\n## execution\n\nVerify counts."
        ),
        "original_characters": 65,
        "retained_characters": 65,
        "retained_character_limit": 16_384,
        "dropped_characters": 0,
        "truncated": False,
    }
    assert declined.status is SkillDistillationStatus.DECLINED
    assert declined.status.value == "declined"
    assert declined.reason == "the evidence shows no repeatable failure"
    assert declined.response is not None
    assert declined.response.excerpt == (
        "DECLINE: the evidence shows no repeatable failure"
    )
    assert declined.response.truncated is False


@pytest.mark.parametrize(
    ("response", "expected_reason", "expected_characters"),
    [
        (
            "The evidence did not yield a procedural skill.",
            "no recognizable required section headings",
            46,
        ),
        (
            "## activation\n\nWhen offsets differ.\n\n## execution\n\nVerify counts.",
            "missing required section: termination",
            65,
        ),
        (
            "## execution\n\nVerify counts.\n\n## activation\n\n"
            "When offsets differ.\n\n"
            "## termination\n\nStop after validation.",
            "sections must appear in activation, execution, termination order",
            105,
        ),
        (
            "## activation\n\nWhen offsets differ.\n\n## activation\n\nAgain.\n\n"
            "## execution\n\nVerify counts.\n\n## termination\n\n"
            "Stop after validation.",
            "duplicate section: activation",
            128,
        ),
    ],
)
async def test_callable_distiller_retains_each_malformed_shape(
    response: str, expected_reason: str, expected_characters: int
) -> None:
    async def complete(_prompt: str) -> str:
        return response

    result = await CallableSkillDistiller(complete).distill({"trajectory": []})

    assert result.status is SkillDistillationStatus.MALFORMED
    assert expected_reason in result.reason
    assert result.response is not None
    assert result.response.excerpt == response
    assert result.response.original_characters == expected_characters
    assert result.response.dropped_characters == 0
    assert result.response.truncated is False


async def test_callable_distiller_requires_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    async def complete(prompt: str) -> str:
        assert "<evidence>" in prompt
        return "DECLINE: synthetic evidence is intentionally empty"

    result = await CallableSkillDistiller(complete).distill({"trajectory": []})

    assert result.status is SkillDistillationStatus.DECLINED


async def test_callable_distiller_does_not_retain_a_generated_response() -> None:
    async def complete(_prompt: str) -> str:
        return (
            "## activation\n\nWhen offsets differ.\n\n"
            "## execution\n\nVerify counts.\n\n"
            "## termination\n\nStop after validation."
        )

    result = await CallableSkillDistiller(complete).distill({"trajectory": []})

    assert result.status is SkillDistillationStatus.GENERATED
    assert result.response is None
    assert DISTILLATION_RESPONSE_CHARACTER_LIMIT == 16_384


async def test_callable_distiller_bounds_a_long_decline_and_its_reason() -> None:
    async def complete(_prompt: str) -> str:
        return "DECLINE: " + ("d" * 20_000)

    result = await CallableSkillDistiller(complete).distill({"trajectory": []})

    assert result.status is SkillDistillationStatus.DECLINED
    assert len(result.reason) == 1_086
    assert result.reason.endswith(
        " [... 18976 response characters omitted; see response excerpt]"
    )
    assert result.response is not None
    assert result.response.original_characters == 20_009
    assert result.response.retained_characters == 16_384
    assert result.response.dropped_characters == 3_625
    assert result.response.truncated is True


@pytest.mark.parametrize(
    ("status_type", "expected_values"),
    [
        (
            SkillDistillationStatus,
            ["generated", "declined", "malformed", "failed"],
        ),
        (
            FineJudgeStatus,
            [
                "verdict",
                "failed",
                "budget_exhausted",
                "not_configured",
                "not_invoked",
            ],
        ),
        (
            DriftTriggerOutcome,
            [
                "rolled_back",
                "vetoed",
                "rollback_limit_refused",
                "suppressed",
                "judge_failed",
                "judge_budget_exhausted",
            ],
        ),
        (
            JudgeReliabilityStatus,
            ["not_assessed", "reliable", "failed", "inconclusive"],
        ),
        (
            RunStatus,
            ["completed", "step_limit", "token_limit", "rollback_limit"],
        ),
    ],
)
def test_status_enums_have_distinct_stable_serialized_values(
    status_type: type[StrEnum], expected_values: list[str]
) -> None:
    declared_values = [member.value for member in status_type.__members__.values()]

    assert declared_values == expected_values
    assert len(set(declared_values)) == len(expected_values)
