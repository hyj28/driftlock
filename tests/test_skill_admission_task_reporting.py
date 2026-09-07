from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftlock.lhtb_experiment import main
from driftlock.skill_admission import (
    SkillAdmissionCandidate,
    TaskMetadataCondition,
    assemble_admission_report,
    decide_skill_admission,
    load_admission_candidates,
    render_admission_report,
)
from driftlock.skill_distillation import Skill, serialize_skill


def _skill(label: str) -> Skill:
    return Skill(
        activation=f"When {label} applies.",
        execution="Inspect the paired measurements.",
        termination="Stop after reporting the result.",
    )


def _candidate(
    candidate_id: str,
    deltas: list[float | None],
    flags: list[bool | None] | None,
    *,
    task_name: str | None,
) -> SkillAdmissionCandidate:
    return SkillAdmissionCandidate(
        candidate_id=candidate_id,
        arm="localized",
        skill=_skill(candidate_id),
        paired_deltas=tuple(deltas),
        injection_flags=None if flags is None else tuple(flags),
        task_name=task_name,
        source_task_name=task_name,
    )


def _task_groups(report: dict[str, object]) -> dict[str | None, dict[str, object]]:
    null_channel = report["null_channel"]
    assert isinstance(null_channel, dict)
    per_task = null_channel["per_task"]
    assert isinstance(per_task, list)
    return {group["task_name"]: group for group in per_task}


def test_report_has_independent_per_task_channel_statistics_and_contrasts() -> None:
    report = assemble_admission_report(
        [
            _candidate(
                "mixed-task",
                [1.0, 3.0, 5.0, 7.0] + [None] * 6,
                [False, False, True, True] + [None] * 6,
                task_name="mixed",
            ),
            _candidate(
                "injected-task",
                [2.0, 4.0] + [None] * 8,
                [True, True] + [None] * 8,
                task_name="injected-only",
            ),
            _candidate(
                "null-task",
                [-2.0, 0.0] + [None] * 8,
                [False, False] + [None] * 8,
                task_name="null-only",
            ),
        ]
    )

    groups = _task_groups(report)
    assert set(groups) == {"mixed", "injected-only", "null-only"}
    assert groups["mixed"]["no_skill_injected"] == {
        "n": 2,
        "mean_delta": 2.0,
        "sample_standard_deviation": pytest.approx(1.4142135623730951),
        "positive_count": 2,
        "negative_count": 0,
        "zero_count": 0,
    }
    assert groups["mixed"]["skill_injected"] == {
        "n": 2,
        "mean_delta": 6.0,
        "sample_standard_deviation": pytest.approx(1.4142135623730951),
        "positive_count": 2,
        "negative_count": 0,
        "zero_count": 0,
    }
    assert groups["mixed"]["within_task_contrast"] == {
        "availability": "available",
        "injected_minus_no_skill_mean_delta": 4.0,
    }
    assert groups["injected-only"]["no_skill_injected"] == {
        "n": 0,
        "mean_delta": None,
        "sample_standard_deviation": None,
        "positive_count": 0,
        "negative_count": 0,
        "zero_count": 0,
    }
    assert groups["injected-only"]["skill_injected"] == {
        "n": 2,
        "mean_delta": 3.0,
        "sample_standard_deviation": pytest.approx(1.4142135623730951),
        "positive_count": 2,
        "negative_count": 0,
        "zero_count": 0,
    }
    assert groups["null-only"]["no_skill_injected"] == {
        "n": 2,
        "mean_delta": -1.0,
        "sample_standard_deviation": pytest.approx(1.4142135623730951),
        "positive_count": 0,
        "negative_count": 1,
        "zero_count": 1,
    }
    assert groups["null-only"]["skill_injected"] == {
        "n": 0,
        "mean_delta": None,
        "sample_standard_deviation": None,
        "positive_count": 0,
        "negative_count": 0,
        "zero_count": 0,
    }
    assert groups["injected-only"]["within_task_contrast"]["reason"] == (
        "no_skill_injected_channel_missing"
    )
    assert groups["null-only"]["within_task_contrast"]["reason"] == (
        "skill_injected_channel_missing"
    )
    null_channel = report["null_channel"]
    assert "no_skill_injected" not in null_channel
    assert "skill_injected" not in null_channel

    rendered = render_admission_report(report)
    assert rendered.count("  task ") == 3
    assert (
        "no contrast available on this task, only the skill-injected channel "
        "exists (the no-skill-injected channel is missing)"
    ) in rendered
    assert (
        "no contrast available on this task, only the no-skill-injected channel "
        "exists (the skill-injected channel is missing)"
    ) in rendered
    assert "within-task difference (skill injected minus no skill injected): +4" in (
        rendered
    )
    assert "no skill injected (noise floor): n=0, mean=null, sample sd=null" in (
        rendered
    )


def test_headline_uses_retrieved_complete_candidates_as_pass_rate_denominator() -> None:
    candidates = [
        _candidate(
            "retrieved-admitted",
            [0.02] * 9 + [0.0],
            [True] * 10,
            task_name="riscv",
        )
    ]
    candidates.extend(
        _candidate(
            f"retrieved-rejected-{index}",
            [0.0] * 10,
            [True] * 10,
            task_name="riscv" if index < 5 else "spice",
        )
        for index in range(7)
    )
    candidates.extend(
        _candidate(
            f"never-retrieved-{index}",
            [0.0] * 10,
            [False] * 10,
            task_name="alp",
        )
        for index in range(5)
    )
    candidates.append(
        _candidate(
            "retrieval-unknown",
            [0.0] * 10,
            [None] * 10,
            task_name="spice",
        )
    )

    report = assemble_admission_report(candidates)
    split = report["retrieval_split"]

    assert report["schema_version"] == 2
    assert report["tested_candidate_count"] == 14
    assert report["admitted_candidate_count"] == 1
    assert report["rejected_candidate_count"] == 13
    assert report["pass_rate"] == 0.125
    assert report["pass_rate_numerator"] == 1
    assert report["pass_rate_denominator"] == 8
    assert report["pass_rate_denominator_description"] == (
        "retrieved_complete_candidates"
    )
    assert split == {
        "availability": "partial",
        "complete_candidate_count": 14,
        "retrieved_candidate_count": 8,
        "never_retrieved_candidate_count": 5,
        "unknown_retrieval_candidate_count": 1,
        "retrieved_admitted_candidate_count": 1,
        "retrieved_and_unhelpful_candidate_count": 7,
        "never_retrieved_admitted_candidate_count": 0,
        "unknown_retrieval_admitted_candidate_count": 0,
    }
    headline = render_admission_report(report).splitlines()[0]
    assert "never retrieved 5" in headline
    assert "retrieved and unhelpful 7" in headline
    assert "retrieved and admitted 1" in headline
    assert "pass rate 1/8 (12.5%) among retrieved complete candidates" in headline
    assert (
        "field reference 55/388 (14.2%) under a different validation filter "
        "(not like-for-like)"
    ) in headline


def test_loader_threads_task_metadata_and_missing_task_is_explicit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "validation.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "candidate_id": "known-task",
                        "arm": "localized",
                        "skill": serialize_skill(_skill("known-task")),
                        "paired_deltas": [0.0] * 10,
                        "injection_flags": [True] * 10,
                        "validation_observation_summary": {
                            "task_name": "validation-task",
                            "source_task_name": "source-task",
                        },
                    },
                    {
                        "candidate_id": "unknown-task",
                        "arm": "localized",
                        "skill": serialize_skill(_skill("unknown-task")),
                        "paired_deltas": [0.0] * 10,
                        "injection_flags": [False] * 10,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_admission_candidates(source)
    report = assemble_admission_report(candidates)

    assert candidates[0].task_name == "validation-task"
    assert candidates[0].source_task_name == "source-task"
    assert candidates[1].task_name is None
    unknown = _task_groups(report)[None]
    assert unknown["task_label"] == "task unknown"
    assert unknown["task_identity"] == "unknown"
    assert (
        "task unknown (validation_observation_summary_missing)"
        in render_admission_report(report)
    )


def test_task_and_injection_metadata_do_not_change_verdict_or_effect() -> None:
    deltas = [0.03] * 9 + [-0.01]
    first = decide_skill_admission(
        _candidate("same", deltas, [False] * 10, task_name="task-a")
    )
    second = decide_skill_admission(
        _candidate("same", deltas, [True] * 10, task_name="task-b")
    )

    assert first["status"] == second["status"] == "admitted"
    assert first["measurement"]["effect"] == second["measurement"]["effect"]
    assert json.dumps(first["measurement"]["effect"], sort_keys=True) == json.dumps(
        second["measurement"]["effect"], sort_keys=True
    )


def test_cli_reports_three_task_blocks_without_a_pooled_contrast(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "validation.json"
    output = tmp_path / "admission.json"
    candidates = []
    for candidate_id, task_name, deltas, flags in (
        ("mixed", "mixed", [1.0, 3.0, 5.0, 7.0], [False, False, True, True]),
        ("injected", "injected-only", [2.0], [True]),
        ("null", "null-only", [-2.0], [False]),
    ):
        candidates.append(
            {
                "candidate_id": candidate_id,
                "arm": "localized",
                "skill": serialize_skill(_skill(candidate_id)),
                "paired_deltas": deltas,
                "injection_flags": flags,
                "validation_observation_summary": {
                    "task_name": task_name,
                    "source_task_name": task_name,
                },
            }
        )
    source.write_text(
        json.dumps({"schema_version": 1, "candidates": candidates}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "admit-skills",
            str(source),
            "--library-dir",
            str(tmp_path / "fresh-library"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["null_channel"]["task_count"] == 3
    assert "no_skill_injected" not in report["null_channel"]
    assert "skill_injected" not in report["null_channel"]
    groups = _task_groups(report)
    assert (
        groups["injected-only"]["skill_injected"]["sample_standard_deviation"] is None
    )
    assert groups["injected-only"]["no_skill_injected"]["mean_delta"] is None
    rendered = capsys.readouterr().out
    assert rendered.count("  task ") == 3
    assert "within-task difference (skill injected minus no skill injected): +4" in (
        rendered
    )
    assert rendered.count("no contrast available on this task") == 2


def _raw_candidate(
    candidate_id: str,
    task_name: str | None,
    deltas: list[float],
    flags: list[bool],
) -> dict[str, object]:
    raw: dict[str, object] = {
        "candidate_id": candidate_id,
        "arm": "localized",
        "skill": serialize_skill(_skill(candidate_id)),
        "paired_deltas": deltas,
        "injection_flags": flags,
    }
    if task_name is not None:
        raw["validation_observation_summary"] = {
            "task_name": task_name,
            "source_task_name": task_name,
        }
    return raw


def _run_cli_report(
    tmp_path: Path,
    run_name: str,
    candidates: list[dict[str, object]],
) -> tuple[int, dict[str, object]]:
    run_dir = tmp_path / run_name
    run_dir.mkdir()
    source = run_dir / "validation.json"
    output = run_dir / "admission.json"
    source.write_text(
        json.dumps({"schema_version": 1, "candidates": candidates}),
        encoding="utf-8",
    )
    exit_code = main(
        [
            "admit-skills",
            str(source),
            "--library-dir",
            str(run_dir / "fresh-library"),
            "--output",
            str(output),
        ]
    )
    return exit_code, json.loads(output.read_text(encoding="utf-8"))


def test_cli_real_shape_reports_retrieval_split_and_task_local_contrasts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    alp_half_span = 0.2294
    candidates = [
        _raw_candidate(
            f"alp-{index}",
            "alp-paper-reproduction",
            [0.13 + alp_half_span, 0.13 - alp_half_span] * 5,
            [False] * 10,
        )
        for index in range(4)
    ]
    candidates.append(
        _raw_candidate(
            "riscv-admitted",
            "riscv-core-debug",
            [0.06] * 10,
            [True] * 10,
        )
    )
    candidates.extend(
        _raw_candidate(
            f"riscv-rejected-{index}",
            "riscv-core-debug",
            [0.18, -0.07475] * 5,
            [True] * 10,
        )
        for index in range(4)
    )
    candidates.append(
        _raw_candidate(
            "riscv-never",
            "riscv-core-debug",
            [0.13, -0.0056] * 5,
            [False] * 10,
        )
    )
    candidates.extend(
        _raw_candidate(
            f"spice-injected-{index}",
            "spice-ephemeris-regression",
            [0.05, -0.0864] * 5,
            [True] * 10,
        )
        for index in range(2)
    )
    spice_mixed_deltas = [
        0.05,
        -0.0864,
        0.05,
        -0.0864,
        -0.0182,
        0.02,
        -0.111,
        0.02,
        -0.111,
        -0.0455,
    ]
    candidates.extend(
        _raw_candidate(
            f"spice-mixed-{index}",
            "spice-ephemeris-regression",
            spice_mixed_deltas,
            [True] * 5 + [False] * 5,
        )
        for index in range(2)
    )

    exit_code, report = _run_cli_report(tmp_path, "real-shape", candidates)

    assert exit_code == 0
    assert report["tested_candidate_count"] == 14
    assert report["admitted_candidate_count"] == 1
    assert report["retrieval_split"]["never_retrieved_candidate_count"] == 5
    assert report["retrieval_split"]["retrieved_and_unhelpful_candidate_count"] == 8
    groups = _task_groups(report)
    assert groups["alp-paper-reproduction"]["no_skill_injected"][
        "mean_delta"
    ] == pytest.approx(0.13)
    assert groups["alp-paper-reproduction"]["within_task_contrast"]["reason"] == (
        "skill_injected_channel_missing"
    )
    assert groups["riscv-core-debug"]["skill_injected"]["mean_delta"] == pytest.approx(
        0.0541
    )
    assert groups["riscv-core-debug"]["no_skill_injected"][
        "mean_delta"
    ] == pytest.approx(0.0622)
    assert groups["riscv-core-debug"]["within_task_contrast"][
        "injected_minus_no_skill_mean_delta"
    ] == pytest.approx(-0.0081)
    assert groups["spice-ephemeris-regression"]["skill_injected"][
        "mean_delta"
    ] == pytest.approx(-0.0182)
    assert groups["spice-ephemeris-regression"]["no_skill_injected"][
        "mean_delta"
    ] == pytest.approx(-0.0455)
    assert groups["spice-ephemeris-regression"]["within_task_contrast"][
        "injected_minus_no_skill_mean_delta"
    ] == pytest.approx(0.0273)
    headline = capsys.readouterr().out.splitlines()[0]
    assert "never retrieved 5" in headline
    assert "retrieved and unhelpful 8" in headline
    assert "pass rate 1/9 (11.1%) among retrieved complete candidates" in headline


def test_cli_missing_task_metadata_exits_zero_and_uses_unknown_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code, report = _run_cli_report(
        tmp_path,
        "unknown-task",
        [_raw_candidate("unknown", None, [0.0] * 10, [False] * 10)],
    )

    assert exit_code == 0
    unknown = _task_groups(report)[None]
    assert unknown["task_label"] == "task unknown"
    assert unknown["no_skill_injected"]["n"] == 10
    assert (
        "task unknown (validation_observation_summary_missing)"
        in capsys.readouterr().out
    )


def test_cli_task_and_flags_leave_admission_result_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deltas = [0.03] * 9 + [-0.01]
    first_exit, first_report = _run_cli_report(
        tmp_path,
        "first-decision",
        [_raw_candidate("same", "task-a", deltas, [False] * 10)],
    )
    capsys.readouterr()
    second_exit, second_report = _run_cli_report(
        tmp_path,
        "second-decision",
        [_raw_candidate("same", "task-b", deltas, [True] * 10)],
    )

    assert first_exit == second_exit == 0
    first = first_report["decisions"][0]
    second = second_report["decisions"][0]
    assert first["status"] == second["status"] == "admitted"
    assert json.dumps(first["measurement"]["effect"], sort_keys=True) == json.dumps(
        second["measurement"]["effect"], sort_keys=True
    )


def test_same_short_name_under_different_sources_never_gets_a_contrast() -> None:
    skill = _skill("qualified identity")
    report = assemble_admission_report(
        [
            SkillAdmissionCandidate(
                candidate_id="suite-a",
                arm="localized",
                skill=skill,
                paired_deltas=(1.0, 3.0),
                injection_flags=(False, False),
                task_name="same-label",
                source_task_name="suiteA/same-label",
            ),
            SkillAdmissionCandidate(
                candidate_id="suite-b",
                arm="localized",
                skill=skill,
                paired_deltas=(5.0, 7.0),
                injection_flags=(True, True),
                task_name="same-label",
                source_task_name="suiteB/same-label",
            ),
        ]
    )

    groups = report["null_channel"]["per_task"]
    assert [group["task_label"] for group in groups] == [
        "suiteA/same-label",
        "suiteB/same-label",
    ]
    assert [group["within_task_contrast"]["availability"] for group in groups] == [
        "unavailable",
        "unavailable",
    ]
    assert groups[0]["no_skill_injected"]["mean_delta"] == 2.0
    assert groups[1]["skill_injected"]["mean_delta"] == 6.0


def test_task_identity_normalizes_case_and_whitespace_before_grouping() -> None:
    skill = _skill("normalized identity")
    report = assemble_admission_report(
        [
            SkillAdmissionCandidate(
                candidate_id="normalized-null",
                arm="localized",
                skill=skill,
                paired_deltas=(1.0, 3.0),
                injection_flags=(False, False),
                task_name="Task-Q",
                source_task_name="Suite/Task-Q",
            ),
            SkillAdmissionCandidate(
                candidate_id="normalized-injected",
                arm="localized",
                skill=skill,
                paired_deltas=(5.0, 7.0),
                injection_flags=(True, True),
                task_name="task-q ",
                source_task_name=" suite/task-q ",
            ),
        ]
    )

    groups = report["null_channel"]["per_task"]
    assert len(groups) == 1
    assert groups[0]["within_task_contrast"] == {
        "availability": "available",
        "injected_minus_no_skill_mean_delta": 4.0,
    }
    assert groups[0]["identity_condition"] == ("fully_qualified_task_identity_recorded")


def test_same_source_with_disagreeing_short_names_is_one_stated_condition() -> None:
    skill = _skill("metadata disagreement")
    report = assemble_admission_report(
        [
            SkillAdmissionCandidate(
                candidate_id="disagree-null",
                arm="localized",
                skill=skill,
                paired_deltas=(1.0,),
                injection_flags=(False,),
                task_name="short-a",
                source_task_name="suite/qualified",
            ),
            SkillAdmissionCandidate(
                candidate_id="disagree-injected",
                arm="localized",
                skill=skill,
                paired_deltas=(3.0,),
                injection_flags=(True,),
                task_name="short-b",
                source_task_name="suite/qualified",
            ),
        ]
    )

    groups = report["null_channel"]["per_task"]
    assert len(groups) == 1
    assert groups[0]["task_names"] == ["short-a", "short-b"]
    assert groups[0]["identity_condition"] == (
        "task_name_disagreement_within_source_identity"
    )
    assert "task identity condition: task_name_disagreement" in (
        render_admission_report(report)
    )


def test_loader_uses_top_level_fallback_and_splits_unknown_conditions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "task-metadata.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        **_raw_candidate("fallback", None, [0.0] * 10, [False] * 10),
                        "task_name": " Suite/Fallback ",
                    },
                    _raw_candidate("summary-missing", None, [0.0] * 10, [False] * 10),
                    {
                        **_raw_candidate(
                            "summary-null", None, [0.0] * 10, [False] * 10
                        ),
                        "validation_observation_summary": {
                            "task_name": None,
                            "source_task_name": None,
                        },
                    },
                    {
                        **_raw_candidate(
                            "top-level-disagrees", None, [0.0] * 10, [False] * 10
                        ),
                        "task_name": "other-task",
                        "validation_observation_summary": {
                            "task_name": "short-task",
                            "source_task_name": "suite/short-task",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_admission_candidates(source)
    report = assemble_admission_report(candidates)

    assert candidates[0].task_name == " Suite/Fallback "
    assert candidates[0].source_task_name == " Suite/Fallback "
    assert candidates[0].task_metadata_condition.value == (
        "top_level_fallback_summary_missing"
    )
    assert candidates[3].task_metadata_condition.value == (
        "summary_top_level_task_name_disagreement"
    )
    groups = report["null_channel"]["per_task"]
    assert len(groups) == 4
    unknown_conditions = {
        group["identity_condition"]
        for group in groups
        if group["task_identity"] == "unknown"
    }
    assert unknown_conditions == {
        "validation_observation_summary_missing",
        "validation_observation_summary_task_name_null",
    }
    rendered = render_admission_report(report)
    assert "task metadata condition(s): top_level_fallback_summary_missing" in rendered
    assert "summary_and_top_level_task_name_disagree" in rendered
    assert "task unknown (validation_observation_summary_missing)" in rendered
    assert "task unknown (validation_observation_summary_task_name_null)" in rendered


@pytest.mark.parametrize("flags", [None, [None] * 10])
def test_no_flags_headline_and_json_state_all_complete_denominator(
    flags: list[bool | None] | None,
) -> None:
    report = assemble_admission_report(
        [_candidate("unknown-retrieval", [0.0] * 10, flags, task_name="task")]
    )

    assert report["pass_rate"] == 0.0
    assert report["pass_rate_numerator"] == 0
    assert report["pass_rate_denominator"] == 1
    assert report["pass_rate_denominator_description"] == ("all_complete_candidates")
    assert "pass_rate" not in report["retrieval_split"]
    headline = render_admission_report(report).splitlines()[0]
    assert "retrieval unknown 1; retrieval could not be determined" in headline
    assert "pass rate 0/1 (0.0%) among all complete candidates" in headline


def test_never_retrieved_admission_is_explained_outside_rate_denominator() -> None:
    report = assemble_admission_report(
        [
            _candidate(
                "noise-admission",
                [0.02] * 9 + [0.0],
                [False] * 10,
                task_name="task",
            )
        ]
    )

    assert report["admitted_candidate_count"] == 1
    assert report["pass_rate"] is None
    assert report["pass_rate_numerator"] == 0
    assert report["pass_rate_denominator"] == 0
    assert report["admitted_outside_pass_rate_denominator_count"] == 1
    headline = render_admission_report(report).splitlines()[0]
    assert "pass rate 0/0 (not defined) among retrieved complete candidates" in headline
    assert "admitted outside the retrieval pass-rate denominator 1" in headline
    assert "(never retrieved 1, retrieval unknown 0)" in headline


def test_null_channel_shape_is_constant_and_task_entries_are_not_nested_reports() -> (
    None
):
    one_task = assemble_admission_report(
        [_candidate("one", [0.0] * 10, [False] * 10, task_name="one")]
    )["null_channel"]
    two_tasks = assemble_admission_report(
        [
            _candidate("two-a", [0.0] * 10, [False] * 10, task_name="one"),
            _candidate("two-b", [0.0] * 10, [True] * 10, task_name="two"),
        ]
    )["null_channel"]

    assert set(one_task) == set(two_tasks)
    assert "no_skill_injected" not in one_task
    assert "skill_injected" not in one_task
    assert "no_skill_injected" not in two_tasks
    assert "skill_injected" not in two_tasks
    for group in [*one_task["per_task"], *two_tasks["per_task"]]:
        assert "schema_version" not in group
        assert "rationale" not in group


def test_null_channel_states_incomplete_observation_scope() -> None:
    report = assemble_admission_report(
        [
            _candidate(
                "incomplete-observations",
                [1.0, 3.0] + [None] * 8,
                [False, True] + [None] * 8,
                task_name="task",
            )
        ]
    )

    scope = report["null_channel"]["observation_scope"]
    assert scope["includes_incomplete_candidates"] is True
    assert scope["incomplete_candidate_measured_observation_count"] == 2
    assert "need not equal tested_candidate_count multiplied by 10" in scope["note"]
    assert "including observations from incomplete candidates" in (
        render_admission_report(report)
    )


def test_task_metadata_condition_values_do_not_alias() -> None:
    values = [member.value for member in TaskMetadataCondition.__members__.values()]

    assert len(values) == len(set(values))
