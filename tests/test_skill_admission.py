from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftlock.lhtb_experiment import main
from driftlock.skill_admission import (
    MIN_POSITIVE_TASKS,
    NULL_ADMISSION_PROBABILITY_UPPER_BOUND,
    VALIDATION_TASK_COUNT,
    SkillAdmissionCandidate,
    SkillAdmissionStatus,
    SkillLibrary,
    assemble_admission_report,
    decide_skill_admission,
    render_admission_report,
)
from driftlock.skill_distillation import Skill, serialize_skill


def _skill(label: str = "paired validation") -> Skill:
    return Skill(
        activation=f"When {label} applies.",
        execution="Do not guess from an aggregate; inspect paired tasks instead.",
        termination="Stop after every validation task has a result.",
    )


def _candidate(
    candidate_id: str,
    deltas: list[float | None],
    *,
    arm: str = "localized",
) -> SkillAdmissionCandidate:
    return SkillAdmissionCandidate(
        candidate_id=candidate_id,
        arm=arm,
        skill=_skill(candidate_id),
        paired_deltas=tuple(deltas),
    )


def test_nine_small_paired_improvements_and_one_zero_are_admitted() -> None:
    decision = decide_skill_admission(
        _candidate("consistent-small", [0.01] * 9 + [0.0])
    )

    assert decision["status"] == "admitted"
    assert decision["rule_id"] == "paired-direction-v1"
    assert decision["measurement"]["effect"] == {
        "mean_delta": 0.009,
        "median_delta": 0.01,
        "total_delta": 0.09,
        "minimum_delta": 0.0,
        "maximum_delta": 0.01,
        "positive_task_count": 9,
        "zero_task_count": 1,
        "negative_task_count": 0,
    }


def test_eight_paired_improvements_and_two_zeros_are_rejected() -> None:
    decision = decide_skill_admission(
        _candidate("one-below-boundary", [0.01] * 8 + [0.0, 0.0])
    )

    assert decision["status"] == "rejected"
    assert decision["measurement"]["effect"]["positive_task_count"] == 8
    assert decision["refusal"]["reason"] == "inconsistent_improvement"
    assert decision["refusal"]["detail"] == (
        "improved on 8 of 10 paired tasks; the shared rule requires at least 9 "
        "so one high-variance task cannot decide admission"
    )


def test_null_admission_bound_is_derived_for_exactly_nine_of_ten() -> None:
    assert VALIDATION_TASK_COUNT == 10
    assert MIN_POSITIVE_TASKS == 9
    assert NULL_ADMISSION_PROBABILITY_UPPER_BOUND == 0.0107421875


def test_mixed_paired_noise_around_zero_is_rejected() -> None:
    decision = decide_skill_admission(_candidate("mixed-noise", [0.02, -0.02] * 5))

    assert decision["status"] == "rejected"
    assert decision["refusal"]["reason"] == "inconsistent_improvement"
    assert decision["measurement"]["effect"]["positive_task_count"] == 5
    assert decision["measurement"]["effect"]["mean_delta"] == 0.0


def test_one_large_swing_cannot_admit_nine_small_regressions() -> None:
    decision = decide_skill_admission(_candidate("single-swing", [0.5] + [-0.05] * 9))

    assert decision["measurement"]["effect"]["mean_delta"] == pytest.approx(0.005)
    assert decision["status"] == "rejected"
    assert decision["refusal"]["reason"] == "inconsistent_improvement"
    assert "improved on 1 of 10 paired tasks" in decision["refusal"]["detail"]


def test_positive_mean_cannot_hide_one_loss_larger_than_nine_tiny_gains() -> None:
    decision = decide_skill_admission(_candidate("net-negative", [0.01] * 9 + [-0.1]))

    assert decision["measurement"]["effect"]["positive_task_count"] == 9
    assert decision["measurement"]["effect"]["mean_delta"] == pytest.approx(-0.001)
    assert decision["status"] == "rejected"
    assert decision["refusal"]["reason"] == "nonpositive_mean_effect"


def test_report_preserves_effect_size_difference_between_small_and_strong_wins() -> (
    None
):
    report = assemble_admission_report(
        [
            _candidate("consistent-small", [0.01] * 9 + [0.0]),
            _candidate("uniform-strong", [0.3] * 10),
        ]
    )

    assert report["tested_candidate_count"] == 2
    assert report["admitted_candidate_count"] == 2
    assert report["pass_rate"] == 1.0
    small, strong = report["decisions"]
    assert small["measurement"]["effect"]["mean_delta"] == 0.009
    assert strong["measurement"]["effect"]["mean_delta"] == pytest.approx(0.3)
    assert small["measurement"]["effect"]["zero_task_count"] == 1
    assert strong["measurement"]["effect"]["zero_task_count"] == 0


def test_admitted_candidate_line_and_library_record_carry_null_context(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate("survivor", [0.03] * 10),
        _candidate("ordinary-rejection", [0.02, -0.02] * 5),
    ]
    library = SkillLibrary(tmp_path / "library")

    report = library.submit_cohort(candidates)
    record = library.read_decision("survivor")

    expected_context = {
        "single_candidate_null_admission_probability_upper_bound": 0.0107421875,
        "interpretation": (
            "Admission means this candidate passed a directional effect screen; "
            "it is not an individual statistical certification. Null candidates "
            "can survive, and the screen protects expected library composition "
            "only when read with cohort context."
        ),
        "cohort": {
            "tested_candidate_count": 2,
            "observed_admitted_candidate_count": 1,
            "all_null_expected_chance_admissions_upper_bound": 0.021484375,
            "interpretation": (
                "Under the stated null model this many admissions are expected by "
                "chance across the cohort; it does not identify which individual "
                "admissions are false."
            ),
        },
    }
    assert report["decisions"][0]["admission_context"] == expected_context
    assert record["admission_context"] == expected_context
    survivor_line = next(
        line
        for line in render_admission_report(report).splitlines()
        if "survivor" in line
    )
    assert "directional screen only, not individual certification" in survivor_line
    assert "null admission upper bound 1.074% per candidate" in survivor_line
    assert "cohort all-null expectation at most 0.021 across 2 tests" in survivor_line


def test_render_qualifies_field_reference_beside_our_pass_rate() -> None:
    report = assemble_admission_report(
        [
            _candidate("one-admitted", [0.03] * 10),
            _candidate("one-rejected", [0.02, -0.02] * 5),
        ]
    )

    first_line = render_admission_report(report).splitlines()[0]

    assert first_line == (
        "tested 2 complete candidate(s); admitted 1; rejected 1; incomplete 0; "
        "pass rate 50.0%; field reference 55/388 (14.2%) under a different "
        "validation filter (not like-for-like)"
    )
    assert report["field_reference"]["comparison_note"] == (
        "The study used a different validation filter; its pass rate and this "
        "directional-screen pass rate are juxtaposed as references, not treated "
        "as like-for-like estimates."
    )


@pytest.mark.parametrize(
    "deltas",
    [
        [0.1] * 7,
        [0.1, 0.1, 0.1, None, None, None, 0.1, 0.1, 0.1, 0.1],
    ],
)
def test_incomplete_validation_is_not_averaged(deltas: list[float | None]) -> None:
    decision = decide_skill_admission(_candidate("incomplete", deltas))

    assert decision["status"] == "incomplete"
    assert decision["measurement"]["measured_task_count"] == 7
    assert decision["measurement"]["missing_task_count"] == 3
    assert decision["measurement"]["effect"] is None
    assert decision["refusal"]["reason"] == "incomplete_validation"
    assert "missing results are not zeros" in decision["refusal"]["detail"]


def test_eight_results_are_incomplete_not_a_smaller_sign_test() -> None:
    candidate = _candidate("eight-results", [0.1] * 8)

    decision = decide_skill_admission(candidate)
    report = assemble_admission_report([candidate])

    assert decision["status"] == "incomplete"
    assert decision["measurement"] == {
        "expected_task_count": 10,
        "measured_task_count": 8,
        "missing_task_count": 2,
        "paired_deltas": [0.1] * 8,
        "effect": None,
    }
    assert report["tested_candidate_count"] == 0
    assert report["incomplete_candidate_count"] == 1
    assert report["pass_rate"] is None
    assert report["multiple_comparisons"]["candidate_tests"] == 0
    assert (
        report["multiple_comparisons"][
            "all_null_expected_chance_admissions_upper_bound"
        ]
        == 0.0
    )


def test_hundred_null_candidates_report_the_multiple_comparisons_expectation() -> None:
    candidates = [
        _candidate(f"null-{index:03d}", [0.02, -0.02] * 5) for index in range(100)
    ]

    report = assemble_admission_report(candidates)

    assert report["submitted_candidate_count"] == 100
    assert report["tested_candidate_count"] == 100
    assert report["admitted_candidate_count"] == 0
    assert report["rejected_candidate_count"] == 100
    assert report["pass_rate"] == 0.0
    assert report["multiple_comparisons"] == {
        "candidate_tests": 100,
        "single_candidate_null_admission_probability_upper_bound": 0.0107421875,
        "all_null_expected_chance_admissions_upper_bound": 1.07421875,
        "interpretation": (
            "Expectation assumes each complete candidate has independent, symmetric "
            "paired task-level null signs; ties can only lower the bound. Linearity "
            "of expectation does not require candidates tested on the shared split "
            "to be independent. Correlated task signs invalidate the bound. It is "
            "not a per-candidate p-value or a family-wise significance claim."
        ),
    }
    human = render_admission_report(report)
    assert "across 100 tests (1.074% per candidate upper bound)" in human
    assert "not a multiplicity-corrected significance claim" in human


def test_both_distillation_arms_use_the_identical_admission_rule() -> None:
    deltas = [0.02] * 9 + [0.0]
    baseline = decide_skill_admission(
        _candidate("baseline-candidate", deltas, arm="baseline")
    )
    localized = decide_skill_admission(
        _candidate("localized-candidate", deltas, arm="localized")
    )

    assert baseline["status"] == localized["status"] == "admitted"
    assert baseline["rule_id"] == localized["rule_id"] == "paired-direction-v1"
    assert baseline["measurement"] == localized["measurement"]


def test_library_stores_admitted_skill_and_retrievable_rejection(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    admitted = _candidate("admitted", [0.02] * 10)
    rejected = _candidate("rejected", [0.5] + [-0.05] * 9)

    assert library.submit(admitted)["status"] == "admitted"
    assert library.submit(rejected)["status"] == "rejected"

    assert library.read_skill("admitted") == admitted.skill
    with pytest.raises(FileNotFoundError, match="has no admitted skill"):
        library.read_skill("rejected")
    refusal = library.read_decision("rejected")["refusal"]
    assert refusal["reason"] == "inconsistent_improvement"
    assert "improved on 1 of 10 paired tasks" in refusal["detail"]


def test_admit_skills_cli_updates_library_and_writes_quoteable_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "validation.json"
    output_path = tmp_path / "admission.json"
    library_path = tmp_path / "library"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "candidate_id": "cli-admitted",
                        "arm": "baseline",
                        "skill": serialize_skill(_skill("CLI admission")),
                        "paired_deltas": [0.04] * 9 + [0.0],
                    },
                    {
                        "candidate_id": "cli-incomplete",
                        "arm": "localized",
                        "skill": serialize_skill(_skill("CLI refusal")),
                        "paired_deltas": [0.04] * 7,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "admit-skills",
                str(input_path),
                "--library-dir",
                str(library_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["tested_candidate_count"] == 1
    assert report["incomplete_candidate_count"] == 1
    assert report["admitted_candidate_count"] == 1
    assert SkillLibrary(library_path).read_skill("cli-admitted") == _skill(
        "CLI admission"
    )
    output = capsys.readouterr().out
    assert "tested 1 complete candidate(s); admitted 1" in output
    assert "all-null chance expectation" in output
    assert f"wrote {output_path}" in output


def test_skill_admission_status_values_are_stable() -> None:
    assert [status.value for status in SkillAdmissionStatus.__members__.values()] == [
        "admitted",
        "rejected",
        "incomplete",
    ]
