from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftlock.checkpoint_localization import (
    assemble_localization_report,
    localize_scored_timeline,
)
from driftlock.lhtb_experiment import main


def _timeline(
    task: str,
    points: list[tuple[int, float | None]],
    *,
    final_reward: float | None = None,
) -> dict[str, object]:
    return {
        "trial_name": f"{task}__trial",
        "task_name": task,
        "final_reward": final_reward,
        "checkpoints": [
            {"phase": 0, "step": step, "reward": reward} for step, reward in points
        ],
    }


def test_riscv_timeline_localizes_only_its_six_flat_gaps() -> None:
    timeline = _timeline(
        "riscv-core-debug",
        [
            (0, 0.28),
            (5, 0.28),
            (10, 0.389091),
            (15, 0.389091),
            (20, 0.389091),
            (25, 0.607273),
            (31, 0.716364),
            (41, 0.770909),
            (46, 0.825455),
            (53, 0.825455),
            (58, 0.88),
            (79, 0.88),
            (84, 0.88),
        ],
    )

    result = localize_scored_timeline(timeline)

    assert result["status"] == "usable"
    assert [
        (
            segment["type"],
            segment["start"]["step"],
            segment["end"]["step"],
            segment["gap_count"],
        )
        for segment in result["segments"]
    ] == [
        ("flat", 0, 5, 1),
        ("flat", 10, 20, 2),
        ("flat", 46, 53, 1),
        ("flat", 58, 84, 2),
    ]


def test_uniform_checkpoint_scores_are_refused_even_with_a_different_final() -> None:
    result = localize_scored_timeline(
        _timeline(
            "2048",
            [(0, 0.0), (5, 0.0), (10, 0.0), (15, 0.0)],
            final_reward=0.712,
        )
    )

    assert result["status"] == "refused"
    assert result["refusal"]["reason"] == "uniform_checkpoint_scores"
    assert "all 4 scored checkpoints have reward 0" in result["refusal"]["detail"]
    assert "insensitive to intermediate state" in result["refusal"]["detail"]
    assert result["segments"] == []


def test_two_point_timeline_is_refused_for_insufficient_points_not_uniformity() -> None:
    result = localize_scored_timeline(
        _timeline("epidemic-inverse-control-audit", [(0, 0.027), (5, 0.027)])
    )

    assert result["status"] == "refused"
    assert result["refusal"]["reason"] == "insufficient_scored_checkpoints"
    assert "only 2 of 2 checkpoints are scored" in result["refusal"]["detail"]


def test_unscored_checkpoint_breaks_adjacency_instead_of_becoming_zero() -> None:
    result = localize_scored_timeline(
        _timeline(
            "partially-scored",
            [(0, 0.5), (5, None), (10, 0.25), (15, 0.5)],
        )
    )

    assert result["status"] == "usable"
    assert result["scored_checkpoint_count"] == 3
    assert result["unscored_checkpoint_count"] == 1
    assert result["segments"] == []


def test_flat_and_regressing_work_are_reported_as_different_segments() -> None:
    alp = localize_scored_timeline(
        _timeline(
            "alp-paper-reproduction",
            [(0, 0.2), (48, 0.2), (53, 0.0), (58, 0.3)],
        )
    )
    spice = localize_scored_timeline(
        _timeline(
            "spice-ephemeris-regression",
            [
                (0, 0.727273),
                (5, 0.757576),
                (10, 0.848485),
                (15, 0.757576),
                (20, 0.757576),
            ],
        )
    )

    assert [
        (item["type"], item["start"]["step"], item["end"]["step"])
        for item in alp["segments"]
    ] == [("flat", 0, 48), ("regression", 48, 53)]
    assert [
        (item["type"], item["start"]["step"], item["end"]["step"])
        for item in spice["segments"]
    ] == [("regression", 10, 15), ("flat", 15, 20)]


def test_job_report_states_coverage_and_refusal_reason_counts() -> None:
    score_report = {
        "schema_version": 1,
        "mode": "checkpoint-scoring",
        "source_job_dir": "/jobs/round-five",
        "trials": [
            _timeline("2048", [(0, 0.0), (5, 0.0), (10, 0.0)]),
            _timeline("sudoku-recovery", [(0, 0.0), (9, 0.0), (54, 0.0)]),
            _timeline("epidemic-inverse-control-audit", [(0, 0.027), (5, 0.027)]),
            _timeline(
                "alp-paper-reproduction",
                [(0, 0.2), (48, 0.2), (53, 0.0), (58, 0.3)],
            ),
            _timeline(
                "spice-ephemeris-regression",
                [(0, 0.73), (5, 0.76), (10, 0.85), (15, 0.76), (20, 0.76)],
            ),
            _timeline("riscv-core-debug", [(0, 0.28), (5, 0.28), (10, 0.39)]),
        ],
    }

    report = assemble_localization_report(score_report)

    assert report["task_count"] == 6
    assert report["usable_task_count"] == 3
    assert report["refused_task_count"] == 3
    assert report["coverage"] == 0.5
    assert report["refusal_reason_counts"] == {
        "insufficient_scored_checkpoints": 1,
        "uniform_checkpoint_scores": 2,
    }
    assert report["source_job_dir"] == "/jobs/round-five"


def test_malformed_timeline_refuses_only_its_task_in_job_report() -> None:
    score_report = {
        "schema_version": 1,
        "mode": "checkpoint-scoring",
        "trials": [
            _timeline(
                "riscv-core-debug",
                [(5, 0.28), (0, 0.28), (10, 0.389091)],
            ),
            _timeline(
                "spice-ephemeris-regression",
                [
                    (0, 0.7272727272727273),
                    (5, 0.7575757575757576),
                    (10, 0.8484848484848485),
                    (15, 0.7575757575757576),
                    (20, 0.7575757575757576),
                ],
            ),
        ],
    }

    report = assemble_localization_report(score_report)

    assert report["usable_task_count"] == 1
    assert report["refused_task_count"] == 1
    assert report["refusal_reason_counts"] == {"malformed_timeline": 1}
    bad, good = report["tasks"]
    assert bad["status"] == "refused"
    assert bad["refusal"] == {
        "reason": "malformed_timeline",
        "detail": (
            "timeline 'riscv-core-debug__trial' checkpoints are not strictly "
            "ordered by phase and step"
        ),
    }
    assert good["status"] == "usable"
    assert [item["type"] for item in good["segments"]] == ["regression", "flat"]


def test_float_round_trip_noise_is_flat_while_real_regression_remains() -> None:
    result = localize_scored_timeline(
        _timeline(
            "riscv-core-debug",
            [(0, 0.2800000000000001), (5, 0.28), (10, 0.389091)],
        )
    )

    assert result["status"] == "usable"
    assert len(result["segments"]) == 1
    segment = result["segments"][0]
    assert segment["type"] == "flat"
    assert (segment["start"]["step"], segment["end"]["step"]) == (0, 5)
    assert segment["reward_change"] < 0.0


def test_localize_checkpoints_cli_writes_machine_report_and_human_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    score_path = tmp_path / "checkpoint-scores.json"
    output_path = tmp_path / "localized.json"
    score_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "checkpoint-scoring",
                "trials": [
                    _timeline("usable", [(0, 0.1), (5, 0.1), (10, 0.2)]),
                    _timeline("refused", [(0, 0.0), (5, 0.0), (10, 0.0)]),
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "localize-checkpoints",
                str(score_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["mode"] == "checkpoint-localization"
    assert report["usable_task_count"] == 1
    assert report["refused_task_count"] == 1
    output = capsys.readouterr().out
    assert "usable 1/2 tasks (50.0%); refused 1" in output
    assert "usable: 1 non-improving segment(s)" in output
    assert "refused (uniform_checkpoint_scores)" in output
