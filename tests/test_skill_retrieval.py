from __future__ import annotations

import shutil
import socket
from collections.abc import Sequence
from pathlib import Path

import pytest

from driftlock.skill_admission import SkillAdmissionCandidate, SkillLibrary
from driftlock.skill_distillation import Skill
from driftlock.skill_retrieval import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_MIN_SIMILARITY_CALIBRATION,
    ActivationSkillRetriever,
    SkillRetrievalConfig,
    SkillRetrievalStatus,
    retrieve_for_distillation_arms,
)


class LiteralEmbedder:
    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, texts: Sequence[str]) -> list[Sequence[float]]:
        call = tuple(texts)
        self.calls.append(call)
        return [self.vectors[text] for text in call]


def _skill(activation: str, *, execution: str = "Use the applicable repair.") -> Skill:
    return Skill(
        activation=activation,
        execution=execution,
        termination="Stop when the repair is verified or the condition disappears.",
    )


def _admit(
    library: SkillLibrary,
    candidate_id: str,
    activation: str,
    *,
    arm: str = "baseline",
    execution: str = "Use the applicable repair.",
) -> None:
    decision = library.submit(
        SkillAdmissionCandidate(
            candidate_id=candidate_id,
            arm=arm,
            skill=_skill(activation, execution=execution),
            paired_deltas=(0.02,) * 10,
        )
    )
    assert decision["status"] == "admitted"


def test_default_minimum_similarity_and_calibration_provenance_are_pinned() -> None:
    assert DEFAULT_MIN_SIMILARITY == 0.35
    assert DEFAULT_MIN_SIMILARITY_CALIBRATION["model"] == {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "dimensions": 384,
    }
    assert DEFAULT_MIN_SIMILARITY_CALIBRATION["sample"] == {
        "candidate_count": 14,
        "task_count": 5,
        "pair_count": 70,
        "label_rule": (
            "A candidate distilled from task T is labelled applicable to T and "
            "inapplicable to each of the other four tasks."
        ),
        "applicable": {
            "count": 14,
            "minimum": 0.147,
            "median": 0.415,
            "maximum": 0.620,
        },
        "inapplicable": {
            "count": 56,
            "minimum": -0.013,
            "median": 0.188,
            "maximum": 0.330,
        },
    }
    assert DEFAULT_MIN_SIMILARITY_CALIBRATION["threshold_sweep"] == (
        {"threshold": 0.30, "recall_percent": 79, "false_retrieval_percent": 12},
        {"threshold": 0.35, "recall_percent": 71, "false_retrieval_percent": 0},
        {"threshold": 0.40, "recall_percent": 50, "false_retrieval_percent": 0},
        {"threshold": 0.50, "recall_percent": 7, "false_retrieval_percent": 0},
        {"threshold": 0.75, "recall_percent": 0, "false_retrieval_percent": 0},
    )
    assert DEFAULT_MIN_SIMILARITY_CALIBRATION["selection"] == {
        "threshold": 0.35,
        "recall_percent": 71,
        "false_retrieval_percent": 0,
        "rationale": (
            "0.35 and 0.40 both measured zero false retrievals; choose 0.35 for "
            "its higher measured recall."
        ),
    }


def test_default_similarity_boundary_and_old_dead_zone(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")
    activations = {
        "old-dead-zone": "activation at measured applicable maximum",
        "just-above": "activation just above calibrated boundary",
        "at-threshold": "activation exactly at calibrated boundary",
        "just-below": "activation just below calibrated boundary",
        "worst-inapplicable": "activation at measured inapplicable maximum",
    }
    for candidate_id, activation in activations.items():
        _admit(library, candidate_id, activation)
    query = "literal calibration query"
    embedder = LiteralEmbedder(
        {
            # Each vector has an integer norm: the resulting cosines against the
            # query are independently known literals 0.62, 0.36, 0.35, 0.34, 0.33.
            activations["old-dead-zone"]: (31, 39, 3, 3, 0, 0),
            activations["just-above"]: (9, 22, 7, 3, 1, 1),
            activations["at-threshold"]: (7, 15, 11, 2, 1, 0),
            activations["just-below"]: (17, 47, 1, 1, 0, 0),
            activations["worst-inapplicable"]: (33, 94, 8, 3, 1, 1),
            query: (1, 0, 0, 0, 0, 0),
        }
    )

    result = ActivationSkillRetriever(library, embedder).retrieve(query)

    assert result.config.minimum_similarity == 0.35
    assert [match.candidate_id for match in result.matches] == [
        "old-dead-zone",
        "just-above",
        "at-threshold",
    ]
    assert [match.similarity for match in result.matches] == [0.62, 0.36, 0.35]
    assert [
        candidate["candidate_id"] for candidate in result.near_threshold_candidates
    ] == [
        "just-below",
        "worst-inapplicable",
    ]
    assert [
        candidate["similarity"] for candidate in result.near_threshold_candidates
    ] == [
        0.34,
        0.33,
    ]


def test_empty_library_is_a_usable_no_match_without_embedding(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")

    def unexpected_embedding(_texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("an empty index must not invoke the embedder")

    result = ActivationSkillRetriever(library, unexpected_embedding).retrieve(
        "any query"
    )

    assert result.status is SkillRetrievalStatus.USABLE
    assert result.matches == ()
    assert result.to_report()["selected_skill_count"] == 0
    assert "refusal" not in result.to_report()
    assert result.to_report()["library_snapshot"]["candidate_count"] == 0


def test_clearly_applicable_single_skill_is_retrieved(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")
    activation = "When a generated parser rejects a valid trailing comma."
    query = "The generated parser rejects a valid trailing comma."
    _admit(library, "parser-repair", activation)
    embedder = LiteralEmbedder({activation: (1.0, 0.0), query: (1.0, 0.0)})

    result = ActivationSkillRetriever(library, embedder).retrieve(query)

    assert result.status is SkillRetrievalStatus.USABLE
    assert [match.candidate_id for match in result.matches] == ["parser-repair"]
    assert result.matches[0].similarity == 1.0
    assert "## activation" in result.injection_text
    assert '<retrieved-skill id="parser-repair">' in result.injection_text


def test_best_available_but_inapplicable_single_skill_is_not_retrieved(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    activation = "When a database migration deadlocks."
    query = "A CSS grid overflows its card."
    _admit(library, "database-deadlock", activation)
    embedder = LiteralEmbedder({activation: (1.0, 0.0), query: (0.0, 1.0)})

    result = ActivationSkillRetriever(library, embedder).retrieve(query)

    assert result.status is SkillRetrievalStatus.USABLE
    assert result.considered_candidate_count == 1
    assert result.applicable_candidate_count == 0
    assert result.matches == ()


def test_many_applicable_skills_are_score_ordered_and_bounded(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")
    activations = {
        "exact": "activation exact",
        "near-a": "activation near a",
        "near-b": "activation near b",
        "near-c": "activation near c",
        "irrelevant": "activation irrelevant",
    }
    for candidate_id, activation in activations.items():
        _admit(library, candidate_id, activation)
    query = "bounded query"
    embedder = LiteralEmbedder(
        {
            "activation exact": (1.0, 0.0),
            "activation near a": (5.0, 1.0),
            "activation near b": (3.0, 1.0),
            "activation near c": (4.0, 3.0),
            "activation irrelevant": (0.0, 1.0),
            query: (1.0, 0.0),
        }
    )

    result = ActivationSkillRetriever(library, embedder).retrieve(query)

    assert result.applicable_candidate_count == 4
    assert [match.candidate_id for match in result.matches] == [
        "exact",
        "near-a",
        "near-b",
    ]
    assert len(result.matches) == 3
    assert result.exclusions == (
        {
            "similarity_rank": 4,
            "candidate_id": "near-c",
            "similarity": 0.8,
            "reason": "max_skills",
        },
    )
    assert result.to_report()["configuration"]["max_skills"] == 3
    assert result.to_report()["configuration"]["max_injection_characters"] == 6000
    assert "context rot" in result.to_report()["configuration"]["context_policy"]


def test_injection_character_budget_is_a_second_hard_context_bound(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    activation = "activation too large for the configured injection budget"
    query = "small-budget query"
    _admit(library, "over-budget", activation)
    embedder = LiteralEmbedder({activation: (1.0, 0.0), query: (1.0, 0.0)})

    result = ActivationSkillRetriever(
        library,
        embedder,
        config=SkillRetrievalConfig(max_injection_characters=1),
    ).retrieve(query)

    assert result.status is SkillRetrievalStatus.USABLE
    assert result.applicable_candidate_count == 1
    assert result.matches == ()
    assert result.exclusions == (
        {
            "similarity_rank": 1,
            "candidate_id": "over-budget",
            "similarity": 1.0,
            "reason": "injection_character_budget",
        },
    )
    assert set(result.to_report()["exclusions"][0]) == {
        "similarity_rank",
        "candidate_id",
        "similarity",
        "reason",
    }


def test_similarity_ties_use_candidate_id_and_repeat_exactly(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")
    activation_alpha = "activation alpha"
    activation_zulu = "activation zulu"
    query = "tie query"
    _admit(library, "zulu", activation_zulu)
    _admit(library, "alpha", activation_alpha)
    embedder = LiteralEmbedder(
        {
            activation_alpha: (1.0, 0.0),
            activation_zulu: (1.0, 0.0),
            query: (1.0, 0.0),
        }
    )
    retriever = ActivationSkillRetriever(library, embedder)

    first = retriever.retrieve(query)
    second = retriever.retrieve(query)

    assert [match.candidate_id for match in first.matches] == ["alpha", "zulu"]
    assert first.to_report() == second.to_report()
    assert first.injection_text == second.injection_text
    assert embedder.calls == [
        (activation_alpha, activation_zulu),
        (query,),
    ]


def test_embedding_exception_is_failed_not_a_usable_empty_result(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    activation = "activation failure"
    _admit(library, "embedding-failure", activation)

    def failing_embedding(_texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("literal embedder outage")

    result = ActivationSkillRetriever(library, failing_embedding).retrieve("query")
    report = result.to_report()

    assert result.status is SkillRetrievalStatus.FAILED
    assert result.matches == ()
    assert report["refusal"] == {
        "reason": "embedding_callable_failed",
        "detail": (
            "activation index embedding call failed: RuntimeError: "
            "literal embedder outage"
        ),
        "stage": "index",
    }


@pytest.mark.parametrize(
    ("bad_query_vector", "detail"),
    [
        ((1.0, 0.0, 0.0), "query: vector 0 has length 3; expected 2"),
        (("not-a-number", 0.0), "query: vector 0 value 0 is not numeric"),
    ],
)
def test_unusable_embedding_vector_is_refused(
    tmp_path: Path,
    bad_query_vector: Sequence[object],
    detail: str,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    activation = "activation valid"
    query = "query invalid"
    _admit(library, "invalid-vector", activation)

    def embed(texts: Sequence[str]) -> list[Sequence[object]]:
        if tuple(texts) == (activation,):
            return [(1.0, 0.0)]
        return [bad_query_vector]

    result = ActivationSkillRetriever(library, embed).retrieve(query)

    assert result.status is SkillRetrievalStatus.FAILED
    assert result.to_report()["refusal"] == {
        "reason": "invalid_embedding",
        "detail": detail,
        "stage": "query",
    }


def test_both_arms_receive_the_same_single_retrieval_result(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")
    activation = "activation shared"
    query = "shared query"
    _admit(library, "shared-skill", activation, arm="localized")
    embedder = LiteralEmbedder({activation: (1.0, 0.0), query: (1.0, 0.0)})
    retriever = ActivationSkillRetriever(
        library,
        embedder,
        config=SkillRetrievalConfig(minimum_similarity=0.8, max_skills=2),
    )

    by_arm = retrieve_for_distillation_arms(retriever, query)

    assert tuple(by_arm) == ("baseline", "localized")
    assert by_arm["baseline"] is by_arm["localized"]
    assert by_arm["baseline"].to_report() == by_arm["localized"].to_report()
    assert embedder.calls == [(activation,), (query,)]


def test_library_growth_is_excluded_from_and_recorded_against_fixed_snapshot(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    first_activation = "activation round one"
    later_activation = "activation round three"
    query = "round query"
    _admit(library, "round-one", first_activation)
    embedder = LiteralEmbedder(
        {
            first_activation: (1.0, 0.0),
            later_activation: (1.0, 0.0),
            query: (1.0, 0.0),
        }
    )
    retriever = ActivationSkillRetriever(library, embedder)

    before_growth = retriever.retrieve(query)
    _admit(library, "round-three", later_activation)
    after_growth = retriever.retrieve(query)

    before_snapshot = before_growth.to_report()["library_snapshot"]
    after_snapshot = after_growth.to_report()["library_snapshot"]
    assert before_snapshot["growth_policy"] == "fixed_at_first_retrieval"
    assert before_snapshot["candidate_count"] == 1
    assert before_snapshot["candidate_ids"] == ["round-one"]
    assert before_snapshot == after_snapshot
    assert after_growth.to_report()["library_observation"] == {
        "current_candidate_count": 2,
        "current_candidate_ids": ["round-one", "round-three"],
        "added_after_snapshot": ["round-three"],
        "removed_after_snapshot": [],
        "snapshot_still_current": False,
    }
    assert [match.candidate_id for match in after_growth.matches] == ["round-one"]
    assert embedder.calls == [(first_activation,), (query,)]


def test_existing_retriever_honors_removals_but_keeps_additions_frozen(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    first_activation = "activation first"
    second_activation = "activation second"
    added_activation = "activation added later"
    query = "revocation query"
    _admit(library, "first", first_activation)
    _admit(library, "second", second_activation)
    embedder = LiteralEmbedder(
        {
            first_activation: (1.0, 0.0),
            second_activation: (1.0, 0.0),
            added_activation: (1.0, 0.0),
            query: (1.0, 0.0),
        }
    )
    retriever = ActivationSkillRetriever(library, embedder)

    before_change = retriever.retrieve(query)
    shutil.rmtree(library.entries / "first")
    _admit(library, "added", added_activation)
    after_change = retriever.retrieve(query)

    assert [match.candidate_id for match in before_change.matches] == [
        "first",
        "second",
    ]
    assert [match.candidate_id for match in after_change.matches] == ["second"]
    assert '<retrieved-skill id="second">' in after_change.injection_text
    assert '<retrieved-skill id="first">' not in after_change.injection_text
    assert '<retrieved-skill id="added">' not in after_change.injection_text
    assert after_change.to_report()["library_snapshot"]["growth_policy"] == (
        "fixed_at_first_retrieval"
    )
    assert after_change.to_report()["library_snapshot"]["removal_policy"] == (
        "honor_immediately"
    )
    assert after_change.to_report()["library_observation"] == {
        "current_candidate_count": 2,
        "current_candidate_ids": ["added", "second"],
        "added_after_snapshot": ["added"],
        "removed_after_snapshot": ["first"],
        "snapshot_still_current": False,
    }
    assert embedder.calls == [
        (first_activation, second_activation),
        (query,),
    ]


def test_threshold_diagnostics_record_nearest_misses_and_explicit_truncation(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    exact_activation = "activation exact diagnostic"
    near_activation = "activation near diagnostic"
    orthogonal_activation = "activation orthogonal diagnostic"
    opposite_activation = "activation opposite diagnostic"
    query = "diagnostic query"
    _admit(library, "exact", exact_activation)
    _admit(library, "near", near_activation)
    _admit(library, "orthogonal", orthogonal_activation)
    _admit(library, "opposite", opposite_activation)
    vectors = {
        exact_activation: (1.0, 0.0, 0.0, 0.0),
        near_activation: (3.0, 9.0, 3.0, 1.0),
        orthogonal_activation: (0.0, 1.0, 0.0, 0.0),
        opposite_activation: (-1.0, 0.0, 0.0, 0.0),
        query: (1.0, 0.0, 0.0, 0.0),
    }

    complete = ActivationSkillRetriever(
        library,
        LiteralEmbedder(vectors),
        config=SkillRetrievalConfig(max_diagnostic_candidates=3),
    ).retrieve(query)
    truncated = ActivationSkillRetriever(
        library,
        LiteralEmbedder(vectors),
        config=SkillRetrievalConfig(max_diagnostic_candidates=2),
    ).retrieve(query)

    assert complete.to_report()["threshold_diagnostics"] == {
        "below_threshold_candidate_count": 3,
        "reported_candidate_count": 3,
        "unreported_candidate_count": 0,
        "complete": True,
        "lowest_reported_similarity": -1.0,
        "candidates": [
            {
                "similarity_rank": 2,
                "candidate_id": "near",
                "similarity": 0.3,
                "activation": near_activation,
            },
            {
                "similarity_rank": 3,
                "candidate_id": "orthogonal",
                "similarity": 0.0,
                "activation": orthogonal_activation,
            },
            {
                "similarity_rank": 4,
                "candidate_id": "opposite",
                "similarity": -1.0,
                "activation": opposite_activation,
            },
        ],
    }
    assert truncated.to_report()["threshold_diagnostics"] == {
        "below_threshold_candidate_count": 3,
        "reported_candidate_count": 2,
        "unreported_candidate_count": 1,
        "complete": False,
        "lowest_reported_similarity": 0.0,
        "candidates": [
            {
                "similarity_rank": 2,
                "candidate_id": "near",
                "similarity": 0.3,
                "activation": near_activation,
            },
            {
                "similarity_rank": 3,
                "candidate_id": "orthogonal",
                "similarity": 0.0,
                "activation": orthogonal_activation,
            },
        ],
    }


def test_character_budget_can_select_a_shorter_lower_similarity_rank(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    oversized_activation = "activation oversized higher rank"
    shorter_activation = "activation shorter lower rank"
    query = "rank divergence query"
    _admit(
        library,
        "oversized",
        oversized_activation,
        execution="X" * 400,
    )
    _admit(library, "shorter", shorter_activation)
    embedder = LiteralEmbedder(
        {
            oversized_activation: (1.0, 0.0),
            shorter_activation: (3.0, 4.0),
            query: (1.0, 0.0),
        }
    )

    result = ActivationSkillRetriever(
        library,
        embedder,
        config=SkillRetrievalConfig(
            minimum_similarity=0.5,
            max_skills=2,
            max_injection_characters=300,
        ),
    ).retrieve(query)
    report = result.to_report()

    assert [match.candidate_id for match in result.matches] == ["shorter"]
    assert '<retrieved-skill id="shorter">' in result.injection_text
    assert '<retrieved-skill id="oversized">' not in result.injection_text
    assert result.exclusions == (
        {
            "similarity_rank": 1,
            "candidate_id": "oversized",
            "similarity": 1.0,
            "reason": "injection_character_budget",
        },
    )
    assert report["selection_matches_similarity_prefix"] is False
    assert report["selected_skills"][0]["selection_rank"] == 1
    assert report["selected_skills"][0]["similarity_rank"] == 2
    assert report["selected_skills"][0]["similarity"] == 0.6


def test_selected_skill_report_uses_only_explicit_order_names(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")
    activation = "activation explicit ranks"
    query = "explicit-rank query"
    _admit(library, "explicit-ranks", activation)
    embedder = LiteralEmbedder({activation: (1.0, 0.0), query: (1.0, 0.0)})

    selected = (
        ActivationSkillRetriever(library, embedder)
        .retrieve(query)
        .to_report()["selected_skills"][0]
    )

    assert "rank" not in selected
    assert set(selected) == {
        "selection_rank",
        "similarity_rank",
        "candidate_id",
        "similarity",
        "basis",
        "activation",
        "skill_sha256",
        "injection_characters",
        "injected_text",
    }
    assert selected["selection_rank"] == 1
    assert selected["similarity_rank"] == 1


def test_report_separates_frozen_snapshot_from_live_eligibility_after_removal(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    keep_activation = "activation keep live"
    old_activation = "activation revoke old"
    query = "live eligibility query"
    _admit(library, "keepskill", keep_activation)
    _admit(library, "oldskill", old_activation)
    embedder = LiteralEmbedder(
        {
            keep_activation: (1.0, 0.0),
            old_activation: (1.0, 0.0),
            query: (1.0, 0.0),
        }
    )
    retriever = ActivationSkillRetriever(library, embedder)
    retriever.retrieve(query)

    shutil.rmtree(library.entries / "oldskill")
    report = retriever.retrieve(query).to_report()

    assert report["library_snapshot"]["scope"] == (
        "frozen_initial_candidate_pool_not_live_eligibility"
    )
    assert report["library_snapshot"]["candidate_count"] == 2
    assert report["library_snapshot"]["candidate_ids"] == ["keepskill", "oldskill"]
    assert report["live_eligible_candidate_count"] == 1
    assert report["live_eligible_candidate_ids"] == ["keepskill"]
    assert report["considered_candidate_count"] == 1
    assert report["library_observation"]["removed_after_snapshot"] == ["oldskill"]


def test_rejected_library_decisions_never_enter_the_activation_index(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    admitted_activation = "activation admitted"
    rejected_activation = "activation rejected"
    query = "admission-filter query"
    _admit(library, "admitted", admitted_activation)
    rejected = library.submit(
        SkillAdmissionCandidate(
            candidate_id="rejected",
            arm="localized",
            skill=_skill(rejected_activation),
            paired_deltas=(0.02, -0.02) * 5,
        )
    )
    assert rejected["status"] == "rejected"
    embedder = LiteralEmbedder({admitted_activation: (1.0, 0.0), query: (1.0, 0.0)})

    result = ActivationSkillRetriever(library, embedder).retrieve(query)

    assert result.considered_candidate_count == 1
    assert [match.candidate_id for match in result.matches] == ["admitted"]
    assert embedder.calls == [(admitted_activation,), (query,)]


def test_audit_report_records_injected_skill_and_selection_basis(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    activation = "activation reportable"
    query = "reportable query"
    _admit(library, "reportable", activation)
    embedder = LiteralEmbedder({activation: (1.0, 0.0), query: (1.0, 0.0)})

    report = ActivationSkillRetriever(library, embedder).retrieve(query).to_report()

    assert report["selected_skills"][0]["candidate_id"] == "reportable"
    assert report["selected_skills"][0]["similarity"] == 1.0
    assert report["selected_skills"][0]["basis"] == (
        "activation cosine similarity met the configured threshold"
    )
    assert report["selected_skills"][0]["activation"] == activation
    assert report["selected_skills"][0]["injected_text"].startswith(
        '<retrieved-skill id="reportable">\n## activation'
    )


def test_retrieval_has_no_network_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = SkillLibrary(tmp_path / "library")
    activation = "activation offline"
    query = "offline query"
    _admit(library, "offline", activation)
    embedder = LiteralEmbedder({activation: (1.0, 0.0), query: (1.0, 0.0)})

    def refuse_socket(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("retrieval attempted network access")

    monkeypatch.setattr(socket, "socket", refuse_socket)

    result = ActivationSkillRetriever(library, embedder).retrieve(query)

    assert result.status is SkillRetrievalStatus.USABLE
    assert [match.candidate_id for match in result.matches] == ["offline"]
