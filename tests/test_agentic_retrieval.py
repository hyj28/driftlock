from __future__ import annotations

import importlib
import importlib.util
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from driftlock.agent import (
    AgentCompletion,
    AgentCompletionRequest,
    ToolCall,
    ToolCallingAgent,
)
from driftlock.agentic_retrieval import (
    AgenticRetrievalConfig,
    AgenticRetrievalStatus,
    AgenticRetrievalTool,
    RetrievalCorpusBuilder,
    RetrievalCorpusStatus,
    RetrievalDocumentKind,
)
from driftlock.local import LocalEnvironment, LocalWorkspaceDeltaObserver
from driftlock.models import StepContext
from driftlock.skill_admission import SkillAdmissionCandidate, SkillLibrary
from driftlock.skill_distillation import Skill
from driftlock.skill_injection import TaskSkillInjector
from driftlock.skill_retrieval import ActivationSkillRetriever


class LiteralEmbedder:
    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, texts: Sequence[str]) -> list[Sequence[float]]:
        call = tuple(texts)
        self.calls.append(call)
        return [self.vectors[text] for text in call]


def _cosine_vector(similarity: float) -> tuple[float, ...]:
    return (
        similarity,
        math.sqrt(1.0 - similarity * similarity),
        *((0.0,) * 382),
    )


_QUERY_VECTOR = (1.0, *((0.0,) * 383))


def _admit(library: SkillLibrary, candidate_id: str, activation: str) -> None:
    decision = library.submit(
        SkillAdmissionCandidate(
            candidate_id=candidate_id,
            arm="baseline",
            skill=Skill(
                activation=activation,
                execution="Inspect the placeholder and implement the external format.",
                termination="Stop after the externally specified output matches.",
            ),
            paired_deltas=(0.02,) * 10,
        )
    )
    assert decision["status"] == "admitted"


def _library(tmp_path: Path) -> SkillLibrary:
    return SkillLibrary(tmp_path / "library")


def test_live_context_routes_through_tool_while_legacy_instruction_misses(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    library = _library(tmp_path)
    activation = (
        "When you are in a placeholder repository and need to match an externally "
        "specified output format."
    )
    situation = (
        "I am in a placeholder repository trying to match the externally specified "
        "output format."
    )
    task_instruction = (
        "Reproduce the ALP paper implementation and submit the required artifact."
    )
    _admit(library, "placeholder-output", activation)

    old_embedder = LiteralEmbedder(
        {activation: (1.0, 0.0), task_instruction: (0.0, 1.0)}
    )
    injector = TaskSkillInjector(
        ActivationSkillRetriever(library, old_embedder), "baseline"
    )
    old_result = injector.retrieve_for_task(task_instruction)

    new_embedder = LiteralEmbedder(
        {
            activation: (1.0, 0.0),
            situation: (1.0, 0.0),
            task_instruction: (0.0, 1.0),
        }
    )
    tool = AgenticRetrievalTool.from_workspace(workspace, library, new_embedder)
    live_result = tool.retrieve(situation)
    new_task_result = tool.retrieve(task_instruction)

    assert old_result.matches == ()
    assert injector.to_report()["policy"] == {
        "retrieval_frequency": "once_per_task",
        "query_source": "original_task_instruction",
        "injection_position": "prepended_to_each_phase_entry_prompt",
        "retrieval_failure": "fail_trial_after_recording",
    }
    assert [match.document_id for match in live_result.matches] == [
        "skill:placeholder-output"
    ]
    assert live_result.matches[0].kind is RetrievalDocumentKind.SKILL
    assert live_result.matches[0].origin == "placeholder-output"
    assert live_result.matches[0].rank == 1
    assert new_task_result.matches == ()


def test_semantic_paraphrase_needs_no_shared_content_word(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    library = _library(tmp_path)
    activation = (
        "When you are in a placeholder repository and need to match an externally "
        "specified output format."
    )
    query = "The repo here is only a stub and the required shape is dictated elsewhere."
    distractors = {
        "database": "When a database migration deadlocks during schema changes.",
        "layout": "When CSS grid cards overflow their responsive container.",
        "parser": "When a generated parser rejects a valid trailing comma.",
    }
    _admit(library, "semantic-only", activation)
    for candidate_id, candidate_activation in distractors.items():
        _admit(library, candidate_id, candidate_activation)
    vectors = {
        activation: _cosine_vector(0.72),
        query: _QUERY_VECTOR,
        distractors["database"]: _cosine_vector(0.28),
        distractors["layout"]: _cosine_vector(0.24),
        distractors["parser"]: _cosine_vector(0.21),
    }
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        library,
        LiteralEmbedder(vectors),
    )

    result = tool.retrieve(query)

    assert [match.document_id for match in result.matches] == ["skill:semantic-only"]
    assert result.matches[0].lexical_evidence is False
    assert result.matches[0].semantic_evidence is True
    selected = next(
        item
        for item in result.to_report()["considered_documents"]
        if item["document_id"] == "skill:semantic-only"
    )
    assert selected["overlap_terms"] == []
    assert selected["similarity"] == pytest.approx(0.72)


def test_pinned_embedder_separates_live_situation_from_alp_instruction(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip(
            "install sentence-transformers in the project-local environment to "
            "enable the pinned driftlock.st_embedder integration test"
        )
    module = importlib.import_module("driftlock.st_embedder")
    embed = module.embed

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    library = _library(tmp_path)
    activation = (
        "When you are in a placeholder repository and need to match an externally "
        "specified output format."
    )
    for candidate_id, candidate_activation in (
        ("placeholder-output", activation),
        (
            "database-deadlock",
            "When a database migration deadlocks during schema changes.",
        ),
        ("grid-overflow", "When CSS grid cards overflow their responsive container."),
        ("parser-comma", "When a generated parser rejects a valid trailing comma."),
    ):
        _admit(library, candidate_id, candidate_activation)
    situation = (
        "The repo here is only a stub and the required shape is dictated elsewhere."
    )
    task_instruction = (
        "Reproduce the ALP paper implementation and submit the required artifact."
    )
    tool = AgenticRetrievalTool.from_workspace(workspace, library, embed)

    live_result = tool.retrieve(situation)
    task_result = tool.retrieve(task_instruction)

    assert "skill:placeholder-output" in {
        match.document_id for match in live_result.matches
    }
    assert "skill:placeholder-output" not in {
        match.document_id for match in task_result.matches
    }


@pytest.mark.parametrize(
    "scores",
    [
        (0.20, 0.10, 0.09, 0.08),
        (0.70, 0.62, 0.55, 0.49),
        (0.70, 0.50, 0.30, 0.29),
    ],
)
def test_semantic_rule_abstains_without_floor_or_distinct_cluster(
    tmp_path: Path, scores: tuple[float, ...]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    texts = tuple(f"document topic {index}" for index in range(4))
    for index, text in enumerate(texts):
        (workspace / f"doc-{index}.txt").write_text(text, encoding="utf-8")
    query = "sourdough starter maintenance in a cold kitchen"
    vectors = {
        **{
            text: _cosine_vector(similarity)
            for text, similarity in zip(texts, scores, strict=True)
        },
        query: _QUERY_VECTOR,
    }
    tool = AgenticRetrievalTool.from_workspace(
        workspace, _library(tmp_path), LiteralEmbedder(vectors)
    )

    result = tool.retrieve(query)

    assert result.matches == ()
    assert result.status is AgenticRetrievalStatus.USABLE
    assert result.exclusion_reason_counts == {
        "no_lexical_evidence_or_semantic_separation": 4
    }


def test_lexical_coverage_breaks_equal_cosine_ties(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    complete = "alpha beta implementation"
    partial = "alpha implementation"
    (workspace / "complete.txt").write_text(complete, encoding="utf-8")
    (workspace / "partial.txt").write_text(partial, encoding="utf-8")
    query = "alpha beta"
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        _library(tmp_path),
        LiteralEmbedder({complete: (1.0, 0.0), partial: (1.0, 0.0), query: (0.0, 1.0)}),
    )

    result = tool.retrieve(query)

    assert [match.origin for match in result.matches] == [
        "complete.txt",
        "partial.txt",
    ]
    assert [match.lexical_coverage for match in result.matches] == [1.0, 0.5]


def test_builder_indexes_skills_and_workspace_with_distinct_origins(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "The coordinator uses quorum lease fencing before failover.\n"
    (workspace / "coordinator.txt").write_text(source, encoding="utf-8")
    library = _library(tmp_path)
    activation = "When a placeholder repository must emit an external artifact."
    skill_query = "placeholder repository external artifact"
    _admit(library, "placeholder", activation)
    query = "quorum lease fencing failure"
    embedder = LiteralEmbedder(
        {
            activation: (0.0, 1.0),
            source: (1.0, 0.0),
            query: (1.0, 0.0),
            skill_query: (0.0, 1.0),
        }
    )

    corpus = RetrievalCorpusBuilder(workspace, library, embedder).build()
    tool = AgenticRetrievalTool(corpus)
    result = tool.retrieve(query)
    skill_result = tool.retrieve(skill_query)

    assert corpus.status is RetrievalCorpusStatus.READY
    assert corpus.build_report["indexed_kind_counts"] == {
        "skill": 1,
        "workspace": 1,
    }
    assert [(match.kind.value, match.origin) for match in result.matches] == [
        ("workspace", "coordinator.txt")
    ]
    assert result.matches[0].content == source
    assert [(match.kind.value, match.origin) for match in skill_result.matches] == [
        ("skill", "placeholder")
    ]
    report = result.to_report()
    assert report["considered_document_count"] == 2
    assert {item["kind"] for item in report["considered_documents"]} == {
        "skill",
        "workspace",
    }
    assert report["exclusion_reason_counts"] == {
        "no_lexical_evidence_or_semantic_separation": 1
    }


def test_tokenizer_strips_sentence_punctuation_and_splits_compounds(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "The quorum-lease uses owner_id."
    (workspace / "lease.txt").write_text(source, encoding="utf-8")
    query = "quorum lease owner id"
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        _library(tmp_path),
        LiteralEmbedder({source: (1.0, 0.0), query: (0.0, 1.0)}),
    )

    result = tool.retrieve(query)

    assert [match.origin for match in result.matches] == ["lease.txt"]
    assert result.matches[0].lexical_evidence is True
    assert result.to_report()["considered_documents"][0]["overlap_terms"] == [
        "id",
        "lease",
        "owner",
        "quorum",
    ]

    library = SkillLibrary(tmp_path / "second-library")
    (tmp_path / "empty-workspace").mkdir()
    activation = "When the agent cannot reach the hypervisor."
    _admit(library, "hypervisor", activation)
    hypervisor_query = "the hypervisor is unreachable"
    second = AgenticRetrievalTool.from_workspace(
        tmp_path / "empty-workspace",
        library,
        LiteralEmbedder({activation: (1.0, 0.0), hypervisor_query: (0.0, 1.0)}),
    )

    hypervisor_result = second.retrieve(hypervisor_query)

    assert [match.origin for match in hypervisor_result.matches] == ["hypervisor"]
    assert hypervisor_result.to_report()["considered_documents"][0][
        "overlap_terms"
    ] == ["hypervisor"]


def test_lexical_evidence_normalizes_nfd_workspace_text(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "réplica configuración"
    query = "re\u0301plica configuracio\u0301n"
    (workspace / "unicode.txt").write_text(source, encoding="utf-8")
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        _library(tmp_path),
        LiteralEmbedder({source: (1.0, 0.0), query: (0.0, 1.0)}),
    )

    result = tool.retrieve(query)

    assert [match.origin for match in result.matches] == ["unicode.txt"]
    assert result.to_report()["considered_documents"][0]["overlap_terms"] == [
        "configuración",
        "réplica",
    ]


def test_ignored_directories_are_configurable_and_audited(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ignored = (".git", ".hg", ".tox", ".venv", "__pycache__", "node_modules")
    for name in ignored:
        directory = workspace / name
        directory.mkdir()
        (directory / "dropped.txt").write_text("dropped content", encoding="utf-8")
    indexed = {
        "root.txt": "root content",
        "build/output.txt": "build content",
        "dist/output.txt": "dist content",
        "src/module.txt": "source content",
    }
    for relative, content in indexed.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    vectors = {content: (1.0, 0.0) for content in indexed.values()}

    corpus = RetrievalCorpusBuilder(
        workspace, _library(tmp_path), LiteralEmbedder(vectors)
    ).build()

    assert corpus.build_report["indexed_document_count"] == 4
    assert corpus.build_report["excluded_source_count"] == 6
    assert corpus.build_report["exclusion_reason_counts"] == {"ignored_directory": 6}
    assert [item["origin"] for item in corpus.build_report["excluded_sources"]] == [
        ".git/",
        ".hg/",
        ".tox/",
        ".venv/",
        "__pycache__/",
        "node_modules/",
    ]
    assert {document.origin for document in corpus.documents} == set(indexed)

    custom_workspace = tmp_path / "custom-workspace"
    custom_workspace.mkdir()
    (custom_workspace / "vendor").mkdir()
    (custom_workspace / "vendor" / "drop.txt").write_text(
        "vendor content", encoding="utf-8"
    )
    config = AgenticRetrievalConfig(ignored_directory_names=frozenset({"vendor"}))
    custom = RetrievalCorpusBuilder(
        custom_workspace,
        SkillLibrary(tmp_path / "custom-library"),
        LiteralEmbedder({}),
        config,
    ).build()

    assert custom.config.to_report()["ignored_directory_names"] == ["vendor"]
    assert custom.build_report["excluded_sources"] == [
        {
            "kind": "workspace",
            "origin": "vendor/",
            "reason": "ignored_directory",
        }
    ]


def test_default_cost_bounds_and_observation_shape_are_fingerprinted() -> None:
    config = AgenticRetrievalConfig()

    assert config.max_results_per_call == 4
    assert config.max_characters_per_task == 18_000
    assert config.workspace_chunk_overlap == 300
    assert config.to_report()["max_observation_exclusions"] == 20
    assert len(config.fingerprint) == 64
    assert (
        config.fingerprint
        != AgenticRetrievalConfig(max_observation_exclusions=1).fingerprint
    )


def test_every_workspace_file_exclusion_path_is_recorded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (workspace / "00-link.txt").symlink_to(outside)
    (workspace / "01-binary.bin").write_bytes(b"a\x00b")
    (workspace / "02-empty.txt").write_text("", encoding="utf-8")
    (workspace / "03-large.txt").write_text("123456", encoding="utf-8")
    (workspace / "04-indexed.txt").write_text("ok", encoding="utf-8")
    (workspace / "05-over-count.txt").write_text("no", encoding="utf-8")
    corpus = RetrievalCorpusBuilder(
        workspace,
        _library(tmp_path),
        LiteralEmbedder({"ok": (1.0, 0.0)}),
        AgenticRetrievalConfig(
            max_workspace_files=1,
            max_workspace_file_bytes=5,
        ),
    ).build()

    assert [document.origin for document in corpus.documents] == ["04-indexed.txt"]
    assert corpus.build_report["exclusion_reason_counts"] == {
        "binary_file": 1,
        "empty_text_file": 1,
        "symlink": 1,
        "workspace_file_byte_limit": 1,
        "workspace_file_limit": 1,
    }
    assert corpus.build_report["excluded_source_count"] == 5


def test_private_key_and_credential_names_are_never_indexed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sensitive = (
        ".env",
        ".git-credentials",
        ".npmrc",
        ".pypirc",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "server.pem",
    )
    for name in sensitive:
        (workspace / name).write_text("secret material", encoding="utf-8")
    (workspace / "id_ed25519.pub").write_text("public material", encoding="utf-8")
    corpus = RetrievalCorpusBuilder(
        workspace,
        _library(tmp_path),
        LiteralEmbedder({"public material": (1.0, 0.0)}),
    ).build()

    assert [document.origin for document in corpus.documents] == ["id_ed25519.pub"]
    assert corpus.build_report["exclusion_reason_counts"] == {"sensitive_filename": 8}
    assert {item["origin"] for item in corpus.build_report["excluded_sources"]} == set(
        sensitive
    )


def test_different_queries_are_stateless_and_return_different_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_text = "cache eviction uses a generation counter\n"
    lease_text = "lease renewal uses a fencing token\n"
    (workspace / "cache.txt").write_text(cache_text, encoding="utf-8")
    (workspace / "lease.txt").write_text(lease_text, encoding="utf-8")
    library = _library(tmp_path)
    cache_query = "generation counter cache eviction"
    lease_query = "fencing token lease renewal"
    embedder = LiteralEmbedder(
        {
            cache_text: (1.0, 0.0),
            lease_text: (0.0, 1.0),
            cache_query: (1.0, 0.0),
            lease_query: (0.0, 1.0),
        }
    )
    tool = AgenticRetrievalTool.from_workspace(workspace, library, embedder)

    first = tool.retrieve(cache_query)
    second = tool.retrieve(lease_query)

    assert [match.origin for match in first.matches] == ["cache.txt"]
    assert [match.origin for match in second.matches] == ["lease.txt"]
    assert [record.query for record in tool.audit_records] == [
        cache_query,
        lease_query,
    ]
    assert embedder.calls == [
        (cache_text, lease_text),
        (cache_query,),
        (lease_query,),
    ]


def test_no_match_is_empty_and_well_formed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "quorum lease fencing\n"
    (workspace / "notes.txt").write_text(source, encoding="utf-8")
    query = "watercolor pigment granulation"
    embedder = LiteralEmbedder({source: (1.0, 0.0), query: (0.0, 1.0)})
    tool = AgenticRetrievalTool.from_workspace(workspace, _library(tmp_path), embedder)

    result = tool.retrieve(query)
    observation = json.loads(result.to_observation())

    assert result.status is AgenticRetrievalStatus.USABLE
    assert result.matches == ()
    assert observation["status"] == "usable"
    assert observation["results"] == []
    assert observation["exclusion_reason_counts"] == {
        "no_lexical_evidence_or_semantic_separation": 1
    }
    assert "refusal" not in observation


def test_observation_cap_preserves_json_and_prioritizes_results(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    texts = {}
    for index in range(30):
        text = f"needle document {index:02d} " + ("x" * 180)
        texts[f"doc-{index:02d}.txt"] = text
        (workspace / f"doc-{index:02d}.txt").write_text(text, encoding="utf-8")
    query = "needle " + ("q" * 1_983)
    vectors = {text: (1.0, 0.0) for text in texts.values()}
    vectors[query] = (1.0, 0.0)
    tool = AgenticRetrievalTool.from_workspace(
        workspace, _library(tmp_path), LiteralEmbedder(vectors)
    )

    result = tool.retrieve(query)
    observation = result.to_observation()
    parsed = json.loads(observation)

    assert len(observation) <= 6_000
    assert len(parsed["results"]) == 4
    assert parsed["results"][0]["kind"] == "workspace"
    assert parsed["results"][0]["origin"] == "doc-00.txt"
    assert parsed["results"][0]["content"]
    assert parsed["results"][0]["chunk"] == {
        "start_character": 0,
        "end_character": 199,
    }
    assert len(parsed["exclusion_examples"]) == 6
    assert parsed["unreported_exclusion_count"] == 20
    assert "tool output truncated" not in observation
    assert [
        item["similarity_rank"] for item in result.to_report()["considered_documents"]
    ] == list(range(1, 31))


def test_per_call_result_cap_and_task_budget_are_explicit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    texts = {
        "one.txt": "shared alpha",
        "two.txt": "shared beta",
        "three.txt": "shared gamma",
    }
    for name, text in texts.items():
        (workspace / name).write_text(text, encoding="utf-8")
    shared_query = "shared evidence"
    alpha_query = "alpha"
    embedder = LiteralEmbedder(
        {
            "shared alpha": (1.0, 0.0),
            "shared beta": (4.0, 3.0),
            "shared gamma": (3.0, 4.0),
            shared_query: (1.0, 0.0),
            alpha_query: (1.0, 0.0),
        }
    )
    config = AgenticRetrievalConfig(
        max_results_per_call=1,
        max_characters_per_call=1_000,
        max_characters_per_task=13,
    )
    tool = AgenticRetrievalTool.from_workspace(
        workspace, _library(tmp_path), embedder, config=config
    )

    capped = tool.retrieve(shared_query)
    exhausted = tool.retrieve(alpha_query)

    assert [match.origin for match in capped.matches] == ["one.txt"]
    assert capped.per_call_result_cap_hit is True
    assert capped.exclusion_reason_counts == {"per_call_result_cap": 2}
    assert capped.to_report()["limits"]["per_call_result_cap"] == {
        "limit": 1,
        "hit": True,
        "excluded_count": 2,
    }
    assert exhausted.status is AgenticRetrievalStatus.TASK_BUDGET_INSUFFICIENT
    assert exhausted.matches == ()
    assert exhausted.per_task_character_budget_hit is True
    assert exhausted.to_report()["limits"]["per_task_character_budget"] == {
        "limit": 13,
        "before": 12,
        "contributed": 0,
        "after": 12,
        "hit": True,
        "excluded_count": 1,
    }
    assert exhausted.refusal == {
        "reason": "task_character_budget_insufficient",
        "stage": "selection",
        "detail": (
            "matching documents remained, but none fit the remaining task retrieval "
            "character budget"
        ),
    }


def test_partial_task_budget_is_insufficient_not_exhausted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    short = "match alpha"
    large = "match " + ("x" * 30)
    (workspace / "a-short.txt").write_text(short, encoding="utf-8")
    (workspace / "b-large.txt").write_text(large, encoding="utf-8")
    query = "match"
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        _library(tmp_path),
        LiteralEmbedder({short: (1.0, 0.0), large: (4.0, 3.0), query: (1.0, 0.0)}),
        config=AgenticRetrievalConfig(
            max_characters_per_call=1_000,
            max_characters_per_task=40,
        ),
    )

    result = tool.retrieve(query)

    assert [match.origin for match in result.matches] == ["a-short.txt"]
    assert result.status is AgenticRetrievalStatus.TASK_BUDGET_INSUFFICIENT
    assert result.task_characters_after == 11
    assert result.to_report()["limits"]["per_task_character_budget"] == {
        "limit": 40,
        "before": 0,
        "contributed": 11,
        "after": 11,
        "hit": True,
        "excluded_count": 1,
    }


def test_per_call_character_cap_is_explicit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "matching document " + ("x" * 1_100)
    (workspace / "large.txt").write_text(source, encoding="utf-8")
    query = "matching document"
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        _library(tmp_path),
        LiteralEmbedder({source: (1.0, 0.0), query: (1.0, 0.0)}),
        config=AgenticRetrievalConfig(
            max_characters_per_call=1_000,
            max_characters_per_task=2_000,
        ),
    )

    result = tool.retrieve(query)

    assert result.matches == ()
    assert result.per_call_character_cap_hit is True
    assert result.exclusion_reason_counts == {"per_call_character_cap": 1}
    assert result.to_report()["limits"]["per_call_character_cap"] == {
        "limit": 1_000,
        "used": 0,
        "hit": True,
        "excluded_count": 1,
    }


def test_exact_call_and_task_budget_boundaries_are_usable_then_exhausted(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alpha = "alpha12345"
    bravo = "bravo12345"
    (workspace / "alpha.txt").write_text(alpha, encoding="utf-8")
    (workspace / "bravo.txt").write_text(bravo, encoding="utf-8")
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        _library(tmp_path),
        LiteralEmbedder(
            {
                alpha: (1.0, 0.0),
                bravo: (0.0, 1.0),
            }
        ),
        config=AgenticRetrievalConfig(
            max_characters_per_call=10,
            max_characters_per_task=20,
        ),
    )

    first = tool.retrieve(alpha)
    second = tool.retrieve(bravo)
    third = tool.retrieve(alpha)

    assert [match.origin for match in first.matches] == ["alpha.txt"]
    assert first.returned_character_count == 10
    assert first.per_call_character_cap_hit is False
    assert first.status is AgenticRetrievalStatus.USABLE
    assert [match.origin for match in second.matches] == ["bravo.txt"]
    assert second.returned_character_count == 10
    assert second.task_characters_after == 20
    assert second.status is AgenticRetrievalStatus.TASK_BUDGET_EXHAUSTED
    assert second.per_task_character_budget_hit is True
    assert third.status is AgenticRetrievalStatus.TASK_BUDGET_EXHAUSTED
    assert third.matches == ()
    assert third.refusal is not None
    assert third.refusal["reason"] == "task_character_budget_exhausted"


def test_direct_invalid_query_uses_the_same_soft_result_contract(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = AgenticRetrievalTool.from_workspace(
        workspace, _library(tmp_path), LiteralEmbedder({})
    )

    empty = tool.retrieve("")
    wrong_type = tool.retrieve(7)
    overlong = tool.retrieve("x" * 2_001)

    assert empty.status is AgenticRetrievalStatus.FAILED
    assert empty.refusal == {
        "reason": "invalid_query",
        "stage": "input",
        "detail": "retrieval query must be non-empty text",
    }
    assert wrong_type.status is AgenticRetrievalStatus.FAILED
    assert wrong_type.query == "7"
    assert overlong.status is AgenticRetrievalStatus.FAILED
    assert overlong.refusal is not None
    assert overlong.refusal["detail"] == (
        "retrieval query exceeds max_query_characters (2000)"
    )
    assert [record.status.value for record in tool.audit_records] == [
        "failed",
        "failed",
        "failed",
    ]


def test_empty_corpus_is_usable_without_embedding(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = AgenticRetrievalTool.from_workspace(
        workspace, _library(tmp_path), LiteralEmbedder({})
    )

    result = tool.retrieve("valid query")

    assert result.status is AgenticRetrievalStatus.USABLE
    assert result.matches == ()
    assert result.corpus["status"] == "ready"


def test_query_stage_embedder_failure_is_soft_and_audited(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "query failure source"
    (workspace / "source.txt").write_text(source, encoding="utf-8")

    class QueryFailureEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, texts: Sequence[str]) -> list[Sequence[float]]:
            self.calls += 1
            if self.calls == 1:
                assert tuple(texts) == (source,)
                return [(1.0, 0.0)]
            raise RuntimeError("query embedding failed")

    tool = AgenticRetrievalTool.from_workspace(
        workspace, _library(tmp_path), QueryFailureEmbedder()
    )

    result = tool.retrieve("query failure")

    assert result.status is AgenticRetrievalStatus.FAILED
    assert result.refusal == {
        "reason": "embedding_callable_failed",
        "stage": "query",
        "detail": "RuntimeError: query embedding failed",
    }
    assert tool.audit_records == (result,)


async def test_agent_tool_observation_and_full_step_audit_are_well_formed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "parser recovery handles a trailing comma\n"
    (workspace / "parser.txt").write_text(source, encoding="utf-8")
    query = "parser trailing comma recovery"
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        _library(tmp_path),
        LiteralEmbedder({source: (1.0, 0.0), query: (1.0, 0.0)}),
    )

    class Provider:
        def __init__(self) -> None:
            self.requests: list[AgentCompletionRequest] = []

        async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
            self.requests.append(request)
            return AgentCompletion(
                tool_calls=(
                    ToolCall("retrieve_context", {"query": query}, "retrieve-1"),
                ),
                tokens=7,
            )

    provider = Provider()
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        retrieval_tool=tool,
    )
    context = StepContext(
        goal="repair parser",
        plan="inspect, repair, verify",
        state=agent.initial_state(),
        sequence=1,
        logical_step=1,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=None,
    )

    outcome = await agent(context)

    assert "retrieve_context" in {item.name for item in provider.requests[0].tools}
    observation = json.loads(outcome.tool_observations[0].split("\n", 1)[1])
    assert observation["status"] == "usable"
    assert observation["results"][0]["kind"] == "workspace"
    assert observation["results"][0]["origin"] == "parser.txt"
    assert observation["results"][0]["chunk"] == {
        "start_character": 0,
        "end_character": 41,
    }
    assert outcome.action == "Retrieve context for: parser trailing comma recovery"
    assert outcome.error is None
    assert len(outcome.tool_audits) == 1
    assert outcome.tool_audits[0]["tool_call"] == {
        "id": "retrieve-1",
        "name": "retrieve_context",
        "arguments": {"query": query},
    }
    assert outcome.tool_audits[0]["result"]["selected_document_count"] == 1
    encoded_state = json.dumps(outcome.state)
    assert '"name": "retrieve_context"' in encoded_state
    assert "parser.txt" in encoded_state


async def test_agent_receives_bounded_parseable_retrieval_json_with_results_first(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    texts = []
    for index in range(30):
        text = f"needle document {index:02d} " + ("x" * 180)
        texts.append(text)
        (workspace / f"doc-{index:02d}.txt").write_text(text, encoding="utf-8")
    query = "needle " + ("q" * 1_983)
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        _library(tmp_path),
        LiteralEmbedder(
            {
                **{text: (1.0, 0.0) for text in texts},
                query: (1.0, 0.0),
            }
        ),
    )

    async def provider(_request: AgentCompletionRequest) -> AgentCompletion:
        return AgentCompletion(
            tool_calls=(ToolCall("retrieve_context", {"query": query}, "bounded"),)
        )

    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        retrieval_tool=tool,
    )
    outcome = await agent(
        StepContext(
            goal="inspect needle documents",
            plan="retrieve",
            state=agent.initial_state(),
            sequence=1,
            logical_step=1,
            attempt=1,
            rollback_feedback=None,
            tokens_remaining=None,
        )
    )

    observation = outcome.tool_observations[0].split("\n", 1)[1]
    parsed = json.loads(observation)
    assert len(observation) <= 6_000
    assert len(parsed["results"]) == 4
    assert parsed["results"][0]["origin"] == "doc-00.txt"
    assert parsed["results"][0]["content"]
    assert parsed["unreported_exclusion_count"] > 0
    assert "tool output truncated" not in observation


async def test_unconfigured_agent_keeps_exact_legacy_five_tool_surface(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class Provider:
        def __init__(self) -> None:
            self.request: AgentCompletionRequest | None = None

        async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
            self.request = request
            return AgentCompletion()

    provider = Provider()
    agent = ToolCallingAgent(
        LocalEnvironment(workspace), LocalWorkspaceDeltaObserver(workspace), provider
    )
    await agent(
        StepContext(
            goal="legacy",
            plan="inspect",
            state=agent.initial_state(),
            sequence=1,
            logical_step=1,
            attempt=1,
            rollback_feedback=None,
            tokens_remaining=None,
        )
    )

    assert provider.request is not None
    assert [definition.name for definition in provider.request.tools] == [
        "run_shell",
        "read_file",
        "write_file",
        "search_files",
        "complete",
    ]


async def test_lone_surrogate_query_cannot_crash_agent_step(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "safe source"
    (workspace / "source.txt").write_text(source, encoding="utf-8")
    query = "\ud800"
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        _library(tmp_path),
        LiteralEmbedder({source: (1.0, 0.0), query: (0.0, 1.0)}),
    )

    async def provider(_request: AgentCompletionRequest) -> AgentCompletion:
        return AgentCompletion(
            tool_calls=(ToolCall("retrieve_context", {"query": query}, "surrogate"),)
        )

    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        retrieval_tool=tool,
    )
    outcome = await agent(
        StepContext(
            goal="survive malformed unicode",
            plan="retrieve",
            state=agent.initial_state(),
            sequence=1,
            logical_step=1,
            attempt=1,
            rollback_feedback=None,
            tokens_remaining=None,
        )
    )

    assert outcome.error is None
    assert len(outcome.tool_audits) == 1
    assert outcome.tool_audits[0]["result"]["query"]["character_count"] == 1


async def test_unexpected_retrieval_exception_is_audited_without_crashing_step(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class UnexpectedFailureTool(AgenticRetrievalTool):
        def retrieve(self, query: object):
            del query
            raise UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")

    base_tool = AgenticRetrievalTool.from_workspace(
        workspace, _library(tmp_path), LiteralEmbedder({})
    )
    tool = UnexpectedFailureTool(base_tool.corpus)

    async def provider(_request: AgentCompletionRequest) -> AgentCompletion:
        return AgentCompletion(
            tool_calls=(
                ToolCall("retrieve_context", {"query": "trigger"}, "unexpected"),
            )
        )

    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        retrieval_tool=tool,
    )
    outcome = await agent(
        StepContext(
            goal="survive retrieval exception",
            plan="retrieve",
            state=agent.initial_state(),
            sequence=1,
            logical_step=1,
            attempt=1,
            rollback_feedback=None,
            tokens_remaining=None,
        )
    )

    assert "UnicodeEncodeError" in (outcome.error or "")
    assert len(tool.audit_records) == 1
    assert tool.audit_records[0].status is AgenticRetrievalStatus.FAILED
    assert tool.audit_records[0].refusal is not None
    assert tool.audit_records[0].refusal["reason"] == "retrieval_execution_failed"
    assert outcome.tool_audits[0]["result"]["status"] == "failed"


async def test_raising_embedder_is_recorded_without_crashing_agent_step(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.txt").write_text("retrieval source\n", encoding="utf-8")

    def raising_embedder(_texts: Sequence[str]) -> list[Sequence[float]]:
        raise RuntimeError("deterministic embedder outage")

    tool = AgenticRetrievalTool.from_workspace(
        workspace, _library(tmp_path), raising_embedder
    )

    async def provider(_request: AgentCompletionRequest) -> AgentCompletion:
        return AgentCompletion(
            tool_calls=(
                ToolCall("retrieve_context", {"query": "retrieval source"}, "bad"),
            ),
            tokens=3,
        )

    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        retrieval_tool=tool,
    )
    context = StepContext(
        goal="inspect source",
        plan="retrieve",
        state=agent.initial_state(),
        sequence=1,
        logical_step=1,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=None,
    )

    outcome = await agent(context)
    observation = json.loads(outcome.tool_observations[0].split("\n", 1)[1])

    assert outcome.error is None
    assert observation["status"] == "failed"
    assert observation["refusal"] == {
        "detail": "RuntimeError: deterministic embedder outage",
        "reason": "embedding_callable_failed",
        "stage": "corpus_index",
    }
    assert outcome.tool_audits[0]["result"]["status"] == "failed"


async def test_malformed_agent_retrieval_call_is_counted_in_both_audits(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = AgenticRetrievalTool.from_workspace(
        workspace, _library(tmp_path), LiteralEmbedder({})
    )

    async def provider(_request: AgentCompletionRequest) -> AgentCompletion:
        return AgentCompletion(
            tool_calls=(ToolCall("retrieve_context", {"unexpected": "value"}, "bad"),)
        )

    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        retrieval_tool=tool,
    )
    outcome = await agent(
        StepContext(
            goal="inspect source",
            plan="retrieve",
            state=agent.initial_state(),
            sequence=1,
            logical_step=1,
            attempt=1,
            rollback_feedback=None,
            tokens_remaining=None,
        )
    )

    assert "malformed arguments for retrieve_context" in (outcome.error or "")
    assert len(tool.audit_records) == 1
    assert tool.audit_records[0].refusal is not None
    assert tool.audit_records[0].refusal["reason"] == "malformed_tool_arguments"
    assert len(outcome.tool_audits) == 1
    assert outcome.tool_audits[0]["result"]["refusal"]["reason"] == (
        "malformed_tool_arguments"
    )


def test_retrieval_strenum_values_are_all_unique() -> None:
    for enum_type in (
        RetrievalDocumentKind,
        RetrievalCorpusStatus,
        AgenticRetrievalStatus,
    ):
        values = [member.value for member in enum_type]
        assert len(values) == len(set(values))
