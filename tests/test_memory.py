from __future__ import annotations

import json
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from driftlock.agent import (
    AgentCompletion,
    AgentCompletionRequest,
    ToolCall,
    ToolCallingAgent,
)
from driftlock.agentic_retrieval import (
    AgenticRetrievalTool,
    RetrievalCorpusBuilder,
    RetrievalDocumentKind,
)
from driftlock.local import LocalEnvironment, LocalWorkspaceDeltaObserver
from driftlock.memory import (
    MemoryEntryStatus,
    MemoryMutationStatus,
    MemoryOperation,
    MemoryProvenance,
    MemoryRecoveryKind,
    MemoryStore,
    MemoryStoreConfig,
    MemoryStoreFormatError,
)
from driftlock.models import StepContext
from driftlock.skill_admission import SkillAdmissionCandidate, SkillLibrary
from driftlock.skill_distillation import Skill


class LiteralEmbedder:
    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, texts: Sequence[str]) -> list[Sequence[float]]:
        call = tuple(texts)
        self.calls.append(call)
        return [self.vectors[text] for text in call]


class ScriptedProvider:
    def __init__(self, *responses: AgentCompletion) -> None:
        self.responses = list(responses)
        self.requests: list[AgentCompletionRequest] = []

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
        self.requests.append(request)
        return self.responses.pop(0)


def _vector(similarity: float) -> tuple[float, ...]:
    return (
        similarity,
        math.sqrt(1.0 - similarity * similarity),
        *((0.0,) * 382),
    )


_RELATED_QUERY_VECTOR = (1.0, *((0.0,) * 383))
_UNRELATED_QUERY_VECTOR = (0.0, 0.0, 1.0, *((0.0,) * 381))


def _provenance(
    task_id: str = "task-parser", run_id: str = "run-07"
) -> MemoryProvenance:
    return MemoryProvenance(task_id, run_id, 3, 2, 1)


def _context(agent: ToolCallingAgent) -> StepContext:
    return StepContext(
        goal="repair parser",
        plan="inspect, patch, verify",
        state=agent.initial_state(),
        sequence=3,
        logical_step=2,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=None,
    )


def _admit(library: SkillLibrary, candidate_id: str, activation: str) -> None:
    decision = library.submit(
        SkillAdmissionCandidate(
            candidate_id=candidate_id,
            arm="baseline",
            skill=Skill(
                activation=activation,
                execution="Inspect the relevant subsystem.",
                termination="Stop after its checks pass.",
            ),
            paired_deltas=(0.02,) * 10,
        )
    )
    assert decision["status"] == "admitted"


def test_record_persists_with_provenance_across_fresh_store(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    store = MemoryStore(root)
    provenance = _provenance()

    result = store.record("The repository test entry point is make check.", provenance)
    fresh = MemoryStore(root)
    entry = fresh.read("memory-000001")

    assert result.status is MemoryMutationStatus.APPLIED
    assert result.memory_id == "memory-000001"
    assert (root / "entries" / "memory-000001.json").is_file()
    assert entry.current_content == "The repository test entry point is make check."
    assert entry.status is MemoryEntryStatus.ACTIVE
    assert entry.revision == 1
    assert entry.current_provenance == provenance
    assert entry.to_dict()["events"][0]["provenance"] == {
        "task_id": "task-parser",
        "run_id": "run-07",
        "sequence": 3,
        "logical_step": 2,
        "attempt": 1,
    }


def test_correction_retains_audit_history_but_only_new_claim_is_current(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    first = _provenance("task-one", "run-one")
    second = MemoryProvenance("task-two", "run-two", 8, 5, 2)
    store.record("Run pytest directly.", first)

    result = store.correct(
        "memory-000001",
        "Run make check; it configures required fixtures.",
        "The Makefile is the current repository authority.",
        second,
    )
    entry = MemoryStore(tmp_path / "memory").read("memory-000001")

    assert result.status is MemoryMutationStatus.APPLIED
    assert entry.current_content == "Run make check; it configures required fixtures."
    assert entry.current_provenance == second
    assert [event.operation for event in entry.events] == [
        MemoryOperation.RECORD,
        MemoryOperation.CORRECT,
    ]
    assert [event.content for event in entry.events] == [
        "Run pytest directly.",
        "Run make check; it configures required fixtures.",
    ]
    assert entry.events[1].reason == (
        "The Makefile is the current repository authority."
    )
    assert result.changes == (
        {
            "kind": "corrected",
            "revision": 2,
            "superseded_revision": 1,
            "before_content_sha256": (
                "c20a394847abbc05b5ee6a9061ae2a15c0ed1eb51679d4589f4252f4c2da07d2"
            ),
            "after_content_sha256": (
                "d99fd4a0e443f9f003e0aa27ae4bdd5fcc7818c32d3a63322fdaf5c5c9eb6b4e"
            ),
            "content_character_count": 48,
            "reason_sha256": (
                "4fe18824cd275f89f7385bfd86fd48f2987c415a1a49dcbfe9378780a9f7bbda"
            ),
            "reason_character_count": 49,
        },
    )


def test_revoke_is_audited_and_removes_entry_from_current_set(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    provenance = _provenance()
    store.record("The integration database is pre-seeded.", provenance)

    result = store.revoke(
        "memory-000001", "A clean run showed the database is empty.", provenance
    )
    entry = MemoryStore(tmp_path / "memory").read("memory-000001")

    assert result.status is MemoryMutationStatus.APPLIED
    assert entry.status is MemoryEntryStatus.REVOKED
    assert entry.current_content is None
    assert entry.events[-1].reason == "A clean run showed the database is empty."
    assert store.current_entries() == ()


@pytest.mark.parametrize("starting_status", ["missing", "active", "revoked"])
@pytest.mark.parametrize("operation", [MemoryOperation.CORRECT, MemoryOperation.REVOKE])
def test_complete_target_status_operation_matrix(
    tmp_path: Path, starting_status: str, operation: MemoryOperation
) -> None:
    store = MemoryStore(tmp_path / f"{starting_status}-{operation.value}")
    provenance = _provenance()
    target = "memory-999999"
    if starting_status != "missing":
        recorded = store.record("Original fact.", provenance)
        assert recorded.memory_id is not None
        target = recorded.memory_id
    if starting_status == "revoked":
        store.revoke(target, "No longer true.", provenance)

    if operation is MemoryOperation.CORRECT:
        result = store.correct(target, "Replacement fact.", "Observed now.", provenance)
    else:
        result = store.revoke(target, "Observed now.", provenance)

    expected = (
        MemoryMutationStatus.APPLIED
        if starting_status == "active"
        else MemoryMutationStatus.REJECTED
    )
    assert result.status is expected
    if starting_status == "revoked":
        assert result.error == "revoked memory is terminal"
    elif starting_status == "missing":
        assert result.error == "memory 'memory-999999' does not exist"


def test_each_independent_store_cap_is_rejected_and_recorded(tmp_path: Path) -> None:
    provenance = _provenance("t", "r")
    count_store = MemoryStore(
        tmp_path / "count",
        config=MemoryStoreConfig(max_entries=1, max_store_bytes=10_000),
    )
    assert (
        count_store.record("first", provenance).status is MemoryMutationStatus.APPLIED
    )
    count_result = count_store.record("second", provenance)

    length_store = MemoryStore(
        tmp_path / "length",
        config=MemoryStoreConfig(max_content_characters=5, max_store_bytes=10_000),
    )
    length_result = length_store.record("123456", provenance)

    size_store = MemoryStore(
        tmp_path / "size", config=MemoryStoreConfig(max_store_bytes=300)
    )
    size_result = size_store.record("small", provenance)

    assert count_result.status is MemoryMutationStatus.REJECTED
    assert count_result.limits["entry_count"]["hit"] is True
    assert count_store.memory_ids() == ("memory-000001",)
    assert length_result.status is MemoryMutationStatus.REJECTED
    assert length_result.limits["entry_length"] == {
        "limit": 5,
        "attempted": 6,
        "hit": True,
    }
    assert length_store.memory_ids() == ()
    assert size_result.status is MemoryMutationStatus.REJECTED
    assert size_result.memory_id is None
    assert size_result.limits["store_size"] == {
        "limit": 300,
        "before": 0,
        "after": 0,
        "hit": True,
    }
    assert size_store.memory_ids() == ()


@pytest.mark.parametrize(
    ("operation", "ordinary_cap", "used_before"),
    [
        (MemoryOperation.CORRECT, 376, 0),
        (MemoryOperation.CORRECT, 375, 1),
        (MemoryOperation.REVOKE, 376, 0),
        (MemoryOperation.REVOKE, 375, 1),
    ],
)
def test_remediation_reserve_allows_mutation_at_or_over_ordinary_byte_cap(
    tmp_path: Path,
    operation: MemoryOperation,
    ordinary_cap: int,
    used_before: int,
) -> None:
    root = tmp_path / f"{operation.value}-{ordinary_cap}"
    provenance = _provenance("t", "r")
    initial = MemoryStore(root, config=MemoryStoreConfig(max_store_bytes=1_000))
    first = initial.record("Original fact.", provenance)
    store = MemoryStore(root, config=MemoryStoreConfig(max_store_bytes=ordinary_cap))

    if operation is MemoryOperation.CORRECT:
        result = store.correct(
            "memory-000001",
            "Replacement fact.",
            "Observed in current files.",
            provenance,
        )
    else:
        result = store.revoke("memory-000001", "Observed in current files.", provenance)

    assert first.status is MemoryMutationStatus.APPLIED
    assert result.status is MemoryMutationStatus.APPLIED
    assert result.limits["remediation_reserve"]["used_before"] == used_before
    assert result.limits["remediation_reserve"]["used_after"] > 0


def test_identical_correction_is_rejected_without_appending_an_event(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    provenance = _provenance()
    store.record("Current fact.", provenance)

    result = store.correct(
        "memory-000001", "  Current fact.  ", "No actual change.", provenance
    )

    assert result.status is MemoryMutationStatus.REJECTED
    assert result.error == "memory correction content is identical to the current claim"
    assert store.read("memory-000001").revision == 1


def test_absolute_remediation_reserve_cap_is_recorded_without_losing_entry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    provenance = _provenance("t", "r")
    initial = MemoryStore(root, config=MemoryStoreConfig(max_store_bytes=1_000))
    initial.record("Original fact.", provenance)
    store = MemoryStore(
        root,
        config=MemoryStoreConfig(
            max_store_bytes=376,
            max_remediation_bytes=1,
        ),
    )

    result = store.revoke("memory-000001", "Observed now.", provenance)

    assert result.status is MemoryMutationStatus.REJECTED
    assert result.memory_id == "memory-000001"
    assert result.limits["store_size"]["hit"] is True
    assert store.read("memory-000001").current_content == "Original fact."


@pytest.mark.parametrize(
    "content",
    [
        "api_key=abcd1234",
        "client-secret: topsecret",
        "Authorization: Bearer abcdefghijklmnop",
        "AWS access AKIAABCDEFGHIJKLMNOP",
        "GitHub ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "OpenAI sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN PGP PRIVATE KEY BLOCK-----",
        "github_pat_11AA22BB33CC44DD55EE66FF77GG88HH",
        "xoxb-" + "123456789012-" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCY",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123",
        "postgres://user:s3cretpassword@db.internal:5432/app",
    ],
)
def test_every_credential_signature_is_refused_without_disk_write(
    tmp_path: Path, content: str
) -> None:
    store = MemoryStore(tmp_path / "memory")

    result = store.record(content, _provenance())

    assert result.status is MemoryMutationStatus.REJECTED
    assert result.error == "memory content looks like a credential"
    assert store.memory_ids() == ()


def test_credential_words_without_values_are_permitted(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")

    result = store.record(
        "The service uses token bucket throttling and password rotation tests.",
        _provenance(),
    )

    assert result.status is MemoryMutationStatus.APPLIED


@pytest.mark.parametrize("field", ["content", "reason", "task_id", "run_id"])
def test_credential_like_text_is_refused_from_every_durable_text_role(
    tmp_path: Path, field: str
) -> None:
    store = MemoryStore(tmp_path / field)
    credential = "token=abcdefghijklmnop"
    provenance = MemoryProvenance(
        credential if field == "task_id" else "task",
        credential if field == "run_id" else "run",
        1,
        1,
        1,
    )
    if field == "content":
        result = store.record(credential, provenance)
    elif field == "reason":
        store.record("Original fact.", provenance)
        result = store.correct(
            "memory-000001", "Replacement fact.", credential, provenance
        )
    else:
        result = store.record("Original fact.", provenance)

    assert result.status is MemoryMutationStatus.REJECTED
    if field in {"task_id", "run_id"}:
        assert "provenance" not in result.to_report()
        assert store.memory_ids() == ()
    else:
        assert credential not in json.dumps(result.to_report())


@pytest.mark.parametrize("field", ["content", "reason", "task_id", "run_id"])
def test_lone_surrogate_is_refused_from_every_durable_text_role(
    tmp_path: Path, field: str
) -> None:
    store = MemoryStore(tmp_path / field)
    invalid = "\ud800"
    provenance = MemoryProvenance(
        invalid if field == "task_id" else "task",
        invalid if field == "run_id" else "run",
        1,
        1,
        1,
    )
    if field == "content":
        result = store.record(invalid, provenance)
    elif field == "reason":
        store.record("Original fact.", provenance)
        result = store.correct(
            "memory-000001", "Replacement fact.", invalid, provenance
        )
    else:
        result = store.record("Original fact.", provenance)

    assert result.status is MemoryMutationStatus.REJECTED


@pytest.mark.parametrize(
    "document",
    [
        "{",
        json.dumps({"schema_version": 1, "memory_id": "memory-000001"}),
        json.dumps(
            {
                "schema_version": 1,
                "memory_id": "memory-000001",
                "events": 7,
            }
        ),
    ],
    ids=["bad-json", "missing-field", "wrong-type"],
)
def test_malformed_disk_store_raises_specific_typed_error(
    tmp_path: Path, document: str
) -> None:
    store = MemoryStore(tmp_path / "memory")
    (store.entries / "memory-000001.json").write_text(document, encoding="utf-8")

    with pytest.raises(MemoryStoreFormatError):
        store.current_entries()


def test_malformed_memory_store_is_audited_as_corpus_build_exclusion(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "workspace remains available"
    (workspace / "source.txt").write_text(source, encoding="utf-8")
    store = MemoryStore(tmp_path / "memory")
    (store.entries / "memory-000001.json").write_text("{", encoding="utf-8")

    corpus = RetrievalCorpusBuilder(
        workspace,
        SkillLibrary(tmp_path / "library"),
        LiteralEmbedder({source: _vector(0.2)}),
        memory_store=store,
    ).build()

    assert corpus.build_report["indexed_kind_counts"] == {"workspace": 1}
    assert corpus.build_report["excluded_sources"] == [
        {
            "kind": "memory",
            "origin": "memory-store",
            "reason": "memory_store_read_failed",
            "detail": (
                "MemoryStoreFormatError: memory entry is not valid UTF-8 JSON: "
                f"{store.entries / 'memory-000001.json'}"
            ),
        }
    ]


def test_corpus_keeps_healthy_memory_when_another_entry_is_corrupt(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "current workspace evidence"
    (workspace / "source.txt").write_text(source, encoding="utf-8")
    store = MemoryStore(tmp_path / "memory")
    content = "historical memory hint"
    store.record(content, _provenance())
    corrupt = store.entries / "memory-000002.json"
    corrupt.write_text("{", encoding="utf-8")

    corpus = RetrievalCorpusBuilder(
        workspace,
        SkillLibrary(tmp_path / "library"),
        LiteralEmbedder({source: _vector(0.2), content: _vector(0.3)}),
        memory_store=store,
    ).build()

    assert corpus.build_report["indexed_kind_counts"] == {
        "memory": 1,
        "workspace": 1,
    }
    assert corpus.build_report["excluded_sources"] == [
        {
            "kind": "memory",
            "origin": "memory-000002",
            "reason": "memory_entry_read_failed",
            "detail": (
                "memory entry is not valid UTF-8 JSON: "
                f"{store.entries / 'memory-000002.json'}"
            ),
        }
    ]


def test_stale_temp_is_cleaned_recorded_and_does_not_block_store(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    store.record("Healthy fact.", _provenance())
    temporary = store.entries / ".tmp-memory-000002.json"
    temporary.write_text("partial", encoding="utf-8")

    assert store.memory_ids() == ("memory-000001",)
    assert temporary.exists()
    assert [entry.current_content for entry in store.current_entries()] == [
        "Healthy fact."
    ]
    result = store.record("Later fact.", _provenance())
    assert result.status is MemoryMutationStatus.APPLIED
    assert result.memory_id == "memory-000002"
    assert not temporary.exists()
    report = store.recovery_report
    assert report["total_event_count"] == 3
    assert report["kind_counts"] == {"stale_temporary_file": 3}
    assert report["recent_examples"][-1] == {
        "kind": "stale_temporary_file",
        "origin": ".tmp-memory-000002.json",
        "detail": "removed while holding the mutation lock",
    }


def test_one_corrupt_entry_does_not_deny_healthy_entries_or_new_records(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    store.record("Healthy fact.", _provenance())
    corrupt = store.entries / "memory-000002.json"
    corrupt.write_text("{", encoding="utf-8")

    assert store.read("memory-000001").current_content == "Healthy fact."
    with pytest.raises(MemoryStoreFormatError):
        store.read("memory-000002")
    assert [entry.memory_id for entry in store.current_entries()] == ["memory-000001"]
    result = store.record("New healthy fact.", _provenance())

    assert result.status is MemoryMutationStatus.APPLIED
    assert result.memory_id == "memory-000003"
    assert [entry.memory_id for entry in store.current_entries()] == [
        "memory-000001",
        "memory-000003",
    ]
    assert store.recovery_report["kind_counts"] == {"malformed_entry": 3}


@pytest.mark.parametrize(
    "anomaly_kind",
    ["invalid-json-name", "non-json-file", "unexpected-directory"],
)
def test_directory_anomaly_does_not_deny_healthy_entries_or_new_records(
    tmp_path: Path, anomaly_kind: str
) -> None:
    store = MemoryStore(tmp_path / anomaly_kind)
    store.record("Healthy fact.", _provenance())
    if anomaly_kind == "invalid-json-name":
        (store.entries / "not-a-memory.json").write_text("{}", encoding="utf-8")
    elif anomaly_kind == "non-json-file":
        (store.entries / "unexpected.txt").write_text("junk", encoding="utf-8")
    else:
        (store.entries / "unexpected").mkdir()

    assert store.memory_ids() == ("memory-000001",)
    assert [entry.current_content for entry in store.current_entries()] == [
        "Healthy fact."
    ]
    result = store.record("Later fact.", _provenance())

    assert result.status is MemoryMutationStatus.APPLIED
    assert result.memory_id == "memory-000002"
    assert store.recovery_report["kind_counts"]["malformed_entry"] >= 1


def test_concurrent_records_report_applied_only_for_durable_unique_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    provenance = _provenance("concurrent-task", "concurrent-run")

    def write_claims(worker: int) -> list[tuple[str, MemoryMutationStatus, str | None]]:
        store = MemoryStore(root)
        results = []
        for index in range(5):
            content = f"claim {worker}-{index}"
            result = store.record(content, provenance)
            results.append((content, result.status, result.memory_id))
        return results

    with ThreadPoolExecutor(max_workers=8) as executor:
        groups = tuple(executor.map(write_claims, range(8)))
    results = [result for group in groups for result in group]
    fresh = MemoryStore(root)
    entries = fresh.current_entries()

    assert len(results) == 40
    assert all(status is MemoryMutationStatus.APPLIED for _, status, _ in results)
    assert len({memory_id for _, _, memory_id in results}) == 40
    assert len(entries) == 40
    assert {entry.current_content for entry in entries} == {
        content for content, _, _ in results
    }


def test_memory_shares_corpus_and_384_dimension_relevance_floor(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_text = "The parser accepts trailing commas."
    (workspace / "parser.txt").write_text(workspace_text, encoding="utf-8")
    second_workspace_text = "The scheduler uses incremental backoff."
    (workspace / "scheduler.txt").write_text(second_workspace_text, encoding="utf-8")
    library = SkillLibrary(tmp_path / "library")
    activation = "When quorum lease fencing fails."
    _admit(library, "lease-fencing", activation)
    memory_content = "Continuous integration is invoked by the repository check target."
    store = MemoryStore(tmp_path / "memory")
    store.record(memory_content, _provenance())
    related = "How should I execute automated verification?"
    unrelated = "Explain photosynthesis in pond algae."
    embedder = LiteralEmbedder(
        {
            activation: _vector(0.20),
            memory_content: _vector(0.91),
            workspace_text: _vector(0.10),
            second_workspace_text: _vector(0.15),
            related: _RELATED_QUERY_VECTOR,
            unrelated: _UNRELATED_QUERY_VECTOR,
        }
    )
    tool = AgenticRetrievalTool.from_workspace(
        workspace, library, embedder, memory_store=store
    )

    related_result = tool.retrieve(related)
    unrelated_result = tool.retrieve(unrelated)

    assert tool.corpus.build_report["indexed_kind_counts"] == {
        "memory": 1,
        "skill": 1,
        "workspace": 2,
    }
    assert len(tool.corpus.documents[0].vector) == 384
    assert [match.kind for match in related_result.matches] == [
        RetrievalDocumentKind.MEMORY
    ]
    match = related_result.matches[0]
    assert match.origin == "memory-000001"
    assert match.indexed_span == "memory_current_claim"
    assert match.epistemic_status == (
        "unvalidated_claim_verify_against_current_observation"
    )
    assert match.provenance == {
        "task_id": "task-parser",
        "run_id": "run-07",
        "sequence": 3,
        "logical_step": 2,
        "attempt": 1,
        "revision": 1,
    }
    assert match.content.startswith("UNVALIDATED MEMORY:")
    assert "current observations win" in match.content
    assert related_result.semantic_relevance_floor == pytest.approx(0.23294584840212837)
    assert unrelated_result.matches == ()


def test_one_result_labels_skill_workspace_and_memory_as_distinct_kinds(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_text = "sharedneedle workspace evidence"
    (workspace / "source.txt").write_text(workspace_text, encoding="utf-8")
    library = SkillLibrary(tmp_path / "library")
    activation = "sharedneedle validated skill"
    _admit(library, "validated", activation)
    memory_content = "sharedneedle historical claim"
    store = MemoryStore(tmp_path / "memory")
    store.record(memory_content, _provenance())
    query = "sharedneedle"
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        library,
        LiteralEmbedder(
            {
                activation: _vector(0.20),
                workspace_text: _vector(0.10),
                memory_content: _vector(0.30),
                query: _RELATED_QUERY_VECTOR,
            }
        ),
        memory_store=store,
    )

    result = tool.retrieve(query)

    assert [(match.kind.value, match.origin) for match in result.matches] == [
        ("memory", "memory-000001"),
        ("skill", "validated"),
        ("workspace", "source.txt"),
    ]
    assert [match.epistemic_status for match in result.matches] == [
        "unvalidated_claim_verify_against_current_observation",
        None,
        None,
    ]
    report = result.to_report()["selected_documents"]
    assert [item["kind"] for item in report] == ["memory", "skill", "workspace"]
    assert "epistemic_status" in report[0]
    assert "epistemic_status" not in report[1]
    assert "epistemic_status" not in report[2]


def test_workspace_wins_equal_similarity_tie_against_memory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "sharedterm current observation"
    memory_content = "sharedterm historical claim"
    query = "sharedterm"
    (workspace / "f0.txt").write_text(source, encoding="utf-8")
    store = MemoryStore(tmp_path / "memory")
    store.record(memory_content, _provenance())
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        SkillLibrary(tmp_path / "library"),
        LiteralEmbedder(
            {
                source: _vector(0.9),
                memory_content: _vector(0.9),
                query: _RELATED_QUERY_VECTOR,
            }
        ),
        memory_store=store,
    )

    result = tool.retrieve(query)

    assert [(match.kind.value, match.origin) for match in result.matches] == [
        ("workspace", "f0.txt"),
        ("memory", "memory-000001"),
    ]


def test_memory_cannot_fill_all_result_slots_when_it_scores_higher(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = MemoryStore(tmp_path / "memory")
    vectors: dict[str, Sequence[float]] = {}
    for index in range(6):
        source = f"sharedterm workspace {index}"
        (workspace / f"f{index}.txt").write_text(source, encoding="utf-8")
        vectors[source] = _vector(0.9)
        memory_content = f"sharedterm memory {index}"
        store.record(memory_content, _provenance())
        vectors[memory_content] = _vector(0.95)
    query = "sharedterm"
    vectors[query] = _RELATED_QUERY_VECTOR
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        SkillLibrary(tmp_path / "library"),
        LiteralEmbedder(vectors),
        memory_store=store,
    )

    result = tool.retrieve(query)

    assert [match.kind.value for match in result.matches] == [
        "memory",
        "workspace",
        "workspace",
        "workspace",
    ]
    assert result.exclusion_reason_counts["per_call_memory_result_cap"] == 5
    assert result.per_call_memory_result_cap_hit is True
    assert result.to_report()["limits"]["per_call_memory_result_cap"] == {
        "limit": 1,
        "hit": True,
        "excluded_count": 5,
    }


def test_memory_cannot_consume_whole_shared_task_character_budget(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    content = "memoryneedle " + ("x" * 1_880)
    query = "memoryneedle"
    store = MemoryStore(tmp_path / "memory")
    store.record(content, _provenance())
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        SkillLibrary(tmp_path / "library"),
        LiteralEmbedder({content: _vector(0.95), query: _RELATED_QUERY_VECTOR}),
        memory_store=store,
    )

    results = [tool.retrieve(query) for _ in range(9)]

    assert [len(result.matches) for result in results] == [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert tool.returned_memory_characters == 4_278
    assert tool.returned_characters == 4_278
    assert tool.returned_characters < 18_000
    assert results[2].exclusion_reason_counts == {"per_task_memory_character_budget": 1}
    assert results[2].per_task_memory_character_budget_hit is True


def test_corrected_memory_is_retrieved_without_superseded_content(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    library = SkillLibrary(tmp_path / "library")
    store = MemoryStore(tmp_path / "memory")
    old = "Tests run through pytest directly."
    corrected = "Tests run through make check with seeded fixtures."
    query = "What command runs the repository tests?"
    store.record(old, _provenance())
    store.correct("memory-000001", corrected, "Read the Makefile.", _provenance())
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        library,
        LiteralEmbedder(
            {
                corrected: _vector(0.92),
                query: _RELATED_QUERY_VECTOR,
            }
        ),
        memory_store=store,
    )

    result = tool.retrieve(query)

    assert len(tool.corpus.documents) == 1
    assert [match.origin for match in result.matches] == ["memory-000001"]
    assert corrected in result.matches[0].content
    assert old not in result.matches[0].content
    assert result.matches[0].provenance is not None
    assert result.matches[0].provenance["revision"] == 2


def test_revoked_memory_is_not_added_to_retrieval_corpus(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = MemoryStore(tmp_path / "memory")
    store.record("A stale operational fact.", _provenance())
    store.revoke("memory-000001", "Disproved by current files.", _provenance())
    tool = AgenticRetrievalTool.from_workspace(
        workspace,
        SkillLibrary(tmp_path / "library"),
        LiteralEmbedder({}),
        memory_store=store,
    )

    result = tool.retrieve("operational fact")

    assert tool.corpus.documents == ()
    assert result.matches == ()


def test_empty_memory_store_leaves_retrieval_behavior_byte_identical(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "quorum lease fencing"
    query = "quorum lease"
    (workspace / "lease.txt").write_text(source, encoding="utf-8")
    library = SkillLibrary(tmp_path / "library")
    first_embedder = LiteralEmbedder(
        {source: _vector(0.8), query: _RELATED_QUERY_VECTOR}
    )
    second_embedder = LiteralEmbedder(
        {source: _vector(0.8), query: _RELATED_QUERY_VECTOR}
    )
    without_memory = RetrievalCorpusBuilder(workspace, library, first_embedder).build()
    with_empty_memory = RetrievalCorpusBuilder(
        workspace,
        library,
        second_embedder,
        memory_store=MemoryStore(tmp_path / "empty-memory"),
    ).build()

    first_result = AgenticRetrievalTool(without_memory).retrieve(query)
    second_result = AgenticRetrievalTool(with_empty_memory).retrieve(query)

    assert without_memory.build_report == with_empty_memory.build_report
    assert without_memory.documents == with_empty_memory.documents
    assert first_embedder.calls == second_embedder.calls
    assert (
        json.dumps(first_result.to_report(), sort_keys=True).encode()
        == json.dumps(second_result.to_report(), sort_keys=True).encode()
    )


async def test_agent_records_memory_with_well_formed_observation_and_audit(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    content = "The repository test entry point is make check."
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(
                ToolCall(
                    "manage_memory",
                    {"operation": "record", "content": content},
                    "memory-1",
                ),
            ),
            tokens=9,
        )
    )
    store = MemoryStore(tmp_path / "memory")
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        memory_store=store,
        memory_task_id="task-parser",
        memory_run_id="run-07",
    )

    outcome = await agent(_context(agent))

    assert [tool.name for tool in provider.requests[0].tools] == [
        "run_shell",
        "read_file",
        "write_file",
        "search_files",
        "complete",
        "manage_memory",
    ]
    observation = json.loads(outcome.tool_observations[0].split("\n", 1)[1])
    assert observation["status"] == "applied"
    assert observation["memory_id"] == "memory-000001"
    assert outcome.action == "Manage memory: record"
    assert outcome.error is None
    assert outcome.tool_audits[0]["result"]["status"] == "applied"
    assert outcome.tool_audits[0]["tool_call"]["arguments"] == {
        "operation": "record",
        "content": {
            "sha256": (
                "b907ea7fb4aabc18b6fd0007bde2eb6401b8b0cbeee753479ef9416ec5d0d9fd"
            ),
            "character_count": 46,
            "value_type": "str",
        },
    }
    assert content not in json.dumps(outcome.tool_audits)
    assert (
        MemoryStore(tmp_path / "memory").read("memory-000001").current_content
        == content
    )


@pytest.mark.parametrize(
    "arguments",
    [
        "{",
        {},
        {"operation": "record"},
        {"operation": "correct", "memory_id": "memory-000001"},
        {"operation": "revoke", "memory_id": "memory-000001"},
        {"operation": "invent", "content": "claim"},
        {"operation": 7, "content": "claim"},
    ],
)
async def test_every_malformed_memory_tool_call_is_audited_and_contained(
    tmp_path: Path, arguments: object
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider(
        AgentCompletion(
            tool_calls=(ToolCall("manage_memory", arguments, "malformed"),),
            tokens=3,
        )
    )
    store = MemoryStore(tmp_path / "memory")
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        memory_store=store,
        memory_task_id="task-parser",
        memory_run_id="run-07",
    )

    outcome = await agent(_context(agent))

    assert outcome.completed is False
    assert outcome.error is not None
    assert outcome.error.startswith("malformed arguments for manage_memory:")
    assert len(outcome.tool_audits) == 1
    assert outcome.tool_audits[0]["result"]["status"] == "rejected"
    observation = json.loads(outcome.tool_observations[0].split("\n", 1)[1])
    assert observation["status"] == "rejected"
    assert store.memory_ids() == ()
    if isinstance(arguments, dict) and arguments.get("operation") == 7:
        assert outcome.action == "Manage memory with malformed operation"
        assert outcome.error == (
            "malformed arguments for manage_memory: "
            "operation must be a non-empty string"
        )
        assert outcome.tool_audits[0]["tool_call"]["arguments"]["operation"] == {
            "sha256": (
                "7902699be42c8a8e46fbbb4501726517e86b22c56a189f7625a6da49081b2451"
            ),
            "character_count": 1,
            "value_type": "int",
        }


def test_all_memory_and_retrieval_kind_strenum_values_are_unique() -> None:
    for enum_type in (
        MemoryEntryStatus,
        MemoryOperation,
        MemoryMutationStatus,
        MemoryRecoveryKind,
        RetrievalDocumentKind,
    ):
        values = [member.value for member in enum_type]
        assert len(values) == len(set(values))
