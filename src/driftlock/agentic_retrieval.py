"""Agent-initiated retrieval over one skill-and-workspace corpus.

The embedding model is always supplied by the host.  This module uses only the
standard library, never opens a network connection, and keeps the completed
once-per-task experiment path in :mod:`driftlock.skill_retrieval` separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from pathlib import Path
from statistics import NormalDist, median
from typing import Any

from driftlock.skill_admission import SkillLibrary
from driftlock.skill_distillation import serialize_skill
from driftlock.skill_retrieval import (
    _coerce_embeddings,
    _cosine,
    _EmbeddingValidationError,
)

# This identifier keeps paid records from pooling the new hybrid, agent-driven
# rule with the completed activation-threshold experiment.
AGENTIC_RETRIEVAL_RULE_ID = "agent-query-adaptive-semantic-rank-v2"

# Four results let one call surface both evidence kinds plus alternatives without
# turning a single exploratory query into a large prompt append.
DEFAULT_MAX_RESULTS_PER_CALL = 4

# Six thousand characters bounds both selected document text and the complete
# serialized agent observation; audit-only diagnostics remain out of context.
DEFAULT_MAX_CHARACTERS_PER_CALL = 6_000

# Three full-size calls permit re-querying while fixing total document context at
# a task-level ceiling that does not grow with trajectory length or corpus size.
DEFAULT_MAX_CHARACTERS_PER_TASK = 18_000

# Three-thousand-character chunks usually preserve a complete function or config
# section while leaving room for more than one result under the per-call ceiling.
DEFAULT_WORKSPACE_CHUNK_CHARACTERS = 3_000

# A three-hundred-character overlap keeps declarations near a chunk boundary
# retrievable without duplicating a material fraction of every indexed file.
DEFAULT_WORKSPACE_CHUNK_OVERLAP = 300

# A 256 KiB per-file read bound prevents generated logs or vendored bundles from
# dominating index construction; skipped files remain explicit in the build audit.
DEFAULT_MAX_WORKSPACE_FILE_BYTES = 256 * 1024

# Two thousand files cover the measured task repositories while bounding path
# traversal, embedding batch size, and the full diagnostic record.
DEFAULT_MAX_WORKSPACE_FILES = 2_000

# Two million indexed workspace characters bound embedding work independently of
# file count; files that would cross the ceiling are recorded and skipped whole.
DEFAULT_MAX_WORKSPACE_CHARACTERS = 2_000_000

# Query text is capped at a situation-sized paragraph so an accidental transcript
# dump cannot become an unbounded embedding input or audit payload.
DEFAULT_MAX_QUERY_CHARACTERS = 2_000

# Twenty exclusion examples are enough for the ordinary agent observation; the
# complete per-document decisions remain in the out-of-context step audit.
DEFAULT_MAX_OBSERVATION_EXCLUSIONS = 20

# A one-in-one-hundred-thousand family-wise null-match allowance makes semantic
# retrieval conservative enough to abstain on unrelated MiniLM queries while the
# resulting cosine floor still adapts to vector dimension and corpus size.
DEFAULT_SEMANTIC_FALSE_MATCH_PROBABILITY = 0.00001

# A largest semantic score gap must be at least twice the median positive gap to
# identify a distinct cluster rather than manufacture one from ordinary noise.
_SEMANTIC_GAP_DOMINANCE_FACTOR = 2.0

# A tiny absolute tolerance distinguishes mathematical ties and cosine identity
# from floating-point roundoff; it is never used as a relevance-score threshold.
_SCORE_EQUALITY_TOLERANCE = 1e-12

# NFC-normalized Unicode alphanumeric runs deliberately stop at punctuation,
# hyphens, and underscores. Lexical evidence should match sentence-final words,
# decomposed macOS filenames/text, and either component of a compound; embeddings
# remain the primary paraphrase mechanism.
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

# These ubiquitous instruction words otherwise make unrelated task prose appear
# matched.  Technical nouns and verbs are intentionally not filtered.
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "when",
        "with",
    }
)

# Repository internals, dependency caches, and the explicitly out-of-scope ops
# tree are not task evidence and can contain credentials or enormous generated data.
DEFAULT_IGNORED_DIRECTORY_NAMES = frozenset(
    {".git", ".hg", ".svn", ".tox", ".venv", "__pycache__", "node_modules", "ops"}
)

# Common credential filenames are excluded before opening a file; retrieval never
# needs private-key or environment-secret contents to answer a code question.
_SENSITIVE_FILE_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials",
        "credentials.json",
    }
)


class RetrievalDocumentKind(StrEnum):
    """The evidence authority represented by a corpus document."""

    SKILL = "skill"
    WORKSPACE = "workspace"


class RetrievalCorpusStatus(StrEnum):
    """Whether the shared corpus has usable vectors."""

    READY = "ready"
    FAILED = "failed"


class AgenticRetrievalStatus(StrEnum):
    """The outcome of one tool invocation."""

    USABLE = "usable"
    FAILED = "failed"
    TASK_BUDGET_EXHAUSTED = "task_budget_exhausted"
    TASK_BUDGET_INSUFFICIENT = "task_budget_insufficient"


@dataclass(frozen=True, slots=True)
class AgenticRetrievalConfig:
    """Fixed index and context bounds for one task-scoped retrieval tool."""

    max_results_per_call: int = DEFAULT_MAX_RESULTS_PER_CALL
    max_characters_per_call: int = DEFAULT_MAX_CHARACTERS_PER_CALL
    max_characters_per_task: int = DEFAULT_MAX_CHARACTERS_PER_TASK
    workspace_chunk_characters: int = DEFAULT_WORKSPACE_CHUNK_CHARACTERS
    workspace_chunk_overlap: int = DEFAULT_WORKSPACE_CHUNK_OVERLAP
    max_workspace_file_bytes: int = DEFAULT_MAX_WORKSPACE_FILE_BYTES
    max_workspace_files: int = DEFAULT_MAX_WORKSPACE_FILES
    max_workspace_characters: int = DEFAULT_MAX_WORKSPACE_CHARACTERS
    max_query_characters: int = DEFAULT_MAX_QUERY_CHARACTERS
    max_observation_exclusions: int = DEFAULT_MAX_OBSERVATION_EXCLUSIONS
    semantic_false_match_probability: float = DEFAULT_SEMANTIC_FALSE_MATCH_PROBABILITY
    ignored_directory_names: frozenset[str] = DEFAULT_IGNORED_DIRECTORY_NAMES

    def __post_init__(self) -> None:
        for name in (
            "max_results_per_call",
            "max_characters_per_call",
            "max_characters_per_task",
            "workspace_chunk_characters",
            "max_workspace_file_bytes",
            "max_workspace_files",
            "max_workspace_characters",
            "max_query_characters",
            "max_observation_exclusions",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        overlap = self.workspace_chunk_overlap
        if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
            raise ValueError("workspace_chunk_overlap must be a non-negative integer")
        if overlap >= self.workspace_chunk_characters:
            raise ValueError(
                "workspace_chunk_overlap must be smaller than "
                "workspace_chunk_characters"
            )
        probability = self.semantic_false_match_probability
        if (
            isinstance(probability, bool)
            or not isinstance(probability, Real)
            or not math.isfinite(float(probability))
            or not 0.0 < float(probability) < 1.0
        ):
            raise ValueError(
                "semantic_false_match_probability must be finite and in (0, 1)"
            )
        object.__setattr__(self, "semantic_false_match_probability", float(probability))
        ignored = self.ignored_directory_names
        if not isinstance(ignored, frozenset) or any(
            not isinstance(name, str) or not name or "/" in name or "\x00" in name
            for name in ignored
        ):
            raise ValueError(
                "ignored_directory_names must be a frozenset of non-empty "
                "single path-component strings"
            )

    def to_report(self) -> dict[str, Any]:
        """Describe the replayable selection and context policies."""

        return {
            "rule_id": AGENTIC_RETRIEVAL_RULE_ID,
            "max_results_per_call": self.max_results_per_call,
            "max_characters_per_call": self.max_characters_per_call,
            "max_characters_per_task": self.max_characters_per_task,
            "workspace_chunk_characters": self.workspace_chunk_characters,
            "workspace_chunk_overlap": self.workspace_chunk_overlap,
            "max_workspace_file_bytes": self.max_workspace_file_bytes,
            "max_workspace_files": self.max_workspace_files,
            "max_workspace_characters": self.max_workspace_characters,
            "max_query_characters": self.max_query_characters,
            "max_observation_exclusions": self.max_observation_exclusions,
            "semantic_false_match_probability": (self.semantic_false_match_probability),
            "ignored_directory_names": sorted(self.ignored_directory_names),
            "indexing_policy": (
                "Skill entries index activation text only and return the complete "
                "activation/execution/termination document. Workspace entries index "
                "and return the same bounded text chunk."
            ),
            "selection_policy": (
                "A document is eligible through either non-stopword lexical overlap "
                "with its indexed span or query-relative semantic separation. "
                "Semantic evidence must first clear a family-wise geometric null "
                "floor derived from embedding dimension, corpus size, and the "
                "configured false-match probability. For three or more documents it "
                "must also be in the prefix above a unique largest adjacent cosine "
                "gap at least twice the median positive gap; a flat or ambiguous "
                "distribution abstains. Eligible documents rank by cosine, lexical "
                "coverage, and stable document id. No fixed global cosine threshold "
                "is used."
            ),
            "requery_policy": (
                "Queries are scored statelessly against the immutable corpus. A "
                "prior query cannot alter later ranking; only the explicit cumulative "
                "task character budget carries across calls."
            ),
            "context_policy": (
                "Result count and returned document characters are bounded per call, "
                "and returned document characters are also bounded cumulatively per "
                "task because PLAN section 2.5 identifies context rot as a "
                "long-horizon failure mode. Every binding cap is reported."
            ),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_report(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class _CorpusDocument:
    document_id: str
    kind: RetrievalDocumentKind
    origin: str
    indexed_span: str
    index_text: str
    returned_text: str
    vector: tuple[float, ...]
    terms: frozenset[str]
    sha256: str
    chunk_start: int | None = None
    chunk_end: int | None = None


@dataclass(frozen=True, slots=True)
class _UnembeddedDocument:
    document_id: str
    kind: RetrievalDocumentKind
    origin: str
    indexed_span: str
    index_text: str
    returned_text: str
    chunk_start: int | None = None
    chunk_end: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    """One ranked agent-facing skill or workspace excerpt."""

    document_id: str
    kind: RetrievalDocumentKind
    origin: str
    indexed_span: str
    similarity: float
    lexical_coverage: float
    lexical_evidence: bool
    semantic_evidence: bool
    rank: int
    content: str
    chunk_start: int | None = None
    chunk_end: int | None = None

    def to_report(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rank": self.rank,
            "document_id": self.document_id,
            "kind": self.kind.value,
            "origin": self.origin,
            "indexed_span": self.indexed_span,
            "similarity": self.similarity,
            "lexical_coverage": self.lexical_coverage,
            "eligibility_evidence": {
                "lexical_overlap": self.lexical_evidence,
                "adaptive_semantic_separation": self.semantic_evidence,
            },
            "basis": _retrieval_basis(self.lexical_evidence, self.semantic_evidence),
            "character_count": len(self.content),
            "content": self.content,
        }
        if self.chunk_start is not None:
            result["chunk"] = {
                "start_character": self.chunk_start,
                "end_character": self.chunk_end,
            }
        return result


@dataclass(frozen=True, slots=True)
class AgenticRetrievalResult:
    """A complete audit result with a separately compact agent observation."""

    status: AgenticRetrievalStatus
    query: str
    query_sha256: str
    config: AgenticRetrievalConfig
    corpus: Mapping[str, Any]
    task_characters_before: int
    task_characters_after: int
    semantic_relevance_floor: float | None = None
    matches: tuple[RetrievedContext, ...] = ()
    considered: tuple[Mapping[str, Any], ...] = ()
    exclusion_reason_counts: Mapping[str, int] = field(default_factory=dict)
    per_call_result_cap_hit: bool = False
    per_call_character_cap_hit: bool = False
    per_task_character_budget_hit: bool = False
    refusal: Mapping[str, str] | None = None

    @property
    def returned_character_count(self) -> int:
        return sum(len(match.content) for match in self.matches)

    def to_report(self) -> dict[str, Any]:
        """Return full paid-run evidence, including every considered document."""

        report: dict[str, Any] = {
            "schema_version": 1,
            "mode": "agentic-context-retrieval",
            "status": self.status.value,
            "query": {
                "text": self.query,
                "sha256": self.query_sha256,
                "character_count": len(self.query),
            },
            "configuration": {
                **self.config.to_report(),
                "fingerprint": self.config.fingerprint,
            },
            "corpus": dict(self.corpus),
            "semantic_relevance_floor": self.semantic_relevance_floor,
            "considered_document_count": len(self.considered),
            "considered_documents": [dict(candidate) for candidate in self.considered],
            "selected_document_count": len(self.matches),
            "selected_documents": [match.to_report() for match in self.matches],
            "excluded_document_count": sum(self.exclusion_reason_counts.values()),
            "exclusion_reason_counts": dict(self.exclusion_reason_counts),
            "limits": {
                "per_call_result_cap": {
                    "limit": self.config.max_results_per_call,
                    "hit": self.per_call_result_cap_hit,
                    "excluded_count": self.exclusion_reason_counts.get(
                        "per_call_result_cap", 0
                    ),
                },
                "per_call_character_cap": {
                    "limit": self.config.max_characters_per_call,
                    "used": self.returned_character_count,
                    "hit": self.per_call_character_cap_hit,
                    "excluded_count": self.exclusion_reason_counts.get(
                        "per_call_character_cap", 0
                    ),
                },
                "observation_character_cap": {
                    "limit": self.config.max_characters_per_call,
                    "results_before_diagnostics": True,
                    "serialization": "complete_json_never_mid_string_truncation",
                },
                "per_task_character_budget": {
                    "limit": self.config.max_characters_per_task,
                    "before": self.task_characters_before,
                    "contributed": self.returned_character_count,
                    "after": self.task_characters_after,
                    "hit": self.per_task_character_budget_hit,
                    "excluded_count": self.exclusion_reason_counts.get(
                        "per_task_character_budget", 0
                    ),
                },
            },
        }
        if self.refusal is not None:
            report["refusal"] = dict(self.refusal)
        return report

    def to_observation(self, *, max_characters: int | None = None) -> str:
        """Serialize bounded JSON, allocating space to results before diagnostics."""

        limit = self.config.max_characters_per_call
        if max_characters is not None:
            if (
                not isinstance(max_characters, int)
                or isinstance(max_characters, bool)
                or max_characters <= 0
            ):
                raise ValueError("max_characters must be a positive integer or None")
            limit = min(limit, max_characters)
        excluded = tuple(
            _observation_exclusion(candidate)
            for candidate in self.considered
            if candidate.get("outcome") == "excluded"
        )
        results = [_observation_match(match, content="") for match in self.matches]
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": self.status.value,
            "query": self.query,
            "results": results,
            "considered_document_count": len(self.considered),
            "exclusion_reason_counts": dict(self.exclusion_reason_counts),
            "exclusion_examples": [],
            "unreported_exclusion_count": len(excluded),
            "limits": self.to_report()["limits"],
        }
        observation_limit = payload["limits"]["observation_character_cap"]
        observation_limit["limit"] = limit
        # Start pessimistically so every content-allocation probe includes the
        # longer boolean representation; clearing it later can only save space.
        observation_limit["hit"] = True
        if self.refusal is not None:
            payload["refusal"] = dict(self.refusal)
        query_omitted = False
        if len(_observation_json(payload)) > limit:
            payload["query"] = {
                "sha256": self.query_sha256,
                "character_count": len(self.query),
                "text_omitted_for_observation_cap": True,
            }
            query_omitted = True
        for index, match in enumerate(self.matches):
            _allocate_result_content(payload, index, match.content, limit)
        exclusions_omitted_for_character_cap = False
        if all(result["content_omitted_character_count"] == 0 for result in results):
            for exclusion in excluded[: self.config.max_observation_exclusions]:
                payload["exclusion_examples"].append(exclusion)
                payload["unreported_exclusion_count"] -= 1
                if len(_observation_json(payload)) > limit:
                    payload["exclusion_examples"].pop()
                    payload["unreported_exclusion_count"] += 1
                    exclusions_omitted_for_character_cap = True
                    break
        content_omitted = any(
            result["content_omitted_character_count"] > 0 for result in results
        )
        observation_limit["hit"] = (
            query_omitted or content_omitted or exclusions_omitted_for_character_cap
        )
        rendered = _observation_json(payload)
        if len(rendered) <= limit:
            return rendered
        # Extremely small caller caps or unusually long origins can make even the
        # normal compact schema impossible. Return a parseable failure rather than
        # slicing JSON; the full result and origins remain in the step audit.
        fallback = {
            "schema_version": 1,
            "status": AgenticRetrievalStatus.FAILED.value,
            "results": [],
            "refusal": {
                "reason": "observation_character_cap_too_small",
                "configured_limit": limit,
            },
        }
        rendered = _observation_json(fallback)
        if len(rendered) <= limit:
            return rendered
        minimal_failure = {
            "status": AgenticRetrievalStatus.FAILED.value,
            "results": [],
            "error": "observation_character_cap_too_small",
        }
        rendered = _observation_json(minimal_failure)
        if len(rendered) <= limit:
            return rendered
        return "{}"


@dataclass(frozen=True, slots=True)
class RetrievalCorpus:
    """One immutable vector index over skills and workspace text chunks."""

    status: RetrievalCorpusStatus
    config: AgenticRetrievalConfig
    documents: tuple[_CorpusDocument, ...]
    build_report: Mapping[str, Any]
    embed: Callable[[Sequence[str]], Iterable[Iterable[Real]]]
    refusal: Mapping[str, str] | None = None

    def snapshot_report(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "document_count": len(self.documents),
            "kind_counts": dict(self.build_report["indexed_kind_counts"]),
            "fingerprint": self.build_report["fingerprint"],
            "build_exclusion_count": self.build_report["excluded_source_count"],
            "build_exclusion_reason_counts": dict(
                self.build_report["exclusion_reason_counts"]
            ),
            "build_exclusions": [
                dict(exclusion) for exclusion in self.build_report["excluded_sources"]
            ],
        }


@dataclass(frozen=True, slots=True)
class RetrievalCorpusBuilder:
    """Build the shared corpus through canonical skill and safe file readers."""

    workspace_root: Path
    skill_library: SkillLibrary
    embed: Callable[[Sequence[str]], Iterable[Iterable[Real]]]
    config: AgenticRetrievalConfig = AgenticRetrievalConfig()

    def __post_init__(self) -> None:
        root = Path(self.workspace_root).resolve()
        if not root.is_dir():
            raise ValueError("workspace_root must be an existing directory")
        if not isinstance(self.skill_library, SkillLibrary):
            raise TypeError("skill_library must be a SkillLibrary")
        if not callable(self.embed):
            raise TypeError("embed must be callable")
        if not isinstance(self.config, AgenticRetrievalConfig):
            raise TypeError("config must be an AgenticRetrievalConfig")
        object.__setattr__(self, "workspace_root", root)

    def build(self) -> RetrievalCorpus:
        """Read, chunk, and embed both evidence kinds into one fixed snapshot."""

        pending: list[_UnembeddedDocument] = []
        exclusions: list[dict[str, str]] = []
        self._append_skills(pending, exclusions)
        self._append_workspace(pending, exclusions)
        fingerprint = _corpus_fingerprint(pending)
        kind_counts = Counter(document.kind.value for document in pending)
        exclusion_counts = Counter(item["reason"] for item in exclusions)
        base_report: dict[str, Any] = {
            "schema_version": 1,
            "mode": "agentic-retrieval-corpus-build",
            "workspace_root": self.workspace_root.as_posix(),
            "indexed_document_count": len(pending),
            "indexed_kind_counts": dict(sorted(kind_counts.items())),
            "excluded_source_count": len(exclusions),
            "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
            "excluded_sources": exclusions,
            "fingerprint": fingerprint,
        }
        if not pending:
            return RetrievalCorpus(
                RetrievalCorpusStatus.READY,
                self.config,
                (),
                base_report,
                self.embed,
            )
        try:
            raw_vectors = self.embed(tuple(document.index_text for document in pending))
        except Exception as error:
            refusal = {
                "reason": "embedding_callable_failed",
                "stage": "corpus_index",
                "detail": f"{type(error).__name__}: {error}",
            }
        else:
            try:
                vectors = _coerce_embeddings(raw_vectors, len(pending))
            except _EmbeddingValidationError as error:
                refusal = {
                    "reason": "invalid_embedding",
                    "stage": "corpus_index",
                    "detail": str(error),
                }
            else:
                documents = tuple(
                    _CorpusDocument(
                        document_id=document.document_id,
                        kind=document.kind,
                        origin=document.origin,
                        indexed_span=document.indexed_span,
                        index_text=document.index_text,
                        returned_text=document.returned_text,
                        vector=vector,
                        terms=_terms(document.index_text),
                        sha256=hashlib.sha256(
                            document.returned_text.encode()
                        ).hexdigest(),
                        chunk_start=document.chunk_start,
                        chunk_end=document.chunk_end,
                    )
                    for document, vector in zip(pending, vectors, strict=True)
                )
                return RetrievalCorpus(
                    RetrievalCorpusStatus.READY,
                    self.config,
                    documents,
                    base_report,
                    self.embed,
                )
        failed_report = {**base_report, "refusal": refusal}
        return RetrievalCorpus(
            RetrievalCorpusStatus.FAILED,
            self.config,
            (),
            failed_report,
            self.embed,
            refusal,
        )

    def _append_skills(
        self,
        pending: list[_UnembeddedDocument],
        exclusions: list[dict[str, str]],
    ) -> None:
        try:
            candidate_ids = self.skill_library.admitted_skill_ids()
        except Exception as error:
            exclusions.append(
                {
                    "kind": RetrievalDocumentKind.SKILL.value,
                    "origin": "skill-library",
                    "reason": "skill_library_read_failed",
                    "detail": f"{type(error).__name__}: {error}",
                }
            )
            return
        for candidate_id in candidate_ids:
            try:
                skill = self.skill_library.read_skill(candidate_id)
            except Exception as error:
                exclusions.append(
                    {
                        "kind": RetrievalDocumentKind.SKILL.value,
                        "origin": candidate_id,
                        "reason": "skill_read_failed",
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            pending.append(
                _UnembeddedDocument(
                    document_id=f"skill:{candidate_id}",
                    kind=RetrievalDocumentKind.SKILL,
                    origin=candidate_id,
                    indexed_span="skill_activation",
                    index_text=skill.activation,
                    returned_text=serialize_skill(skill),
                )
            )

    def _append_workspace(
        self,
        pending: list[_UnembeddedDocument],
        exclusions: list[dict[str, str]],
    ) -> None:
        indexed_files = 0
        indexed_characters = 0
        library_entries = self.skill_library.entries
        for directory, names, filenames in os.walk(
            self.workspace_root, topdown=True, followlinks=False
        ):
            directory_path = Path(directory)
            retained_directories = []
            for name in sorted(names):
                path = directory_path / name
                relative = path.relative_to(self.workspace_root).as_posix() + "/"
                if name in self.config.ignored_directory_names:
                    exclusions.append(
                        _workspace_exclusion(relative, "ignored_directory")
                    )
                elif path.is_symlink():
                    exclusions.append(
                        _workspace_exclusion(relative, "symlink_directory")
                    )
                elif _is_within(path.resolve(), library_entries):
                    exclusions.append(
                        _workspace_exclusion(relative, "skill_library_entries")
                    )
                else:
                    retained_directories.append(name)
            names[:] = retained_directories
            for filename in sorted(filenames):
                path = directory_path / filename
                relative = path.relative_to(self.workspace_root).as_posix()
                if path.is_symlink():
                    exclusions.append(_workspace_exclusion(relative, "symlink"))
                    continue
                if _sensitive_filename(filename):
                    exclusions.append(
                        _workspace_exclusion(relative, "sensitive_filename")
                    )
                    continue
                if indexed_files >= self.config.max_workspace_files:
                    exclusions.append(
                        _workspace_exclusion(relative, "workspace_file_limit")
                    )
                    continue
                try:
                    data = path.read_bytes()
                except OSError as error:
                    exclusions.append(
                        _workspace_exclusion(
                            relative,
                            "workspace_file_read_failed",
                            f"{type(error).__name__}: {error}",
                        )
                    )
                    continue
                if len(data) > self.config.max_workspace_file_bytes:
                    exclusions.append(
                        _workspace_exclusion(relative, "workspace_file_byte_limit")
                    )
                    continue
                if b"\x00" in data:
                    exclusions.append(_workspace_exclusion(relative, "binary_file"))
                    continue
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    exclusions.append(_workspace_exclusion(relative, "non_utf8_file"))
                    continue
                if not text.strip():
                    exclusions.append(_workspace_exclusion(relative, "empty_text_file"))
                    continue
                chunks = _chunks(
                    text,
                    self.config.workspace_chunk_characters,
                    self.config.workspace_chunk_overlap,
                )
                embedded_characters = sum(len(chunk) for _, _, chunk in chunks)
                if (
                    indexed_characters + embedded_characters
                    > self.config.max_workspace_characters
                ):
                    exclusions.append(
                        _workspace_exclusion(relative, "workspace_character_limit")
                    )
                    continue
                indexed_files += 1
                indexed_characters += embedded_characters
                for chunk_number, (start, end, chunk) in enumerate(chunks, start=1):
                    pending.append(
                        _UnembeddedDocument(
                            document_id=f"workspace:{relative}#chunk-{chunk_number}",
                            kind=RetrievalDocumentKind.WORKSPACE,
                            origin=relative,
                            indexed_span="workspace_chunk",
                            index_text=chunk,
                            returned_text=chunk,
                            chunk_start=start,
                            chunk_end=end,
                        )
                    )


class AgenticRetrievalTool:
    """Task-scoped public retrieval entry point with a cumulative context budget."""

    def __init__(self, corpus: RetrievalCorpus) -> None:
        if not isinstance(corpus, RetrievalCorpus):
            raise TypeError("corpus must be a RetrievalCorpus")
        self.corpus = corpus
        self._returned_characters = 0
        self._audit_records: list[AgenticRetrievalResult] = []

    @classmethod
    def from_workspace(
        cls,
        workspace_root: Path | str,
        skill_library: SkillLibrary,
        embed: Callable[[Sequence[str]], Iterable[Iterable[Real]]],
        *,
        config: AgenticRetrievalConfig | None = None,
    ) -> AgenticRetrievalTool:
        """Build a corpus and return the agent-callable task-scoped tool."""

        corpus = RetrievalCorpusBuilder(
            Path(workspace_root),
            skill_library,
            embed,
            config or AgenticRetrievalConfig(),
        ).build()
        return cls(corpus)

    @property
    def audit_records(self) -> tuple[AgenticRetrievalResult, ...]:
        return tuple(self._audit_records)

    @property
    def returned_characters(self) -> int:
        return self._returned_characters

    def retrieve(self, query: object) -> AgenticRetrievalResult:
        """Rank one fresh query and represent every failure as an audited result."""

        before = self._returned_characters
        if not isinstance(query, str) or not query.strip():
            result = self._failure(
                _audit_input_text(query),
                before,
                {
                    "reason": "invalid_query",
                    "stage": "input",
                    "detail": "retrieval query must be non-empty text",
                },
            )
        elif len(query) > self.corpus.config.max_query_characters:
            result = self._failure(
                query,
                before,
                {
                    "reason": "invalid_query",
                    "stage": "input",
                    "detail": (
                        "retrieval query exceeds max_query_characters "
                        f"({self.corpus.config.max_query_characters})"
                    ),
                },
            )
        elif self.corpus.status is RetrievalCorpusStatus.FAILED:
            result = self._failure(query, before, self.corpus.refusal or {})
        elif before >= self.corpus.config.max_characters_per_task:
            result = AgenticRetrievalResult(
                status=AgenticRetrievalStatus.TASK_BUDGET_EXHAUSTED,
                query=query,
                query_sha256=_text_sha256(query),
                config=self.corpus.config,
                corpus=self.corpus.snapshot_report(),
                task_characters_before=before,
                task_characters_after=before,
                per_task_character_budget_hit=True,
                refusal={
                    "reason": "task_character_budget_exhausted",
                    "stage": "selection",
                    "detail": "no task retrieval character budget remains",
                },
            )
        else:
            result = self._retrieve_usable(query, before)
        self._returned_characters = result.task_characters_after
        self._audit_records.append(result)
        return result

    def record_rejected_attempt(
        self,
        attempted_input: object,
        detail: str,
        *,
        reason: str = "malformed_tool_arguments",
    ) -> AgenticRetrievalResult:
        """Record malformed tool arguments that could not reach normal retrieval."""

        before = self._returned_characters
        result = self._failure(
            _audit_input_text(attempted_input),
            before,
            {
                "reason": reason,
                "stage": "input",
                "detail": detail,
            },
        )
        self._audit_records.append(result)
        return result

    def _failure(
        self, query: str, before: int, refusal: Mapping[str, str]
    ) -> AgenticRetrievalResult:
        return AgenticRetrievalResult(
            status=AgenticRetrievalStatus.FAILED,
            query=query,
            query_sha256=_text_sha256(query),
            config=self.corpus.config,
            corpus=self.corpus.snapshot_report(),
            task_characters_before=before,
            task_characters_after=before,
            refusal=refusal,
        )

    def _retrieve_usable(self, query: str, before: int) -> AgenticRetrievalResult:
        documents = self.corpus.documents
        if not documents:
            return self._empty(query, before)
        try:
            raw_vectors = self.corpus.embed((query,))
        except Exception as error:
            return self._failure(
                query,
                before,
                {
                    "reason": "embedding_callable_failed",
                    "stage": "query",
                    "detail": f"{type(error).__name__}: {error}",
                },
            )
        try:
            query_vector = _coerce_embeddings(
                raw_vectors,
                1,
                expected_dimension=len(documents[0].vector),
            )[0]
        except _EmbeddingValidationError as error:
            return self._failure(
                query,
                before,
                {
                    "reason": "invalid_embedding",
                    "stage": "query",
                    "detail": str(error),
                },
            )

        query_terms = _terms(query)
        denominator = max(1, len(query_terms))
        scored = [
            (
                _cosine(query_vector, document.vector),
                len(query_terms & document.terms) / denominator,
                query_terms & document.terms,
                document,
            )
            for document in documents
        ]
        semantic_floor = _semantic_relevance_floor(
            dimension=len(query_vector),
            candidate_count=len(scored),
            false_match_probability=(
                self.corpus.config.semantic_false_match_probability
            ),
        )
        semantic_document_ids = _semantic_document_ids(scored, semantic_floor)
        eligible = sorted(
            (
                item
                for item in scored
                if item[2] or item[3].document_id in semantic_document_ids
            ),
            key=lambda item: (-item[0], -item[1], item[3].document_id),
        )
        selected: list[RetrievedContext] = []
        reasons: dict[str, str] = {
            document.document_id: "no_lexical_evidence_or_semantic_separation"
            for _, _, overlap, document in scored
            if not overlap and document.document_id not in semantic_document_ids
        }
        per_call_characters = 0
        task_remaining = self.corpus.config.max_characters_per_task - before
        for rank, (similarity, lexical_coverage, _overlap, document) in enumerate(
            eligible, start=1
        ):
            size = len(document.returned_text)
            if len(selected) >= self.corpus.config.max_results_per_call:
                reasons[document.document_id] = "per_call_result_cap"
                continue
            if per_call_characters + size > self.corpus.config.max_characters_per_call:
                reasons[document.document_id] = "per_call_character_cap"
                continue
            if per_call_characters + size > task_remaining:
                reasons[document.document_id] = "per_task_character_budget"
                continue
            selected.append(
                RetrievedContext(
                    document_id=document.document_id,
                    kind=document.kind,
                    origin=document.origin,
                    indexed_span=document.indexed_span,
                    similarity=similarity,
                    lexical_coverage=lexical_coverage,
                    lexical_evidence=bool(_overlap),
                    semantic_evidence=(document.document_id in semantic_document_ids),
                    rank=rank,
                    content=document.returned_text,
                    chunk_start=document.chunk_start,
                    chunk_end=document.chunk_end,
                )
            )
            per_call_characters += size

        selected_ids = {match.document_id for match in selected}
        similarity_ranks = {
            document.document_id: rank
            for rank, (_similarity, _coverage, _overlap, document) in enumerate(
                sorted(scored, key=lambda item: (-item[0], item[3].document_id)),
                start=1,
            )
        }
        considered = tuple(
            {
                "document_id": document.document_id,
                "kind": document.kind.value,
                "origin": document.origin,
                "indexed_span": document.indexed_span,
                "similarity": similarity,
                "similarity_rank": similarity_ranks[document.document_id],
                "semantic_relevance_floor": semantic_floor,
                "lexical_coverage": lexical_coverage,
                "overlap_terms": sorted(overlap),
                "outcome": (
                    "selected" if document.document_id in selected_ids else "excluded"
                ),
                "reason": (
                    "selected"
                    if document.document_id in selected_ids
                    else reasons[document.document_id]
                ),
            }
            for similarity, lexical_coverage, overlap, document in sorted(
                scored, key=lambda item: (-item[0], item[3].document_id)
            )
        )
        reason_counts = Counter(reasons.values())
        after = before + per_call_characters
        task_budget_blocked = reason_counts["per_task_character_budget"] > 0
        task_budget_exhausted = after >= self.corpus.config.max_characters_per_task
        task_budget_hit = task_budget_blocked or task_budget_exhausted
        if task_budget_exhausted:
            status = AgenticRetrievalStatus.TASK_BUDGET_EXHAUSTED
        elif task_budget_blocked:
            status = AgenticRetrievalStatus.TASK_BUDGET_INSUFFICIENT
        else:
            status = AgenticRetrievalStatus.USABLE
        return AgenticRetrievalResult(
            status=status,
            query=query,
            query_sha256=_text_sha256(query),
            config=self.corpus.config,
            corpus=self.corpus.snapshot_report(),
            task_characters_before=before,
            task_characters_after=after,
            semantic_relevance_floor=semantic_floor,
            matches=tuple(selected),
            considered=considered,
            exclusion_reason_counts=dict(sorted(reason_counts.items())),
            per_call_result_cap_hit=reason_counts["per_call_result_cap"] > 0,
            per_call_character_cap_hit=(reason_counts["per_call_character_cap"] > 0),
            per_task_character_budget_hit=task_budget_hit,
            refusal=(
                {
                    "reason": "task_character_budget_insufficient",
                    "stage": "selection",
                    "detail": (
                        "matching documents remained, but none fit the remaining "
                        "task retrieval character budget"
                    ),
                }
                if task_budget_blocked and not selected
                else None
            ),
        )

    def _empty(self, query: str, before: int) -> AgenticRetrievalResult:
        return AgenticRetrievalResult(
            status=AgenticRetrievalStatus.USABLE,
            query=query,
            query_sha256=_text_sha256(query),
            config=self.corpus.config,
            corpus=self.corpus.snapshot_report(),
            task_characters_before=before,
            task_characters_after=before,
        )


def _terms(text: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFC", text)
    return frozenset(
        token
        for token in (
            match.group(0).casefold() for match in _TOKEN.finditer(normalized)
        )
        if len(token) > 1 and token not in _STOP_WORDS
    )


def _semantic_document_ids(
    scored: Sequence[tuple[float, float, frozenset[str], _CorpusDocument]],
    semantic_floor: float,
) -> frozenset[str]:
    """Return a query-specific semantic cluster above a geometric null floor.

    A fixed cosine threshold cannot serve a heterogeneous corpus. The caller's
    floor instead comes from vector dimension and a corpus-size-corrected null
    probability, providing the non-scale-invariant evidence a pure gap rule lacks.
    Three-or-more-document corpora additionally require one gap to dominate their
    own positive-gap distribution. Lexical evidence remains an independent path.
    """

    ordered = sorted(scored, key=lambda item: (-item[0], item[3].document_id))
    above_floor = tuple(item for item in ordered if item[0] >= semantic_floor)
    if len(ordered) < 3:
        return frozenset(item[3].document_id for item in above_floor)
    gaps = tuple(
        ordered[index][0] - ordered[index + 1][0] for index in range(len(ordered) - 1)
    )
    largest_gap = max(gaps)
    if largest_gap <= _SCORE_EQUALITY_TOLERANCE:
        return frozenset()
    largest_positions = tuple(
        index
        for index, gap in enumerate(gaps)
        if math.isclose(
            gap,
            largest_gap,
            rel_tol=0.0,
            abs_tol=_SCORE_EQUALITY_TOLERANCE,
        )
    )
    if len(largest_positions) != 1:
        return frozenset()
    positive_gaps = tuple(gap for gap in gaps if gap > _SCORE_EQUALITY_TOLERANCE)
    if not positive_gaps or largest_gap < (
        _SEMANTIC_GAP_DOMINANCE_FACTOR * median(positive_gaps)
    ):
        return frozenset()
    cluster_end = largest_positions[0] + 1
    return frozenset(
        item[3].document_id
        for item in ordered[:cluster_end]
        if item[0] >= semantic_floor
    )


def _semantic_relevance_floor(
    *, dimension: int, candidate_count: int, false_match_probability: float
) -> float:
    """Approximate a family-wise cosine bound under an isotropic null model."""

    per_candidate_tail = false_match_probability / max(1, candidate_count)
    z_score = NormalDist().inv_cdf(1.0 - per_candidate_tail)
    return min(1.0, z_score / math.sqrt(dimension))


def _retrieval_basis(lexical_evidence: bool, semantic_evidence: bool) -> str:
    if lexical_evidence and semantic_evidence:
        return "eligible through both lexical overlap and the semantic relevance rule"
    if lexical_evidence:
        return "eligible through lexical overlap with the indexed span"
    return "eligible through the semantic relevance rule without lexical overlap"


def _observation_match(match: RetrievedContext, *, content: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rank": match.rank,
        "document_id": match.document_id,
        "kind": match.kind.value,
        "origin": match.origin,
        "indexed_span": match.indexed_span,
        "similarity": match.similarity,
        "lexical_coverage": match.lexical_coverage,
        "eligibility_evidence": {
            "lexical_overlap": match.lexical_evidence,
            "adaptive_semantic_separation": match.semantic_evidence,
        },
        "content": content,
        "content_character_count": len(match.content),
        "content_omitted_character_count": len(match.content) - len(content),
    }
    if match.chunk_start is not None:
        result["chunk"] = {
            "start_character": match.chunk_start,
            "end_character": match.chunk_end,
        }
    return result


def _observation_exclusion(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "document_id",
            "kind",
            "origin",
            "similarity",
            "similarity_rank",
            "reason",
        )
        if key in candidate
    }


def _allocate_result_content(
    payload: dict[str, Any], result_index: int, content: str, limit: int
) -> None:
    results = payload["results"]
    assert isinstance(results, list)
    result = results[result_index]
    assert isinstance(result, dict)
    low = 0
    high = len(content)
    while low < high:
        length = (low + high + 1) // 2
        result["content"] = content[:length]
        result["content_omitted_character_count"] = len(content) - length
        if len(_observation_json(payload)) <= limit:
            low = length
        else:
            high = length - 1
    result["content"] = content[:low]
    result["content_omitted_character_count"] = len(content) - low


def _observation_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _audit_input_text(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _text_sha256(value: str) -> str:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        encoded = value.encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(encoded).hexdigest()


def _chunks(text: str, size: int, overlap: int) -> tuple[tuple[int, int, str], ...]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append((start, end, text[start:end]))
        if end == len(text):
            break
        start = end - overlap
    return tuple(chunks)


def _sensitive_filename(filename: str) -> bool:
    lowered = filename.casefold()
    return (
        lowered in _SENSITIVE_FILE_NAMES
        or lowered.startswith(".env.")
        or (lowered.startswith("id_") and not lowered.endswith(".pub"))
        or lowered.endswith((".key", ".pem", ".p12", ".pfx"))
    )


def _workspace_exclusion(origin: str, reason: str, detail: str = "") -> dict[str, str]:
    result = {
        "kind": RetrievalDocumentKind.WORKSPACE.value,
        "origin": origin,
        "reason": reason,
    }
    if detail:
        result["detail"] = detail
    return result


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _corpus_fingerprint(documents: Sequence[_UnembeddedDocument]) -> str:
    payload = "\n".join(
        "\0".join(
            (
                document.document_id,
                document.kind.value,
                document.origin,
                document.indexed_span,
                hashlib.sha256(document.index_text.encode()).hexdigest(),
                hashlib.sha256(document.returned_text.encode()).hexdigest(),
            )
        )
        for document in documents
    )
    return hashlib.sha256(payload.encode()).hexdigest()
