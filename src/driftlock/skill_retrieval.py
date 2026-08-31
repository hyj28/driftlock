"""Dependency-free activation indexing and deterministic skill retrieval.

The host injects an embedding callable.  This module never imports a model SDK,
opens a socket, or reads a skill outside :class:`SkillLibrary`'s canonical
reader.  A retriever fixes its library snapshot at its first query so an
evolution round cannot acquire more context merely because the library grows.
Later additions and removals are recorded in every result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import Any

from driftlock.skill_admission import DISTILLATION_ARMS, SkillLibrary
from driftlock.skill_distillation import Skill, serialize_skill

# This identifier makes reports from materially different retrieval semantics
# impossible to pool accidentally.  Bump it if the applicability gate, ordering,
# context bounds, or snapshot policy changes; those are experimental conditions,
# not implementation details.
RETRIEVAL_RULE_ID = "activation-cosine-threshold-v1"

# §4.3 compares two distillation arms across three evolution rounds.  Fixing the
# candidate pool at a retriever's first query prevents later additions from giving a
# long-lived round/arm path more candidates or more possible prompt context.  The cost
# is deliberate staleness: a newly admitted skill cannot be used until the host creates
# a new retriever (and therefore a new, reported snapshot).  Removals are different:
# they are honored immediately because deletion can revoke harmful instructions, and
# safety takes precedence over preserving a now-invalid pool.  `library_snapshot` is
# therefore the frozen initial pool, not current eligibility: it deliberately retains
# revoked ids for replay/audit.  Consumers must use the result's `live_eligible_*`
# fields to reconstruct what could actually be scored and injected.  Shrinkage and
# additions are both reported, so the write-up can identify changed round conditions.
LIBRARY_GROWTH_POLICY = "fixed_at_first_retrieval"
LIBRARY_REMOVAL_POLICY = "honor_immediately"

# This default is a measured boundary for one pinned model and task family, not a
# model-independent semantic constant.  On
# sentence-transformers/all-MiniLM-L6-v2 revision
# c9745ed1d9f207416be6d2e6f8de32d1f16199bf (384 dimensions), 14 real candidates
# were paired with five tasks: 14 source-task pairs were labelled applicable and
# 56 cross-task pairs inapplicable.  Applicable similarities were min 0.147,
# median 0.415, max 0.620; inapplicable similarities were min -0.013, median
# 0.188, max 0.330.  The threshold sweep (threshold: recall, false retrieval) was
# 0.30: 79%, 12%; 0.35: 71%, 0%; 0.40: 50%, 0%; 0.50: 7%, 0%; 0.75: 0%, 0%.
# Because 0.35 and 0.40 tied at the declared zero measured false-retrieval rate,
# 0.35 wins on recall.  The positive, inclusive gate remains a safety requirement.
#
# These 70 labels are provenance proxies, not human semantic judgements: a skill
# distilled from task T was labelled applicable to T and inapplicable to the other
# four tasks.  A procedure can genuinely transfer (for example, the same stale-cache
# failure can recur elsewhere), so some of the 56 negatives may really be positives
# and the measured false-retrieval rate may be biased downward.  The small,
# single-family sample is not a law.  Recalibrate if the model or revision, query or
# activation construction, task family, or label audit changes, and re-check on a
# broader labelled set before inheriting this value elsewhere.  Original task
# instruction remains the query source: its best observed similarity was 0.613,
# versus 0.385/0.620/0.613 for the last 3/10/25 trajectory steps, so the best
# alternative gained only 0.007.  Both distillation arms consume this same default
# through one retrieval rule, so the calibrated change cannot favor either arm.
DEFAULT_MIN_SIMILARITY = 0.35

# Machine-readable provenance for the measurements summarized above.  Keep the
# observed values as literals: this host cannot reproduce them because the pinned
# embedding model is deliberately a host-local experiment dependency.
DEFAULT_MIN_SIMILARITY_CALIBRATION: Mapping[str, Any] = {
    "schema_version": 1,
    "model": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "dimensions": 384,
    },
    "sample": {
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
    },
    "threshold_sweep": (
        {"threshold": 0.30, "recall_percent": 79, "false_retrieval_percent": 12},
        {"threshold": 0.35, "recall_percent": 71, "false_retrieval_percent": 0},
        {"threshold": 0.40, "recall_percent": 50, "false_retrieval_percent": 0},
        {"threshold": 0.50, "recall_percent": 7, "false_retrieval_percent": 0},
        {"threshold": 0.75, "recall_percent": 0, "false_retrieval_percent": 0},
    ),
    "selection": {
        "threshold": 0.35,
        "recall_percent": 71,
        "false_retrieval_percent": 0,
        "rationale": (
            "0.35 and 0.40 both measured zero false retrievals; choose 0.35 for "
            "its higher measured recall."
        ),
    },
    "query_source_check": {
        "selected": "original_task_instruction",
        "best_similarity": {
            "original_task_instruction": 0.613,
            "last_3_trajectory_steps": 0.385,
            "last_10_trajectory_steps": 0.620,
            "last_25_trajectory_steps": 0.613,
        },
        "best_alternative_gain": 0.007,
    },
    "limitations": (
        "The 70 pairs come from one model and one task family.",
        "Provenance labels are proxies rather than human semantic judgements.",
        "Transferable cross-task skills can make the measured false-retrieval rate "
        "optimistically low.",
    ),
    "recalibrate_if": (
        "embedding model or revision changes",
        "query or activation construction changes",
        "task family changes",
        "labels receive a semantic audit or the labelled sample is broadened",
    ),
    "arm_policy": "Both distillation arms use the identical retrieval rule.",
}

# §2.5's context-rot measurement falls from 98.1 in a clean prompt to 64.1 when
# information is distributed through a multi-turn run.  It proves extra context has
# a material cost, but it does not derive an optimal k or character allowance.  The
# top-three cap is therefore a provisional conservative bound on the number of
# three-section preventative instructions, and 6,000 characters independently bounds
# their aggregate size when skill documents vary.  Selection scans similarity order,
# skips an over-budget skill whole, and continues so a shorter lower-ranked skill may
# use the remaining budget; reports preserve both ranks and say whether that happened.
# Neither cap grows with the library.  The settling measurement is a validation sweep
# over k and injected characters for the pinned agent/embedder, recording task reward
# and false-skill injection rate; choose the smallest budget that preserves measured
# benefit, then freeze it before held-out evaluation.
DEFAULT_MAX_SKILLS = 3
DEFAULT_MAX_INJECTION_CHARACTERS = 6_000

# Threshold calibration needs scores below the current gate, but an unbounded score
# dump recreates §2.5's context problem in reports.  Record the 20 nearest misses and
# at most 20 selection exclusions, plus omitted counts and the lowest recorded score.
# Twenty is a provisional reporting budget, not a statistical derivation.  Calibrate
# it by measuring serialized report size and whether validation threshold sweeps reach
# below the recorded floor; increase it only if the sweep is truncated in practice.
DEFAULT_MAX_DIAGNOSTIC_CANDIDATES = 20


class SkillRetrievalStatus(StrEnum):
    """Whether retrieval ran, independently of whether anything applied."""

    USABLE = "usable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SkillRetrievalConfig:
    """One shared retrieval rule for every distillation arm."""

    minimum_similarity: float = DEFAULT_MIN_SIMILARITY
    max_skills: int = DEFAULT_MAX_SKILLS
    max_injection_characters: int = DEFAULT_MAX_INJECTION_CHARACTERS
    max_diagnostic_candidates: int = DEFAULT_MAX_DIAGNOSTIC_CANDIDATES

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_similarity, bool)
            or not isinstance(self.minimum_similarity, Real)
            or not math.isfinite(float(self.minimum_similarity))
            or not 0.0 < float(self.minimum_similarity) <= 1.0
        ):
            raise ValueError("minimum similarity must be finite and in (0, 1]")
        if (
            isinstance(self.max_skills, bool)
            or not isinstance(self.max_skills, int)
            or self.max_skills <= 0
        ):
            raise ValueError("maximum skills must be a positive integer")
        if (
            isinstance(self.max_injection_characters, bool)
            or not isinstance(self.max_injection_characters, int)
            or self.max_injection_characters <= 0
        ):
            raise ValueError("maximum injection characters must be a positive integer")
        if (
            isinstance(self.max_diagnostic_candidates, bool)
            or not isinstance(self.max_diagnostic_candidates, int)
            or self.max_diagnostic_candidates <= 0
        ):
            raise ValueError("maximum diagnostic candidates must be a positive integer")
        object.__setattr__(self, "minimum_similarity", float(self.minimum_similarity))

    def to_report(self) -> dict[str, Any]:
        """Describe the replayable rule and its context-cost rationale."""

        return {
            "rule_id": RETRIEVAL_RULE_ID,
            "minimum_similarity": self.minimum_similarity,
            "max_skills": self.max_skills,
            "max_injection_characters": self.max_injection_characters,
            "max_diagnostic_candidates": self.max_diagnostic_candidates,
            "similarity": "cosine(embedded query, embedded activation)",
            "tie_break": "candidate_id ascending",
            "selection_policy": (
                "Scan applicable candidates in similarity order. Check the skill-count "
                "cap before the character cap, so an exclusion reason names the first "
                "binding constraint under that priority, not every binding constraint. "
                "Skip a skill whole when it does not fit the remaining character "
                "budget, then continue to lower-ranked candidates."
            ),
            "context_policy": (
                "At most three preventative skills and 6000 injected characters "
                "by default. The cap is fixed rather than library-relative because "
                "PLAN.md section 2.5 identifies context rot as a long-horizon "
                "failure mode; library growth must not grow prompt context."
            ),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_report(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RetrievedSkill:
    """One selected skill, including the exact agent-facing text and basis."""

    candidate_id: str
    skill: Skill
    similarity: float
    similarity_rank: int
    skill_sha256: str
    injected_text: str

    def to_report(self, selection_rank: int) -> dict[str, Any]:
        return {
            "selection_rank": selection_rank,
            "similarity_rank": self.similarity_rank,
            "candidate_id": self.candidate_id,
            "similarity": self.similarity,
            "basis": "activation cosine similarity met the configured threshold",
            "activation": self.skill.activation,
            "skill_sha256": self.skill_sha256,
            "injection_characters": len(self.injected_text),
            "injected_text": self.injected_text,
        }


@dataclass(frozen=True, slots=True)
class SkillRetrievalResult:
    """Auditable retrieval output that keeps no-match separate from failure."""

    status: SkillRetrievalStatus
    query_sha256: str
    query_character_count: int
    config: SkillRetrievalConfig
    library_snapshot: Mapping[str, Any]
    library_observation: Mapping[str, Any]
    live_eligible_candidate_ids: tuple[str, ...] | None = None
    considered_candidate_count: int = 0
    applicable_candidate_count: int = 0
    matches: tuple[RetrievedSkill, ...] = ()
    exclusions: tuple[Mapping[str, Any], ...] = ()
    excluded_candidate_count: int = 0
    near_threshold_candidates: tuple[Mapping[str, Any], ...] = ()
    below_threshold_candidate_count: int = 0
    refusal: Mapping[str, str] | None = None

    @property
    def injection_text(self) -> str:
        """Return exactly the bounded text a caller may inject into an agent."""

        return "\n\n".join(match.injected_text for match in self.matches)

    def to_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "schema_version": 1,
            "mode": "skill-retrieval",
            "status": self.status.value,
            "query": {
                "sha256": self.query_sha256,
                "character_count": self.query_character_count,
            },
            "configuration": {
                **self.config.to_report(),
                "fingerprint": self.config.fingerprint,
            },
            "library_snapshot": dict(self.library_snapshot),
            "library_observation": dict(self.library_observation),
            "live_eligible_candidate_count": (
                len(self.live_eligible_candidate_ids)
                if self.live_eligible_candidate_ids is not None
                else None
            ),
            "live_eligible_candidate_ids": (
                list(self.live_eligible_candidate_ids)
                if self.live_eligible_candidate_ids is not None
                else None
            ),
            "considered_candidate_count": self.considered_candidate_count,
            "applicable_candidate_count": self.applicable_candidate_count,
            "selected_skill_count": len(self.matches),
            "injection_character_count": len(self.injection_text),
            "selection_matches_similarity_prefix": [
                match.similarity_rank for match in self.matches
            ]
            == list(range(1, len(self.matches) + 1)),
            "selected_skills": [
                match.to_report(rank)
                for rank, match in enumerate(self.matches, start=1)
            ],
            "exclusions": [dict(exclusion) for exclusion in self.exclusions],
            "excluded_candidate_count": self.excluded_candidate_count,
            "unreported_exclusion_count": (
                self.excluded_candidate_count - len(self.exclusions)
            ),
            "threshold_diagnostics": {
                "below_threshold_candidate_count": (
                    self.below_threshold_candidate_count
                ),
                "reported_candidate_count": len(self.near_threshold_candidates),
                "unreported_candidate_count": (
                    self.below_threshold_candidate_count
                    - len(self.near_threshold_candidates)
                ),
                "complete": (
                    self.below_threshold_candidate_count
                    == len(self.near_threshold_candidates)
                ),
                "lowest_reported_similarity": (
                    self.near_threshold_candidates[-1]["similarity"]
                    if self.near_threshold_candidates
                    else None
                ),
                "candidates": [
                    dict(candidate) for candidate in self.near_threshold_candidates
                ],
            },
        }
        if self.refusal is not None:
            report["refusal"] = dict(self.refusal)
        return report


@dataclass(frozen=True, slots=True)
class _IndexedSkill:
    candidate_id: str
    skill: Skill
    vector: tuple[float, ...]
    skill_sha256: str
    injected_text: str


@dataclass(frozen=True, slots=True)
class _IndexFailure:
    reason: str
    detail: str
    stage: str


class _EmbeddingValidationError(ValueError):
    pass


class ActivationSkillRetriever:
    """Index admitted activation text and retrieve applicable skills only."""

    def __init__(
        self,
        library: SkillLibrary,
        embed: Callable[[Sequence[str]], Iterable[Iterable[Real]]],
        *,
        config: SkillRetrievalConfig | None = None,
    ) -> None:
        if not isinstance(library, SkillLibrary):
            raise TypeError("library must be a SkillLibrary")
        if not callable(embed):
            raise TypeError("embed must be callable")
        self._library = library
        self._embed = embed
        self.config = config or SkillRetrievalConfig()
        self._index: tuple[_IndexedSkill, ...] | None = None
        self._index_failure: _IndexFailure | None = None
        self._snapshot: dict[str, Any] | None = None
        self._query_vectors: dict[str, tuple[float, ...] | _IndexFailure] = {}

    def retrieve(self, query_context: str) -> SkillRetrievalResult:
        """Retrieve above-threshold skills in stable score/id order."""

        if not isinstance(query_context, str) or not query_context.strip():
            raise ValueError("retrieval query context must be non-empty text")
        query_hash = hashlib.sha256(query_context.encode()).hexdigest()
        self._ensure_index()
        snapshot = self._snapshot or _snapshot_report(())

        try:
            current_ids = self._library.admitted_skill_ids()
        except Exception as error:
            return self._failure_result(
                query_context,
                snapshot,
                _IndexFailure(
                    "library_read_failed",
                    f"could not observe current admitted skills: {error}",
                    "library_observation",
                ),
            )
        observation = _library_observation(snapshot, current_ids)

        if self._index_failure is not None:
            return self._failure_result(
                query_context, snapshot, self._index_failure, observation
            )
        current_id_set = set(current_ids)
        # This is the live post-filter pool. Similarity ranks below are local to this
        # result and can shift after a revocation; candidate_id is the only stable key
        # for joining candidates across retrieval reports.
        index = tuple(
            item for item in (self._index or ()) if item.candidate_id in current_id_set
        )
        if not index:
            return SkillRetrievalResult(
                status=SkillRetrievalStatus.USABLE,
                query_sha256=query_hash,
                query_character_count=len(query_context),
                config=self.config,
                library_snapshot=snapshot,
                library_observation=observation,
                live_eligible_candidate_ids=(),
            )

        query_vector = self._query_vector(query_context, len(index[0].vector))
        if isinstance(query_vector, _IndexFailure):
            return self._failure_result(
                query_context, snapshot, query_vector, observation
            )

        scored = sorted(
            ((_cosine(query_vector, item.vector), item) for item in index),
            key=lambda pair: (-pair[0], pair[1].candidate_id),
        )
        applicable = [
            (rank, similarity, item)
            for rank, (similarity, item) in enumerate(scored, start=1)
            # The applicability gate is deliberately inclusive. A cosine exactly at
            # the configured threshold is applicable, never a near miss.
            if similarity >= self.config.minimum_similarity
        ]
        below_threshold = [
            {
                "similarity_rank": rank,
                "candidate_id": item.candidate_id,
                "similarity": similarity,
                "activation": item.skill.activation,
            }
            for rank, (similarity, item) in enumerate(scored, start=1)
            if similarity < self.config.minimum_similarity
        ]
        selected: list[RetrievedSkill] = []
        exclusions: list[dict[str, Any]] = []
        used_characters = 0
        for similarity_rank, similarity, item in applicable:
            separator = 2 if selected else 0
            # Cap priority is fixed: count first, then characters. The recorded reason
            # is the first cap hit under this order, not an exhaustive constraint list.
            if len(selected) >= self.config.max_skills:
                exclusions.append(
                    {
                        "similarity_rank": similarity_rank,
                        "candidate_id": item.candidate_id,
                        "similarity": similarity,
                        "reason": "max_skills",
                    }
                )
                continue
            if (
                used_characters + separator + len(item.injected_text)
                > self.config.max_injection_characters
            ):
                exclusions.append(
                    {
                        "similarity_rank": similarity_rank,
                        "candidate_id": item.candidate_id,
                        "similarity": similarity,
                        "reason": "injection_character_budget",
                    }
                )
                continue
            selected.append(
                RetrievedSkill(
                    candidate_id=item.candidate_id,
                    skill=item.skill,
                    similarity=similarity,
                    similarity_rank=similarity_rank,
                    skill_sha256=item.skill_sha256,
                    injected_text=item.injected_text,
                )
            )
            used_characters += separator + len(item.injected_text)

        return SkillRetrievalResult(
            status=SkillRetrievalStatus.USABLE,
            query_sha256=query_hash,
            query_character_count=len(query_context),
            config=self.config,
            library_snapshot=snapshot,
            library_observation=observation,
            live_eligible_candidate_ids=tuple(item.candidate_id for item in index),
            considered_candidate_count=len(index),
            applicable_candidate_count=len(applicable),
            matches=tuple(selected),
            exclusions=tuple(exclusions[: self.config.max_diagnostic_candidates]),
            excluded_candidate_count=len(exclusions),
            near_threshold_candidates=tuple(
                below_threshold[: self.config.max_diagnostic_candidates]
            ),
            below_threshold_candidate_count=len(below_threshold),
        )

    def _ensure_index(self) -> None:
        if self._index is not None or self._index_failure is not None:
            return
        try:
            candidate_ids = self._library.admitted_skill_ids()
            skills = tuple(
                (candidate_id, self._library.read_skill(candidate_id))
                for candidate_id in candidate_ids
            )
        except Exception as error:
            self._snapshot = _snapshot_report(())
            self._index_failure = _IndexFailure(
                "library_read_failed",
                f"could not build the activation index: {error}",
                "index",
            )
            return

        documents = tuple(
            (candidate_id, skill, serialize_skill(skill))
            for candidate_id, skill in skills
        )
        self._snapshot = _snapshot_report(
            tuple(
                (candidate_id, hashlib.sha256(document.encode()).hexdigest())
                for candidate_id, _, document in documents
            )
        )
        if not documents:
            self._index = ()
            return
        try:
            raw_vectors = self._embed(
                tuple(skill.activation for _, skill, _ in documents)
            )
            vectors = _coerce_embeddings(raw_vectors, len(documents))
        except _EmbeddingValidationError as error:
            self._index_failure = _IndexFailure(
                "invalid_embedding", f"activation index: {error}", "index"
            )
            return
        except Exception as error:
            self._index_failure = _IndexFailure(
                "embedding_callable_failed",
                "activation index embedding call failed: "
                f"{type(error).__name__}: {error}",
                "index",
            )
            return

        self._index = tuple(
            _IndexedSkill(
                candidate_id=candidate_id,
                skill=skill,
                vector=vector,
                skill_sha256=hashlib.sha256(document.encode()).hexdigest(),
                injected_text=_injected_skill_text(candidate_id, document),
            )
            for (candidate_id, skill, document), vector in zip(
                documents, vectors, strict=True
            )
        )

    def _query_vector(
        self, query_context: str, expected_dimension: int
    ) -> tuple[float, ...] | _IndexFailure:
        cached = self._query_vectors.get(query_context)
        if cached is not None:
            return cached
        try:
            raw_vectors = self._embed((query_context,))
            vector = _coerce_embeddings(
                raw_vectors, 1, expected_dimension=expected_dimension
            )[0]
        except _EmbeddingValidationError as error:
            result: tuple[float, ...] | _IndexFailure = _IndexFailure(
                "invalid_embedding", f"query: {error}", "query"
            )
        except Exception as error:
            result = _IndexFailure(
                "embedding_callable_failed",
                f"query embedding call failed: {type(error).__name__}: {error}",
                "query",
            )
        else:
            result = vector
        self._query_vectors[query_context] = result
        return result

    def _failure_result(
        self,
        query_context: str,
        snapshot: Mapping[str, Any],
        failure: _IndexFailure,
        observation: Mapping[str, Any] | None = None,
    ) -> SkillRetrievalResult:
        return SkillRetrievalResult(
            status=SkillRetrievalStatus.FAILED,
            query_sha256=hashlib.sha256(query_context.encode()).hexdigest(),
            query_character_count=len(query_context),
            config=self.config,
            library_snapshot=snapshot,
            library_observation=observation or _library_observation(snapshot, ()),
            considered_candidate_count=int(snapshot["candidate_count"]),
            refusal={
                "reason": failure.reason,
                "detail": failure.detail,
                "stage": failure.stage,
            },
        )


def retrieve_for_distillation_arms(
    retriever: ActivationSkillRetriever, query_context: str
) -> dict[str, SkillRetrievalResult]:
    """Fan one retrieval result out to both arms, preventing path divergence."""

    result = retriever.retrieve(query_context)
    return {arm: result for arm in DISTILLATION_ARMS}


def _coerce_embeddings(
    raw_embeddings: Iterable[Iterable[Real]],
    expected_count: int,
    *,
    expected_dimension: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(raw_embeddings, (str, bytes, Mapping)):
        raise _EmbeddingValidationError("embedding result must be a vector sequence")
    try:
        raw_vectors = tuple(raw_embeddings)
    except TypeError as error:
        raise _EmbeddingValidationError("embedding result must be iterable") from error
    if len(raw_vectors) != expected_count:
        raise _EmbeddingValidationError(
            f"expected {expected_count} vector(s), received {len(raw_vectors)}"
        )

    vectors = []
    dimension = expected_dimension
    for vector_index, raw_vector in enumerate(raw_vectors):
        if isinstance(raw_vector, (str, bytes, Mapping)):
            raise _EmbeddingValidationError(
                f"vector {vector_index} must contain numeric values"
            )
        try:
            raw_values = tuple(raw_vector)
        except TypeError as error:
            raise _EmbeddingValidationError(
                f"vector {vector_index} must be iterable"
            ) from error
        if not raw_values:
            raise _EmbeddingValidationError(f"vector {vector_index} is empty")
        values = []
        for value_index, value in enumerate(raw_values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise _EmbeddingValidationError(
                    f"vector {vector_index} value {value_index} is not numeric"
                )
            number = float(value)
            if not math.isfinite(number):
                raise _EmbeddingValidationError(
                    f"vector {vector_index} value {value_index} is not finite"
                )
            values.append(number)
        if dimension is None:
            dimension = len(values)
        if len(values) != dimension:
            raise _EmbeddingValidationError(
                f"vector {vector_index} has length {len(values)}; expected {dimension}"
            )
        if math.fsum(value * value for value in values) == 0.0:
            raise _EmbeddingValidationError(f"vector {vector_index} has zero magnitude")
        vectors.append(tuple(values))
    return tuple(vectors)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def _snapshot_report(entries: Sequence[tuple[str, str]]) -> dict[str, Any]:
    candidate_ids = tuple(entry[0] for entry in entries)
    payload = "\n".join(f"{candidate_id}\0{digest}" for candidate_id, digest in entries)
    return {
        "growth_policy": LIBRARY_GROWTH_POLICY,
        "removal_policy": LIBRARY_REMOVAL_POLICY,
        "scope": "frozen_initial_candidate_pool_not_live_eligibility",
        "candidate_count": len(candidate_ids),
        "candidate_ids": list(candidate_ids),
        "fingerprint": hashlib.sha256(payload.encode()).hexdigest(),
    }


def _library_observation(
    snapshot: Mapping[str, Any], current_ids: Sequence[str]
) -> dict[str, Any]:
    snapshot_ids = tuple(snapshot["candidate_ids"])
    snapshot_set = set(snapshot_ids)
    current_set = set(current_ids)
    return {
        "current_candidate_count": len(current_ids),
        "current_candidate_ids": list(current_ids),
        "added_after_snapshot": sorted(current_set - snapshot_set),
        "removed_after_snapshot": sorted(snapshot_set - current_set),
        "snapshot_still_current": snapshot_set == current_set,
    }


def _injected_skill_text(candidate_id: str, document: str) -> str:
    return f'<retrieved-skill id="{candidate_id}">\n{document}\n</retrieved-skill>'
