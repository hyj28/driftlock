"""Once-per-task skill retrieval and auditable phase-entry prompt injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from driftlock.skill_admission import DISTILLATION_ARMS
from driftlock.skill_retrieval import (
    ActivationSkillRetriever,
    SkillRetrievalResult,
    SkillRetrievalStatus,
    retrieve_for_distillation_arms,
)

# Retrieval is deliberately once per task, with the original task instruction as
# the query.  This keeps every checkpoint and verifier-resume phase in one trial
# under the same retrieved-skill condition, at the cost of not adapting to failures
# discovered partway through.  The instruction is the only pre-execution situation
# description available at this seam.  A better query would combine it with frozen,
# provider-independent environment signals that are available before the first call;
# labelled query/activation pairs would settle which signals help without leaking
# trajectory-dependent treatment changes into checkpoint comparisons.
SKILL_RETRIEVAL_FREQUENCY = "once_per_task"
SKILL_RETRIEVAL_QUERY_SOURCE = "original_task_instruction"

# §2.5's 98.1-to-64.1 context-rot result makes position an experimental condition.
# Put one conspicuously delimited block before the phase-entry prompt, where it is
# visible before task or verifier text, and repeat that same fixed block only when a
# new phase starts.  The cost is prompt tokens on every verifier resume; a held-out
# position/repetition sweep for the pinned agent would settle whether initial-only or
# appended context preserves more benefit.  Retrieval itself is not repeated, so the
# selected skills never vary within the trajectory.
SKILL_INJECTION_POSITION = "prepended_to_each_phase_entry_prompt"
SKILL_INJECTION_START = "<driftlock-retrieved-skill-context>"
SKILL_INJECTION_END = "</driftlock-retrieved-skill-context>"
SKILL_RETRIEVAL_FAILURE_POLICY = "fail_trial_after_recording"


class SkillRetrievalFailed(RuntimeError):
    """The configured skill layer could not decide applicability safely."""


@dataclass(slots=True)
class TaskSkillInjector:
    """Retrieve once, then reuse one exact injection block across task phases."""

    retriever: ActivationSkillRetriever
    distillation_arm: str
    _query: str | None = None
    _result: SkillRetrievalResult | None = None

    def __post_init__(self) -> None:
        if self.distillation_arm not in DISTILLATION_ARMS:
            raise ValueError(
                "skill distillation arm must be one of " + ", ".join(DISTILLATION_ARMS)
            )

    @property
    def result(self) -> SkillRetrievalResult | None:
        return self._result

    def retrieve_for_task(self, instruction: str) -> SkillRetrievalResult:
        """Fix and return the task's sole retrieval result."""

        if self._query is not None:
            if instruction != self._query:
                raise RuntimeError("one task skill injector cannot switch queries")
            assert self._result is not None
            return self._result
        by_arm = retrieve_for_distillation_arms(self.retriever, instruction)
        self._query = instruction
        self._result = by_arm[self.distillation_arm]
        return self._result

    @property
    def injection_block(self) -> str:
        result = self._require_result()
        if result.status is SkillRetrievalStatus.FAILED or not result.injection_text:
            return ""
        return (
            f"{SKILL_INJECTION_START}\n{result.injection_text}\n{SKILL_INJECTION_END}"
        )

    def inject_phase_entry(self, prompt: str) -> str:
        """Prepend the fixed block, preserving no-match prompts byte-for-byte."""

        result = self._require_result()
        if result.status is SkillRetrievalStatus.FAILED:
            refusal = result.refusal or {}
            reason = refusal.get("reason", "unknown_retrieval_failure")
            raise SkillRetrievalFailed(f"skill retrieval failed: {reason}")
        block = self.injection_block
        return prompt if not block else f"{block}\n\n{prompt}"

    def to_report(self) -> dict[str, Any]:
        """Return the exact selection and injection evidence for a trial record."""

        result = self._require_result()
        block = self.injection_block
        if result.status is SkillRetrievalStatus.FAILED:
            injection_status = "retrieval_failed"
        elif block:
            injection_status = "injected"
        else:
            injection_status = "no_applicable_skills"
        return {
            "schema_version": 1,
            "mode": "task-skill-injection",
            "status": result.status.value,
            "distillation_arm": self.distillation_arm,
            "policy": {
                "retrieval_frequency": SKILL_RETRIEVAL_FREQUENCY,
                "query_source": SKILL_RETRIEVAL_QUERY_SOURCE,
                "injection_position": SKILL_INJECTION_POSITION,
                "retrieval_failure": SKILL_RETRIEVAL_FAILURE_POLICY,
            },
            "retrieval": result.to_report(),
            "injection": {
                "status": injection_status,
                "candidate_ids": [match.candidate_id for match in result.matches],
                "delimiter_start": SKILL_INJECTION_START,
                "delimiter_end": SKILL_INJECTION_END,
                "character_count": len(block),
                "text": block,
            },
        }

    def phase_report(self) -> dict[str, Any]:
        """Return compact evidence that this phase received the fixed block."""

        report = self.to_report()
        injection = report["injection"]
        return {
            "status": injection["status"],
            "candidate_ids": list(injection["candidate_ids"]),
            "position": SKILL_INJECTION_POSITION,
            "character_count": injection["character_count"],
        }

    def _require_result(self) -> SkillRetrievalResult:
        if self._result is None:
            raise RuntimeError("skill retrieval has not run for this task")
        return self._result
