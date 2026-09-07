"""Paired validation admission and persistent skill-library bookkeeping.

This module consumes validation results; it never runs an agent or verifier.  It
is therefore Harbor-free and deliberately has no provider or credential path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import median, stdev
from typing import Any

from driftlock.skill_distillation import (
    Skill,
    parse_skill,
    serialize_skill,
    validate_skill,
)

ADMISSION_REPORT_NAME = "skill-admission.json"
ADMISSION_RULE_ID = "paired-direction-v1"
# §4.3 fixes the held-out validation split at ten tasks, one attempt each.  The
# rule refuses any smaller denominator rather than silently changing its power.
VALIDATION_TASK_COUNT = 10
DISTILLATION_ARMS = ("baseline", "localized")

# Round five (§2.3a) found that between-task variance dwarfed arm differences and
# one task could move the complete ranking.  Requiring improvement on nine of the
# ten paired tasks makes that observed failure shape ineligible: one task cannot
# carry nine non-improving tasks.  Under continuous symmetric null signs, at least
# nine positive signs occur with probability (C(10, 9) + C(10, 10)) / 2**10 =
# 11/1024 = 1.074% under independent, symmetric task-level null signs (ties only
# reduce it).  That is well below the field's 55/388 = 14.2% pass rate and bounds
# all-null chance admissions at 4.17 across 388 tests, instead of roughly 19 at
# an uncorrected 5% threshold.  Ten pairs cannot support a useful family-wise-
# corrected per-candidate significance claim across hundreds of candidates, so
# this is explicitly an effect/direction screen, not a p-value.  The report names
# the null assumption because correlated task signs would invalidate this bound.
MIN_POSITIVE_TASKS = 9
NULL_ADMISSION_PROBABILITY_UPPER_BOUND = sum(
    math.comb(VALIDATION_TASK_COUNT, positive_count)
    for positive_count in range(MIN_POSITIVE_TASKS, VALIDATION_TASK_COUNT + 1)
) / (2**VALIDATION_TASK_COUNT)

# JSON round trips can perturb decimal rewards at about 1e-16.  This tolerance is
# ten orders below §2.3a's smallest real measured task swing (+0.091), so it
# suppresses representation noise without reclassifying the measured effect.
DELTA_ABS_TOLERANCE = 1e-12

# An uninjected treatment is byte-identical to its paired control, so its delta
# measures run-to-run noise rather than skill effect.  Keep this explanation in
# both machine-readable reports instead of relying on operator interpretation.
NULL_CHANNEL_RATIONALE = (
    "No-skill-injected treatments were byte-identical to their controls, so "
    "their measured deltas are a run-to-run noise floor. Skill-injected effects "
    "must be read against that noise floor, not against zero. Because task "
    "distributions can differ, channel contrasts are valid only within a task; "
    "a pooled cross-task contrast with different task mixes is not reported."
)

# Older validation reports do not carry task metadata.  A visible label prevents
# their observations from being dropped or silently merged with a recorded task.
UNKNOWN_TASK_LABEL = "task unknown"

# The two channels can contain different task mixtures, making a pooled mean
# difference a task-composition artifact rather than an interpretable effect.
NULL_CHANNEL_POOLING_INVALID_REASON = (
    "The injected and no-skill-injected channels can contain different task "
    "mixtures, so a pooled cross-task mean difference would confound channel "
    "with task composition."
)

_SAFE_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class SkillAdmissionStatus(StrEnum):
    """Disposition of one candidate's validation measurement."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    INCOMPLETE = "incomplete"


class TaskMetadataCondition(StrEnum):
    """How admission obtained a candidate's reporting-only task identity."""

    DIRECT = "direct_candidate_metadata"
    SUMMARY_RECORDED = "validation_observation_summary_recorded"
    SUMMARY_ABSENT = "validation_observation_summary_missing"
    SUMMARY_TASK_NULL = "validation_observation_summary_task_name_null"
    SUMMARY_SOURCE_ABSENT = "validation_observation_summary_source_task_name_null"
    TOP_LEVEL_FALLBACK_SUMMARY_ABSENT = "top_level_fallback_summary_missing"
    TOP_LEVEL_FALLBACK_SUMMARY_TASK_NULL = "top_level_fallback_summary_task_null"
    SUMMARY_TOP_LEVEL_DISAGREEMENT = "summary_top_level_task_name_disagreement"


class CandidateRetrievalStatus(StrEnum):
    """Whether any measured observation established skill retrieval."""

    RETRIEVED = "retrieved"
    NEVER_RETRIEVED = "never_retrieved"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SkillAdmissionCandidate:
    """One distilled skill and its already-paired per-task reward deltas."""

    candidate_id: str
    arm: str
    skill: Skill
    paired_deltas: tuple[float | None, ...]
    injection_flags: tuple[bool | None, ...] | None = None
    task_name: str | None = None
    source_task_name: str | None = None
    task_metadata_condition: TaskMetadataCondition = TaskMetadataCondition.DIRECT
    top_level_task_name: str | None = None

    def __post_init__(self) -> None:
        _validate_candidate_id(self.candidate_id)
        if self.arm not in DISTILLATION_ARMS:
            raise ValueError(
                f"unknown distillation arm {self.arm!r}; expected one of "
                f"{', '.join(DISTILLATION_ARMS)}"
            )
        validate_skill(self.skill)
        normalized = tuple(
            _optional_delta(value, self.candidate_id, index)
            for index, value in enumerate(self.paired_deltas)
        )
        if len(normalized) > VALIDATION_TASK_COUNT:
            raise ValueError(
                f"candidate {self.candidate_id!r} has {len(normalized)} paired "
                f"deltas; validation split has {VALIDATION_TASK_COUNT} tasks"
            )
        object.__setattr__(self, "paired_deltas", normalized)
        if self.injection_flags is not None:
            normalized_flags = tuple(
                _optional_injection_flag(value, self.candidate_id, index)
                for index, value in enumerate(self.injection_flags)
            )
            if len(normalized_flags) != len(normalized):
                raise ValueError(
                    f"candidate {self.candidate_id!r} has "
                    f"{len(normalized_flags)} injection flags for "
                    f"{len(normalized)} paired deltas"
                )
            object.__setattr__(self, "injection_flags", normalized_flags)
        _validate_optional_task_name(self.task_name, self.candidate_id, "task_name")
        _validate_optional_task_name(
            self.source_task_name, self.candidate_id, "source_task_name"
        )
        _validate_optional_task_name(
            self.top_level_task_name, self.candidate_id, "top_level_task_name"
        )
        if not isinstance(self.task_metadata_condition, TaskMetadataCondition):
            try:
                condition = TaskMetadataCondition(self.task_metadata_condition)
            except (TypeError, ValueError):
                raise ValueError(
                    f"candidate {self.candidate_id!r} has unknown task metadata "
                    f"condition {self.task_metadata_condition!r}"
                ) from None
            object.__setattr__(self, "task_metadata_condition", condition)


def decide_skill_admission(candidate: SkillAdmissionCandidate) -> dict[str, Any]:
    """Apply the single shared paired rule to either distillation arm."""

    deltas = candidate.paired_deltas
    measured = sum(delta is not None for delta in deltas)
    missing = VALIDATION_TASK_COUNT - measured
    decision: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "arm": candidate.arm,
        "status": SkillAdmissionStatus.INCOMPLETE.value,
        "rule_id": ADMISSION_RULE_ID,
        "skill_application": _skill_application_report(
            candidate.injection_flags, candidate.paired_deltas
        ),
        "measurement": {
            "expected_task_count": VALIDATION_TASK_COUNT,
            "measured_task_count": measured,
            "missing_task_count": missing,
            "paired_deltas": list(deltas),
            "effect": None,
        },
    }
    if len(deltas) != VALIDATION_TASK_COUNT or missing:
        decision["refusal"] = {
            "reason": "incomplete_validation",
            "detail": (
                f"only {measured} of {VALIDATION_TASK_COUNT} paired task deltas "
                "are measured; missing results are not zeros, and an incomplete "
                "candidate is not averaged"
            ),
        }
        return decision

    complete = [delta for delta in deltas if delta is not None]
    positive = sum(delta > DELTA_ABS_TOLERANCE for delta in complete)
    negative = sum(delta < -DELTA_ABS_TOLERANCE for delta in complete)
    zero = VALIDATION_TASK_COUNT - positive - negative
    total = math.fsum(complete)
    mean = total / VALIDATION_TASK_COUNT
    decision["measurement"]["effect"] = {
        "mean_delta": mean,
        "median_delta": median(complete),
        "total_delta": total,
        "minimum_delta": min(complete),
        "maximum_delta": max(complete),
        "positive_task_count": positive,
        "zero_task_count": zero,
        "negative_task_count": negative,
    }

    if positive < MIN_POSITIVE_TASKS:
        decision["status"] = SkillAdmissionStatus.REJECTED.value
        decision["refusal"] = {
            "reason": "inconsistent_improvement",
            "detail": (
                f"improved on {positive} of {VALIDATION_TASK_COUNT} paired tasks; "
                f"the shared rule requires at least {MIN_POSITIVE_TASKS} so one "
                "high-variance task cannot decide admission"
            ),
        }
        return decision
    # The directional threshold alone can admit nine tiny gains plus one larger
    # loss.  Positive mean is therefore a separate effect-size gate: the skill
    # must improve aggregate measured reward as well as win consistently.
    if mean <= DELTA_ABS_TOLERANCE:
        decision["status"] = SkillAdmissionStatus.REJECTED.value
        decision["refusal"] = {
            "reason": "nonpositive_mean_effect",
            "detail": (
                f"mean paired delta {mean:+.6g} is not above the "
                f"{DELTA_ABS_TOLERANCE:g} representation-noise tolerance"
            ),
        }
        return decision

    decision["status"] = SkillAdmissionStatus.ADMITTED.value
    decision["admission_context"] = {
        "single_candidate_null_admission_probability_upper_bound": (
            NULL_ADMISSION_PROBABILITY_UPPER_BOUND
        ),
        "interpretation": (
            "Admission means this candidate passed a directional effect screen; "
            "it is not an individual statistical certification. Null candidates "
            "can survive, and the screen protects expected library composition "
            "only when read with cohort context."
        ),
    }
    return decision


def assemble_admission_report(
    candidates: Sequence[SkillAdmissionCandidate],
) -> dict[str, Any]:
    """Decide a cohort and make its multiplicity and denominator explicit."""

    if not candidates:
        raise ValueError("skill admission cohort must contain at least one candidate")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("skill admission cohort has duplicate candidate ids")

    decisions = [decide_skill_admission(candidate) for candidate in candidates]
    statuses = Counter(decision["status"] for decision in decisions)
    tested = (
        statuses[SkillAdmissionStatus.ADMITTED.value]
        + statuses[SkillAdmissionStatus.REJECTED.value]
    )
    admitted = statuses[SkillAdmissionStatus.ADMITTED.value]
    reasons = Counter(
        decision["refusal"]["reason"] for decision in decisions if "refusal" in decision
    )
    expected_chance = tested * NULL_ADMISSION_PROBABILITY_UPPER_BOUND
    retrieval_split = _retrieval_split(candidates, decisions)
    null_channel = _task_null_channel_summary(candidates, decisions)
    if retrieval_split["availability"] == "unavailable":
        pass_rate_numerator = admitted
        pass_rate_denominator = tested
        pass_rate_denominator_description = "all_complete_candidates"
    else:
        pass_rate_numerator = retrieval_split["retrieved_admitted_candidate_count"]
        pass_rate_denominator = retrieval_split["retrieved_candidate_count"]
        pass_rate_denominator_description = "retrieved_complete_candidates"
    pass_rate = (
        pass_rate_numerator / pass_rate_denominator if pass_rate_denominator else None
    )
    admitted_outside_pass_rate_denominator = (
        admitted - pass_rate_numerator
        if pass_rate_denominator_description == "retrieved_complete_candidates"
        else 0
    )
    cohort_context = {
        "tested_candidate_count": tested,
        "observed_admitted_candidate_count": admitted,
        "all_null_expected_chance_admissions_upper_bound": expected_chance,
        "interpretation": (
            "Under the stated null model this many admissions are expected by "
            "chance across the cohort; it does not identify which individual "
            "admissions are false."
        ),
    }
    for decision in decisions:
        if decision["status"] == SkillAdmissionStatus.ADMITTED.value:
            decision["admission_context"]["cohort"] = cohort_context
    return {
        "schema_version": 2,
        "mode": "skill-admission",
        "rule": _rule_report(),
        "submitted_candidate_count": len(decisions),
        "tested_candidate_count": tested,
        "incomplete_candidate_count": statuses[SkillAdmissionStatus.INCOMPLETE.value],
        "admitted_candidate_count": admitted,
        "rejected_candidate_count": statuses[SkillAdmissionStatus.REJECTED.value],
        "pass_rate": pass_rate,
        "pass_rate_numerator": pass_rate_numerator,
        "pass_rate_denominator": pass_rate_denominator,
        "pass_rate_denominator_description": pass_rate_denominator_description,
        "admitted_outside_pass_rate_denominator_count": (
            admitted_outside_pass_rate_denominator
        ),
        "retrieval_split": retrieval_split,
        "refusal_reason_counts": dict(sorted(reasons.items())),
        "null_channel": null_channel,
        "multiple_comparisons": {
            "candidate_tests": tested,
            "single_candidate_null_admission_probability_upper_bound": (
                NULL_ADMISSION_PROBABILITY_UPPER_BOUND
            ),
            "all_null_expected_chance_admissions_upper_bound": expected_chance,
            "interpretation": (
                "Expectation assumes each complete candidate has independent, "
                "symmetric paired task-level null signs; ties can only lower the "
                "bound. Linearity of expectation does not require candidates "
                "tested on the shared split to be independent. Correlated task "
                "signs invalidate the bound. It is not a per-candidate p-value or "
                "a family-wise significance claim."
            ),
        },
        "field_reference": {
            "tested_candidate_count": 388,
            "admitted_candidate_count": 55,
            "pass_rate": 55 / 388,
            "comparison_note": (
                "The study used a different validation filter; its pass rate and "
                "this directional-screen pass rate are juxtaposed as references, "
                "not treated as like-for-like estimates."
            ),
        },
        "decisions": decisions,
    }


def render_admission_report(report: Mapping[str, Any]) -> str:
    """Render the cohort denominator, effects, and chance expectation for humans."""

    tested = report["tested_candidate_count"]
    admitted = report["admitted_candidate_count"]
    incomplete = report["incomplete_candidate_count"]
    pass_rate = report["pass_rate"]
    rate_text = "not defined" if pass_rate is None else f"{pass_rate:.1%}"
    pass_rate_numerator = report.get("pass_rate_numerator", admitted)
    pass_rate_denominator = report.get("pass_rate_denominator", tested)
    pass_rate_denominator_description = report.get(
        "pass_rate_denominator_description", "all_complete_candidates"
    )
    multiple = report["multiple_comparisons"]
    field = report["field_reference"]
    null_probability = multiple[
        "single_candidate_null_admission_probability_upper_bound"
    ]
    retrieval_split = report.get("retrieval_split")
    if isinstance(retrieval_split, Mapping):
        retrieval_unavailable = retrieval_split["availability"] == "unavailable"
        denominator_label = (
            "all complete candidates"
            if pass_rate_denominator_description == "all_complete_candidates"
            else "retrieved complete candidates"
        )
        headline = (
            f"tested {tested} complete candidate(s); admitted {admitted} admission "
            "verdict(s); "
            f"rejected {report['rejected_candidate_count']}; incomplete "
            f"{incomplete}; retrieval split among complete candidates: never "
            f"retrieved {retrieval_split['never_retrieved_candidate_count']}; "
            "retrieved and unhelpful "
            f"{retrieval_split['retrieved_and_unhelpful_candidate_count']}; "
            f"retrieved and admitted "
            f"{retrieval_split['retrieved_admitted_candidate_count']}; "
            f"retrieval unknown "
            f"{retrieval_split['unknown_retrieval_candidate_count']}"
        )
        if retrieval_unavailable:
            headline += "; retrieval could not be determined"
        elif retrieval_split["availability"] == "partial":
            headline += (
                "; retrieval could not be determined for "
                f"{retrieval_split['unknown_retrieval_candidate_count']} complete "
                "candidate(s)"
            )
        headline += (
            f"; pass rate {pass_rate_numerator}/{pass_rate_denominator} "
            f"({rate_text}) among {denominator_label}"
        )
        admitted_outside_denominator = report.get(
            "admitted_outside_pass_rate_denominator_count",
            admitted - retrieval_split["retrieved_admitted_candidate_count"]
            if pass_rate_denominator_description == "retrieved_complete_candidates"
            else 0,
        )
        if admitted_outside_denominator:
            headline += (
                "; admitted outside the retrieval pass-rate denominator "
                f"{admitted_outside_denominator} "
                f"(never retrieved "
                f"{retrieval_split['never_retrieved_admitted_candidate_count']}, "
                "retrieval unknown "
                f"{retrieval_split['unknown_retrieval_admitted_candidate_count']})"
            )
        headline += (
            "; field "
            f"reference {field['admitted_candidate_count']}/"
            f"{field['tested_candidate_count']} ({field['pass_rate']:.1%}) under "
            "a different validation filter (not like-for-like)"
        )
    else:
        # Schema-version-1 inputs lack retrieval flags.  Preserve their headline
        # while newer reports expose the effective retrieval denominator above.
        headline = (
            f"tested {tested} complete candidate(s); admitted {admitted}; "
            f"rejected {report['rejected_candidate_count']}; incomplete "
            f"{incomplete}; pass rate {rate_text}; field reference "
            f"{field['admitted_candidate_count']}/{field['tested_candidate_count']} "
            f"({field['pass_rate']:.1%}) under a different validation filter "
            "(not like-for-like)"
        )
    lines = [
        headline,
        (
            "all-null chance expectation: at most "
            f"{multiple['all_null_expected_chance_admissions_upper_bound']:.3f} "
            f"admission(s) across {multiple['candidate_tests']} tests "
            f"({null_probability:.3%} "
            "per candidate upper bound)"
        ),
        (
            "this directional effect screen is not a multiplicity-corrected "
            "significance claim"
        ),
    ]
    null_channel = report.get("null_channel")
    if isinstance(null_channel, Mapping) and isinstance(
        null_channel.get("per_task"), list
    ):
        if null_channel.get("availability") == "unavailable":
            if null_channel.get("unavailability_reason") == "injection_flags_unknown":
                lines.append(
                    "null channel: unavailable (per-observation injection flags "
                    "were recorded, but all measured values are unknown)"
                )
            else:
                lines.append(
                    "null channel: unavailable (per-observation injection flags "
                    "were not recorded)"
                )
            lines.append(f"null channel rationale: {null_channel['rationale']}")
        else:
            lines.append(f"null channel: {null_channel['rationale']}")
        lines.append(
            "null channel totals: "
            f"{null_channel['measured_observation_count']} measured observation(s) "
            f"across {null_channel['task_count']} task(s); no pooled cross-task "
            f"contrast is reported: {null_channel['pooling']['reason']}"
        )
        lines.append(
            "null channel observation scope: "
            f"{null_channel['observation_scope']['note']}"
        )
        for task_group in null_channel["per_task"]:
            task_label = task_group["task_label"]
            if task_group["task_identity"] == "unknown":
                lines.append(f"  {task_label} ({task_group['identity_condition']}):")
            else:
                lines.append(f"  task {task_label}:")
            if task_group["identity_condition"] != (
                "fully_qualified_task_identity_established"
            ):
                lines.append(
                    f"    task identity condition: {task_group['identity_condition']}"
                )
            nonstandard_metadata = [
                condition
                for condition in task_group["task_metadata_conditions"]
                if condition
                not in {
                    TaskMetadataCondition.DIRECT.value,
                    TaskMetadataCondition.SUMMARY_RECORDED.value,
                }
            ]
            if nonstandard_metadata:
                lines.append(
                    f"    task metadata condition(s): {', '.join(nonstandard_metadata)}"
                )
            if task_group["availability"] == "unavailable":
                lines.append(
                    "    channels unavailable: per-observation injection flags "
                    "are not known on this task"
                )
            else:
                _append_channel_group_lines(lines, task_group, indent="    ")
            contrast = task_group["within_task_contrast"]
            if contrast["availability"] == "available":
                lines.append(
                    "    within-task difference (skill injected minus no skill "
                    "injected): "
                    f"{contrast['injected_minus_no_skill_mean_delta']:+.6g}"
                )
            else:
                lines.append(f"    {contrast['detail']}")
            if task_group["unknown_injection_observation_count"]:
                lines.append(
                    "    measured observations with unknown injection: "
                    f"{task_group['unknown_injection_observation_count']}"
                )
    elif (
        not isinstance(null_channel, Mapping)
        or null_channel.get("availability") == "unavailable"
    ):
        reason = (
            null_channel.get("unavailability_reason")
            if isinstance(null_channel, Mapping)
            else None
        )
        if reason == "injection_flags_unknown":
            lines.append(
                "null channel: unavailable (per-observation injection flags were "
                "recorded, but all measured values are unknown)"
            )
        else:
            lines.append(
                "null channel: unavailable (per-observation injection flags were "
                "not recorded)"
            )
    else:
        lines.append(f"null channel: {null_channel['rationale']}")
        _append_channel_group_lines(lines, null_channel, indent="  ")
        if null_channel["unknown_injection_observation_count"]:
            lines.append(
                "  measured observations with unknown injection: "
                f"{null_channel['unknown_injection_observation_count']}"
            )
    for decision in report["decisions"]:
        application = decision.get(
            "skill_application", {"status": "unavailable", "ever_injected": None}
        )
        effect = decision["measurement"]["effect"]
        application_status = application["status"]
        if application_status == "never_injected":
            if application["ever_injected"] is True:
                application_text = (
                    "skill was not injected in any measured observation; it was "
                    "injected only in "
                    f"{application['skill_injected_unpaired_treatment_count']} "
                    "unpaired treatment(s)"
                )
            else:
                application_text = (
                    "skill never retrieved/injected (reporting only; admission rule "
                    "unchanged)"
                )
        elif application_status == "mixed_injection":
            application_text = (
                "skill mixed injection: injected in "
                f"{application['skill_injected_measured_observation_count']} "
                f"of {application['measured_observation_count']} measured "
                "observations; "
            )
            application_text += (
                "reported mean mixes skill-injected effects with byte-identical "
                "no-skill noise"
                if effect is not None
                else "measured deltas mix skill-injected effects with byte-identical "
                "no-skill noise"
            )
            if decision["status"] == SkillAdmissionStatus.REJECTED.value:
                application_text += (
                    "; skill retrieved/injected but did not help enough for admission"
                )
        elif application_status == "always_injected":
            application_text = (
                "skill retrieved/injected but did not help enough for admission"
                if decision["status"] == SkillAdmissionStatus.REJECTED.value
                else "skill retrieved/injected"
            )
        elif application["ever_injected"] is False:
            application_text = (
                "skill never retrieved/injected (reporting only; admission rule "
                "unchanged)"
            )
        else:
            application_text = "skill retrieval/injection unavailable"
        if effect is None:
            refusal = decision["refusal"]
            lines.append(
                f"  {decision['candidate_id']} [{decision['arm']}]: incomplete "
                f"({refusal['reason']}): {refusal['detail']}; {application_text}"
            )
            continue
        line = (
            f"  {decision['candidate_id']} [{decision['arm']}]: "
            f"{decision['status']}; mean {effect['mean_delta']:+.6g}, "
            f"median {effect['median_delta']:+.6g}, range "
            f"{effect['minimum_delta']:+.6g}..{effect['maximum_delta']:+.6g}, "
            f"signs +{effect['positive_task_count']} "
            f"/ 0:{effect['zero_task_count']} / -{effect['negative_task_count']}; "
            f"{application_text}"
        )
        if decision["status"] == SkillAdmissionStatus.ADMITTED.value:
            context = decision["admission_context"]
            cohort = context["cohort"]
            candidate_null_probability = context[
                "single_candidate_null_admission_probability_upper_bound"
            ]
            line += (
                "; directional screen only, not individual certification; "
                "null admission upper bound "
                f"{candidate_null_probability:.3%} "
                "per candidate, cohort all-null expectation at most "
                f"{cohort['all_null_expected_chance_admissions_upper_bound']:.3f} "
                f"across {cohort['tested_candidate_count']} tests"
            )
        lines.append(line)
    return "\n".join(lines)


class SkillLibrary:
    """Persistent admitted skills plus auditable decisions for every submission."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.entries = self.root / "entries"
        self.entries.mkdir(parents=True, exist_ok=True)

    def submit(self, candidate: SkillAdmissionCandidate) -> dict[str, Any]:
        """Submit one candidate as a one-candidate cohort."""

        report = self.submit_cohort([candidate])
        return report["decisions"][0]

    def submit_cohort(
        self, candidates: Sequence[SkillAdmissionCandidate]
    ) -> dict[str, Any]:
        """Apply one shared cohort rule and persist those exact decisions."""

        report = assemble_admission_report(candidates)
        for candidate in candidates:
            final = self.entries / candidate.candidate_id
            temporary = self.entries / f".tmp-{candidate.candidate_id}"
            if final.exists():
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} already exists in the "
                    "library"
                )
            if temporary.exists():
                raise ValueError(
                    "temporary library entry already exists for "
                    f"{candidate.candidate_id!r}"
                )
        for candidate, decision in zip(candidates, report["decisions"], strict=True):
            self._record(candidate, decision)
        return report

    def _record(
        self, candidate: SkillAdmissionCandidate, decision: Mapping[str, Any]
    ) -> None:
        final = self.entries / candidate.candidate_id
        temporary = self.entries / f".tmp-{candidate.candidate_id}"
        temporary.mkdir()
        try:
            document = serialize_skill(candidate.skill)
            record = {
                **decision,
                "skill_sha256": hashlib.sha256(document.encode()).hexdigest(),
            }
            (temporary / "decision.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if decision["status"] == SkillAdmissionStatus.ADMITTED.value:
                (temporary / "skill.md").write_text(document + "\n", encoding="utf-8")
            temporary.rename(final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def read_skill(self, candidate_id: str) -> Skill:
        """Read an admitted skill through the canonical parser."""

        _validate_candidate_id(candidate_id)
        path = self.entries / candidate_id / "skill.md"
        try:
            document = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"candidate {candidate_id!r} has no admitted skill"
            ) from None
        return parse_skill(document)

    def candidate_ids(self) -> tuple[str, ...]:
        """List recorded candidates in deterministic identifier order."""

        candidate_ids = []
        for entry in self.entries.iterdir():
            if not entry.is_dir() or entry.name.startswith(".tmp-"):
                continue
            _validate_candidate_id(entry.name)
            candidate_ids.append(entry.name)
        return tuple(sorted(candidate_ids))

    def admitted_skill_ids(self) -> tuple[str, ...]:
        """List admitted candidates through their canonical decision records."""

        return tuple(
            candidate_id
            for candidate_id in self.candidate_ids()
            if self.read_decision(candidate_id).get("status")
            == SkillAdmissionStatus.ADMITTED.value
        )

    def read_decision(self, candidate_id: str) -> dict[str, Any]:
        """Read the recorded admission or refusal reason."""

        _validate_candidate_id(candidate_id)
        path = self.entries / candidate_id / "decision.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"candidate {candidate_id!r} has no library decision"
            ) from None
        if not isinstance(data, dict):
            raise ValueError(f"library decision is not an object: {path}")
        return data


def load_admission_candidates(path: Path | str) -> list[SkillAdmissionCandidate]:
    """Load a schema-version-1 cohort whose deltas were computed by the host."""

    source = Path(path).expanduser().resolve()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"skill admission input does not exist: {source}"
        ) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"skill admission input is invalid JSON: {source}") from error
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("skill admission input must be a schema-version-1 object")
    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("skill admission input candidates must be a non-empty list")

    candidates = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise ValueError(f"skill admission candidate {index} must be an object")
        candidate_id = raw.get("candidate_id")
        arm = raw.get("arm")
        document = raw.get("skill")
        deltas = raw.get("paired_deltas")
        raw_injection_flags = raw.get("injection_flags")
        raw_observation_summary = raw.get("validation_observation_summary")
        top_level_task_name = raw.get("task_name")
        if not isinstance(candidate_id, str) or not isinstance(arm, str):
            raise ValueError(f"skill admission candidate {index} needs text id and arm")
        if not isinstance(document, str):
            raise ValueError(
                f"skill admission candidate {candidate_id!r} needs skill text"
            )
        if not isinstance(deltas, list):
            raise ValueError(
                f"skill admission candidate {candidate_id!r} paired_deltas must "
                "be a list"
            )
        if raw_injection_flags is not None and not isinstance(
            raw_injection_flags, list
        ):
            raise ValueError(
                f"skill admission candidate {candidate_id!r} injection_flags "
                "must be a list or null"
            )
        if raw_observation_summary is not None and not isinstance(
            raw_observation_summary, dict
        ):
            raise ValueError(
                f"skill admission candidate {candidate_id!r} validation "
                "observation summary must be an object or null"
            )
        if top_level_task_name is not None and not isinstance(top_level_task_name, str):
            raise ValueError(
                f"skill admission candidate {candidate_id!r} top-level task_name "
                "must be text or null"
            )
        if raw_observation_summary is not None:
            for field_name in ("task_name", "source_task_name"):
                field_value = raw_observation_summary.get(field_name)
                if field_value is not None and not isinstance(field_value, str):
                    raise ValueError(
                        f"skill admission candidate {candidate_id!r} validation "
                        f"observation {field_name} must be text or null"
                    )
        task_name, source_task_name, task_metadata_condition = _load_task_metadata(
            raw_observation_summary,
            top_level_task_name,
        )
        candidates.append(
            SkillAdmissionCandidate(
                candidate_id=candidate_id,
                arm=arm,
                skill=parse_skill(document),
                paired_deltas=tuple(deltas),
                injection_flags=(
                    tuple(raw_injection_flags)
                    if raw_injection_flags is not None
                    else None
                ),
                task_name=task_name,
                source_task_name=source_task_name,
                task_metadata_condition=task_metadata_condition,
                top_level_task_name=top_level_task_name,
            )
        )
    return candidates


def write_admission_report(path: Path | str, report: Mapping[str, Any]) -> None:
    """Atomically write a machine-readable admission cohort report."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _load_task_metadata(
    observation_summary: Mapping[str, Any] | None,
    top_level_task_name: str | None,
) -> tuple[str | None, str | None, TaskMetadataCondition]:
    if observation_summary is None:
        if top_level_task_name is None:
            return None, None, TaskMetadataCondition.SUMMARY_ABSENT
        return (
            top_level_task_name,
            top_level_task_name,
            TaskMetadataCondition.TOP_LEVEL_FALLBACK_SUMMARY_ABSENT,
        )

    summary_task_name = observation_summary.get("task_name")
    summary_source_task_name = observation_summary.get("source_task_name")
    if summary_task_name is None:
        if top_level_task_name is None:
            return (
                None,
                summary_source_task_name,
                TaskMetadataCondition.SUMMARY_TASK_NULL,
            )
        return (
            top_level_task_name,
            summary_source_task_name,
            TaskMetadataCondition.TOP_LEVEL_FALLBACK_SUMMARY_TASK_NULL,
        )

    if summary_source_task_name is None:
        return (
            summary_task_name,
            None,
            TaskMetadataCondition.SUMMARY_SOURCE_ABSENT,
        )
    source_task_name = summary_source_task_name or top_level_task_name
    if top_level_task_name is not None and not (
        _task_names_compatible(
            _normalize_task_name(top_level_task_name),
            _normalize_task_name(summary_task_name),
        )
        or _task_names_compatible(
            _normalize_task_name(top_level_task_name),
            _normalize_task_name(source_task_name),
        )
        or _task_names_compatible(
            _normalize_task_name(source_task_name),
            _normalize_task_name(top_level_task_name),
        )
    ):
        condition = TaskMetadataCondition.SUMMARY_TOP_LEVEL_DISAGREEMENT
    else:
        condition = TaskMetadataCondition.SUMMARY_RECORDED
    return summary_task_name, source_task_name, condition


def _rule_report() -> dict[str, Any]:
    return {
        "rule_id": ADMISSION_RULE_ID,
        "paired": True,
        "expected_task_count": VALIDATION_TASK_COUNT,
        "minimum_positive_task_count": MIN_POSITIVE_TASKS,
        "requires_positive_mean_delta": True,
        "delta_absolute_tolerance": DELTA_ABS_TOLERANCE,
        "statistical_claim": (
            "Directional consistency plus observed effect size; not a "
            "per-candidate significance test. Ten paired tasks cannot support a "
            "useful family-wise-corrected claim across candidate search."
        ),
        "null_expectation_assumption": (
            "Independent, symmetric paired task-level null signs; ties can only "
            "lower the bound, while correlated task signs invalidate it."
        ),
    }


def _task_null_channel_summary(
    candidates: Sequence[SkillAdmissionCandidate],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[SkillAdmissionCandidate, Mapping[str, Any]]]] = {}
    for candidate, decision in zip(candidates, decisions, strict=True):
        grouped.setdefault(_task_identity_key(candidate), []).append(
            (candidate, decision)
        )

    per_task = []
    for identity_key in sorted(grouped):
        members = grouped[identity_key]
        observations = [
            (delta, flag)
            for candidate, _ in members
            for delta, flag in zip(
                candidate.paired_deltas,
                candidate.injection_flags
                if candidate.injection_flags is not None
                else (None,) * len(candidate.paired_deltas),
                strict=True,
            )
        ]
        flat = _flat_null_channel_summary(
            observations,
            injection_data_available=any(
                candidate.injection_flags is not None for candidate, _ in members
            ),
        )
        identity = _task_identity_report(
            [candidate for candidate, _ in members], identity_key
        )
        per_task.append(
            {
                **identity,
                "candidate_count": len(members),
                "candidate_ids": sorted(
                    candidate.candidate_id for candidate, _ in members
                ),
                "complete_candidate_count": sum(
                    decision["status"] != SkillAdmissionStatus.INCOMPLETE.value
                    for _, decision in members
                ),
                "incomplete_candidate_count": sum(
                    decision["status"] == SkillAdmissionStatus.INCOMPLETE.value
                    for _, decision in members
                ),
                "availability": flat["availability"],
                "unavailability_reason": flat.get("unavailability_reason"),
                "unknown_injection_observation_count": flat[
                    "unknown_injection_observation_count"
                ],
                "no_skill_injected": flat["no_skill_injected"],
                "skill_injected": flat["skill_injected"],
                "within_task_contrast": _contrast_for_task_identity(flat, identity),
            }
        )

    all_observations = [
        (delta, flag)
        for candidate in candidates
        for delta, flag in zip(
            candidate.paired_deltas,
            candidate.injection_flags
            if candidate.injection_flags is not None
            else (None,) * len(candidate.paired_deltas),
            strict=True,
        )
    ]
    measured_count = sum(delta is not None for delta, _ in all_observations)
    unknown_count = sum(
        delta is not None and flag is None for delta, flag in all_observations
    )
    known_count = measured_count - unknown_count
    injection_data_available = any(
        candidate.injection_flags is not None for candidate in candidates
    )
    if not injection_data_available:
        availability = "unavailable"
        unavailability_reason = "injection_flags_not_recorded"
    elif not known_count and unknown_count:
        availability = "unavailable"
        unavailability_reason = "injection_flags_unknown"
    else:
        availability = "partial" if unknown_count else "available"
        unavailability_reason = None
    incomplete_observation_count = sum(
        delta is not None
        for candidate, decision in zip(candidates, decisions, strict=True)
        if decision["status"] == SkillAdmissionStatus.INCOMPLETE.value
        for delta in candidate.paired_deltas
    )
    return {
        "schema_version": 2,
        "availability": availability,
        "unavailability_reason": unavailability_reason,
        "rationale": NULL_CHANNEL_RATIONALE,
        "measured_observation_count": measured_count,
        "unknown_injection_observation_count": unknown_count,
        "task_count": len(per_task),
        "observation_scope": {
            "includes_incomplete_candidates": True,
            "incomplete_candidate_measured_observation_count": (
                incomplete_observation_count
            ),
            "note": (
                "Channel statistics include every measured paired observation, "
                "including observations from incomplete candidates. Their counts "
                "therefore need not equal tested_candidate_count multiplied by "
                f"{VALIDATION_TASK_COUNT}."
            ),
        },
        "pooling": {
            "cross_task_contrast": "not_reported",
            "reason": NULL_CHANNEL_POOLING_INVALID_REASON,
        },
        "per_task": per_task,
    }


def _task_identity_key(candidate: SkillAdmissionCandidate) -> str:
    if candidate.source_task_name is not None:
        if candidate.task_metadata_condition in {
            TaskMetadataCondition.TOP_LEVEL_FALLBACK_SUMMARY_ABSENT,
            TaskMetadataCondition.TOP_LEVEL_FALLBACK_SUMMARY_TASK_NULL,
        }:
            return (
                f"fallback:{_normalize_task_name(candidate.source_task_name)}:"
                f"{candidate.candidate_id}"
            )
        return f"recorded:{_normalize_task_name(candidate.source_task_name)}"
    if candidate.task_name is not None:
        return (
            f"unqualified:{_normalize_task_name(candidate.task_name)}:"
            f"{candidate.candidate_id}"
        )
    return f"unknown:{candidate.candidate_id}"


def _task_identity_report(
    candidates: Sequence[SkillAdmissionCandidate], identity_key: str
) -> dict[str, Any]:
    source_names = _distinct_task_names(
        candidate.source_task_name for candidate in candidates
    )
    task_names = _distinct_task_names(candidate.task_name for candidate in candidates)
    top_level_names = _distinct_task_names(
        candidate.top_level_task_name for candidate in candidates
    )
    metadata_conditions = sorted(
        {candidate.task_metadata_condition.value for candidate in candidates}
    )
    if identity_key.startswith("unknown:"):
        return {
            "task_identity": "unknown",
            "task_label": UNKNOWN_TASK_LABEL,
            "task_name": None,
            "source_task_name": None,
            "task_names": [],
            "source_task_names": [],
            "top_level_task_names": top_level_names,
            "task_metadata_conditions": metadata_conditions,
            "identity_condition": "task_identity_unknown",
        }
    if identity_key.startswith("unqualified:"):
        return {
            "task_identity": "unqualified",
            "task_label": task_names[0],
            "task_name": task_names[0],
            "source_task_name": None,
            "task_names": task_names,
            "source_task_names": [],
            "top_level_task_names": top_level_names,
            "task_metadata_conditions": metadata_conditions,
            "identity_condition": "fully_qualified_task_identity_missing",
        }

    normalized_sources = {_normalize_task_name(name) for name in source_names}
    normalized_tasks = {_normalize_task_name(name) for name in task_names}
    conflicting_metadata = any(
        condition
        not in {
            TaskMetadataCondition.DIRECT.value,
            TaskMetadataCondition.SUMMARY_RECORDED.value,
        }
        for condition in metadata_conditions
    )
    if len(normalized_tasks) > 1:
        identity_condition = "task_name_disagreement_within_source_identity"
    elif (
        normalized_sources
        and normalized_tasks
        and not all(
            _task_names_compatible(source, task)
            for source in normalized_sources
            for task in normalized_tasks
        )
    ):
        identity_condition = "source_and_task_name_disagree"
    elif any(
        condition == TaskMetadataCondition.SUMMARY_TOP_LEVEL_DISAGREEMENT.value
        for condition in metadata_conditions
    ):
        identity_condition = "summary_and_top_level_task_name_disagree"
    elif conflicting_metadata:
        identity_condition = "task_identity_metadata_not_fully_verified"
    else:
        identity_condition = "fully_qualified_task_identity_established"
    task_label = source_names[0] if source_names else task_names[0]
    return {
        "task_identity": (
            "fully_qualified"
            if identity_condition == "fully_qualified_task_identity_established"
            else "conflicted"
        ),
        "task_label": task_label,
        "task_name": task_names[0] if len(task_names) == 1 else None,
        "source_task_name": source_names[0] if len(source_names) == 1 else None,
        "task_names": task_names,
        "source_task_names": source_names,
        "top_level_task_names": top_level_names,
        "task_metadata_conditions": metadata_conditions,
        "identity_condition": identity_condition,
    }


def _distinct_task_names(names: Iterable[str | None]) -> list[str]:
    distinct = {name.strip() for name in names if name is not None}
    return sorted(distinct, key=lambda name: (_normalize_task_name(name), name))


def _normalize_task_name(task_name: str) -> str:
    return " ".join(task_name.strip().split()).casefold()


def _task_names_compatible(source_name: str, task_name: str) -> bool:
    return source_name == task_name or source_name.rsplit("/", 1)[-1] == task_name


def build_null_channel_summary(
    observations: Sequence[tuple[float | None, bool | None]],
    *,
    injection_data_available: bool,
) -> dict[str, Any]:
    """Summarize measured deltas by injection without influencing admission."""

    return _flat_null_channel_summary(
        observations, injection_data_available=injection_data_available
    )


def _flat_null_channel_summary(
    observations: Sequence[tuple[float | None, bool | None]],
    *,
    injection_data_available: bool,
) -> dict[str, Any]:
    """Build one task's channel summaries, or the legacy ungrouped summary."""

    measured = [(delta, flag) for delta, flag in observations if delta is not None]
    unknown_count = sum(flag is None for _, flag in measured)
    known = [(float(delta), flag) for delta, flag in measured if flag is not None]
    if not injection_data_available or (not known and unknown_count):
        unavailability_reason = (
            "injection_flags_not_recorded"
            if not injection_data_available
            else "injection_flags_unknown"
        )
        return {
            "schema_version": 1,
            "availability": "unavailable",
            "unavailability_reason": unavailability_reason,
            "rationale": NULL_CHANNEL_RATIONALE,
            "unknown_injection_observation_count": unknown_count,
            "no_skill_injected": None,
            "skill_injected": None,
        }
    return {
        "schema_version": 1,
        "availability": "partial" if unknown_count else "available",
        "rationale": NULL_CHANNEL_RATIONALE,
        "unknown_injection_observation_count": unknown_count,
        "no_skill_injected": _delta_group_summary(
            [delta for delta, flag in known if flag is False]
        ),
        "skill_injected": _delta_group_summary(
            [delta for delta, flag in known if flag is True]
        ),
    }


def _contrast_for_task_identity(
    summary: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    if not (
        identity["task_identity"] == "fully_qualified"
        and identity["identity_condition"]
        == "fully_qualified_task_identity_established"
    ):
        return {
            "availability": "unavailable",
            "reason": "task_identity_not_established",
            "detail": (
                "no contrast available for this group because a shared, "
                "fully-qualified task identity was not established"
            ),
        }
    return _within_task_contrast(summary)


def _within_task_contrast(summary: Mapping[str, Any]) -> dict[str, Any]:
    no_skill = summary["no_skill_injected"]
    injected = summary["skill_injected"]
    if no_skill is None or injected is None:
        return {
            "availability": "unavailable",
            "reason": "injection_flags_unavailable",
            "detail": (
                "no contrast available on this task because injection flags are "
                "unavailable"
            ),
        }
    if no_skill["n"] and injected["n"]:
        return {
            "availability": "available",
            "injected_minus_no_skill_mean_delta": (
                injected["mean_delta"] - no_skill["mean_delta"]
            ),
        }
    if injected["n"]:
        return {
            "availability": "unavailable",
            "reason": "no_skill_injected_channel_missing",
            "detail": (
                "no contrast available on this task, only the skill-injected "
                "channel exists (the no-skill-injected channel is missing)"
            ),
        }
    if no_skill["n"]:
        return {
            "availability": "unavailable",
            "reason": "skill_injected_channel_missing",
            "detail": (
                "no contrast available on this task, only the no-skill-injected "
                "channel exists (the skill-injected channel is missing)"
            ),
        }
    return {
        "availability": "unavailable",
        "reason": "both_channels_empty",
        "detail": "no contrast available on this task: both channels are empty",
    }


def _delta_group_summary(deltas: Sequence[float]) -> dict[str, Any]:
    count = len(deltas)
    positive = sum(delta > DELTA_ABS_TOLERANCE for delta in deltas)
    negative = sum(delta < -DELTA_ABS_TOLERANCE for delta in deltas)
    return {
        "n": count,
        "mean_delta": math.fsum(deltas) / count if count else None,
        "sample_standard_deviation": stdev(deltas) if count >= 2 else None,
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": count - positive - negative,
    }


def _append_channel_group_lines(
    lines: list[str], summary: Mapping[str, Any], *, indent: str
) -> None:
    for group_name, label in (
        ("no_skill_injected", "no skill injected (noise floor)"),
        ("skill_injected", "skill injected"),
    ):
        group = summary[group_name]
        if group is None:
            continue
        mean = "null" if group["mean_delta"] is None else f"{group['mean_delta']:+.6g}"
        sample_sd = (
            "null"
            if group["sample_standard_deviation"] is None
            else f"{group['sample_standard_deviation']:.6g}"
        )
        lines.append(
            f"{indent}{label}: n={group['n']}, mean={mean}, sample sd={sample_sd}, "
            f"signs +{group['positive_count']} / 0:{group['zero_count']} / "
            f"-{group['negative_count']}"
        )


def _retrieval_split(
    candidates: Sequence[SkillAdmissionCandidate],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    retrieved = 0
    never_retrieved = 0
    unknown = 0
    retrieved_admitted = 0
    retrieved_unhelpful = 0
    never_retrieved_admitted = 0
    unknown_retrieval_admitted = 0
    for candidate, decision in zip(candidates, decisions, strict=True):
        if decision["status"] == SkillAdmissionStatus.INCOMPLETE.value:
            continue
        retrieval_status = _measured_retrieval_status(
            candidate.injection_flags, candidate.paired_deltas
        )
        if retrieval_status is CandidateRetrievalStatus.RETRIEVED:
            retrieved += 1
            if decision["status"] == SkillAdmissionStatus.ADMITTED.value:
                retrieved_admitted += 1
            else:
                retrieved_unhelpful += 1
        elif retrieval_status is CandidateRetrievalStatus.NEVER_RETRIEVED:
            never_retrieved += 1
            if decision["status"] == SkillAdmissionStatus.ADMITTED.value:
                never_retrieved_admitted += 1
        else:
            unknown += 1
            if decision["status"] == SkillAdmissionStatus.ADMITTED.value:
                unknown_retrieval_admitted += 1

    if unknown and not retrieved and not never_retrieved:
        availability = "unavailable"
    elif unknown:
        availability = "partial"
    else:
        availability = "available"
    return {
        "availability": availability,
        "complete_candidate_count": retrieved + never_retrieved + unknown,
        "retrieved_candidate_count": retrieved,
        "never_retrieved_candidate_count": never_retrieved,
        "unknown_retrieval_candidate_count": unknown,
        "retrieved_admitted_candidate_count": retrieved_admitted,
        "retrieved_and_unhelpful_candidate_count": retrieved_unhelpful,
        "never_retrieved_admitted_candidate_count": never_retrieved_admitted,
        "unknown_retrieval_admitted_candidate_count": unknown_retrieval_admitted,
    }


def _skill_application_report(
    injection_flags: tuple[bool | None, ...] | None,
    paired_deltas: tuple[float | None, ...],
) -> dict[str, Any]:
    if injection_flags is None:
        return {
            "status": "unavailable",
            "ever_injected": None,
            "injection_flags": None,
            "measured_observation_count": sum(
                delta is not None for delta in paired_deltas
            ),
            "skill_injected_measured_observation_count": None,
            "skill_injected_unpaired_treatment_count": None,
        }
    measured_flags = tuple(
        flag
        for delta, flag in zip(paired_deltas, injection_flags, strict=True)
        if delta is not None
    )
    all_observed = tuple(flag for flag in injection_flags if flag is not None)
    retrieval_status = _measured_retrieval_status(injection_flags, paired_deltas)
    if retrieval_status is CandidateRetrievalStatus.UNKNOWN:
        status = "unmeasured"
    elif retrieval_status is CandidateRetrievalStatus.RETRIEVED:
        status = (
            "mixed_injection"
            if any(flag is False for flag in measured_flags)
            else "always_injected"
        )
    else:
        status = "never_injected"
    ever_injected = True if any(all_observed) else False if all_observed else None
    return {
        "status": status,
        "ever_injected": ever_injected,
        "injection_flags": list(injection_flags),
        "measured_observation_count": len(measured_flags),
        "skill_injected_measured_observation_count": sum(
            flag is True for flag in measured_flags
        ),
        "skill_injected_unpaired_treatment_count": sum(
            delta is None and flag is True
            for delta, flag in zip(paired_deltas, injection_flags, strict=True)
        ),
    }


def _measured_retrieval_status(
    injection_flags: tuple[bool | None, ...] | None,
    paired_deltas: tuple[float | None, ...],
) -> CandidateRetrievalStatus:
    if injection_flags is None:
        return CandidateRetrievalStatus.UNKNOWN
    measured_flags = tuple(
        flag
        for delta, flag in zip(paired_deltas, injection_flags, strict=True)
        if delta is not None
    )
    if any(flag is True for flag in measured_flags):
        return CandidateRetrievalStatus.RETRIEVED
    if any(flag is False for flag in measured_flags):
        return CandidateRetrievalStatus.NEVER_RETRIEVED
    return CandidateRetrievalStatus.UNKNOWN


def _validate_candidate_id(candidate_id: str) -> None:
    if (
        not isinstance(candidate_id, str)
        or _SAFE_CANDIDATE_ID.fullmatch(candidate_id) is None
    ):
        raise ValueError(f"unsafe candidate id: {candidate_id!r}")


def _validate_optional_task_name(
    task_name: str | None, candidate_id: str, field_name: str
) -> None:
    if task_name is not None and (
        not isinstance(task_name, str) or not task_name.strip()
    ):
        raise ValueError(
            f"candidate {candidate_id!r} {field_name} must be non-empty text or null"
        )


def _optional_delta(value: object, candidate_id: str, index: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"candidate {candidate_id!r} paired delta {index} must be numeric or null"
        )
    delta = float(value)
    if not math.isfinite(delta):
        raise ValueError(
            f"candidate {candidate_id!r} paired delta {index} must be finite"
        )
    return delta


def _optional_injection_flag(
    value: object, candidate_id: str, index: int
) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(
        f"candidate {candidate_id!r} injection flag {index} must be boolean or null"
    )
