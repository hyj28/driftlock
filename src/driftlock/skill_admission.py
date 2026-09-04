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
from collections.abc import Mapping, Sequence
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
    "must be read against that noise floor, not against zero."
)

_SAFE_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class SkillAdmissionStatus(StrEnum):
    """Disposition of one candidate's validation measurement."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class SkillAdmissionCandidate:
    """One distilled skill and its already-paired per-task reward deltas."""

    candidate_id: str
    arm: str
    skill: Skill
    paired_deltas: tuple[float | None, ...]
    injection_flags: tuple[bool | None, ...] | None = None

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
    null_observations = [
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
    null_channel = build_null_channel_summary(
        null_observations,
        injection_data_available=any(
            candidate.injection_flags is not None for candidate in candidates
        ),
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
        "schema_version": 1,
        "mode": "skill-admission",
        "rule": _rule_report(),
        "submitted_candidate_count": len(decisions),
        "tested_candidate_count": tested,
        "incomplete_candidate_count": statuses[SkillAdmissionStatus.INCOMPLETE.value],
        "admitted_candidate_count": admitted,
        "rejected_candidate_count": statuses[SkillAdmissionStatus.REJECTED.value],
        "pass_rate": admitted / tested if tested else None,
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
    multiple = report["multiple_comparisons"]
    field = report["field_reference"]
    null_probability = multiple[
        "single_candidate_null_admission_probability_upper_bound"
    ]
    lines = [
        (
            f"tested {tested} complete candidate(s); admitted {admitted}; "
            f"rejected {report['rejected_candidate_count']}; incomplete "
            f"{incomplete}; pass rate {rate_text}; field reference "
            f"{field['admitted_candidate_count']}/{field['tested_candidate_count']} "
            f"({field['pass_rate']:.1%}) under a different validation filter "
            "(not like-for-like)"
        ),
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
    if (
        not isinstance(null_channel, Mapping)
        or null_channel.get("availability") == "unavailable"
    ):
        lines.append(
            "null channel: unavailable (per-observation injection flags were not "
            "recorded)"
        )
    else:
        lines.append(f"null channel: {null_channel['rationale']}")
        for group_name, label in (
            ("no_skill_injected", "no skill injected (noise floor)"),
            ("skill_injected", "skill injected"),
        ):
            group = null_channel[group_name]
            mean = (
                "unavailable"
                if group["mean_delta"] is None
                else f"{group['mean_delta']:+.6g}"
            )
            sample_sd = (
                "null"
                if group["sample_standard_deviation"] is None
                else f"{group['sample_standard_deviation']:.6g}"
            )
            lines.append(
                f"  {label}: n={group['n']}, mean={mean}, sample sd={sample_sd}, "
                f"signs +{group['positive_count']} / 0:{group['zero_count']} / "
                f"-{group['negative_count']}"
            )
        if null_channel["unknown_injection_observation_count"]:
            lines.append(
                "  measured observations with unknown injection: "
                f"{null_channel['unknown_injection_observation_count']}"
            )
    for decision in report["decisions"]:
        application = decision.get(
            "skill_application", {"status": "unavailable", "ever_injected": None}
        )
        if application["ever_injected"] is False:
            application_text = (
                "skill never retrieved/injected (reporting only; admission rule "
                "unchanged)"
            )
        elif application["ever_injected"] is True:
            if application["status"] == "mixed_injection":
                application_text = (
                    "skill mixed injection: injected in "
                    f"{application['skill_injected_measured_observation_count']} "
                    f"of {application['measured_observation_count']} measured "
                    "observations; reported mean mixes skill-injected effects "
                    "with byte-identical no-skill noise"
                )
                if decision["status"] == SkillAdmissionStatus.REJECTED.value:
                    application_text += (
                        "; skill retrieved/injected but did not help enough for "
                        "admission"
                    )
            else:
                application_text = (
                    "skill retrieved/injected but did not help enough for admission"
                    if decision["status"] == SkillAdmissionStatus.REJECTED.value
                    else "skill retrieved/injected"
                )
        else:
            application_text = "skill retrieval/injection unavailable"
        effect = decision["measurement"]["effect"]
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


def build_null_channel_summary(
    observations: Sequence[tuple[float | None, bool | None]],
    *,
    injection_data_available: bool,
) -> dict[str, Any]:
    """Summarize measured deltas by injection without influencing admission."""

    measured = [(delta, flag) for delta, flag in observations if delta is not None]
    unknown_count = sum(flag is None for _, flag in measured)
    known = [(float(delta), flag) for delta, flag in measured if flag is not None]
    if not injection_data_available or (not known and unknown_count):
        return {
            "schema_version": 1,
            "availability": "unavailable",
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
        }
    observed = tuple(flag for flag in injection_flags if flag is not None)
    if not observed:
        status = "unmeasured"
        ever_injected = None
    elif any(observed):
        status = "always_injected" if all(observed) else "mixed_injection"
        ever_injected = True
    else:
        status = "never_injected"
        ever_injected = False
    measured_flags = tuple(
        flag
        for delta, flag in zip(paired_deltas, injection_flags, strict=True)
        if delta is not None
    )
    return {
        "status": status,
        "ever_injected": ever_injected,
        "injection_flags": list(injection_flags),
        "measured_observation_count": len(measured_flags),
        "skill_injected_measured_observation_count": sum(
            flag is True for flag in measured_flags
        ),
    }


def _validate_candidate_id(candidate_id: str) -> None:
    if (
        not isinstance(candidate_id, str)
        or _SAFE_CANDIDATE_ID.fullmatch(candidate_id) is None
    ):
        raise ValueError(f"unsafe candidate id: {candidate_id!r}")


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
