from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import driftlock.lhtb_experiment as experiment
from driftlock.lhtb_experiment import main
from driftlock.skill_admission import (
    SkillAdmissionCandidate,
    assemble_admission_report,
    decide_skill_admission,
    load_admission_candidates,
    render_admission_report,
)
from driftlock.skill_distillation import Skill, serialize_skill
from driftlock.skill_validation import (
    DEFAULT_MAX_RETRIES,
    ValidationFailureKind,
    ValidationTrial,
    ValidationTrialResult,
    ValidationTrialStatus,
    plan_skill_validation,
    run_skill_validation,
)


def _skill(label: str) -> Skill:
    return Skill(
        activation=f"When {label} is active.",
        execution=f"Apply the {label} procedure.",
        termination="Stop after validation.",
    )


def _lhtb_tree(tmp_path: Path) -> Path:
    root = tmp_path / "LHTB"
    task_dir = root / "tasks" / "task-0"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        "[task]\nname = 'long-horizon-terminal-bench/task-0'\n",
        encoding="utf-8",
    )
    return root


def _candidate_file(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "skill-distillation",
                "candidates": [
                    {
                        "candidate_id": "candidate-0",
                        "arm": "localized",
                        "skill": serialize_skill(_skill("candidate-0")),
                        "paired_deltas": [],
                        "task_name": "long-horizon-terminal-bench/task-0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_harbor_attempt(
    job_dir: Path,
    *,
    reward: float | None,
    injected_candidate_ids: tuple[str, ...],
    exception_name: str | None = None,
) -> None:
    run_record = job_dir / "trial-0" / "agent" / "driftlock-result.json"
    run_record.parent.mkdir(parents=True)
    run_record.write_text(
        json.dumps(
            {
                "skill_layer": {
                    "distillation_arm": "localized",
                    "injection": {
                        "status": (
                            "injected" if injected_candidate_ids else "not_injected"
                        ),
                        "candidate_ids": list(injected_candidate_ids),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    evaluation: dict[str, object] = {
        "reward_stats": {"reward": {} if reward is None else {str(reward): ["trial-0"]}}
    }
    if exception_name is not None:
        evaluation["exception_stats"] = {exception_name: ["trial-0"]}
    (job_dir / "result.json").write_text(
        json.dumps({"stats": {"evals": {"evaluation": evaluation}}}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("exception_name", "expected_kind"),
    [
        ("RateLimitError", "transient_infrastructure"),
        (None, "no_reward"),
        ("RuntimeError", "no_reward"),
    ],
)
@pytest.mark.asyncio
async def test_no_reward_provider_attribution_reaches_the_validation_report(
    tmp_path: Path,
    exception_name: str | None,
    expected_kind: str,
) -> None:
    root = _lhtb_tree(tmp_path)
    plan = plan_skill_validation(_candidate_file(tmp_path), root)
    work_dir = tmp_path / "work"
    target = plan.work_items(work_dir)[0]
    for trial in plan.work_items(work_dir):
        _write_harbor_attempt(
            work_dir / "jobs" / trial.job_name,
            reward=None if trial == target else 0.5,
            injected_candidate_ids=trial.available_candidate_ids,
            exception_name=exception_name if trial == target else None,
        )
    runner = experiment._HarborSkillValidationRunner(
        lhtb_dir=root,
        work_dir=work_dir,
        skill_embedder_import_path="offline_embedder:embed",
        model="offline-model",
        provider="offline-provider",
        api_base="http://offline.invalid/v1",
        judge_api_base=None,
        judge_provider="offline-judge",
        timeout_sec=60,
        max_total_tokens=100,
    )

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=work_dir,
        max_retries=0,
    )

    attempt = next(
        item
        for item in report["validation"]["attempts"]
        if item["trial_id"] == target.trial_id
    )
    assert attempt["failure_kind"] == expected_kind
    assert attempt["audit"]["observed_exception_names"] == (
        [] if exception_name is None else [exception_name]
    )
    if exception_name is None:
        assert attempt["reason"] == "validation job produced no reward (job recovered)"
    else:
        assert exception_name in attempt["reason"]


class _RetryRunner:
    def __init__(
        self,
        *,
        transient_failures: int = 0,
        terminal_failure: bool = False,
        measured_reward: float = 0.5,
    ) -> None:
        self.transient_failures = transient_failures
        self.terminal_failure = terminal_failure
        self.measured_reward = measured_reward
        self.calls: list[ValidationTrial] = []

    async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
        self.calls.append(trial)
        target = trial.condition == "without_skill" and trial.replicate_index == 1
        if target and self.terminal_failure:
            return ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason="terminal no reward",
                failure_kind=ValidationFailureKind.NO_REWARD,
            )
        if target and trial.attempt_number <= self.transient_failures:
            return ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason="RateLimitError before reward",
                failure_kind=ValidationFailureKind.TRANSIENT_INFRASTRUCTURE,
                audit={"observed_exception_names": ["RateLimitError"]},
            )
        return ValidationTrialResult(
            status=ValidationTrialStatus.MEASURED,
            reward=self.measured_reward,
            injected_candidate_ids=trial.available_candidate_ids,
        )


@pytest.mark.asyncio
async def test_transient_failure_is_retried_and_rescued_with_recorded_backoff(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _lhtb_tree(tmp_path))
    runner = _RetryRunner(transient_failures=1)
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
        max_retries=2,
        retry_backoff_seconds=0.25,
        sleep=record_delay,
    )

    target_attempts = [
        attempt
        for attempt in report["validation"]["attempts"]
        if attempt["condition"] == "without_skill" and attempt["replicate_index"] == 1
    ]
    assert [attempt["attempt_number"] for attempt in target_attempts] == [1, 2]
    assert [attempt["status"] for attempt in target_attempts] == ["failed", "measured"]
    assert delays == [0.25]
    summary = report["validation"]["summary"]
    assert summary["retried_trial_count"] == 1
    assert summary["rescued_by_retry_trial_count"] == 1
    assert summary["failed_attempt_counts_by_failure_kind"] == {
        "transient_infrastructure": 1
    }
    assert summary["failed_attempt_counts_by_exception_name"] == {"RateLimitError": 1}


@pytest.mark.asyncio
async def test_transient_retry_budget_is_exact_and_backoff_grows(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _lhtb_tree(tmp_path))
    runner = _RetryRunner(transient_failures=99)
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=runner,
        work_dir=tmp_path / "work",
        max_retries=2,
        retry_backoff_seconds=0.25,
        sleep=record_delay,
    )

    target_attempts = [
        attempt
        for attempt in report["validation"]["attempts"]
        if attempt["condition"] == "without_skill" and attempt["replicate_index"] == 1
    ]
    assert DEFAULT_MAX_RETRIES == 2
    assert [attempt["attempt_number"] for attempt in target_attempts] == [1, 2, 3]
    assert delays == [0.25, 0.5]
    assert report["validation"]["summary"]["rescued_by_retry_trial_count"] == 0


@pytest.mark.asyncio
async def test_forced_resume_retries_terminal_failure_but_default_does_not(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _lhtb_tree(tmp_path))
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"
    first = _RetryRunner(terminal_failure=True)
    await run_skill_validation(plan, output, runner=first, work_dir=work_dir)

    conservative = _RetryRunner()
    conservative_report = await run_skill_validation(
        plan,
        output,
        runner=conservative,
        work_dir=work_dir,
    )

    assert conservative.calls == []
    assert (
        conservative_report["validation"]["summary"]["force_retry_unmeasured_requested"]
        is False
    )

    forced = _RetryRunner()
    forced_report = await run_skill_validation(
        plan,
        output,
        runner=forced,
        work_dir=work_dir,
        force_retry_unmeasured=True,
        sleep=lambda delay: _unexpected_sleep(delay),
    )

    assert len(forced.calls) == 1
    assert forced.calls[0].attempt_number == 2
    target_attempts = [
        attempt
        for attempt in forced_report["validation"]["attempts"]
        if attempt["condition"] == "without_skill" and attempt["replicate_index"] == 1
    ]
    assert [attempt["attempt_number"] for attempt in target_attempts] == [1, 2]
    assert target_attempts[1]["forced_reattempt"] is True
    assert target_attempts[1]["audit"]["forced_reattempt"] is True
    summary = forced_report["validation"]["summary"]
    assert summary["force_retry_unmeasured_requested"] is True
    assert summary["forced_reattempt_trial_count"] == 1


@pytest.mark.asyncio
async def test_forced_resume_retries_after_cumulative_budget_is_spent(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _lhtb_tree(tmp_path))
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"
    exhausted = _RetryRunner(transient_failures=99)
    await run_skill_validation(
        plan,
        output,
        runner=exhausted,
        work_dir=work_dir,
        max_retries=2,
        sleep=_record_no_delay,
    )

    forced = _RetryRunner()
    report = await run_skill_validation(
        plan,
        output,
        runner=forced,
        work_dir=work_dir,
        max_retries=2,
        force_retry_unmeasured=True,
        sleep=lambda delay: _unexpected_sleep(delay),
    )

    assert len(forced.calls) == 1
    assert forced.calls[0].attempt_number == 4
    target_attempts = [
        attempt
        for attempt in report["validation"]["attempts"]
        if attempt["condition"] == "without_skill" and attempt["replicate_index"] == 1
    ]
    assert len(target_attempts) == 4
    assert target_attempts[-1]["status"] == "measured"
    assert target_attempts[-1]["forced_reattempt"] is True


@pytest.mark.asyncio
async def test_terminal_failure_and_measured_reward_are_never_retried(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _lhtb_tree(tmp_path))
    terminal = _RetryRunner(terminal_failure=True)
    terminal_report = await run_skill_validation(
        plan,
        tmp_path / "terminal.json",
        runner=terminal,
        work_dir=tmp_path / "terminal-work",
        sleep=lambda delay: _unexpected_sleep(delay),
    )
    terminal_attempts = [
        attempt
        for attempt in terminal_report["validation"]["attempts"]
        if attempt["condition"] == "without_skill" and attempt["replicate_index"] == 1
    ]
    assert len(terminal_attempts) == 1

    measured = _RetryRunner(measured_reward=0.0)
    measured_report = await run_skill_validation(
        plan,
        tmp_path / "measured.json",
        runner=measured,
        work_dir=tmp_path / "measured-work",
        sleep=lambda delay: _unexpected_sleep(delay),
    )
    measured_attempts = [
        attempt
        for attempt in measured_report["validation"]["attempts"]
        if attempt["condition"] == "without_skill" and attempt["replicate_index"] == 1
    ]
    assert len(measured_attempts) == 1
    assert measured_attempts[0]["reward"] == 0.0


async def _unexpected_sleep(delay: float) -> None:
    raise AssertionError(f"unexpected retry delay: {delay}")


async def _record_no_delay(delay: float) -> None:
    del delay


class _NullChannelRunner:
    async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
        if trial.condition == "without_skill":
            return ValidationTrialResult(
                status=ValidationTrialStatus.MEASURED,
                reward=0.5,
                injected_candidate_ids=(),
            )
        deltas = {
            1: (0.1, ()),
            2: (0.2, ()),
            3: (0.3, ()),
            4: (-0.1, trial.available_candidate_ids),
            5: (0.0, trial.available_candidate_ids),
            6: (0.1, trial.available_candidate_ids),
        }
        observation = deltas.get(trial.replicate_index)
        if observation is None:
            return ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason="terminal synthetic gap",
                failure_kind=ValidationFailureKind.NO_REWARD,
            )
        delta, injected = observation
        return ValidationTrialResult(
            status=ValidationTrialStatus.MEASURED,
            reward=0.5 + delta,
            injected_candidate_ids=injected,
        )


@pytest.mark.asyncio
async def test_validation_summary_separates_null_and_injected_channels(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _lhtb_tree(tmp_path))

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=_NullChannelRunner(),
        work_dir=tmp_path / "work",
    )

    null_channel = report["validation"]["summary"]["null_channel"]
    assert null_channel["no_skill_injected"] == {
        "n": 3,
        "mean_delta": pytest.approx(0.2),
        "sample_standard_deviation": pytest.approx(0.1),
        "positive_count": 3,
        "negative_count": 0,
        "zero_count": 0,
    }
    assert null_channel["skill_injected"] == {
        "n": 3,
        "mean_delta": pytest.approx(0.0, abs=1e-15),
        "sample_standard_deviation": pytest.approx(0.1),
        "positive_count": 1,
        "negative_count": 1,
        "zero_count": 1,
    }
    assert report["candidates"][0]["injection_flags"] == [
        False,
        False,
        False,
        True,
        True,
        True,
        None,
        None,
        None,
        None,
    ]


@pytest.mark.asyncio
async def test_validation_group_with_one_observation_has_null_sample_deviation(
    tmp_path: Path,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _lhtb_tree(tmp_path))

    class SingleObservationRunner:
        async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
            if trial.condition == "without_skill":
                return ValidationTrialResult(
                    status=ValidationTrialStatus.MEASURED,
                    reward=0.4,
                    injected_candidate_ids=(),
                )
            if trial.replicate_index == 1:
                return ValidationTrialResult(
                    status=ValidationTrialStatus.MEASURED,
                    reward=0.6,
                    injected_candidate_ids=(),
                )
            return ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason="terminal synthetic gap",
                failure_kind=ValidationFailureKind.NO_REWARD,
            )

    report = await run_skill_validation(
        plan,
        tmp_path / "validated.json",
        runner=SingleObservationRunner(),
        work_dir=tmp_path / "work",
    )

    group = report["validation"]["summary"]["null_channel"]["no_skill_injected"]
    assert group["n"] == 1
    assert group["mean_delta"] == pytest.approx(0.2)
    assert group["sample_standard_deviation"] is None


def _admission_candidate(
    candidate_id: str,
    deltas: list[float | None],
    flags: list[bool | None] | None,
) -> SkillAdmissionCandidate:
    return SkillAdmissionCandidate(
        candidate_id=candidate_id,
        arm="localized",
        skill=_skill(candidate_id),
        paired_deltas=tuple(deltas),
        injection_flags=None if flags is None else tuple(flags),
    )


def test_one_observation_has_null_sample_deviation_and_flags_do_not_gate() -> None:
    admitted_deltas = [0.02] * 9 + [0.0]
    uninjected = _admission_candidate("uninjected", admitted_deltas, [False] * 10)
    injected = _admission_candidate("injected", admitted_deltas, [True] * 10)

    first = decide_skill_admission(uninjected)
    second = decide_skill_admission(injected)

    assert first["status"] == second["status"] == "admitted"
    assert first["measurement"] == second["measurement"]
    report = assemble_admission_report(
        [
            _admission_candidate(
                "single-null", [0.25] + [None] * 9, [False] + [None] * 9
            ),
            _admission_candidate(
                "single-injected", [-0.25] + [None] * 9, [True] + [None] * 9
            ),
        ]
    )
    assert (
        report["null_channel"]["no_skill_injected"]["sample_standard_deviation"] is None
    )
    assert report["null_channel"]["skill_injected"]["sample_standard_deviation"] is None


def test_loader_preserves_missing_injection_flags_as_unknown(tmp_path: Path) -> None:
    source = tmp_path / "old-validation.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "candidate_id": "old-format",
                        "arm": "baseline",
                        "skill": serialize_skill(_skill("old-format")),
                        "paired_deltas": [0.01] * 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_admission_candidates(source)
    report = assemble_admission_report(candidates)

    assert candidates[0].injection_flags is None
    assert report["null_channel"]["availability"] == "unavailable"
    assert report["null_channel"]["no_skill_injected"] is None
    assert "null channel: unavailable" in render_admission_report(report)


def test_render_distinguishes_mixed_always_and_never_injected_candidates() -> None:
    report = assemble_admission_report(
        [
            _admission_candidate("always", [0.0] * 10, [True] * 10),
            _admission_candidate("mixed", [0.0] * 10, [True, False] * 5),
            _admission_candidate("never", [0.0] * 10, [False] * 10),
        ]
    )

    lines = {
        candidate_id: next(
            line
            for line in render_admission_report(report).splitlines()
            if f"  {candidate_id} [" in line
        )
        for candidate_id in ("always", "mixed", "never")
    }
    assert "skill retrieved/injected but did not help enough" in lines["always"]
    assert (
        "skill mixed injection: injected in 5 of 10 measured observations"
        in lines["mixed"]
    )
    assert "reported mean mixes skill-injected effects" in lines["mixed"]
    assert "skill never retrieved/injected" in lines["never"]
    assert len(set(lines.values())) == 3


def _write_cli_validation(
    path: Path,
    candidates: list[tuple[str, list[float | None], list[bool] | None]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "arm": "localized",
                        "skill": serialize_skill(_skill(candidate_id)),
                        "paired_deltas": deltas,
                        **({"injection_flags": flags} if flags is not None else {}),
                    }
                    for candidate_id, deltas, flags in candidates
                ],
            }
        ),
        encoding="utf-8",
    )


def _run_admit_cli(
    source: Path, library: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("driftlock-lhtb")
    return subprocess.run(
        [
            str(executable),
            "admit-skills",
            str(source),
            "--library-dir",
            str(library),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_validate_skills_cli_forces_legacy_unmeasured_trial_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _lhtb_tree(tmp_path)
    source = _candidate_file(tmp_path)
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"
    current_runner: dict[str, _RetryRunner] = {
        "value": _RetryRunner(terminal_failure=True)
    }
    monkeypatch.setattr(experiment, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        experiment,
        "_HarborSkillValidationRunner",
        lambda **kwargs: current_runner["value"],
    )
    arguments = [
        "validate-skills",
        str(source),
        "--lhtb-dir",
        str(root),
        "--output",
        str(output),
        "--work-dir",
        str(work_dir),
        "--skill-embedder",
        "offline_embedder:embed",
        "--no-provider-probe",
    ]

    assert main(arguments) == 1
    capsys.readouterr()
    legacy_report = json.loads(output.read_text(encoding="utf-8"))
    del legacy_report["validation"]["run"]["max_retries"]
    output.write_text(json.dumps(legacy_report), encoding="utf-8")
    current_runner["value"] = _RetryRunner()

    assert main([*arguments, "--force-retry-unmeasured"]) == 0

    assert len(current_runner["value"].calls) == 1
    assert current_runner["value"].calls[0].attempt_number == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    forced_attempt = next(
        attempt
        for attempt in report["validation"]["attempts"]
        if attempt.get("forced_reattempt") is True
    )
    assert forced_attempt["forced_reattempt"] is True
    assert report["validation"]["summary"]["force_retry_unmeasured_requested"] is True
    assert (
        "operator-forced re-attempt enabled for every unmeasured trial"
        in capsys.readouterr().out
    )


def test_validate_skills_cli_dry_run_executes_no_trial(tmp_path: Path) -> None:
    source = _candidate_file(tmp_path)
    output = tmp_path / "validated.json"
    work_dir = tmp_path / "work"
    executable = Path(sys.executable).with_name("driftlock-lhtb")

    result = subprocess.run(
        [
            str(executable),
            "validate-skills",
            str(source),
            "--lhtb-dir",
            str(_lhtb_tree(tmp_path)),
            "--output",
            str(output),
            "--work-dir",
            str(work_dir),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "20 trial(s) planned" in result.stdout
    assert "dry run; no trials run and no files written" in result.stdout
    assert not output.exists()
    assert not work_dir.exists()


@pytest.mark.parametrize("missing_null_deltas", [False, True])
def test_admit_skills_cli_reports_mixed_cohort_null_channel(
    tmp_path: Path,
    missing_null_deltas: bool,
) -> None:
    source = tmp_path / "validation.json"
    b_deltas: list[float | None] = [0.0] * 10
    if missing_null_deltas:
        b_deltas[-4:] = [None] * 4
    _write_cli_validation(
        source,
        [
            ("candidate-a", [0.02] * 10, [True] * 10),
            ("candidate-b", b_deltas, [False] * 10),
            ("candidate-c", [0.02, -0.02] * 5, [True, False] * 5),
        ],
    )
    output = tmp_path / "admission.json"

    result = _run_admit_cli(source, tmp_path / "library", output)

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    expected_null_n = 11 if missing_null_deltas else 15
    assert report["null_channel"]["no_skill_injected"]["n"] == expected_null_n
    assert report["null_channel"]["skill_injected"]["n"] == 15
    decision_b = next(
        decision
        for decision in report["decisions"]
        if decision["candidate_id"] == "candidate-b"
    )
    assert decision_b["status"] == ("incomplete" if missing_null_deltas else "rejected")
    rendered = result.stdout
    assert "no skill injected (noise floor): n=" in rendered
    assert "sample sd=" in rendered
    assert "candidate-b" in rendered and "skill never retrieved/injected" in rendered
    assert "candidate-c" in rendered
    assert "retrieved/injected but did not help enough" in rendered


def test_admit_skills_cli_keeps_uninjected_admission_and_old_format_unknown(
    tmp_path: Path,
) -> None:
    admitted_source = tmp_path / "uninjected.json"
    _write_cli_validation(
        admitted_source,
        [("admitted-uninjected", [0.02] * 9 + [0.0], [False] * 10)],
    )

    admitted_result = _run_admit_cli(
        admitted_source,
        tmp_path / "admitted-library",
        tmp_path / "admitted-report.json",
    )

    assert admitted_result.returncode == 0, admitted_result.stderr
    admitted_output = admitted_result.stdout
    admitted_report = json.loads(
        (tmp_path / "admitted-report.json").read_text(encoding="utf-8")
    )
    assert admitted_report["decisions"][0]["status"] == "admitted"
    assert (
        admitted_report["decisions"][0]["skill_application"]["ever_injected"] is False
    )
    assert "skill never retrieved/injected" in admitted_output
    assert "admission rule unchanged" in admitted_output

    old_source = tmp_path / "old.json"
    _write_cli_validation(
        old_source,
        [("old-candidate", [0.02] * 9 + [0.0], None)],
    )
    old_result = _run_admit_cli(
        old_source,
        tmp_path / "old-library",
        tmp_path / "old-report.json",
    )

    assert old_result.returncode == 0, old_result.stderr
    old_output = old_result.stdout
    assert "null channel: unavailable" in old_output
    assert "skill retrieval/injection unavailable" in old_output


def test_validation_failure_kind_values_do_not_alias() -> None:
    values = [member.value for member in ValidationFailureKind.__members__.values()]

    assert len(values) == len(set(values))


def test_transient_provider_exception_allowlist_is_exact() -> None:
    assert {
        "RateLimitError",
        "APIError",
        "APIConnectionError",
        "Timeout",
        "ServiceUnavailableError",
        "InternalServerError",
    } == experiment.TRANSIENT_VALIDATION_EXCEPTION_NAMES
