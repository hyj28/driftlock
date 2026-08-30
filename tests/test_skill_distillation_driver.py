from __future__ import annotations

import hashlib
import io
import json
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import driftlock.lhtb_experiment as experiment
from driftlock.lhtb_experiment import main
from driftlock.models import JudgeCompletion
from driftlock.skill_admission import load_admission_candidates
from driftlock.skill_distillation import CallableSkillDistiller
from driftlock.skill_distillation_driver import (
    plan_skill_distillation,
    run_skill_distillation,
)
from driftlock.usage import ReplayUsage

_SKILL = (
    "## activation\n\nWhen checkpoint evidence shows repeated non-improvement.\n\n"
    "## execution\n\nDo not repeat the stalled action; inspect the bounded diff "
    "and choose a verified alternative instead.\n\n"
    "## termination\n\nStop when the alternative produces measured progress."
)


def _archive(files: dict[str, str]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, text in files.items():
            content = text.encode()
            member = tarfile.TarInfo(f"./{name}")
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


def _checkpoint(trial: Path, *, step: int, checkpoint_id: str, source: str) -> None:
    directory = (
        trial / ".driftlock-checkpoints" / "phase-0" / "checkpoints" / checkpoint_id
    )
    directory.mkdir(parents=True)
    archive = _archive({"src/example.py": source})
    state = json.dumps({"phase": 0, "step": step}, separators=(",", ":"))
    digest = hashlib.sha256(archive)
    digest.update(b"\0state\0")
    digest.update(state.encode())
    (directory / "workspace.tar.gz").write_bytes(archive)
    (directory / "state.json").write_text(state, encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "step": step,
                "created_at": "2026-08-28T12:00:00+00:00",
                "digest": digest.hexdigest(),
                "parent_id": None,
                "label": f"step-{step}",
                "remote_workspace": "/app",
            }
        ),
        encoding="utf-8",
    )


def _replace_checkpoint_with_checkout(trial: Path, checkpoint_id: str) -> None:
    directory = (
        trial / ".driftlock-checkpoints" / "phase-0" / "checkpoints" / checkpoint_id
    )
    files = {f"vendor/{index:04d}-{'x' * 340}.py": "x\n" for index in range(2_500)}
    archive = _archive(files)
    state = (directory / "state.json").read_bytes()
    digest = hashlib.sha256(archive)
    digest.update(b"\0state\0")
    digest.update(state)
    (directory / "workspace.tar.gz").write_bytes(archive)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["digest"] = digest.hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _trial(source_job: Path) -> Path:
    trial = source_job / "synthetic__trial"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    steps = [{"step_id": 1, "source": "user", "message": "repair the parser"}]
    steps.extend(
        {
            "step_id": index + 1,
            "source": "agent",
            "message": f"agent-{index}",
            "observation": {"results": [{"content": f"terminal-{index}"}]},
        }
        for index in range(1, 7)
    )
    (agent / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.0", "steps": steps}),
        encoding="utf-8",
    )
    (agent / "driftlock-result.json").write_text(
        json.dumps({"phases": [{"phase": 0, "steps": 6, "status": "completed"}]}),
        encoding="utf-8",
    )
    _checkpoint(trial, step=2, checkpoint_id="a" * 32, source="mode = 'old'\n")
    _checkpoint(trial, step=4, checkpoint_id="b" * 32, source="mode = 'new'\n")
    return trial


def _segment() -> dict[str, object]:
    return {
        "type": "regression",
        "steps": [2, 4],
        "start": {"phase": 0, "step": 2, "checkpoint_id": "a" * 32},
        "end": {"phase": 0, "step": 4, "checkpoint_id": "b" * 32},
        "start_reward": 0.8,
        "end_reward": 0.6,
        "reward_change": -0.2,
    }


def _report(
    tmp_path: Path,
    *,
    usable_tasks: tuple[str, ...] = ("task-a",),
    refused_tasks: tuple[str, ...] = (),
    segment_count: int = 1,
) -> Path:
    source_job = tmp_path / "source-job"
    source_job.mkdir()
    _trial(source_job)
    tasks = [
        {
            "task_name": task_name,
            "trial_name": "synthetic__trial",
            "status": "usable",
            "segments": [_segment() for _ in range(segment_count)],
        }
        for task_name in usable_tasks
    ]
    tasks.extend(
        {
            "task_name": task_name,
            "trial_name": f"{task_name}__refused",
            "status": "refused",
            "segments": [],
            "refusal": {
                "reason": "uniform_checkpoint_scores",
                "detail": f"{task_name} has no localizable score variation",
            },
        }
        for task_name in refused_tasks
    )
    path = tmp_path / "localization.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "checkpoint-localization",
                "source_job_dir": str(source_job),
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )
    return path


class _Meter:
    def __init__(self) -> None:
        self.usage = ReplayUsage(0, 0, 0, 0.0)

    def charge(
        self,
        *,
        input_tokens: int = 1,
        cache_tokens: int = 0,
        output_tokens: int = 1,
        cost_usd: float = 0.001,
    ) -> None:
        self.usage = ReplayUsage(
            self.usage.input_tokens + input_tokens,
            self.usage.cache_tokens + cache_tokens,
            self.usage.output_tokens + output_tokens,
            self.usage.cost_usd + cost_usd,
        )

    def read(self) -> ReplayUsage:
        return self.usage


def _run_console(
    report_path: Path, output: Path, *, dry_run: bool
) -> subprocess.CompletedProcess[str]:
    command = [
        str(Path(sys.executable).with_name("driftlock-lhtb")),
        "distill-skills",
        str(report_path),
        "--output",
        str(output),
    ]
    if dry_run:
        command.append("--dry-run")
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _run_console_with_outcomes(
    report_path: Path,
    output: Path,
    outcomes: tuple[str, str],
) -> subprocess.CompletedProcess[str]:
    harness = Path(__file__).with_name("distillation_cli_harness.py")
    return subprocess.run(
        [
            sys.executable,
            str(harness),
            ",".join(outcomes),
            "distill-skills",
            str(report_path),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


async def test_dry_run_reports_exact_dual_arm_breakdown_without_calling_model(
    tmp_path: Path,
) -> None:
    report_path = _report(
        tmp_path,
        usable_tasks=("task-a", "task-b", "task-c"),
        refused_tasks=("task-d", "task-e"),
    )
    plan = plan_skill_distillation(report_path)

    class ExplodingDistiller:
        async def distill(self, evidence: object) -> object:
            del evidence
            raise AssertionError("dry run made a model call")

    output = tmp_path / "candidates.json"
    result = await run_skill_distillation(
        plan,
        output,
        distiller=ExplodingDistiller(),  # type: ignore[arg-type]
        usage_reader=lambda: ReplayUsage(0, 0, 0, 0.0),
        dry_run=True,
    )

    assert result["summary"] == {
        "planned_call_count": 6,
        "completed_call_count": 0,
        "pending_call_count": 6,
        "reused_call_count": 0,
        "new_call_count": 0,
        "candidate_count": 0,
        "paired_candidate_segment_count": 0,
        "unpaired_segment_count": 0,
        "retryable_failure_count": 0,
        "refused_task_count": 2,
        "evidence_refusal_count": 0,
        "status_counts": {},
    }
    assert [
        (item["task_name"], item["segment_index"], item["arm"])
        for item in result["plan"]["work_items"]
    ] == [
        ("task-a", 0, "localized"),
        ("task-a", 0, "baseline"),
        ("task-b", 0, "localized"),
        ("task-b", 0, "baseline"),
        ("task-c", 0, "localized"),
        ("task-c", 0, "baseline"),
    ]
    assert [item["candidate_id"] for item in result["plan"]["work_items"]] == [
        "skill-1dbdf610e01616c0d537671d",
        "skill-fe8dc871fb192443a9773be1",
        "skill-4717f6d8a9d5a9ff8a9168a2",
        "skill-259286022d6487c8249c4bf1",
        "skill-dd348ea3fb85d228b17aa4b9",
        "skill-2ce79fec4f240e8767dc3a1f",
    ]
    assert [item["task_name"] for item in result["refused_tasks"]] == [
        "task-d",
        "task-e",
    ]
    assert not output.exists()


async def test_real_run_records_each_arms_exact_usage_and_admission_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = _report(tmp_path)
    plan = plan_skill_distillation(report_path)
    meter = _Meter()
    prompts: list[str] = []

    async def complete(prompt: str) -> JudgeCompletion:
        prompts.append(prompt)
        if "workspace_diff" in prompt:
            meter.charge(
                input_tokens=11,
                cache_tokens=2,
                output_tokens=3,
                cost_usd=0.01,
            )
            return JudgeCompletion(_SKILL, tokens=14)
        meter.charge(
            input_tokens=13,
            cache_tokens=1,
            output_tokens=5,
            cost_usd=0.02,
        )
        return JudgeCompletion(_SKILL, tokens=18)

    def socket_forbidden(*args: object, **kwargs: object) -> socket.socket:
        del args, kwargs
        raise AssertionError("the driver opened a socket")

    monkeypatch.setattr(socket, "socket", socket_forbidden)
    output = tmp_path / "candidates.json"
    result = await run_skill_distillation(
        plan,
        output,
        distiller=CallableSkillDistiller(complete),
        usage_reader=meter.read,
    )

    assert len(prompts) == 2
    assert {
        arm: {key: value for key, value in usage.items() if key != "cost_usd"}
        for arm, usage in result["usage_by_arm"].items()
    } == {
        "localized": {
            "input_tokens": 11,
            "cache_tokens": 2,
            "output_tokens": 3,
        },
        "baseline": {
            "input_tokens": 13,
            "cache_tokens": 1,
            "output_tokens": 5,
        },
    }
    assert result["usage_by_arm"]["localized"]["cost_usd"] == pytest.approx(0.01)
    assert result["usage_by_arm"]["baseline"]["cost_usd"] == pytest.approx(0.02)
    assert [candidate["arm"] for candidate in result["candidates"]] == [
        "localized",
        "baseline",
    ]
    assert [candidate["paired_deltas"] for candidate in result["candidates"]] == [
        [],
        [],
    ]
    loaded = load_admission_candidates(output)
    assert [candidate.arm for candidate in loaded] == ["localized", "baseline"]
    assert [candidate.paired_deltas for candidate in loaded] == [(), ()]


async def test_attempt_records_per_call_accounting_provenance(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    plan = plan_skill_distillation(report_path)
    meter = _Meter()
    accounting = {
        "physical_request_count": 0,
        "provider_reported_token_count": 0,
        "provider_reported_cost_count": 0,
        "reported_tokens_priced_count": 0,
        "successful_response_fallback_count": 0,
        "missing_reported_usage_count": 0,
        "invalid_reported_usage_count": 0,
        "error_without_usage_count": 0,
        "conservative_usage_fallbacks": 0,
    }

    async def complete(_prompt: str) -> JudgeCompletion:
        meter.charge(input_tokens=120, cache_tokens=20, output_tokens=10)
        accounting["physical_request_count"] += 1
        accounting["provider_reported_token_count"] += 1
        accounting["reported_tokens_priced_count"] += 1
        accounting["conservative_usage_fallbacks"] += 1
        return JudgeCompletion(_SKILL, tokens=130)

    def usage_reader() -> ReplayUsage:
        return meter.read()

    usage_reader.driftlock_accounting_reader = lambda: dict(accounting)  # type: ignore[attr-defined]

    result = await run_skill_distillation(
        plan,
        tmp_path / "candidates.json",
        distiller=CallableSkillDistiller(complete),
        usage_reader=usage_reader,
    )

    attempt = result["attempts"][0]
    assert attempt["usage_accounting"] == {
        "status": "recorded",
        "physical_request_count": 1,
        "provider_reported_token_count": 1,
        "provider_reported_cost_count": 0,
        "reported_tokens_priced_count": 1,
        "successful_response_fallback_count": 0,
        "missing_reported_usage_count": 0,
        "invalid_reported_usage_count": 0,
        "error_without_usage_count": 0,
        "conservative_usage_fallbacks": 1,
    }
    assert attempt["token_reconciliation"] == {
        "result_tokens": 130,
        "usage_input_plus_output_tokens": 130,
        "matches": True,
    }
    assert attempt["prompt_characters"] == 2_459
    assert attempt["prompt_utf8_bytes"] == 2_465


async def test_distillation_keeps_declined_malformed_and_failed_distinct(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path, segment_count=2)
    plan = plan_skill_distillation(report_path)
    meter = _Meter()
    responses = iter(
        [
            "DECLINE: no repeatable prevention is supported",
            "garbage",
            RuntimeError("provider unavailable"),
            _SKILL,
        ]
    )

    async def complete(_prompt: str) -> str:
        meter.charge()
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    result = await run_skill_distillation(
        plan,
        tmp_path / "candidates.json",
        distiller=CallableSkillDistiller(complete),
        usage_reader=meter.read,
    )

    assert [attempt["status"] for attempt in result["attempts"]] == [
        "declined",
        "malformed",
        "failed",
        "generated",
    ]
    assert result["summary"]["status_counts"] == {
        "declined": 1,
        "failed": 1,
        "generated": 1,
        "malformed": 1,
    }
    assert result["attempts"][0]["reason"] == ("no repeatable prevention is supported")
    assert "malformed response" in result["attempts"][1]["reason"]
    assert "provider unavailable" in result["attempts"][2]["reason"]
    assert result["summary"]["candidate_count"] == 0
    assert result["summary"]["paired_candidate_segment_count"] == 0
    assert result["summary"]["unpaired_segment_count"] == 1
    assert result["summary"]["retryable_failure_count"] == 1


def test_one_sided_call_failure_writes_no_candidate_and_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _report(tmp_path)
    output = tmp_path / "candidates.json"
    meter = _Meter()
    calls: list[str] = []

    async def complete(prompt: str) -> JudgeCompletion:
        calls.append(prompt)
        meter.charge(input_tokens=3, output_tokens=2, cost_usd=0.005)
        if len(calls) == 1:
            raise RuntimeError("transient 502")
        return JudgeCompletion(_SKILL, tokens=5)

    def build_fake(**kwargs: object) -> tuple[object, object, dict[str, object]]:
        return (
            CallableSkillDistiller(complete),
            meter.read,
            {
                "model": kwargs["model"],
                "provider": kwargs["provider"],
                "api_base": kwargs["api_base"],
                "max_output_tokens": kwargs["max_output_tokens"],
                "timeout_sec": kwargs["timeout_sec"],
            },
        )

    monkeypatch.setattr(experiment, "_build_skill_distiller", build_fake)

    assert main(["distill-skills", str(report_path), "--output", str(output)]) == 1

    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == 2
    assert [attempt["status"] for attempt in result["attempts"]] == [
        "failed",
        "generated",
    ]
    assert [attempt["attempt_number"] for attempt in result["attempts"]] == [1, 1]
    assert result["candidates"] == []
    assert result["summary"]["candidate_count"] == 0
    assert result["summary"]["pending_call_count"] == 1
    assert result["summary"]["retryable_failure_count"] == 1
    assert result["summary"]["unpaired_segment_count"] == 1
    printed = capsys.readouterr().out
    assert "task-a segment 0 localized: failed" in printed
    assert "task-a segment 0 baseline: generated" in printed
    assert "1 retryable failure(s); 1 unpaired segment(s)" in printed


def test_console_exit_is_nonzero_for_unpaired_outcome_without_failure(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    output = tmp_path / "candidates.json"

    process = _run_console_with_outcomes(
        report_path,
        output,
        ("declined", "generated"),
    )

    assert process.returncode == 1
    assert "0 retryable failure(s); 1 unpaired segment(s)" in process.stdout
    result = json.loads(output.read_text(encoding="utf-8"))
    assert [attempt["status"] for attempt in result["attempts"]] == [
        "declined",
        "generated",
    ]
    assert result["summary"]["retryable_failure_count"] == 0
    assert result["summary"]["unpaired_segment_count"] == 1
    assert result["candidates"] == []


def test_console_exit_is_nonzero_for_retryable_failure_without_unpaired_outcome(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    output = tmp_path / "candidates.json"

    process = _run_console_with_outcomes(
        report_path,
        output,
        ("failed", "declined"),
    )

    assert process.returncode == 1
    assert "1 retryable failure(s); 0 unpaired segment(s)" in process.stdout
    result = json.loads(output.read_text(encoding="utf-8"))
    assert [attempt["status"] for attempt in result["attempts"]] == [
        "failed",
        "declined",
    ]
    assert result["summary"]["retryable_failure_count"] == 1
    assert result["summary"]["unpaired_segment_count"] == 0
    assert result["candidates"] == []


def test_rerun_retries_persisted_failure_and_restores_paired_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _report(tmp_path)
    output = tmp_path / "candidates.json"
    first_meter = _Meter()
    first_calls = 0

    async def first_complete(_prompt: str) -> JudgeCompletion:
        nonlocal first_calls
        first_calls += 1
        first_meter.charge(input_tokens=3, output_tokens=2, cost_usd=0.005)
        if first_calls == 1:
            raise RuntimeError("transient 502")
        return JudgeCompletion(_SKILL, tokens=5)

    def first_factory(**kwargs: object) -> tuple[object, object, dict[str, object]]:
        return (
            CallableSkillDistiller(first_complete),
            first_meter.read,
            {
                "model": kwargs["model"],
                "provider": kwargs["provider"],
                "api_base": kwargs["api_base"],
                "max_output_tokens": kwargs["max_output_tokens"],
                "timeout_sec": kwargs["timeout_sec"],
            },
        )

    monkeypatch.setattr(experiment, "_build_skill_distiller", first_factory)
    assert main(["distill-skills", str(report_path), "--output", str(output)]) == 1
    assert first_calls == 2
    capsys.readouterr()

    retry_meter = _Meter()
    retry_calls: list[str] = []

    async def retry_complete(prompt: str) -> JudgeCompletion:
        retry_calls.append(prompt)
        retry_meter.charge(input_tokens=7, output_tokens=4, cost_usd=0.009)
        return JudgeCompletion(_SKILL, tokens=11)

    def retry_factory(**kwargs: object) -> tuple[object, object, dict[str, object]]:
        return (
            CallableSkillDistiller(retry_complete),
            retry_meter.read,
            {
                "model": kwargs["model"],
                "provider": kwargs["provider"],
                "api_base": kwargs["api_base"],
                "max_output_tokens": kwargs["max_output_tokens"],
                "timeout_sec": kwargs["timeout_sec"],
            },
        )

    monkeypatch.setattr(experiment, "_build_skill_distiller", retry_factory)

    assert main(["distill-skills", str(report_path), "--output", str(output)]) == 0

    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(retry_calls) == 1
    assert [attempt["status"] for attempt in result["attempts"]] == [
        "failed",
        "generated",
        "generated",
    ]
    assert [attempt["attempt_number"] for attempt in result["attempts"]] == [1, 2, 1]
    assert [attempt["reused"] for attempt in result["attempts"]] == [
        False,
        False,
        True,
    ]
    assert [candidate["arm"] for candidate in result["candidates"]] == [
        "localized",
        "baseline",
    ]
    assert result["summary"]["candidate_count"] == 2
    assert result["summary"]["paired_candidate_segment_count"] == 1
    assert result["summary"]["pending_call_count"] == 0
    assert result["summary"]["retryable_failure_count"] == 0
    assert result["summary"]["unpaired_segment_count"] == 0
    assert result["summary"]["new_call_count"] == 1
    assert result["summary"]["reused_call_count"] == 1
    assert {
        arm: {key: value for key, value in usage.items() if key != "cost_usd"}
        for arm, usage in result["usage_by_arm"].items()
    } == {
        "localized": {
            "input_tokens": 10,
            "cache_tokens": 0,
            "output_tokens": 6,
        },
        "baseline": {
            "input_tokens": 3,
            "cache_tokens": 0,
            "output_tokens": 2,
        },
    }
    assert result["usage_by_arm"]["localized"]["cost_usd"] == pytest.approx(0.014)
    assert result["usage_by_arm"]["baseline"]["cost_usd"] == pytest.approx(0.005)
    printed = capsys.readouterr().out
    assert "task-a segment 0 localized: generated" in printed
    assert "task-a segment 0 baseline: generated (reused)" in printed


async def test_interrupted_run_reuses_every_persisted_paid_attempt(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    plan = plan_skill_distillation(report_path)
    output = tmp_path / "candidates.json"
    first_meter = _Meter()
    first_calls = 0

    async def interrupting_complete(_prompt: str) -> str:
        nonlocal first_calls
        first_calls += 1
        first_meter.charge(input_tokens=2, output_tokens=3, cost_usd=0.004)
        if first_calls == 2:
            raise KeyboardInterrupt
        return _SKILL

    with pytest.raises(KeyboardInterrupt):
        await run_skill_distillation(
            plan,
            output,
            distiller=CallableSkillDistiller(interrupting_complete),
            usage_reader=first_meter.read,
        )

    partial = json.loads(output.read_text(encoding="utf-8"))
    assert len(partial["attempts"]) == 1
    assert partial["attempts"][0]["status"] == "generated"

    second_meter = _Meter()
    second_calls = 0

    async def resumed_complete(_prompt: str) -> str:
        nonlocal second_calls
        second_calls += 1
        second_meter.charge(input_tokens=5, output_tokens=7, cost_usd=0.008)
        return _SKILL

    resumed = await run_skill_distillation(
        plan,
        output,
        distiller=CallableSkillDistiller(resumed_complete),
        usage_reader=second_meter.read,
    )

    assert second_calls == 1
    assert [attempt["reused"] for attempt in resumed["attempts"]] == [True, False]
    assert resumed["summary"]["reused_call_count"] == 1
    assert resumed["summary"]["new_call_count"] == 1
    assert resumed["summary"]["pending_call_count"] == 0


def test_distill_skills_cli_dry_run_lists_calls_and_localization_refusals(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _report(
        tmp_path,
        usable_tasks=("task-a", "task-b", "task-c"),
        refused_tasks=("task-d", "task-e"),
    )
    output = tmp_path / "candidates.json"

    def provider_forbidden(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("dry run initialized the provider")

    monkeypatch.setattr(experiment, "_build_skill_distiller", provider_forbidden)

    assert (
        main(
            [
                "distill-skills",
                str(report_path),
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )

    printed = capsys.readouterr().out
    assert "6 model call(s) to make; 6 planned across 3 segment(s); 0 reused" in printed
    assert "task-a segment 0 localized: planned" in printed
    assert "task-a segment 0 baseline: planned" in printed
    assert "task-d: localization refused (uniform_checkpoint_scores)" in printed
    assert "dry run; no provider calls made and no output written" in printed
    assert not output.exists()


def test_distill_skills_dry_run_surfaces_missing_archive_and_nonzero_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _report(tmp_path)
    output = tmp_path / "candidates.json"
    archive = (
        tmp_path
        / "source-job"
        / "synthetic__trial"
        / ".driftlock-checkpoints"
        / "phase-0"
        / "checkpoints"
        / ("b" * 32)
        / "workspace.tar.gz"
    )
    archive.unlink()

    def provider_forbidden(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("zero-call dry run initialized the provider")

    monkeypatch.setattr(experiment, "_build_skill_distiller", provider_forbidden)

    plan = plan_skill_distillation(report_path)
    planned = plan.as_dict()
    assert planned["localized_segment_count"] == 1
    assert planned["callable_segment_count"] == 0
    assert planned["evidence_refusal_count"] == 1
    assert planned["evidence_refusals"][0]["task_name"] == "task-a"
    assert planned["evidence_refusals"][0]["segment_index"] == 0
    assert (
        planned["evidence_refusals"][0]["refusals_by_arm"]["localized"]["reason"]
        == "checkpoint_archive_unavailable"
    )

    assert (
        main(
            [
                "distill-skills",
                str(report_path),
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        == 1
    )
    printed = capsys.readouterr().out
    assert (
        "0 model call(s) to make; 0 planned across 0 segment(s); 0 reused; "
        "1 evidence-refused segment(s)" in printed
    )
    assert "task-a segment 0: evidence refused" in printed
    assert "localized (checkpoint_archive_unavailable)" in printed
    assert "workspace.tar.gz is missing" in printed
    assert not output.exists()

    refused_process = _run_console(report_path, output, dry_run=True)
    assert refused_process.returncode == 1
    assert "1 evidence-refused segment(s)" in refused_process.stdout
    assert not output.exists()

    localization = json.loads(report_path.read_text(encoding="utf-8"))
    localization["tasks"][0]["segments"] = []
    report_path.write_text(json.dumps(localization), encoding="utf-8")

    assert (
        main(
            [
                "distill-skills",
                str(report_path),
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    genuinely_absent = capsys.readouterr().out
    assert (
        "0 model call(s) to make; 0 planned across 0 segment(s); 0 reused; "
        "0 evidence-refused segment(s)" in genuinely_absent
    )
    assert "evidence refused" not in genuinely_absent
    assert not output.exists()

    empty_process = _run_console(report_path, output, dry_run=True)
    assert empty_process.returncode == 0
    assert "0 evidence-refused segment(s)" in empty_process.stdout
    assert not output.exists()


def test_distill_skills_dry_run_reports_oversized_evidence_and_pairs_refusal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _report(tmp_path)
    output = tmp_path / "candidates.json"
    _replace_checkpoint_with_checkout(
        tmp_path / "source-job" / "synthetic__trial", "b" * 32
    )

    def provider_forbidden(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("dry run initialized the provider")

    monkeypatch.setattr(experiment, "_build_skill_distiller", provider_forbidden)

    assert (
        main(
            [
                "distill-skills",
                str(report_path),
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        == 1
    )

    printed = capsys.readouterr().out
    assert "task-a segment 0 evidence: localized=" in printed
    assert "[AT/OVER BOUND]" in printed
    assert "baseline=" in printed
    assert "localized (evidence_size_limit_exceeded)" in printed
    assert "baseline (paired_arm_refused)" in printed
    assert "0 model call(s) to make" in printed
    assert "dry run; no provider calls made and no output written" in printed
    assert not output.exists()


def test_distill_skills_real_run_records_step_mismatch_and_nonzero_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _report(tmp_path)
    output = tmp_path / "candidates.json"
    manifest_path = (
        tmp_path
        / "source-job"
        / "synthetic__trial"
        / ".driftlock-checkpoints"
        / "phase-0"
        / "checkpoints"
        / ("b" * 32)
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["step"] = 20
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def provider_forbidden(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("zero-call real run initialized the provider")

    monkeypatch.setattr(experiment, "_build_skill_distiller", provider_forbidden)

    assert main(["distill-skills", str(report_path), "--output", str(output)]) == 1

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["dry_run"] is False
    assert result["summary"]["planned_call_count"] == 0
    assert result["summary"]["evidence_refusal_count"] == 1
    assert result["candidates"] == []
    refusal = result["plan"]["evidence_refusals"][0]
    assert refusal["task_name"] == "task-a"
    assert refusal["segment_index"] == 0
    assert refusal["refusals_by_arm"]["localized"] == {
        "reason": "checkpoint_step_mismatch",
        "detail": (f"checkpoint {'b' * 32} records step 20, not localized step 4"),
    }
    printed = capsys.readouterr().out
    assert "completed 0 model call(s); 0 new, 0 reused" in printed
    assert "1 evidence-refused segment(s)" in printed
    assert "task-a segment 0: evidence refused" in printed
    assert "localized (checkpoint_step_mismatch)" in printed
    assert f"wrote {output}" in printed

    refused_process = _run_console(report_path, output, dry_run=False)
    assert refused_process.returncode == 1
    assert "1 evidence-refused segment(s)" in refused_process.stdout


def test_distill_skills_partially_refused_plan_exits_nonzero(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path, segment_count=2)
    output = tmp_path / "candidates.json"
    localization = json.loads(report_path.read_text(encoding="utf-8"))
    localization["tasks"][0]["segments"][1]["end"]["checkpoint_id"] = "c" * 32
    report_path.write_text(json.dumps(localization), encoding="utf-8")

    process = _run_console(report_path, output, dry_run=True)

    assert process.returncode == 1
    assert (
        "2 model call(s) to make; 2 planned across 1 segment(s); 0 reused; "
        "1 evidence-refused segment(s)" in process.stdout
    )
    assert "task-a segment 0 localized: planned" in process.stdout
    assert "task-a segment 0 baseline: planned" in process.stdout
    assert "task-a segment 1: evidence refused" in process.stdout
    assert "localized (checkpoint_archive_unavailable)" in process.stdout
    assert not output.exists()


def test_distill_output_feeds_directly_to_admit_skills_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _report(tmp_path)
    output = tmp_path / "candidates.json"
    meter = _Meter()

    async def complete(_prompt: str) -> JudgeCompletion:
        meter.charge(input_tokens=3, output_tokens=2, cost_usd=0.005)
        return JudgeCompletion(_SKILL, tokens=5)

    monkeypatch.setattr(
        experiment,
        "_build_skill_distiller",
        lambda **kwargs: (
            CallableSkillDistiller(complete),
            meter.read,
            {"model": kwargs["model"], "provider": kwargs["provider"]},
        ),
    )
    assert main(["distill-skills", str(report_path), "--output", str(output)]) == 0

    admission_output = tmp_path / "admission.json"
    assert (
        main(
            [
                "admit-skills",
                str(output),
                "--library-dir",
                str(tmp_path / "library"),
                "--output",
                str(admission_output),
            ]
        )
        == 0
    )
    admission = json.loads(admission_output.read_text(encoding="utf-8"))
    assert admission["incomplete_candidate_count"] == 2
    assert admission["admitted_candidate_count"] == 0
