from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

import driftlock.lhtb_experiment as experiment
from driftlock.skill_distillation import Skill, serialize_skill
from driftlock.skill_validation import (
    ValidationFailureKind,
    ValidationTrial,
    plan_skill_validation,
    run_skill_validation,
)

_SUCCESSFUL_BLOCKING_HARBOR = """
import json
import pathlib
import sys
import time

config = json.loads(pathlib.Path(sys.argv[-1]).read_text(encoding="utf-8"))
time.sleep(0.3)
job_dir = pathlib.Path(config["jobs_dir"]) / config["job_name"]
record = job_dir / "trial-0" / "agent" / "driftlock-result.json"
record.parent.mkdir(parents=True)
record.write_text(
    json.dumps(
        {
            "skill_layer": {
                "distillation_arm": "localized",
                "injection": {"status": "not_injected", "candidate_ids": []},
            }
        }
    ),
    encoding="utf-8",
)
(job_dir / "result.json").write_text(
    json.dumps(
        {
            "stats": {
                "evals": {
                    "evaluation": {
                        "reward_stats": {"reward": {"0.5": ["trial-0"]}}
                    }
                }
            }
        }
    ),
    encoding="utf-8",
)
"""


_BLOCKING_PROCESS_TREE = """
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pid_file = pathlib.Path(os.environ["DRIFTLOCK_TEST_PID_FILE"])
pid_file.write_text(
    json.dumps({"parent": os.getpid(), "child": child.pid}), encoding="utf-8"
)

def stop(signum, frame):
    del frame
    child.terminate()
    child.wait(timeout=2)
    raise SystemExit(128 + signum)

signal.signal(signal.SIGINT, stop)
time.sleep(60)
"""


def _lhtb_tree(tmp_path: Path) -> Path:
    root = tmp_path / "LHTB"
    task = root / "tasks" / "task-0"
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(
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
                        "skill": serialize_skill(
                            Skill(
                                activation="When task 0 is active.",
                                execution="Apply candidate 0 only.",
                                termination="Stop after validation.",
                            )
                        ),
                        "paired_deltas": [],
                        "task_name": "long-horizon-terminal-bench/task-0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _runner(tmp_path: Path, *, work_name: str = "work"):
    return experiment._HarborSkillValidationRunner(
        lhtb_dir=_lhtb_tree(tmp_path),
        work_dir=tmp_path / work_name,
        skill_embedder_import_path="offline_embedder:embed",
        model="offline-model",
        provider="offline-provider",
        api_base="http://offline.invalid/v1",
        judge_api_base=None,
        judge_provider="offline-judge-provider",
        timeout_sec=60,
        max_total_tokens=100,
    )


def _trial(tmp_path: Path) -> ValidationTrial:
    library = tmp_path / "empty-library"
    library.mkdir(exist_ok=True)
    return ValidationTrial(
        trial_id="control-1",
        task_name="task-0",
        replicate_index=1,
        condition="without_skill",
        distillation_arm="localized",
        library_dir=library,
    )


@pytest.mark.asyncio
async def test_runner_preserves_child_cwd_environment_and_nonzero_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = tmp_path / "child.json"
    stand_in = """
import json
import os
import pathlib
import sys

pathlib.Path(os.environ["DRIFTLOCK_TEST_OBSERVATION"]).write_text(
    json.dumps(
        {
            "argv": sys.argv[1:],
            "cwd": os.getcwd(),
            "continue_mode": os.environ.get("HB_CONTINUE_MODE"),
            "process_reward_present": "HB_PROCESS_REWARD" in os.environ,
            "sentinel": os.environ.get("DRIFTLOCK_TEST_SENTINEL"),
        }
    ),
    encoding="utf-8",
)
raise SystemExit(23)
"""
    monkeypatch.setattr(
        experiment,
        "_pinned_harbor_command",
        lambda: [sys.executable, "-c", stand_in],
    )
    monkeypatch.setenv("DRIFTLOCK_TEST_OBSERVATION", str(observation))
    monkeypatch.setenv("DRIFTLOCK_TEST_SENTINEL", "copied")
    monkeypatch.setenv("HB_PROCESS_REWARD", "must-be-removed")
    monkeypatch.setenv("HB_CONTINUE_MODE", "ambient-wrong-value")
    runner = _runner(tmp_path)

    result = await runner.run(_trial(tmp_path))

    child = json.loads(observation.read_text(encoding="utf-8"))
    assert child["argv"][:2] == ["run", "-c"]
    assert child["cwd"] == str(runner.lhtb_dir)
    assert child["continue_mode"] == "same_conversation"
    assert child["process_reward_present"] is False
    assert child["sentinel"] == "copied"
    assert result.failure_kind is ValidationFailureKind.DID_NOT_PRODUCE_RESULT
    assert result.audit is not None
    assert result.audit["process_exit_code"] == 23
    assert child["argv"][2] == result.audit["config"]


@pytest.mark.parametrize(
    ("bound", "minimum_elapsed", "maximum_elapsed"),
    [(4, 0.45, 1.2), (1, 2.0, 4.0)],
)
@pytest.mark.asyncio
async def test_end_to_end_real_blocking_process_respects_concurrency_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bound: int,
    minimum_elapsed: float,
    maximum_elapsed: float,
) -> None:
    plan = plan_skill_validation(_candidate_file(tmp_path), _lhtb_tree(tmp_path))
    eight_trials = replace(plan, replicate_count=4)
    work_dir = tmp_path / f"work-{bound}"
    runner = _runner(tmp_path, work_name=f"work-{bound}")
    monkeypatch.setattr(
        experiment,
        "_pinned_harbor_command",
        lambda: [sys.executable, "-c", _SUCCESSFUL_BLOCKING_HARBOR],
    )

    started = time.perf_counter()
    report = await run_skill_validation(
        eight_trials,
        tmp_path / f"report-{bound}.json",
        runner=runner,
        work_dir=work_dir,
        max_concurrent_trials=bound,
    )
    elapsed = time.perf_counter() - started

    assert minimum_elapsed < elapsed < maximum_elapsed
    assert report["validation"]["summary"]["completed_attempt_count"] == 8
    assert report["validation"]["summary"]["measured_trial_count"] == 8
    assert len(report["validation"]["attempts"]) == 8
    assert (
        len({attempt["trial_id"] for attempt in report["validation"]["attempts"]}) == 8
    )
    assert report["candidates"][0]["paired_deltas"] == [0.0, 0.0, 0.0, 0.0]
    written = json.loads(
        (tmp_path / f"report-{bound}.json").read_text(encoding="utf-8")
    )
    assert written == report


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.asyncio
async def test_cancelling_runner_terminates_harbor_process_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "pids.json"
    monkeypatch.setenv("DRIFTLOCK_TEST_PID_FILE", str(pid_file))
    monkeypatch.setattr(
        experiment,
        "_pinned_harbor_command",
        lambda: [sys.executable, "-c", _BLOCKING_PROCESS_TREE],
    )
    running = asyncio.create_task(_runner(tmp_path).run(_trial(tmp_path)))
    async with asyncio.timeout(2):
        while not pid_file.is_file():
            await asyncio.sleep(0.01)
    pids = json.loads(pid_file.read_text(encoding="utf-8"))

    try:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        async with asyncio.timeout(2):
            while _pid_exists(pids["parent"]) or _pid_exists(pids["child"]):
                await asyncio.sleep(0.01)
    finally:
        for pid in (pids["parent"], pids["child"]):
            if _pid_exists(pid):
                os.kill(pid, signal.SIGKILL)

    assert _pid_exists(pids["parent"]) is False
    assert _pid_exists(pids["child"]) is False
