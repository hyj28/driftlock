"""Score retained checkpoints with fresh hidden-verifier Harbor jobs.

This module deliberately has no Harbor imports.  It validates and enumerates
retained bundles, reads Harbor's job-level reward statistics, builds replay job
configs, and assembles resumable scored timelines.  The agent that performs the
restore lives in :mod:`driftlock.harbor_agent`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from driftlock.oracle import OracleCheckpointError, load_remote_checkpoint_bundle

SCORE_REPORT_NAME = "checkpoint-scores.json"
SCORING_AGENT_IMPORT = "driftlock.harbor_agent:LHTBCheckpointScoringAgent"
_FINGERPRINT = re.compile(r"[0-9a-f]{64}")
_TASK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class CheckpointReplay:
    """One integrity-checked retained checkpoint ready for scoring."""

    candidate_id: str
    trial_name: str
    task_name: str
    task: str
    model_name: str
    experiment_fingerprint: str
    checkpoint_id: str
    phase: int
    step: int
    digest: str
    workspace: str
    checkpoint_dir: Path

    @property
    def job_name(self) -> str:
        return f"checkpoint-score-{self.candidate_id}"


@dataclass(frozen=True, slots=True)
class TrialCheckpointPlan:
    """A source trial and all of its retained checkpoints."""

    trial_name: str
    task_name: str
    final_reward: float | None
    checkpoints: tuple[CheckpointReplay, ...]


@dataclass(frozen=True, slots=True)
class CheckpointScoringPlan:
    """The complete checkpoint-scoring plan for a source Harbor job."""

    source_job_dir: Path
    trials: tuple[TrialCheckpointPlan, ...]

    @property
    def checkpoints(self) -> tuple[CheckpointReplay, ...]:
        return tuple(
            checkpoint for trial in self.trials for checkpoint in trial.checkpoints
        )

    @property
    def trials_without_checkpoints(self) -> tuple[str, ...]:
        return tuple(trial.trial_name for trial in self.trials if not trial.checkpoints)


def extract_job_trial_rewards(data: Mapping[str, Any]) -> dict[str, float]:
    """Return ``trial_name -> reward`` from Harbor job reward statistics.

    Harbor records verifier rewards for timed-out trials here even when their
    per-trial ``metrics.reward`` is absent.  Conflicting entries are rejected so
    iteration order can never decide a trial's score.
    """

    stats = data.get("stats")
    evals = stats.get("evals") if isinstance(stats, Mapping) else None
    if not isinstance(evals, Mapping):
        raise ValueError("Harbor job result lacks stats.evals reward statistics")
    rewards: dict[str, float] = {}
    for eval_name, evaluation in evals.items():
        reward_stats = (
            evaluation.get("reward_stats") if isinstance(evaluation, Mapping) else None
        )
        by_reward = (
            reward_stats.get("reward") if isinstance(reward_stats, Mapping) else None
        )
        if by_reward is None:
            continue
        if not isinstance(by_reward, Mapping):
            raise ValueError(f"invalid reward statistics for eval {eval_name!r}")
        for raw_reward, raw_trial_names in by_reward.items():
            try:
                reward = float(raw_reward)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid reward value {raw_reward!r} for eval {eval_name!r}"
                ) from error
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                raise ValueError(
                    f"reward {raw_reward!r} for eval {eval_name!r} is not in [0, 1]"
                )
            if not isinstance(raw_trial_names, list) or any(
                not isinstance(name, str) or not name for name in raw_trial_names
            ):
                raise ValueError(f"invalid trial names for eval {eval_name!r}")
            for trial_name in raw_trial_names:
                previous = rewards.get(trial_name)
                if previous is not None and previous != reward:
                    raise ValueError(
                        f"conflicting job-level rewards for trial {trial_name!r}"
                    )
                rewards[trial_name] = reward
    return rewards


def load_job_trial_rewards(job_dir: Path | str) -> dict[str, float]:
    """Load per-trial verifier rewards from a Harbor job's ``result.json``."""

    root = Path(job_dir).expanduser().resolve()
    result_file = root / "result.json"
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Harbor job result does not exist: {result_file}"
        ) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Harbor job result is invalid JSON: {result_file}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Harbor job result must be an object: {result_file}")
    return extract_job_trial_rewards(data)


def enumerate_retained_checkpoints(
    source_job_dir: Path | str,
) -> CheckpointScoringPlan:
    """Enumerate and integrity-check every retained checkpoint in a Harbor job."""

    source = Path(source_job_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source job directory does not exist: {source}")
    final_rewards = load_job_trial_rewards(source)
    result_files = sorted(source.glob("*/result.json"))
    if not result_files:
        raise ValueError(f"source job has no trial results: {source}")

    trials: list[TrialCheckpointPlan] = []
    candidate_ids: set[str] = set()
    for result_file in result_files:
        trial_dir = result_file.parent
        if trial_dir.is_symlink() or trial_dir.resolve().parent != source:
            raise ValueError(f"source trial escapes its job directory: {result_file}")
        result = _read_object(result_file, "source trial result")
        task_name, task, model_name, fingerprint = _source_identity(result, result_file)
        checkpoints: list[CheckpointReplay] = []
        checkpoint_root = trial_dir / ".driftlock-checkpoints"
        for checkpoint_dir in sorted(checkpoint_root.glob("phase-*/checkpoints/*")):
            try:
                bundle = load_remote_checkpoint_bundle(checkpoint_dir)
            except OracleCheckpointError as error:
                raise ValueError(
                    f"retained checkpoint is invalid at {checkpoint_dir}: {error}"
                ) from error
            phase_dir = bundle.checkpoint.path.parent.parent
            discovered_phase_dir = checkpoint_dir.parent.parent
            if (
                checkpoint_root.is_symlink()
                or discovered_phase_dir.is_symlink()
                or checkpoint_dir.parent.is_symlink()
                or phase_dir.parent != checkpoint_root.resolve()
                or not re.fullmatch(r"phase-[0-9]+", phase_dir.name)
            ):
                raise ValueError(
                    f"retained checkpoint has an invalid phase path: {checkpoint_dir}"
                )
            phase = int(phase_dir.name.removeprefix("phase-"))
            identity = (
                f"{trial_dir.name}\0{bundle.checkpoint.checkpoint_id}\0"
                f"{bundle.checkpoint.digest}"
            )
            candidate_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
            if candidate_id in candidate_ids:
                raise ValueError("checkpoint scoring candidate ids must be unique")
            candidate_ids.add(candidate_id)
            checkpoints.append(
                CheckpointReplay(
                    candidate_id=candidate_id,
                    trial_name=trial_dir.name,
                    task_name=task_name,
                    task=task,
                    model_name=model_name,
                    experiment_fingerprint=fingerprint,
                    checkpoint_id=bundle.checkpoint.checkpoint_id,
                    phase=phase,
                    step=bundle.checkpoint.step,
                    digest=bundle.checkpoint.digest,
                    workspace=bundle.remote_workspace,
                    checkpoint_dir=bundle.checkpoint.path,
                )
            )
        checkpoints.sort(key=lambda item: (item.phase, item.step, item.checkpoint_id))
        trials.append(
            TrialCheckpointPlan(
                trial_name=trial_dir.name,
                task_name=task_name,
                final_reward=final_rewards.get(trial_dir.name),
                checkpoints=tuple(checkpoints),
            )
        )

    if not any(trial.checkpoints for trial in trials):
        names = ", ".join(trial.trial_name for trial in trials)
        raise ValueError(
            f"source job has no retained checkpoints; trials checked: {names}"
        )
    return CheckpointScoringPlan(source_job_dir=source, trials=tuple(trials))


def build_checkpoint_replay_config(
    *,
    lhtb_dir: Path | str,
    jobs_dir: Path | str,
    replay: CheckpointReplay,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    """Build one token-free Harbor job that scores a retained checkpoint."""

    root = Path(lhtb_dir).expanduser().resolve()
    destination = Path(jobs_dir).expanduser().resolve()
    if timeout_sec <= 0:
        raise ValueError("timeout must be positive")
    task_file = root / "tasks" / replay.task / "task.toml"
    if not task_file.is_file():
        raise ValueError(f"unknown LHTB task: {replay.task}")
    return {
        "job_name": replay.job_name,
        "jobs_dir": str(destination),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "timeout_multiplier": 1.0,
        "retry": {"max_retries": 0},
        "environment": {"type": "docker", "force_build": True, "delete": True},
        "agents": [
            {
                "import_path": SCORING_AGENT_IMPORT,
                "model_name": replay.model_name,
                "override_timeout_sec": timeout_sec,
                "env": {
                    "HB_CONTINUE_MODE": "same_conversation",
                    "DRIFTLOCK_EXPERIMENT_FINGERPRINT": (replay.experiment_fingerprint),
                },
                "kwargs": {
                    "driftlock_scoring_checkpoint_dir": str(replay.checkpoint_dir),
                    "driftlock_scoring_checkpoint_digest": replay.digest,
                    "driftlock_scoring_expected_workspace": replay.workspace,
                },
            }
        ],
        "datasets": [{"path": str(root / "tasks"), "task_names": [replay.task]}],
    }


def assemble_scored_timelines(
    plan: CheckpointScoringPlan,
    scores: Mapping[str, float | None],
) -> dict[str, Any]:
    """Assemble checkpoint scores and final rewards into per-trial timelines."""

    known = {checkpoint.candidate_id for checkpoint in plan.checkpoints}
    unknown = sorted(set(scores) - known)
    if unknown:
        raise ValueError("score data contains unknown checkpoint candidates")
    trials: list[dict[str, Any]] = []
    for trial in plan.trials:
        timeline = []
        for checkpoint in trial.checkpoints:
            reward = scores.get(checkpoint.candidate_id)
            if reward is not None:
                reward = _reward(reward, f"checkpoint {checkpoint.candidate_id}")
            timeline.append(
                {
                    "candidate_id": checkpoint.candidate_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "phase": checkpoint.phase,
                    "step": checkpoint.step,
                    "reward": reward,
                }
            )
        scored = [item for item in timeline if item["reward"] is not None]
        best = max((item["reward"] for item in scored), default=None)
        item: dict[str, Any] = {
            "trial_name": trial.trial_name,
            "task_name": trial.task_name,
            "final_reward": trial.final_reward,
            "checkpoints": timeline,
            "best_checkpoint_reward": best,
        }
        if best is not None and trial.final_reward is not None:
            item["headroom"] = best - trial.final_reward
        trials.append(item)
    return {
        "schema_version": 1,
        "mode": "checkpoint-scoring",
        "source_job_dir": str(plan.source_job_dir),
        "checkpoint_count": len(plan.checkpoints),
        "scored_checkpoint_count": sum(
            checkpoint.candidate_id in scores
            and scores[checkpoint.candidate_id] is not None
            for checkpoint in plan.checkpoints
        ),
        "trials": trials,
    }


def load_completed_scores(
    report_path: Path | str, plan: CheckpointScoringPlan
) -> dict[str, float]:
    """Load completed scores from an earlier compatible partial report."""

    path = Path(report_path)
    if not path.exists():
        return {}
    report = _read_object(path, "checkpoint score report")
    if (
        report.get("schema_version") != 1
        or report.get("mode") != "checkpoint-scoring"
        or report.get("source_job_dir") != str(plan.source_job_dir)
    ):
        raise ValueError(f"existing score report is for a different run: {path}")
    trials = report.get("trials")
    if not isinstance(trials, list):
        raise ValueError(f"existing score report has invalid trials: {path}")
    expected = {checkpoint.candidate_id: checkpoint for checkpoint in plan.checkpoints}
    scores: dict[str, float] = {}
    for trial in trials:
        checkpoints = trial.get("checkpoints") if isinstance(trial, dict) else None
        if not isinstance(checkpoints, list):
            raise ValueError(f"existing score report has invalid checkpoints: {path}")
        for item in checkpoints:
            if not isinstance(item, dict):
                raise ValueError(
                    f"existing score report has invalid checkpoint: {path}"
                )
            candidate_id = item.get("candidate_id")
            checkpoint = expected.get(candidate_id)
            if checkpoint is None:
                raise ValueError(
                    f"existing score report contains an unknown checkpoint: {path}"
                )
            if (
                item.get("checkpoint_id") != checkpoint.checkpoint_id
                or item.get("phase") != checkpoint.phase
                or item.get("step") != checkpoint.step
            ):
                raise ValueError(
                    f"existing score report checkpoint identity changed: {path}"
                )
            if item.get("reward") is not None:
                scores[candidate_id] = _reward(
                    item["reward"], f"checkpoint {candidate_id} in {path}"
                )
    return scores


def write_score_report(
    report_path: Path | str,
    plan: CheckpointScoringPlan,
    scores: Mapping[str, float | None],
) -> dict[str, Any]:
    """Atomically persist the current scored timelines for interruption safety."""

    path = Path(report_path)
    report = assemble_scored_timelines(plan, scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return report


def single_job_reward(job_dir: Path | str) -> float | None:
    """Return the sole trial reward from one checkpoint replay job."""

    rewards = load_job_trial_rewards(job_dir)
    if not rewards:
        return None
    if len(rewards) != 1:
        raise ValueError(
            f"checkpoint replay job has {len(rewards)} trial rewards; expected one"
        )
    return next(iter(rewards.values()))


def _source_identity(
    result: Mapping[str, Any], result_file: Path
) -> tuple[str, str, str, str]:
    task_name = result.get("task_name")
    config = result.get("config")
    agent = config.get("agent") if isinstance(config, Mapping) else None
    model_name = agent.get("model_name") if isinstance(agent, Mapping) else None
    environment = agent.get("env") if isinstance(agent, Mapping) else None
    fingerprint = (
        environment.get("DRIFTLOCK_EXPERIMENT_FINGERPRINT")
        if isinstance(environment, Mapping)
        else None
    )
    if not isinstance(task_name, str) or not _TASK_NAME.fullmatch(task_name):
        raise ValueError(f"source trial has an invalid task name: {result_file}")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"source trial lacks a model name: {result_file}")
    if not isinstance(fingerprint, str) or not _FINGERPRINT.fullmatch(fingerprint):
        raise ValueError(f"source trial lacks an experiment fingerprint: {result_file}")
    return task_name, task_name.rsplit("/", 1)[1], model_name, fingerprint


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _reward(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} reward must be numeric")
    reward = float(value)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"{context} reward is not in [0, 1]")
    return reward
