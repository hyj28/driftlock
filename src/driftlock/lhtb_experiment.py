"""Budget-conscious command line harness for pinned LHTB experiments."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from driftlock.checkpoint_localization import (
    LOCALIZATION_REPORT_NAME,
    load_and_localize_score_report,
    write_localization_report,
)
from driftlock.checkpoint_scoring import (
    SCORE_REPORT_NAME,
    assemble_scored_timelines,
    build_checkpoint_replay_config,
    enumerate_retained_checkpoints,
    load_completed_scores,
    single_job_reward,
    write_score_report,
)
from driftlock.judges import DEFAULT_JUDGE_MAX_OUTPUT_TOKENS
from driftlock.lhtb import (
    DRIFTLOCK_HARBOR_PATCH_VERSION,
    LHTB_LITELLM_VERSION,
    LHTB_REPOSITORY_REVISION,
    lhtb_experiment_fingerprint,
    openrouter_provider_call_kwargs,
    recorded_lhtb_fingerprint,
    require_one_lhtb_fingerprint,
)
from driftlock.lhtb_analysis import (
    analyze_jobs,
    parse_arm_directories,
    task_directory_sha256,
)
from driftlock.oracle import (
    OracleCheckpointError,
    file_sha256,
    load_remote_checkpoint_bundle,
    load_source_trial_provenance,
    validate_checkpoint_source_audit,
)
from driftlock.skill_admission import (
    ADMISSION_REPORT_NAME,
    SkillLibrary,
    load_admission_candidates,
    render_admission_report,
    write_admission_report,
)
from driftlock.skill_distillation import CallableSkillDistiller
from driftlock.skill_distillation_driver import (
    DISTILLATION_ARMS,
    DISTILLATION_REPORT_NAME,
    plan_skill_distillation,
    run_skill_distillation,
)
from driftlock.skill_validation import (
    DEFAULT_ESTIMATED_COST_PER_TRIAL_USD,
    DEFAULT_MAX_CONCURRENT_TRIALS,
    VALIDATION_REPORT_NAME,
    ValidationFailureKind,
    ValidationTrial,
    ValidationTrialResult,
    ValidationTrialStatus,
    plan_skill_validation,
    run_skill_validation,
)
from driftlock.usage import ReplayUsage

# Both identifiers carry an explicit dated build. An unversioned alias such as
# "deepseek-v4-pro" silently follows whatever OpenRouter currently points it at,
# so arms run on different days would use different models while the recorded
# model string stayed identical - the comparison would be void with no evidence.
DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
DEFAULT_PROVIDER = "deepinfra/fp8"
DEFAULT_JUDGE_MODEL = "openrouter/deepseek/deepseek-v4-pro-0813"
DEFAULT_JUDGE_PROVIDER = "alibaba"
# Arms that pay for a fine judge, and therefore need its provider probed too.
_SKILL_DISTILLATION_ARM_BY_RUNNABLE_ARM = {
    "skill-baseline": "baseline",
    "skill-localized": "localized",
}
_FINE_JUDGE_ARMS = frozenset(
    {
        "driftlock",
        "native-driftlock",
        *_SKILL_DISTILLATION_ARM_BY_RUNNABLE_ARM,
    }
)
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_CREDENTIAL_ENV = "OPENROUTER_API_KEY"
RUNNABLE_ARMS = (
    "stock",
    "retry",
    "driftlock-heuristic",
    "driftlock",
    "native-driftlock-heuristic",
    "native-driftlock",
    *_SKILL_DISTILLATION_ARM_BY_RUNNABLE_ARM,
)
DRIFTLOCK_DETECTOR_DEFAULTS = {
    "driftlock_no_change_steps": 4,
    "driftlock_loop_window": 6,
    "driftlock_loop_repetitions": 3,
    "driftlock_error_window": 5,
    "driftlock_error_rate": 0.6,
    "driftlock_command_failure_window": 8,
    "driftlock_command_failure_rate": 1.0,
    "driftlock_reward_stall_steps": 5,
    "driftlock_reward_epsilon": 1e-6,
    "driftlock_corroborating_signals": ["no_file_change"],
}

# SHA-256 of every Harbor file after applying the packaged version-11 patch to the
# pinned LHTB revision.  Preflight also rejects any other Harbor or task-tree change.
_PATCHED_HARBOR_SHA256 = {
    "harbor/src/harbor/_driftlock_pin.py": (
        "9da8ba424621246e0836713932d75a0e94e14aadcb50fe93d008005f57970050"
    ),
    "harbor/src/harbor/agents/terminus_2/terminus_2.py": (
        "7ec452a41b135d1fb1f130a9ff31578653280c7a6a6b12165776bc83080b61e5"
    ),
    "harbor/src/harbor/agents/terminus_2/tmux_session.py": (
        "3efd8216f7c9e276178b474a5c73e4a050026910af8876fa06b4f1bfae8b24f1"
    ),
    "harbor/src/harbor/llms/base.py": (
        "cb8d0e482933eb78c18e95d0f6f4f0e10b36d91522a990ddaab382eb71a38cf0"
    ),
    "harbor/src/harbor/llms/chat.py": (
        "066b0826b4604a4f23762776916b7cde547ee3708ab922669c2d513e73490297"
    ),
    "harbor/src/harbor/llms/lite_llm.py": (
        "f33145a1d3f875239523e9a72ad401b09379364fb38e611d6b0b57c0143b1e1a"
    ),
    "harbor/src/harbor/trial/trial.py": (
        "fc769a6fd7646ec8c3049e16ebb70c31e5a2a7a7ffe010ed30b4d5737184b2c5"
    ),
    "harbor/tests/unit/agents/terminus_2/test_driftlock_quiescence.py": (
        "fab5d8cd139ff8b6fce158a04d09fec77f47ae8cae58a2ba600a41055e13f186"
    ),
    "harbor/tests/unit/agents/terminus_2/test_tmux_session.py": (
        "0412e289ae6e8de7d2fc92fb1d1762761b1862b42102755638ce2a451d737d07"
    ),
    "harbor/tests/unit/llms/test_chat.py": (
        "1b759fbd5ca57a7ef4199dcc958f50d05f8280e8b81c5648df01a3a45112a0a5"
    ),
    "harbor/tests/unit/llms/test_lite_llm.py": (
        "9bc10a5e3b0021a808f56c32aa0ebd2f6d7a27e2b1e3a41413abc194db5103ea"
    ),
}


class PreflightError(RuntimeError):
    """Raised before a paid run when the pinned environment is unsafe."""


def build_job_config(
    *,
    lhtb_dir: Path,
    jobs_dir: Path,
    job_name: str,
    arm: str,
    tasks: Sequence[str],
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    api_base: str = DEFAULT_API_BASE,
    n_concurrent_trials: int = 1,
    timeout_sec: int = 5400,
    max_total_tokens: int = 10_000_000,
    judge_api_base: str | None = None,
    judge_provider: str = DEFAULT_JUDGE_PROVIDER,
    retain_checkpoints: bool = False,
    skill_library_dir: Path | None = None,
    skill_embedder_import_path: str | None = None,
) -> dict[str, Any]:
    """Build the exact JSON-compatible Harbor configuration for one run."""
    root = lhtb_dir.expanduser().resolve()
    task_root = root / "tasks"
    _validate_job_name(job_name)
    selected = _validate_tasks(task_root, tasks)
    if arm == "oracle":
        raise ValueError(
            "oracle requires isolated hidden-verifier checkpoint replay; an online "
            "agent configuration would not be a hindsight-perfect oracle"
        )
    if arm not in RUNNABLE_ARMS:
        raise ValueError("arm must be one of " + ", ".join(RUNNABLE_ARMS))
    skill_distillation_arm = _SKILL_DISTILLATION_ARM_BY_RUNNABLE_ARM.get(arm)
    skill_options = (skill_library_dir, skill_embedder_import_path)
    if skill_distillation_arm is None and any(
        option is not None for option in skill_options
    ):
        raise ValueError("skill configuration requires a skill-* arm")
    if skill_distillation_arm is not None and any(
        option is None for option in skill_options
    ):
        raise ValueError(
            "skill arms require skill_library_dir and skill_embedder_import_path"
        )
    checkpoint_arms = {
        "driftlock",
        "driftlock-heuristic",
        *_SKILL_DISTILLATION_ARM_BY_RUNNABLE_ARM,
    }
    if retain_checkpoints and arm.startswith("native-"):
        raise ValueError(
            "native checkpoint retention is not supported by oracle replay; "
            "native oracle replay is future work"
        )
    if retain_checkpoints and arm not in checkpoint_arms:
        raise ValueError("checkpoint retention requires a driftlock arm")
    if n_concurrent_trials <= 0:
        raise ValueError("n_concurrent_trials must be positive")
    if timeout_sec <= 0 or max_total_tokens <= 0:
        raise ValueError("timeout and token budget must be positive")

    agent: dict[str, Any] = {
        "model_name": model,
        "override_timeout_sec": timeout_sec,
        "kwargs": {
            "api_base": api_base,
            "parser_name": "json",
            "temperature": 0.7,
            "record_terminal_session": True,
            "llm_call_kwargs": {
                "temperature": 0.7,
                "max_tokens": 8192,
                "timeout": 240,
                **openrouter_provider_call_kwargs(provider),
            },
            "model_info": {
                "max_input_tokens": 128000,
                "max_output_tokens": 8192,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
        },
    }
    if arm == "stock":
        agent["name"] = "terminus-2"
        agent["env"] = {
            "HB_CONTINUE_MODE": "fresh",
            "DRIFTLOCK_EXPERIMENT_FINGERPRINT": lhtb_experiment_fingerprint(),
        }
        agent["kwargs"].update(
            {
                "enable_summarize": True,
                "proactive_summarization_threshold": 8000,
            }
        )
        agent["kwargs"]["llm_call_kwargs"]["num_retries"] = 4
    else:
        if arm == "retry":
            agent["import_path"] = "driftlock.harbor_agent:LHTBBlindRetryAgent"
        elif arm.startswith("native-"):
            agent["import_path"] = (
                "driftlock.harbor_native_agent:LHTBNativeDriftlockAgent"
            )
        else:
            agent["import_path"] = "driftlock.harbor_agent:LHTBDriftlockAgent"
        agent["env"] = {
            "HB_CONTINUE_MODE": "same_conversation",
            "DRIFTLOCK_EXPERIMENT_FINGERPRINT": lhtb_experiment_fingerprint(),
        }
        agent["kwargs"].update(
            {
                "enable_summarize": False,
                "driftlock_max_tokens": max_total_tokens,
                "driftlock_max_steps": 500,
                "driftlock_max_rollbacks": 3,
                "driftlock_checkpoint_interval": 5,
            }
        )
        if arm in {
            "driftlock-heuristic",
            "driftlock",
            "native-driftlock-heuristic",
            "native-driftlock",
            *_SKILL_DISTILLATION_ARM_BY_RUNNABLE_ARM,
        }:
            agent["kwargs"].update(DRIFTLOCK_DETECTOR_DEFAULTS)
        if retain_checkpoints:
            agent["kwargs"]["driftlock_retain_checkpoints"] = True
        if arm in _FINE_JUDGE_ARMS:
            agent["kwargs"].update(
                {
                    "driftlock_judge_model": DEFAULT_JUDGE_MODEL,
                    "driftlock_judge_api_base": judge_api_base or api_base,
                    "driftlock_judge_max_output_tokens": (
                        DEFAULT_JUDGE_MAX_OUTPUT_TOKENS
                    ),
                    "driftlock_judge_llm_call_kwargs": (
                        openrouter_provider_call_kwargs(judge_provider)
                    ),
                }
            )
        if skill_distillation_arm is not None:
            assert skill_library_dir is not None
            assert skill_embedder_import_path is not None
            agent["kwargs"].update(
                {
                    "driftlock_skill_library_dir": str(
                        skill_library_dir.expanduser().resolve()
                    ),
                    "driftlock_skill_embedder_import_path": (
                        skill_embedder_import_path
                    ),
                    "driftlock_skill_distillation_arm": skill_distillation_arm,
                }
            )

    return {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir.expanduser().resolve()),
        "n_attempts": 1,
        "n_concurrent_trials": n_concurrent_trials,
        "timeout_multiplier": 1.0,
        "retry": {"max_retries": 0},
        "environment": {"type": "docker", "force_build": True, "delete": True},
        "agents": [agent],
        "datasets": [{"path": str(task_root), "task_names": selected}],
    }


def prepare_oracle_replays(
    *,
    lhtb_dir: Path,
    source_job_dir: Path,
    output_dir: Path,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    """Generate one fresh hidden-verifier Harbor job per retained checkpoint."""
    root = lhtb_dir.expanduser().resolve()
    source = source_job_dir.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if timeout_sec <= 0:
        raise ValueError("timeout must be positive")
    if destination == source or source in destination.parents:
        raise ValueError("oracle output must be outside the source job")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("oracle output directory must be empty")
    configs_dir = destination / "configs"
    jobs_dir = destination / "jobs"

    candidates: list[dict[str, Any]] = []
    pending_configs: list[tuple[Path, dict[str, Any]]] = []
    seen_checkpoints: set[str] = set()
    source_job_fingerprint: str | None = None
    for result_file in sorted(source.glob("*/result.json")):
        if (
            result_file.parent.is_symlink()
            or result_file.resolve().parent.parent != source
        ):
            raise ValueError(f"source trial escapes its job directory: {result_file}")
        try:
            raw_result = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_result = None
        raw_config = raw_result.get("config") if isinstance(raw_result, dict) else None
        raw_agent = raw_config.get("agent") if isinstance(raw_config, dict) else None
        if isinstance(raw_agent, dict) and raw_agent.get("import_path") == (
            "driftlock.harbor_native_agent:LHTBNativeDriftlockAgent"
        ):
            raise ValueError(
                "native retained checkpoints cannot be used for oracle replay; "
                "native oracle replay is not supported"
            )
        provenance = load_source_trial_provenance(result_file)
        result = provenance.data
        source_trial_id = provenance.trial_id
        config = result.get("config")
        agent = config.get("agent") if isinstance(config, dict) else None
        kwargs = agent.get("kwargs") if isinstance(agent, dict) else None
        if (
            not isinstance(kwargs, dict)
            or agent.get("import_path") != "driftlock.harbor_agent:LHTBDriftlockAgent"
        ):
            raise ValueError(
                f"source trial is not a supported driftlock trial: {result_file}"
            )
        source_fingerprint = _validate_oracle_source_agent(
            result, agent, kwargs, result_file
        )
        if source_job_fingerprint is None:
            source_job_fingerprint = _source_job_recorded_fingerprint(source)
        if source_fingerprint != source_job_fingerprint:
            raise ValueError(
                "source trial result fingerprint disagrees with Harbor lock audit: "
                f"{result_file} records {source_fingerprint}, but "
                f"{source / 'lock.json'} records {source_job_fingerprint}"
            )
        model_name = provenance.model_name
        task_name = provenance.task_name
        task = _task_directory_name(root / "tasks", task_name)
        task_checksum = result.get("task_checksum")
        expected_checksum = task_directory_sha256(root / "tasks" / task)
        if task_checksum != expected_checksum:
            raise ValueError(
                f"source task checksum differs from the pinned checkout: {result_file}"
            )
        usage = provenance.usage
        usage_source = provenance.usage_source
        source_result_digest = provenance.result_sha256
        source_audit = result_file.parent / "agent" / "driftlock-result.json"
        source_audit_digest = file_sha256(source_audit)
        checkpoint_root = result_file.parent / ".driftlock-checkpoints"
        checkpoint_dirs = sorted(checkpoint_root.glob("phase-*/checkpoints/*"))
        if not checkpoint_dirs:
            raise ValueError(
                "source trial has no loadable retained checkpoints; looked for "
                f"{checkpoint_root / 'phase-*/checkpoints/*'}"
            )
        loaded_checkpoints = []
        for checkpoint_dir in checkpoint_dirs:
            try:
                bundle = load_remote_checkpoint_bundle(checkpoint_dir)
            except OracleCheckpointError as error:
                raise ValueError(
                    "source trial has no loadable retained checkpoint at "
                    f"{checkpoint_dir}; looked for manifest.json, state.json, and "
                    f"workspace.tar.gz there: {error}"
                ) from error
            phase_audit = validate_checkpoint_source_audit(
                bundle.checkpoint.path,
                source_result=result_file,
                source_audit=source_audit,
                expected_audit_sha256=source_audit_digest,
            )
            checkpoint_phase = int(
                bundle.checkpoint.path.parent.parent.name.removeprefix("phase-")
            )
            if phase_audit.get("phase") != checkpoint_phase:
                raise ValueError(
                    f"source checkpoint audit has an invalid phase id: {source_audit}"
                )
            loaded_checkpoints.append((bundle, checkpoint_phase))
        retained_phases = sorted(
            {checkpoint_phase for _, checkpoint_phase in loaded_checkpoints}
        )
        whole_trial = kwargs.get("driftlock_retain_checkpoints") is True
        checkpoint_coverage = {
            "retained_phases": retained_phases,
            "scope": "whole-trial" if whole_trial else "prefix",
            "whole_trial": whole_trial,
        }
        for bundle, checkpoint_phase in loaded_checkpoints:
            if bundle.checkpoint.checkpoint_id in seen_checkpoints:
                raise ValueError("checkpoint ids must be globally unique")
            seen_checkpoints.add(bundle.checkpoint.checkpoint_id)
            candidate_id = (
                f"{str(source_trial_id)[:8]}-{bundle.checkpoint.checkpoint_id}"
            )
            job_name = f"oracle-{candidate_id}"
            replay_config = _oracle_replay_job_config(
                lhtb_dir=root,
                jobs_dir=jobs_dir,
                job_name=job_name,
                task=task,
                model_name=model_name,
                timeout_sec=timeout_sec,
                checkpoint_dir=bundle.checkpoint.path,
                checkpoint_digest=bundle.checkpoint.digest,
                workspace=bundle.remote_workspace,
                source_trial_id=str(source_trial_id),
                source_task_name=task_name,
                source_result=result_file.resolve(),
                source_result_sha256=source_result_digest,
                source_audit=source_audit.resolve(),
                source_audit_sha256=source_audit_digest,
                source_usage=usage,
                source_usage_source=usage_source,
                checkpoint_coverage=checkpoint_coverage,
                source_fingerprint=source_fingerprint,
            )
            config_path = configs_dir / f"{candidate_id}.json"
            pending_configs.append((config_path, replay_config))
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source_trial_id": str(source_trial_id),
                    "source_result": str(result_file.resolve()),
                    "source_result_sha256": source_result_digest,
                    "source_audit": str(source_audit.resolve()),
                    "source_audit_sha256": source_audit_digest,
                    "task_name": task_name,
                    "model_name": model_name,
                    "checkpoint_id": bundle.checkpoint.checkpoint_id,
                    "checkpoint_phase": checkpoint_phase,
                    "checkpoint_step": bundle.checkpoint.step,
                    "checkpoint_digest": bundle.checkpoint.digest,
                    "checkpoint_dir": str(bundle.checkpoint.path),
                    "archive_sha256": bundle.archive_sha256,
                    "state_sha256": bundle.state_sha256,
                    "workspace": bundle.remote_workspace,
                    "source_usage": usage.as_dict(),
                    "source_usage_source": usage_source,
                    "source_fingerprint": source_fingerprint,
                    "checkpoint_coverage": checkpoint_coverage,
                    "usage_policy": "full-source-trial-conservative",
                    "config": str(config_path),
                    "job_name": job_name,
                }
            )
    if not candidates:
        raise ValueError(f"source job has no trial results: {source}")
    assert source_job_fingerprint is not None
    manifest = {
        "schema_version": 1,
        "mode": "isolated-checkpoint-replay",
        "source_job_dir": str(source),
        "lhtb_revision": LHTB_REPOSITORY_REVISION,
        "harbor_patch": str(DRIFTLOCK_HARBOR_PATCH_VERSION),
        "experiment_fingerprint": source_job_fingerprint,
        "usage_policy": "full-source-trial-conservative",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    configs_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for config_path, replay_config in pending_configs:
        config_path.write_text(
            json.dumps(replay_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest_path = destination / "oracle-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _oracle_replay_job_config(
    *,
    lhtb_dir: Path,
    jobs_dir: Path,
    job_name: str,
    task: str,
    model_name: str,
    timeout_sec: int,
    checkpoint_dir: Path,
    checkpoint_digest: str,
    workspace: str,
    source_trial_id: str,
    source_task_name: str,
    source_result: Path,
    source_result_sha256: str,
    source_audit: Path,
    source_audit_sha256: str,
    source_usage: ReplayUsage,
    source_usage_source: str,
    checkpoint_coverage: dict[str, Any],
    source_fingerprint: str,
) -> dict[str, Any]:
    return {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "timeout_multiplier": 1.0,
        "retry": {"max_retries": 0},
        "environment": {"type": "docker", "force_build": True, "delete": True},
        "agents": [
            {
                "import_path": ("driftlock.harbor_agent:LHTBCheckpointReplayOracle"),
                "model_name": model_name,
                "override_timeout_sec": timeout_sec,
                "env": {
                    "HB_CONTINUE_MODE": "same_conversation",
                    "DRIFTLOCK_EXPERIMENT_FINGERPRINT": source_fingerprint,
                },
                "kwargs": {
                    "driftlock_oracle_mode": "isolated-checkpoint-replay",
                    "driftlock_checkpoint_dir": str(checkpoint_dir),
                    "driftlock_checkpoint_digest": checkpoint_digest,
                    "driftlock_expected_workspace": workspace,
                    "driftlock_source_trial_id": source_trial_id,
                    "driftlock_source_task_name": source_task_name,
                    "driftlock_source_result": str(source_result),
                    "driftlock_source_result_sha256": source_result_sha256,
                    "driftlock_source_audit": str(source_audit),
                    "driftlock_source_audit_sha256": source_audit_sha256,
                    "driftlock_source_usage": source_usage.as_dict(),
                    "driftlock_source_usage_source": source_usage_source,
                    "driftlock_checkpoint_coverage": checkpoint_coverage,
                },
            }
        ],
        "datasets": [{"path": str(lhtb_dir / "tasks"), "task_names": [task]}],
    }


def _validate_oracle_source_agent(
    result: dict[str, Any],
    agent: dict[str, Any],
    kwargs: dict[str, Any],
    path: Path,
) -> str:
    environment = agent.get("env")
    if (
        not isinstance(environment, dict)
        or environment.get("HB_CONTINUE_MODE") != "same_conversation"
    ):
        raise ValueError(f"source trial has the wrong experiment identity: {path}")
    fingerprint = recorded_lhtb_fingerprint(environment, f"source trial result {path}")
    if set(environment) != {
        "HB_CONTINUE_MODE",
        "DRIFTLOCK_EXPERIMENT_FINGERPRINT",
    }:
        raise ValueError(f"source trial has the wrong experiment identity: {path}")
    required = {
        "enable_summarize": False,
        "driftlock_max_steps": 500,
        "driftlock_max_rollbacks": 3,
        "driftlock_checkpoint_interval": 5,
    }
    if any(kwargs.get(name) != expected for name, expected in required.items()):
        raise ValueError(f"source trial has an unsupported driftlock config: {path}")
    retain_checkpoints = kwargs.get("driftlock_retain_checkpoints", False)
    if not isinstance(retain_checkpoints, bool):
        raise ValueError(f"source trial has an unsupported driftlock config: {path}")
    budget = kwargs.get("driftlock_max_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError(f"source trial lacks a positive token budget: {path}")
    info = result.get("agent_info")
    model_info = info.get("model_info") if isinstance(info, dict) else None
    model_name = agent.get("model_name")
    if not isinstance(model_name, str):
        raise ValueError(f"source trial lacks model identity: {path}")
    provider, separator, name = model_name.partition("/")
    if not separator:
        provider, name = None, provider
    if (
        not isinstance(info, dict)
        or info.get("name") != "driftlock-terminus-2"
        or info.get("version") != package_version("driftlock")
        or model_info != {"provider": provider, "name": name}
    ):
        raise ValueError(f"source trial agent identity is inconsistent: {path}")
    return fingerprint


def _source_job_recorded_fingerprint(source: Path) -> str:
    lock_file = source / "lock.json"
    try:
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"source job lacks its fingerprint audit record: {lock_file}"
        ) from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"source fingerprint audit is unreadable: {lock_file}"
        ) from error
    trials = lock.get("trials") if isinstance(lock, dict) else None
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"source fingerprint audit has no trials: {lock_file}")
    fingerprints = set()
    for index, trial in enumerate(trials):
        agent = trial.get("agent") if isinstance(trial, dict) else None
        environment = agent.get("env") if isinstance(agent, dict) else None
        fingerprints.add(
            recorded_lhtb_fingerprint(
                environment,
                f"source Harbor lock audit {lock_file} trial {index}",
            )
        )
    return require_one_lhtb_fingerprint(
        fingerprints, context=f"source Harbor lock audit {lock_file}"
    )


def preflight(
    lhtb_dir: Path,
    *,
    credential_env: str = DEFAULT_CREDENTIAL_ENV,
    require_credential: bool = True,
    require_docker: bool = True,
) -> dict[str, str]:
    """Verify the frozen Harbor, credential presence, and native amd64 Docker."""
    root = lhtb_dir.expanduser().resolve()
    if not (root / "harbor" / "pyproject.toml").is_file():
        raise PreflightError(f"not an LHTB checkout: {root}")
    revision = _run_checked(
        ["git", "rev-parse", "HEAD"], cwd=root, description="read LHTB revision"
    )
    if revision != LHTB_REPOSITORY_REVISION:
        raise PreflightError(
            f"LHTB revision is {revision}, expected {LHTB_REPOSITORY_REVISION}"
        )
    _validate_checkout_contents(root)
    try:
        from importlib.metadata import version

        import harbor
        from harbor._driftlock_pin import (
            DRIFTLOCK_HARBOR_PATCH_VERSION as installed_patch,
        )
        from harbor._driftlock_pin import (
            LHTB_REPOSITORY_REVISION as installed_revision,
        )
    except ImportError as error:
        raise PreflightError(
            "Harbor is not importable or the driftlock companion patch is absent"
        ) from error
    if installed_revision != LHTB_REPOSITORY_REVISION:
        raise PreflightError(
            "installed Harbor revision marker does not match driftlock"
        )
    if installed_patch != DRIFTLOCK_HARBOR_PATCH_VERSION:
        raise PreflightError("installed Harbor patch version does not match driftlock")
    imported_harbor = Path(harbor.__file__ or "").resolve()
    expected_harbor = (root / "harbor" / "src" / "harbor").resolve()
    if not imported_harbor.is_relative_to(expected_harbor):
        raise PreflightError(
            f"imported Harbor is outside the requested LHTB checkout: {imported_harbor}"
        )
    if version("litellm") != LHTB_LITELLM_VERSION:
        raise PreflightError(f"litellm must be exactly {LHTB_LITELLM_VERSION}")
    if require_credential and not os.environ.get(credential_env):
        raise PreflightError(
            f"credential environment variable {credential_env} is not set"
        )

    machine = platform.machine().lower()
    if machine not in {"amd64", "x86_64"}:
        raise PreflightError(
            f"native amd64 host required for measured runs; found {machine}"
        )
    docker_arch = "not-checked"
    if require_docker:
        if shutil.which("docker") is None:
            raise PreflightError("docker executable not found")
        docker_arch = _run_checked(
            ["docker", "info", "--format", "{{.Architecture}}"],
            cwd=root,
            description="query Docker daemon",
        ).lower()
        if docker_arch not in {"amd64", "x86_64"}:
            raise PreflightError(
                f"native amd64 Docker daemon required; found {docker_arch}"
            )
    return {
        "lhtb_revision": revision,
        "harbor_patch": str(installed_patch),
        "litellm": LHTB_LITELLM_VERSION,
        "machine": machine,
        "docker_arch": docker_arch,
        "credential_env": credential_env,
    }


def select_tasks(
    job_dirs: Sequence[Path],
    *,
    limit: int = 12,
    min_reward: float = 0.0,
    max_reward: float = 0.95,
) -> dict[str, Any]:
    """Rank tasks by measured mean partial credit from Harbor trial results."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not 0 <= min_reward < max_reward <= 1:
        raise ValueError("reward bounds must satisfy 0 <= min < max <= 1")
    rewards: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, list[str]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    for job_dir in job_dirs:
        root = job_dir.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for result_file in sorted(root.glob("*/result.json")):
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                task_name = data["task_name"]
                reward = _primary_reward(data)
                if not isinstance(task_name, str) or not task_name:
                    raise ValueError("missing task_name")
                if reward is None:
                    exception = data.get("exception_info") or {}
                    failures.append(
                        {
                            "task": task_name,
                            "result": str(result_file),
                            "reason": str(
                                exception.get("exception_type", "missing reward")
                            ),
                        }
                    )
                    continue
                rewards[task_name].append(reward)
                sources[task_name].append(str(result_file))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                failures.append(
                    {
                        "task": "unknown",
                        "result": str(result_file),
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
    measured = [
        {
            "task": task,
            "mean_reward": sum(values) / len(values),
            "attempts": len(values),
            "rewards": values,
            "result_files": sources[task],
        }
        for task, values in rewards.items()
    ]
    measured.sort(key=lambda item: (-item["mean_reward"], item["task"]))
    eligible = [
        item for item in measured if min_reward < item["mean_reward"] < max_reward
    ]
    return {
        "selection_rule": {
            "minimum_exclusive": min_reward,
            "maximum_exclusive": max_reward,
            "limit": limit,
        },
        "selected_tasks": [item["task"] for item in eligible[:limit]],
        "eligible": eligible,
        "all_measured": measured,
        "failures": failures,
    }


def probe_provider(
    *,
    model: str,
    provider: str,
    api_base: str,
    credential_env: str = DEFAULT_CREDENTIAL_ENV,
    timeout_sec: float = 60.0,
    opener: Any = None,
) -> None:
    """Ask *provider* for one token, so a dead pin fails now instead of in an hour.

    This costs a fraction of a cent and is the only paid call the launcher makes
    itself. It exists because on 2026-08-24 a four-arm round ran for three hours
    against a provider whose shared upstream pool had been returning 429
    continuously since before the round started, and produced nothing at all.
    """
    key = os.environ.get(credential_env)
    if not key:
        raise PreflightError(
            f"credential environment variable {credential_env} is not set"
        )
    body = json.dumps(
        {
            "model": model.removeprefix("openrouter/"),
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            **openrouter_provider_call_kwargs(provider)["extra_body"],
        }
    ).encode()
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    send = opener or urllib.request.urlopen
    try:
        with send(request, timeout=timeout_sec) as response:
            response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:200].replace("\n", " ")
        raise PreflightError(
            f"pinned provider {provider!r} for {model!r} answered HTTP "
            f"{error.code}: {detail}"
        ) from error
    except Exception as error:
        raise PreflightError(
            f"pinned provider {provider!r} for {model!r} is unreachable: "
            f"{type(error).__name__}: {error}"
        ) from error


def _build_skill_distiller(
    *,
    model: str,
    provider: str,
    api_base: str,
    max_output_tokens: int,
    timeout_sec: float,
) -> tuple[CallableSkillDistiller, Any, dict[str, Any]]:
    """Reuse the pinned fine-judge client for one-call skill completions."""

    # Harbor is intentionally an experiment-environment dependency, not a core
    # package dependency.  Keep this import behind the paid command just as the
    # existing agent and fine-judge provider path does.
    from driftlock.harbor_agent import _LHTBJudgeClient

    client = _LHTBJudgeClient(
        model=model,
        api_base=api_base,
        max_output_tokens=max_output_tokens,
        timeout_sec=timeout_sec,
        llm_call_kwargs=openrouter_provider_call_kwargs(provider),
        raise_provider_errors=True,
    )

    async def complete(prompt: str) -> Any:
        return await client.complete(prompt, tokens_remaining=None)

    def usage_reader() -> ReplayUsage:
        return ReplayUsage(
            input_tokens=client.n_input_tokens,
            cache_tokens=client.n_cache_tokens,
            output_tokens=client.n_output_tokens,
            cost_usd=client.cost_usd,
        )

    usage_reader.driftlock_accounting_reader = client.accounting_snapshot  # type: ignore[attr-defined]

    return (
        CallableSkillDistiller(complete),
        usage_reader,
        _skill_distillation_metadata(
            model=model,
            provider=provider,
            api_base=api_base,
            max_output_tokens=max_output_tokens,
            timeout_sec=timeout_sec,
        ),
    )


def _skill_distillation_metadata(
    *,
    model: str,
    provider: str,
    api_base: str,
    max_output_tokens: int,
    timeout_sec: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "provider": provider,
        "api_base": api_base,
        "max_output_tokens": max_output_tokens,
        "timeout_sec": timeout_sec,
    }


class _HarborSkillValidationRunner:
    """Run one isolated validation trial and verify its injection provenance."""

    def __init__(
        self,
        *,
        lhtb_dir: Path,
        work_dir: Path,
        skill_embedder_import_path: str,
        model: str,
        provider: str,
        api_base: str,
        judge_api_base: str | None,
        judge_provider: str,
        timeout_sec: int,
        max_total_tokens: int,
    ) -> None:
        self.lhtb_dir = lhtb_dir.expanduser().resolve()
        self.work_dir = work_dir.expanduser().resolve()
        self.skill_embedder_import_path = skill_embedder_import_path
        self.model = model
        self.provider = provider
        self.api_base = api_base
        self.judge_api_base = judge_api_base
        self.judge_provider = judge_provider
        self.timeout_sec = timeout_sec
        self.max_total_tokens = max_total_tokens

    async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
        configs_dir = self.work_dir / "configs"
        jobs_dir = self.work_dir / "jobs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        config_path = configs_dir / f"{trial.job_name}.json"
        job_dir = jobs_dir / trial.job_name
        config = build_job_config(
            lhtb_dir=self.lhtb_dir,
            jobs_dir=jobs_dir,
            job_name=trial.job_name,
            arm=f"skill-{trial.distillation_arm}",
            tasks=[trial.task_name],
            model=self.model,
            provider=self.provider,
            api_base=self.api_base,
            n_concurrent_trials=1,
            timeout_sec=self.timeout_sec,
            max_total_tokens=self.max_total_tokens,
            judge_api_base=self.judge_api_base,
            judge_provider=self.judge_provider,
            skill_library_dir=trial.library_dir,
            skill_embedder_import_path=self.skill_embedder_import_path,
        )
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        recovered = (job_dir / "result.json").is_file()
        exit_code: int | None = None
        if not recovered:
            child_env = os.environ.copy()
            child_env.pop("HB_PROCESS_REWARD", None)
            child_env["HB_CONTINUE_MODE"] = "same_conversation"
            exit_code = await _run_validation_process(
                [*_pinned_harbor_command(), "run", "-c", str(config_path)],
                cwd=self.lhtb_dir,
                env=child_env,
            )
        audit: dict[str, Any] = {
            "job_name": trial.job_name,
            "job_dir": str(job_dir),
            "config": str(config_path),
            "recovered_existing_job": recovered,
            "process_exit_code": exit_code,
        }
        try:
            run_record = _validation_run_record(job_dir)
        except _ValidationJobEvidenceError as error:
            return ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason=str(error),
                failure_kind=error.failure_kind,
                audit=audit,
            )
        try:
            reward = (
                single_job_reward(job_dir)
                if (job_dir / "result.json").is_file()
                else None
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            return ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason=f"validation reward evidence is unusable: {error}",
                failure_kind=ValidationFailureKind.REWARD_EVIDENCE,
                audit=audit,
            )
        if reward is None:
            detail = (
                f"process exited {exit_code}"
                if exit_code is not None
                else "job recovered"
            )
            return ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason=f"validation job produced no reward ({detail})",
                failure_kind=ValidationFailureKind.NO_REWARD,
                audit=audit,
            )
        try:
            injected_ids, injection_audit = _validation_injection_evidence(
                job_dir, run_record=run_record
            )
            audit["skill_injection"] = injection_audit
        except (OSError, ValueError) as error:
            return ValidationTrialResult(
                status=ValidationTrialStatus.FAILED,
                reason=f"validation skill-layer evidence is unusable: {error}",
                failure_kind=ValidationFailureKind.SKILL_LAYER_EVIDENCE,
                audit=audit,
            )
        return ValidationTrialResult(
            status=ValidationTrialStatus.MEASURED,
            reward=reward,
            injected_candidate_ids=injected_ids,
            audit=audit,
        )


_VALIDATION_PROCESS_INTERRUPT_GRACE_SEC = 10.0
_VALIDATION_PROCESS_TERMINATE_GRACE_SEC = 5.0
_VALIDATION_PROCESS_WRAPPER = """
import os
import sys

pid_fd = int(sys.argv[1])
os.setsid()
os.write(pid_fd, f"{os.getpid()}\\n".encode())
os.close(pid_fd)
os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
"""


async def _run_validation_process(
    command: Sequence[str], *, cwd: Path, env: dict[str, str]
) -> int:
    """Run Harbor without blocking validation workers and reap it on cancellation."""

    pid_reader, pid_writer = os.pipe()
    os.set_blocking(pid_reader, False)

    def run() -> subprocess.CompletedProcess[Any]:
        try:
            return subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _VALIDATION_PROCESS_WRAPPER,
                    str(pid_writer),
                    *command,
                ],
                cwd=cwd,
                check=False,
                env=env,
                pass_fds=(pid_writer,),
            )
        finally:
            os.close(pid_writer)

    run_task = asyncio.create_task(asyncio.to_thread(run))
    pid_task = asyncio.create_task(_read_validation_process_pid(pid_reader, run_task))
    process_group: int | None = None
    try:
        process_group = await asyncio.shield(pid_task)
        completed = await asyncio.shield(run_task)
        return completed.returncode
    except BaseException:
        if process_group is None:
            process_group = await asyncio.shield(pid_task)
        if process_group is not None:
            await _stop_validation_process_group(process_group, run_task)
        else:
            await asyncio.shield(run_task)
        raise
    finally:
        os.close(pid_reader)


async def _stop_validation_process_group(
    process_group: int,
    run_task: asyncio.Task[subprocess.CompletedProcess[Any]],
) -> None:
    """Give Harbor its interrupt teardown, then ensure its process group is gone."""

    if _signal_validation_process_group(
        process_group, signal.SIGINT
    ) and await _wait_for_validation_process(
        process_group, run_task, _VALIDATION_PROCESS_INTERRUPT_GRACE_SEC
    ):
        return
    if _signal_validation_process_group(
        process_group, signal.SIGTERM
    ) and await _wait_for_validation_process(
        process_group, run_task, _VALIDATION_PROCESS_TERMINATE_GRACE_SEC
    ):
        return
    _signal_validation_process_group(process_group, signal.SIGKILL)
    await asyncio.shield(run_task)


def _signal_validation_process_group(
    process_group: int, process_signal: signal.Signals
) -> bool:
    try:
        os.killpg(process_group, process_signal)
    except ProcessLookupError:
        return False
    return True


async def _wait_for_validation_process(
    process_group: int,
    run_task: asyncio.Task[subprocess.CompletedProcess[Any]],
    timeout_sec: float,
) -> bool:
    try:
        async with asyncio.timeout(timeout_sec):
            await asyncio.shield(run_task)
            while _validation_process_group_exists(process_group):
                await asyncio.sleep(0.05)
    except TimeoutError:
        return False
    return True


def _validation_process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _read_validation_process_pid(
    pid_reader: int,
    run_task: asyncio.Task[subprocess.CompletedProcess[Any]],
) -> int | None:
    payload = bytearray()
    while b"\n" not in payload:
        try:
            chunk = os.read(pid_reader, 64)
        except BlockingIOError:
            chunk = None
        if chunk:
            payload.extend(chunk)
            continue
        if chunk == b"" or run_task.done():
            return None
        await asyncio.sleep(0)
    return int(payload.partition(b"\n")[0])


class _ValidationJobEvidenceError(ValueError):
    """A job-level result-cardinality failure with stable attribution."""

    def __init__(self, message: str, failure_kind: ValidationFailureKind) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


def _validation_run_record(job_dir: Path) -> Path:
    records = sorted(job_dir.glob("*/agent/driftlock-result.json"))
    if not records:
        raise _ValidationJobEvidenceError(
            "validation trial did not produce a result: no run record was found",
            ValidationFailureKind.DID_NOT_PRODUCE_RESULT,
        )
    if len(records) != 1:
        raise _ValidationJobEvidenceError(
            f"validation job has {len(records)} run records; expected one",
            ValidationFailureKind.AMBIGUOUS_RESULT,
        )
    return records[0]


def _validation_injection_evidence(
    job_dir: Path,
    *,
    run_record: Path | None = None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    path = run_record if run_record is not None else _validation_run_record(job_dir)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid skill-injection record: {path}") from error
    skill_layer = record.get("skill_layer") if isinstance(record, dict) else None
    injection = skill_layer.get("injection") if isinstance(skill_layer, dict) else None
    candidate_ids = (
        injection.get("candidate_ids") if isinstance(injection, dict) else None
    )
    if not isinstance(candidate_ids, list) or any(
        not isinstance(candidate_id, str) for candidate_id in candidate_ids
    ):
        raise ValueError(f"skill-injection record has invalid candidate ids: {path}")
    return tuple(candidate_ids), {
        "record": str(path),
        "status": injection.get("status"),
        "candidate_ids": candidate_ids,
        "distillation_arm": skill_layer.get("distillation_arm"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            report = preflight(
                args.lhtb_dir,
                credential_env=args.credential_env,
                require_credential=not args.no_credential,
                require_docker=not args.no_docker,
            )
            print(json.dumps(report, indent=2))
            return 0
        if args.command in {"prepare", "run"}:
            if args.command == "run":
                preflight(
                    args.lhtb_dir,
                    credential_env=args.credential_env,
                )
                if args.arm == "stock" and not args.ack_unbounded_stock_tokens:
                    raise PreflightError(
                        "stock Terminus has no total-token ceiling; pass "
                        "--ack-unbounded-stock-tokens after setting a provider "
                        "spend cap"
                    )
                if not args.no_provider_probe:
                    probe_provider(
                        model=args.model,
                        provider=args.provider,
                        api_base=args.api_base,
                        credential_env=args.credential_env,
                    )
                    if args.arm in _FINE_JUDGE_ARMS:
                        probe_provider(
                            model=DEFAULT_JUDGE_MODEL,
                            provider=args.judge_provider,
                            api_base=args.judge_api_base or args.api_base,
                            credential_env=args.credential_env,
                        )
            config = build_job_config(
                lhtb_dir=args.lhtb_dir,
                jobs_dir=args.jobs_dir,
                job_name=args.job_name,
                arm=args.arm,
                tasks=args.tasks,
                model=args.model,
                provider=args.provider,
                api_base=args.api_base,
                n_concurrent_trials=args.concurrency,
                timeout_sec=args.timeout_sec,
                max_total_tokens=args.max_total_tokens,
                judge_api_base=args.judge_api_base,
                judge_provider=args.judge_provider,
                retain_checkpoints=args.retain_checkpoints,
                skill_library_dir=args.skill_library_dir,
                skill_embedder_import_path=args.skill_embedder,
            )
            config_arg = args.config or Path(f"driftlock-job-{args.job_name}.json")
            config_path = config_arg.expanduser().resolve()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
            print(f"wrote {config_path}")
            if args.command == "prepare":
                return 0
            # Re-read what Harbor is actually about to run. An explicit --config
            # shared between concurrent arms, or any other writer, would otherwise
            # hand this process someone else's arm without a word.
            written = json.loads(config_path.read_text(encoding="utf-8"))
            if written != config:
                raise SystemExit(
                    f"{config_path} changed after it was written; another job is "
                    "using the same --config path. Give each concurrent run its "
                    "own --config, or omit --config to get a per-job default."
                )
            harbor_command = _pinned_harbor_command()
            child_env = os.environ.copy()
            child_env.pop("HB_PROCESS_REWARD", None)
            if args.arm != "stock":
                child_env["HB_CONTINUE_MODE"] = "same_conversation"
            else:
                child_env.pop("HB_CONTINUE_MODE", None)
            completed = subprocess.run(
                [*harbor_command, "run", "-c", str(config_path)],
                cwd=args.lhtb_dir.expanduser().resolve(),
                check=False,
                env=child_env,
            )
            return completed.returncode
        if args.command in {"oracle-prepare", "oracle-run"}:
            if args.command == "oracle-run":
                preflight(
                    args.lhtb_dir,
                    credential_env=DEFAULT_CREDENTIAL_ENV,
                    require_credential=False,
                )
            report = prepare_oracle_replays(
                lhtb_dir=args.lhtb_dir,
                source_job_dir=args.source_job_dir,
                output_dir=args.output_dir,
                timeout_sec=args.timeout_sec,
            )
            print(
                f"wrote {report['candidate_count']} replay configs and "
                f"{args.output_dir.expanduser().resolve() / 'oracle-manifest.json'}"
            )
            if args.command == "oracle-prepare":
                return 0
            harbor_command = _pinned_harbor_command()
            child_env = os.environ.copy()
            child_env.pop("HB_PROCESS_REWARD", None)
            child_env["HB_CONTINUE_MODE"] = "same_conversation"
            for candidate in report["candidates"]:
                completed = subprocess.run(
                    [*harbor_command, "run", "-c", candidate["config"]],
                    cwd=args.lhtb_dir.expanduser().resolve(),
                    check=False,
                    env=child_env,
                )
                if completed.returncode != 0:
                    return completed.returncode
            return 0
        if args.command in {"score-checkpoints", "checkpoint-score"}:
            plan = enumerate_retained_checkpoints(args.source_job_dir)
            destination = args.output_dir.expanduser().resolve()
            if destination == plan.source_job_dir or (
                plan.source_job_dir in destination.parents
            ):
                raise ValueError(
                    "checkpoint score output must be outside the source job"
                )
            print(
                f"{len(plan.checkpoints)} checkpoint replays planned across "
                f"{sum(bool(trial.checkpoints) for trial in plan.trials)} trials"
            )
            for trial in plan.trials:
                if not trial.checkpoints:
                    print(f"  {trial.trial_name}: no retained checkpoints")
                    continue
                points = [
                    f"phase {checkpoint.phase} step {checkpoint.step}"
                    for checkpoint in trial.checkpoints
                ]
                print(f"  {trial.trial_name}: " + ", ".join(points))
            if args.dry_run:
                return 0

            preflight(
                args.lhtb_dir,
                credential_env=DEFAULT_CREDENTIAL_ENV,
                require_credential=False,
            )
            configs_dir = destination / "configs"
            jobs_dir = destination / "jobs"
            configs_dir.mkdir(parents=True, exist_ok=True)
            jobs_dir.mkdir(parents=True, exist_ok=True)
            report_path = destination / SCORE_REPORT_NAME
            scores = load_completed_scores(report_path, plan)
            write_score_report(report_path, plan, scores)
            harbor_command = _pinned_harbor_command()
            child_env = os.environ.copy()
            child_env.pop("HB_PROCESS_REWARD", None)
            child_env["HB_CONTINUE_MODE"] = "same_conversation"
            failed = False
            for index, replay in enumerate(plan.checkpoints, 1):
                if replay.candidate_id in scores:
                    print(
                        f"[{index}/{len(plan.checkpoints)}] {replay.trial_name} "
                        f"phase {replay.phase} step {replay.step}: "
                        f"already scored ({scores[replay.candidate_id]:.6g})"
                    )
                    continue
                replay_job_dir = jobs_dir / replay.job_name
                if (replay_job_dir / "result.json").is_file():
                    reward = single_job_reward(replay_job_dir)
                    if reward is not None:
                        scores[replay.candidate_id] = reward
                        write_score_report(report_path, plan, scores)
                        print(
                            f"[{index}/{len(plan.checkpoints)}] "
                            f"{replay.trial_name} phase {replay.phase} "
                            f"step {replay.step}: recovered {reward:.6g}"
                        )
                        continue
                config = build_checkpoint_replay_config(
                    lhtb_dir=args.lhtb_dir,
                    jobs_dir=jobs_dir,
                    replay=replay,
                    timeout_sec=args.timeout_sec,
                )
                config_path = configs_dir / f"{replay.candidate_id}.json"
                config_path.write_text(
                    json.dumps(config, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"[{index}/{len(plan.checkpoints)}] {replay.trial_name} "
                    f"phase {replay.phase} step {replay.step}: running"
                )
                completed = subprocess.run(
                    [*harbor_command, "run", "-c", str(config_path)],
                    cwd=args.lhtb_dir.expanduser().resolve(),
                    check=False,
                    env=child_env,
                )
                reward = (
                    single_job_reward(replay_job_dir)
                    if (replay_job_dir / "result.json").is_file()
                    else None
                )
                if reward is not None:
                    scores[replay.candidate_id] = reward
                    write_score_report(report_path, plan, scores)
                    print(f"  reward {reward:.6g}")
                    continue
                if completed.returncode != 0:
                    failed = True
                    print(f"  replay failed with exit code {completed.returncode}")
                    continue
                failed = True
                print("  replay completed without a job-level reward")

            report = assemble_scored_timelines(plan, scores)
            for trial in report["trials"]:
                best = trial["best_checkpoint_reward"]
                final = trial["final_reward"]
                if best is None:
                    print(f"  {trial['trial_name']}: no checkpoint scored")
                elif final is None:
                    print(
                        f"  {trial['trial_name']}: best checkpoint {best:.6g}; "
                        "final reward unknown"
                    )
                else:
                    print(
                        f"  {trial['trial_name']}: best checkpoint {best:.6g} "
                        f"vs final {final:.6g}; headroom {trial['headroom']:+.6g}"
                    )
            print(f"wrote {report_path}")
            return 1 if failed else 0
        if args.command == "localize-checkpoints":
            report = load_and_localize_score_report(args.score_report)
            output = args.output.expanduser().resolve()
            write_localization_report(output, report)
            print(
                f"usable {report['usable_task_count']}/{report['task_count']} "
                f"tasks ({report['coverage']:.1%}); "
                f"refused {report['refused_task_count']}"
            )
            for task in report["tasks"]:
                if task["status"] == "refused":
                    refusal = task["refusal"]
                    print(
                        f"  {task['task_name']}: refused "
                        f"({refusal['reason']}): {refusal['detail']}"
                    )
                    continue
                print(
                    f"  {task['task_name']}: {len(task['segments'])} "
                    "non-improving segment(s)"
                )
                for segment in task["segments"]:
                    start = segment["start"]
                    end = segment["end"]
                    print(
                        f"    {segment['type']}: phase {start['phase']} step "
                        f"{start['step']} -> phase {end['phase']} step "
                        f"{end['step']} ({segment['reward_change']:+.6g})"
                    )
            print(f"wrote {output}")
            return 0
        if args.command == "distill-skills":
            plan = plan_skill_distillation(args.localization_report)
            output = args.output.expanduser().resolve()
            metadata = _skill_distillation_metadata(
                model=args.model,
                provider=args.provider,
                api_base=args.api_base,
                max_output_tokens=args.max_output_tokens,
                timeout_sec=args.timeout_sec,
            )

            class _UncallableDistiller:
                async def distill(self, evidence: Any) -> Any:
                    del evidence
                    raise AssertionError("zero-call run must not invoke the distiller")

            if args.dry_run or not plan.work_items:
                report = asyncio.run(
                    run_skill_distillation(
                        plan,
                        output,
                        distiller=_UncallableDistiller(),
                        usage_reader=lambda: ReplayUsage(0, 0, 0, 0.0),
                        run_metadata=metadata,
                        dry_run=args.dry_run,
                    )
                )
            else:
                distiller, usage_reader, metadata = _build_skill_distiller(
                    model=args.model,
                    provider=args.provider,
                    api_base=args.api_base,
                    max_output_tokens=args.max_output_tokens,
                    timeout_sec=args.timeout_sec,
                )
                report = asyncio.run(
                    run_skill_distillation(
                        plan,
                        output,
                        distiller=distiller,
                        usage_reader=usage_reader,
                        run_metadata=metadata,
                    )
                )
            summary = report["summary"]
            if args.dry_run:
                print(
                    f"{summary['pending_call_count']} model call(s) to make; "
                    f"{summary['planned_call_count']} planned across "
                    f"{len(plan.work_items) // len(DISTILLATION_ARMS)} segment(s); "
                    f"{summary['reused_call_count']} reused; "
                    f"{summary['evidence_refusal_count']} evidence-refused "
                    "segment(s)"
                )
                for segment in plan.evidence_sizes:
                    rendered_arms = []
                    for arm in DISTILLATION_ARMS:
                        size = segment["arms"][arm]
                        if size["characters"] is None:
                            rendered = "unavailable"
                        else:
                            rendered = f"{size['characters']:,} chars"
                            if size["status"] == "at_or_over_bound":
                                rendered += " [AT/OVER BOUND]"
                        rendered_arms.append(f"{arm}={rendered}")
                    print(
                        f"  {segment['task_name']} segment "
                        f"{segment['segment_index']} evidence: "
                        + "; ".join(rendered_arms)
                    )
            else:
                print(
                    f"completed {summary['completed_call_count']} model call(s); "
                    f"{summary['new_call_count']} new, "
                    f"{summary['reused_call_count']} reused; "
                    f"{len(plan.work_items) // len(DISTILLATION_ARMS)} callable "
                    f"segment(s); {summary['evidence_refusal_count']} "
                    "evidence-refused segment(s); "
                    f"{summary['retryable_failure_count']} retryable failure(s); "
                    f"{summary['unpaired_segment_count']} unpaired segment(s)"
                )
            completed = {
                attempt["candidate_id"]: attempt for attempt in report["attempts"]
            }
            for item in plan.work_items:
                attempt = completed.get(item.candidate_id)
                disposition = (
                    f"{attempt['status']}" + (" (reused)" if attempt["reused"] else "")
                    if attempt is not None
                    else "planned"
                )
                print(
                    f"  {item.task_name} segment {item.segment_index} "
                    f"{item.arm}: {disposition}"
                )
            for task in report["refused_tasks"]:
                refusal = task["refusal"]
                print(
                    f"  {task['task_name']}: localization refused "
                    f"({refusal['reason']}): {refusal['detail']}"
                )
            for segment in report["evidence_refusals"]:
                print(
                    f"  {segment['task_name']} segment "
                    f"{segment['segment_index']}: evidence refused; "
                    f"{segment['detail']}"
                )
                for arm in DISTILLATION_ARMS:
                    refusal = segment["refusals_by_arm"].get(arm)
                    if refusal is not None:
                        print(f"    {arm} ({refusal['reason']}): {refusal['detail']}")
            if args.dry_run:
                print("dry run; no provider calls made and no output written")
            else:
                print(f"wrote {output}")
            # Treat partial evidence refusal, retryable provider failure, and a
            # one-sided generated outcome as failures. Any of them would otherwise
            # let automation validate/admit a different paired cohort than the
            # localization report requested. A fully generated cohort and a genuine
            # nothing-to-do report both remain successful: neither misrepresents
            # arm membership. Persisted successful calls are reused on repair, while
            # failed calls remain audited and are retried automatically.
            invalid_cohort = any(
                summary[key] > 0
                for key in (
                    "evidence_refusal_count",
                    "retryable_failure_count",
                    "unpaired_segment_count",
                )
            )
            return 1 if invalid_cohort else 0
        if args.command == "validate-skills":
            plan = plan_skill_validation(
                args.candidate_file,
                args.lhtb_dir,
                estimated_cost_per_trial_usd=args.estimated_cost_per_trial,
            )
            output = args.output.expanduser().resolve()
            work_dir = (
                args.work_dir.expanduser().resolve()
                if args.work_dir is not None
                else output.with_name(f"{output.stem}-work")
            )
            validation_run_metadata = {
                "model": args.model,
                "provider": args.provider,
                "api_base": args.api_base,
                "judge_model": DEFAULT_JUDGE_MODEL,
                "judge_provider": args.judge_provider,
                "judge_api_base": args.judge_api_base or args.api_base,
                "skill_embedder_import_path": args.skill_embedder,
                "timeout_sec": args.timeout_sec,
                "max_total_tokens": args.max_total_tokens,
            }

            class _UncallableValidationRunner:
                async def run(self, trial: ValidationTrial) -> ValidationTrialResult:
                    del trial
                    raise AssertionError("dry run must not invoke the trial runner")

            if args.dry_run:
                runner: Any = _UncallableValidationRunner()
            else:
                if not args.skill_embedder:
                    raise ValueError("a paid validation run requires --skill-embedder")
                preflight(args.lhtb_dir, credential_env=args.credential_env)
                if not args.no_provider_probe:
                    probe_provider(
                        model=args.model,
                        provider=args.provider,
                        api_base=args.api_base,
                        credential_env=args.credential_env,
                    )
                    probe_provider(
                        model=DEFAULT_JUDGE_MODEL,
                        provider=args.judge_provider,
                        api_base=args.judge_api_base or args.api_base,
                        credential_env=args.credential_env,
                    )
                runner = _HarborSkillValidationRunner(
                    lhtb_dir=args.lhtb_dir,
                    work_dir=work_dir,
                    skill_embedder_import_path=args.skill_embedder,
                    model=args.model,
                    provider=args.provider,
                    api_base=args.api_base,
                    judge_api_base=args.judge_api_base,
                    judge_provider=args.judge_provider,
                    timeout_sec=args.timeout_sec,
                    max_total_tokens=args.max_total_tokens,
                )
            report = asyncio.run(
                run_skill_validation(
                    plan,
                    output,
                    runner=runner,
                    work_dir=work_dir,
                    run_metadata=validation_run_metadata,
                    max_concurrent_trials=args.max_concurrent_trials,
                    dry_run=args.dry_run,
                )
            )
            summary = report["validation"]["summary"]
            print(
                f"{summary['planned_trial_count']} trial(s) planned: "
                f"{summary['shared_control_trial_count']} shared no-skill + "
                f"{summary['treatment_trial_count']} with-skill; estimated "
                f"${summary['estimated_planned_cost_usd']:.3f} at "
                f"${summary['estimated_cost_per_trial_usd']:.3f}/trial; "
                f"{len(plan.tasks)} distinct source task(s) x "
                f"{plan.replicate_count} replicate(s) = "
                f"{plan.shared_control_trial_count} shared control trial(s); "
                f"{len(plan.candidates)} candidate(s) x "
                f"{plan.observation_count} own-task observation(s) = "
                f"{plan.paired_observation_count} paired observation(s); "
                f"{summary['pending_trial_count']} pending, "
                f"{summary['reused_trial_count']} reused; up to "
                f"{args.max_concurrent_trials} concurrent trial(s)"
            )
            print("  shared no-skill controls:")
            for task in plan.tasks:
                for replicate_index in range(1, plan.replicate_count + 1):
                    replicate = (
                        ""
                        if plan.replicate_count == 1
                        else f" replicate {replicate_index}"
                    )
                    print(
                        f"    {task}{replicate}: 1 trial, estimated "
                        f"${plan.estimated_cost_per_trial_usd:.3f}"
                    )
            for candidate in plan.candidates:
                print(
                    f"  {candidate.candidate_id} [{candidate.arm}] from "
                    f"{candidate.source_task_name}:"
                )
                for replicate_index in range(1, plan.replicate_count + 1):
                    replicate = f" replicate {replicate_index}"
                    print(
                        f"    {candidate.task_name}{replicate}: 1 with-skill trial, "
                        f"estimated ${plan.estimated_cost_per_trial_usd:.3f}"
                    )
            if args.dry_run:
                print("dry run; no trials run and no files written")
                return 0
            latest = {
                attempt["trial_id"]: attempt
                for attempt in report["validation"]["attempts"]
            }
            for trial in plan.work_items(work_dir):
                attempt = latest.get(trial.trial_id)
                if attempt is None:
                    disposition = "pending"
                else:
                    disposition = attempt["status"]
                    if attempt["reused"]:
                        disposition += " (reused)"
                label = trial.candidate_id or "shared no-skill"
                print(
                    f"  {label} / {trial.task_name} / replicate "
                    f"{trial.replicate_index}: {disposition}"
                )
            print(f"wrote {output}")
            return 1 if summary["pending_trial_count"] else 0
        if args.command == "admit-skills":
            candidates = load_admission_candidates(args.validation_results)
            library = SkillLibrary(args.library_dir)
            report = library.submit_cohort(candidates)
            output = args.output.expanduser().resolve()
            write_admission_report(output, report)
            print(render_admission_report(report))
            print(f"wrote {output}")
            return 0
        if args.command == "select":
            report = select_tasks(
                args.job_dirs,
                limit=args.limit,
                min_reward=args.min_reward,
                max_reward=args.max_reward,
            )
            serialized = json.dumps(report, indent=2) + "\n"
            args.output.expanduser().resolve().write_text(serialized, encoding="utf-8")
            print(serialized, end="")
            return 0
        if args.command == "analyze":
            report = analyze_jobs(
                lhtb_dir=args.lhtb_dir,
                arm_directories=parse_arm_directories(args.arm_dirs),
                solve_threshold=args.solve_threshold,
                require_complete_matrix=not args.allow_incomplete_matrix,
                exclude_dead_tasks=args.exclude_dead_tasks,
            )
            serialized = json.dumps(report, indent=2) + "\n"
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized, encoding="utf-8")
            print(serialized, end="")
            return 0
    except (FileNotFoundError, OSError, PreflightError, ValueError) as error:
        parser.error(str(error))
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driftlock-lhtb")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight", help="verify the pinned paid-run environment")
    check.add_argument("--lhtb-dir", type=Path, default=Path.cwd())
    check.add_argument("--credential-env", default=DEFAULT_CREDENTIAL_ENV)
    check.add_argument("--no-credential", action="store_true")
    check.add_argument("--no-docker", action="store_true")
    for name in ("prepare", "run"):
        command = sub.add_parser(name, help=f"{name} an exact Harbor job")
        command.add_argument("--lhtb-dir", type=Path, default=Path.cwd())
        command.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
        # Defaults to a per-job filename. A fixed shared path silently merges
        # concurrent arms: on 2026-08-24 four arms launched together each wrote
        # driftlock-job.json and then all four ran the file the last writer left,
        # so every arm executed as driftlock into one job directory.
        command.add_argument("--config", type=Path, default=None)
        command.add_argument("--no-provider-probe", action="store_true")
        command.add_argument("--job-name", required=True)
        command.add_argument("--arm", choices=RUNNABLE_ARMS, required=True)
        command.add_argument("--tasks", nargs="+", required=True)
        command.add_argument("--model", default=DEFAULT_MODEL)
        command.add_argument("--provider", default=DEFAULT_PROVIDER)
        command.add_argument("--api-base", default=DEFAULT_API_BASE)
        command.add_argument("--judge-api-base")
        command.add_argument("--judge-provider", default=DEFAULT_JUDGE_PROVIDER)
        command.add_argument("--credential-env", default=DEFAULT_CREDENTIAL_ENV)
        command.add_argument("--concurrency", type=int, default=1)
        command.add_argument("--timeout-sec", type=int, default=5400)
        command.add_argument("--max-total-tokens", type=int, default=10_000_000)
        command.add_argument("--ack-unbounded-stock-tokens", action="store_true")
        command.add_argument("--retain-checkpoints", action="store_true")
        command.add_argument("--skill-library-dir", type=Path)
        command.add_argument(
            "--skill-embedder",
            help="local embedding callable as module:callable (skill arms only)",
        )
    for name in ("oracle-prepare", "oracle-run"):
        oracle = sub.add_parser(
            name, help=f"{name} isolated retained-checkpoint verifier jobs"
        )
        oracle.add_argument("--lhtb-dir", type=Path, default=Path.cwd())
        oracle.add_argument("--source-job-dir", type=Path, required=True)
        oracle.add_argument("--output-dir", type=Path, required=True)
        oracle.add_argument("--timeout-sec", type=int, default=900)
    scoring = sub.add_parser(
        "score-checkpoints",
        aliases=["checkpoint-score"],
        help="score retained checkpoints with fresh hidden-verifier jobs",
    )
    scoring.add_argument("--lhtb-dir", type=Path, default=Path.cwd())
    scoring.add_argument("--source-job-dir", type=Path, required=True)
    scoring.add_argument("--output-dir", type=Path, required=True)
    scoring.add_argument("--timeout-sec", type=int, default=900)
    scoring.add_argument("--dry-run", action="store_true")
    localization = sub.add_parser(
        "localize-checkpoints",
        help="find non-improving segments in a checkpoint score report",
    )
    localization.add_argument("score_report", type=Path)
    localization.add_argument(
        "--output", type=Path, default=Path(LOCALIZATION_REPORT_NAME)
    )
    distillation = sub.add_parser(
        "distill-skills",
        help="distill paired localized and whole-trajectory skill candidates",
    )
    distillation.add_argument("localization_report", type=Path)
    distillation.add_argument(
        "--output", type=Path, default=Path(DISTILLATION_REPORT_NAME)
    )
    distillation.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    distillation.add_argument("--provider", default=DEFAULT_JUDGE_PROVIDER)
    distillation.add_argument("--api-base", default=DEFAULT_API_BASE)
    distillation.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_JUDGE_MAX_OUTPUT_TOKENS
    )
    distillation.add_argument("--timeout-sec", type=float, default=120.0)
    distillation.add_argument("--dry-run", action="store_true")
    validation = sub.add_parser(
        "validate-skills",
        help="measure resumable own-task paired deltas for skill candidates",
    )
    validation.add_argument("candidate_file", type=Path)
    validation.add_argument("--lhtb-dir", type=Path, default=Path.cwd())
    validation.add_argument("--output", type=Path, default=Path(VALIDATION_REPORT_NAME))
    validation.add_argument("--work-dir", type=Path)
    validation.add_argument(
        "--skill-embedder",
        help="local embedding callable as module:callable (required for paid runs)",
    )
    validation.add_argument("--model", default=DEFAULT_MODEL)
    validation.add_argument("--provider", default=DEFAULT_PROVIDER)
    validation.add_argument("--api-base", default=DEFAULT_API_BASE)
    validation.add_argument("--judge-api-base")
    validation.add_argument("--judge-provider", default=DEFAULT_JUDGE_PROVIDER)
    validation.add_argument("--credential-env", default=DEFAULT_CREDENTIAL_ENV)
    validation.add_argument("--no-provider-probe", action="store_true")
    validation.add_argument("--timeout-sec", type=int, default=5400)
    validation.add_argument("--max-total-tokens", type=int, default=10_000_000)
    validation.add_argument(
        "--estimated-cost-per-trial",
        type=float,
        default=DEFAULT_ESTIMATED_COST_PER_TRIAL_USD,
    )
    validation.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_TRIALS,
        help=(
            "maximum Harbor validation jobs in flight (default: 4; higher values "
            "increase provider load and upstream 429 risk)"
        ),
    )
    validation.add_argument("--dry-run", action="store_true")
    admission = sub.add_parser(
        "admit-skills",
        help="apply the paired validation rule and update a skill library",
    )
    admission.add_argument("validation_results", type=Path)
    admission.add_argument("--library-dir", type=Path, required=True)
    admission.add_argument("--output", type=Path, default=Path(ADMISSION_REPORT_NAME))
    choose = sub.add_parser("select", help="select tasks by measured partial credit")
    choose.add_argument("job_dirs", nargs="+", type=Path)
    choose.add_argument("--limit", type=int, default=12)
    choose.add_argument("--min-reward", type=float, default=0.0)
    choose.add_argument("--max-reward", type=float, default=0.95)
    choose.add_argument("--output", type=Path, default=Path("selected-tasks.json"))
    analyze = sub.add_parser(
        "analyze", help="aggregate auditable multi-arm Harbor results"
    )
    analyze.add_argument("--lhtb-dir", type=Path, default=Path.cwd())
    analyze.add_argument(
        "--arm-dir",
        dest="arm_dirs",
        action="append",
        required=True,
        metavar="ARM=JOB_DIR",
    )
    analyze.add_argument("--solve-threshold", type=float, default=0.95)
    analyze.add_argument("--allow-incomplete-matrix", action="store_true")
    analyze.add_argument(
        "--exclude-dead-tasks",
        action="store_true",
        help=(
            "exclude every arm's result for a task with a non-timeout dead trial; "
            "refuse concentrated deaths or fewer than three surviving tasks"
        ),
    )
    analyze.add_argument("--output", type=Path, default=Path("analysis.json"))
    return parser


def _validate_job_name(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError("job_name contains unsafe characters")


def _validate_tasks(task_root: Path, tasks: Sequence[str]) -> list[str]:
    if not task_root.is_dir():
        raise FileNotFoundError(task_root)
    if not tasks:
        raise ValueError("at least one task is required")
    selected: list[str] = []
    for task in tasks:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task):
            raise ValueError(f"unsafe task name: {task!r}")
        if not (task_root / task / "task.toml").is_file():
            raise ValueError(f"unknown LHTB task: {task}")
        if task not in selected:
            selected.append(task)
    return selected


def _task_directory_name(task_root: Path, canonical_name: str) -> str:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
        canonical_name,
    ):
        raise ValueError(f"invalid canonical task name: {canonical_name!r}")
    matches: list[str] = []
    for task_file in sorted(task_root.glob("*/task.toml")):
        try:
            data = tomllib.loads(task_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise ValueError(f"invalid LHTB task metadata: {task_file}") from error
        task = data.get("task")
        if isinstance(task, dict) and task.get("name") == canonical_name:
            matches.append(task_file.parent.name)
    if len(matches) != 1:
        raise ValueError(
            f"canonical task name must resolve exactly once: {canonical_name}"
        )
    return matches[0]


def _primary_reward(data: dict[str, Any]) -> float | None:
    verifier = data.get("verifier_result")
    if not isinstance(verifier, dict):
        return None
    rewards = verifier.get("rewards")
    if not isinstance(rewards, dict) or "reward" not in rewards:
        return None
    value = rewards["reward"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("reward is not numeric")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError("reward is outside [0, 1]")
    return result


def _run_checked(command: list[str], *, cwd: Path, description: str) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreflightError(f"failed to {description}: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise PreflightError(f"failed to {description}: {detail}")
    return result.stdout.rstrip("\r\n")


def _validate_checkout_contents(root: Path) -> None:
    allowed = set(_PATCHED_HARBOR_SHA256)
    harbor_status = _run_checked(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "harbor",
        ],
        cwd=root,
        description="inspect Harbor checkout",
    )
    changed = {line[3:] for line in harbor_status.splitlines() if len(line) >= 4}
    unexpected = sorted(changed - allowed)
    if unexpected:
        raise PreflightError(
            "Harbor checkout has changes outside the companion patch: "
            + ", ".join(unexpected)
        )
    missing = sorted(allowed - changed)
    if missing:
        raise PreflightError(
            "Harbor companion patch is incomplete: " + ", ".join(missing)
        )
    for relative, expected in _PATCHED_HARBOR_SHA256.items():
        path = root / relative
        if not path.is_file() or _file_sha256(path) != expected:
            raise PreflightError(
                "Harbor file does not match companion patch "
                f"v{DRIFTLOCK_HARBOR_PATCH_VERSION}: {relative}"
            )
    task_status = _run_checked(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "tasks",
        ],
        cwd=root,
        description="inspect benchmark task tree",
    )
    if task_status:
        raise PreflightError(
            "benchmark task tree differs from the pinned revision: "
            + task_status.splitlines()[0]
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_harbor_command() -> list[str]:
    script = Path(sys.executable).parent / "harbor"
    if not script.is_file():
        raise PreflightError(
            "harbor console script is not installed beside the current Python"
        )
    return [sys.executable, str(script)]


if __name__ == "__main__":
    sys.exit(main())
