"""Budget-conscious command line harness for pinned LHTB experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
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

from driftlock.judges import DEFAULT_JUDGE_MAX_OUTPUT_TOKENS
from driftlock.lhtb import (
    DRIFTLOCK_HARBOR_PATCH_VERSION,
    LHTB_LITELLM_VERSION,
    LHTB_REPOSITORY_REVISION,
    lhtb_experiment_fingerprint,
    openrouter_provider_call_kwargs,
)
from driftlock.lhtb_analysis import (
    analyze_jobs,
    parse_arm_directories,
    task_directory_sha256,
)
from driftlock.oracle import (
    ReplayUsage,
    file_sha256,
    load_remote_checkpoint_bundle,
    load_source_trial_provenance,
    validate_checkpoint_source_audit,
)

# Both identifiers carry an explicit dated build. An unversioned alias such as
# "deepseek-v4-pro" silently follows whatever OpenRouter currently points it at,
# so arms run on different days would use different models while the recorded
# model string stayed identical - the comparison would be void with no evidence.
DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
DEFAULT_PROVIDER = "deepinfra/fp8"
DEFAULT_JUDGE_MODEL = "openrouter/deepseek/deepseek-v4-pro-0813"
DEFAULT_JUDGE_PROVIDER = "alibaba"
# Arms that pay for a fine judge, and therefore need its provider probed too.
_FINE_JUDGE_ARMS = frozenset({"driftlock", "native-driftlock"})
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_CREDENTIAL_ENV = "OPENROUTER_API_KEY"
RUNNABLE_ARMS = (
    "stock",
    "retry",
    "driftlock-heuristic",
    "driftlock",
    "native-driftlock-heuristic",
    "native-driftlock",
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
    checkpoint_arms = {"driftlock", "driftlock-heuristic"}
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
        }:
            agent["kwargs"].update(DRIFTLOCK_DETECTOR_DEFAULTS)
        if retain_checkpoints:
            agent["kwargs"]["driftlock_retain_checkpoints"] = True
        if arm in {"driftlock", "native-driftlock"}:
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
            or kwargs.get("driftlock_retain_checkpoints") is not True
        ):
            raise ValueError(
                f"source trial was not configured to retain driftlock checkpoints: "
                f"{result_file}"
            )
        _validate_oracle_source_agent(result, agent, kwargs, result_file)
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
        source_result_digest = provenance.result_sha256
        source_audit = result_file.parent / "agent" / "driftlock-result.json"
        source_audit_digest = file_sha256(source_audit)
        checkpoint_root = result_file.parent / ".driftlock-checkpoints"
        checkpoint_dirs = sorted(checkpoint_root.glob("phase-*/checkpoints/*"))
        if not checkpoint_dirs:
            raise ValueError(f"source trial has no retained checkpoints: {result_file}")
        for checkpoint_dir in checkpoint_dirs:
            bundle = load_remote_checkpoint_bundle(checkpoint_dir)
            validate_checkpoint_source_audit(
                bundle.checkpoint.path,
                source_result=result_file,
                source_audit=source_audit,
                expected_audit_sha256=source_audit_digest,
            )
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
                    "checkpoint_step": bundle.checkpoint.step,
                    "checkpoint_digest": bundle.checkpoint.digest,
                    "checkpoint_dir": str(bundle.checkpoint.path),
                    "archive_sha256": bundle.archive_sha256,
                    "state_sha256": bundle.state_sha256,
                    "workspace": bundle.remote_workspace,
                    "source_usage": usage.as_dict(),
                    "usage_policy": "full-source-trial-conservative",
                    "config": str(config_path),
                    "job_name": job_name,
                }
            )
    if not candidates:
        raise ValueError(f"source job has no trial results: {source}")
    manifest = {
        "schema_version": 1,
        "mode": "isolated-checkpoint-replay",
        "source_job_dir": str(source),
        "lhtb_revision": LHTB_REPOSITORY_REVISION,
        "harbor_patch": str(DRIFTLOCK_HARBOR_PATCH_VERSION),
        "experiment_fingerprint": lhtb_experiment_fingerprint(),
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
                    "DRIFTLOCK_EXPERIMENT_FINGERPRINT": (lhtb_experiment_fingerprint()),
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
) -> None:
    if agent.get("env") != {
        "HB_CONTINUE_MODE": "same_conversation",
        "DRIFTLOCK_EXPERIMENT_FINGERPRINT": lhtb_experiment_fingerprint(),
    }:
        raise ValueError(f"source trial has the wrong experiment identity: {path}")
    required = {
        "enable_summarize": False,
        "driftlock_max_steps": 500,
        "driftlock_max_rollbacks": 3,
        "driftlock_checkpoint_interval": 5,
        "driftlock_retain_checkpoints": True,
    }
    if any(kwargs.get(name) != expected for name, expected in required.items()):
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
    for name in ("oracle-prepare", "oracle-run"):
        oracle = sub.add_parser(
            name, help=f"{name} isolated retained-checkpoint verifier jobs"
        )
        oracle.add_argument("--lhtb-dir", type=Path, default=Path.cwd())
        oracle.add_argument("--source-job-dir", type=Path, required=True)
        oracle.add_argument("--output-dir", type=Path, required=True)
        oracle.add_argument("--timeout-sec", type=int, default=900)
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
