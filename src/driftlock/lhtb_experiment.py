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
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from driftlock.lhtb import (
    DRIFTLOCK_HARBOR_PATCH_VERSION,
    LHTB_LITELLM_VERSION,
    LHTB_REPOSITORY_REVISION,
    lhtb_experiment_fingerprint,
)
from driftlock.lhtb_analysis import analyze_jobs, parse_arm_directories

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-pro"
DEFAULT_JUDGE_MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_CREDENTIAL_ENV = "OPENROUTER_API_KEY"
RUNNABLE_ARMS = ("stock", "retry", "driftlock-heuristic", "driftlock")

# SHA-256 of every Harbor file after applying the packaged version-10 patch to the
# pinned LHTB revision.  Preflight also rejects any other Harbor or task-tree change.
_PATCHED_HARBOR_SHA256 = {
    "harbor/src/harbor/_driftlock_pin.py": (
        "122fbebce971434c4774c26dce15d6382d1c69468c871477d2ab9c2b2d8258d8"
    ),
    "harbor/src/harbor/agents/terminus_2/terminus_2.py": (
        "2b1b9aabac4b4ec40ba6b9c63b17a7d04e5a5a15e78dbf39efa361541e42c2f4"
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
        "6e65a65dd74195ad962401b425504831d412225e011e12f54d6701bd91cb81b5"
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
    api_base: str = DEFAULT_API_BASE,
    n_concurrent_trials: int = 1,
    timeout_sec: int = 5400,
    max_total_tokens: int = 10_000_000,
    judge_api_base: str | None = None,
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
        agent["import_path"] = (
            "driftlock.harbor_agent:LHTBBlindRetryAgent"
            if arm == "retry"
            else "driftlock.harbor_agent:LHTBDriftlockAgent"
        )
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
        if arm == "driftlock":
            agent["kwargs"].update(
                {
                    "driftlock_judge_model": DEFAULT_JUDGE_MODEL,
                    "driftlock_judge_api_base": judge_api_base or api_base,
                    "driftlock_judge_max_output_tokens": 512,
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
            config = build_job_config(
                lhtb_dir=args.lhtb_dir,
                jobs_dir=args.jobs_dir,
                job_name=args.job_name,
                arm=args.arm,
                tasks=args.tasks,
                model=args.model,
                api_base=args.api_base,
                n_concurrent_trials=args.concurrency,
                timeout_sec=args.timeout_sec,
                max_total_tokens=args.max_total_tokens,
                judge_api_base=args.judge_api_base,
            )
            config_path = args.config.expanduser().resolve()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
            print(f"wrote {config_path}")
            if args.command == "prepare":
                return 0
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
        command.add_argument("--config", type=Path, default=Path("driftlock-job.json"))
        command.add_argument("--job-name", required=True)
        command.add_argument("--arm", choices=RUNNABLE_ARMS, required=True)
        command.add_argument("--tasks", nargs="+", required=True)
        command.add_argument("--model", default=DEFAULT_MODEL)
        command.add_argument("--api-base", default=DEFAULT_API_BASE)
        command.add_argument("--judge-api-base")
        command.add_argument("--credential-env", default=DEFAULT_CREDENTIAL_ENV)
        command.add_argument("--concurrency", type=int, default=1)
        command.add_argument("--timeout-sec", type=int, default=5400)
        command.add_argument("--max-total-tokens", type=int, default=10_000_000)
        command.add_argument("--ack-unbounded-stock-tokens", action="store_true")
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
