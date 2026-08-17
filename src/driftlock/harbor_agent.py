"""Harbor agent plugin that runs Terminus-2 through driftlock.

This module intentionally depends on Harbor and is imported only through Harbor's
``AgentConfig.import_path``.  Importing the rest of :mod:`driftlock` remains
dependency-free.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2 import Terminus2

from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.lhtb import HarborWorkspaceDeltaObserver, LHTBTerminusRuntime
from driftlock.models import RunResult, RunStatus
from driftlock.remote import RemoteArchiveCheckpointStore
from driftlock.runner import DriftlockRunner, RunnerConfig
from driftlock.terminus import TerminusConversationCodec, TerminusStepAdapter


class LHTBDriftlockAgent(Terminus2):
    """Pinned Terminus-2 with checkpointed, heuristics-only rollback.

    Configuration uses ``driftlock_*`` keyword arguments so a generated Harbor
    config can distinguish the controller budget from Terminus' own parameters.
    """

    def __init__(
        self,
        *args: Any,
        enable_summarize: bool = False,
        driftlock_max_steps: int = 500,
        driftlock_max_rollbacks: int = 3,
        driftlock_checkpoint_interval: int = 5,
        driftlock_max_tokens: int | None = None,
        driftlock_plan: str = "inspect, implement, verify",
        driftlock_retain_checkpoints: bool = False,
        driftlock_no_change_steps: int = 4,
        driftlock_loop_window: int = 6,
        driftlock_loop_repetitions: int = 3,
        driftlock_error_window: int = 5,
        driftlock_error_rate: float = 0.6,
        **kwargs: Any,
    ) -> None:
        if enable_summarize:
            raise ValueError("driftlock requires enable_summarize=false")
        super().__init__(*args, enable_summarize=False, **kwargs)
        self._driftlock_runner_config = RunnerConfig(
            max_steps=driftlock_max_steps,
            max_rollbacks=driftlock_max_rollbacks,
            checkpoint_interval=driftlock_checkpoint_interval,
            max_tokens=driftlock_max_tokens,
        )
        self._driftlock_heuristic_config = HeuristicConfig(
            no_change_steps=driftlock_no_change_steps,
            loop_window=driftlock_loop_window,
            loop_repetitions=driftlock_loop_repetitions,
            error_window=driftlock_error_window,
            error_rate=driftlock_error_rate,
        )
        self._driftlock_plan = driftlock_plan
        self._driftlock_retain_checkpoints = driftlock_retain_checkpoints
        self._driftlock_runtime: LHTBTerminusRuntime | None = None
        self._driftlock_step: TerminusStepAdapter | None = None
        self._driftlock_environment: Any | None = None
        self._driftlock_workspace: str | None = None
        self._driftlock_store_root: Path | None = None
        self._driftlock_last_result: RunResult | None = None
        self._driftlock_last_context_id: int | None = None
        self._driftlock_accounting = _AccountingSnapshot()
        self._driftlock_phases: list[dict[str, Any]] = []
        self._driftlock_tokens_consumed = 0

    @staticmethod
    def name() -> str:
        return "driftlock-terminus-2"

    def version(self) -> str | None:
        return version("driftlock")

    def set_process_reward_tracker(self, tracker: Any) -> None:
        raise RuntimeError(
            "HB_PROCESS_REWARD is incompatible with driftlock-owned checkpoints"
        )

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        await self._run_driftlock_phase(
            instruction=instruction,
            environment=environment,
            context=context,
            initial_state=None,
        )

    async def resume_after_verifier_rejection(
        self, user_prompt: str, context: Any
    ) -> None:
        if self._driftlock_last_result is None or self._driftlock_environment is None:
            raise RuntimeError("cannot resume driftlock before the initial run")
        if id(context) != self._driftlock_last_context_id:
            raise RuntimeError("same-conversation resume must reuse AgentContext")
        assert self._driftlock_step is not None
        codec = TerminusConversationCodec()
        state = codec.decode(self._driftlock_last_result.state)
        if state is None:
            raise RuntimeError(
                "driftlock result is missing Terminus conversation state"
            )
        continued = replace(
            state,
            next_prompt=(
                f"{user_prompt}\n\nCurrent terminal observation:\n{state.next_prompt}"
            ),
        )
        await self._run_driftlock_phase(
            instruction=self._original_instruction,
            environment=self._driftlock_environment,
            context=context,
            initial_state=codec.encode(continued),
        )

    async def _run_driftlock_phase(
        self,
        *,
        instruction: str,
        environment: Any,
        context: Any,
        initial_state: dict[str, Any] | None,
    ) -> None:
        self._ensure_runtime(environment, context)
        assert self._driftlock_runtime is not None
        assert self._driftlock_step is not None
        assert self._driftlock_store_root is not None

        self._driftlock_runtime.context = context
        same_context = id(context) == self._driftlock_last_context_id
        configured_budget = self._driftlock_runner_config.max_tokens
        remaining_budget = (
            None
            if configured_budget is None
            else configured_budget - self._driftlock_tokens_consumed
        )
        if remaining_budget is not None and remaining_budget <= 0:
            metadata = dict(context.metadata or {})
            metadata["termination_reason"] = "driftlock_token_limit"
            metadata["driftlock"] = {
                "status": RunStatus.TOKEN_LIMIT.value,
                "tokens_used": self._driftlock_tokens_consumed,
                "trial_token_budget": configured_budget,
            }
            context.metadata = metadata
            return
        phase_store = self._driftlock_store_root / (
            f"phase-{len(self._driftlock_phases)}"
        )
        store = RemoteArchiveCheckpointStore(
            environment,
            remote_workspace=self._driftlock_workspace or "",
            store_dir=phase_store,
            user=environment.default_user,
            before_restore=self._driftlock_step.before_workspace_restore,
        )
        runner = DriftlockRunner(
            store,
            HeuristicJudge(self._driftlock_heuristic_config),
            config=replace(self._driftlock_runner_config, max_tokens=remaining_budget),
        )
        try:
            result = await runner.run(
                goal=instruction,
                plan=self._driftlock_plan,
                step=self._driftlock_step,
                initial_state=(
                    initial_state
                    if initial_state is not None
                    else self._driftlock_step.initial_state()
                ),
            )
        except BaseException:
            self._write_phase_record(None, phase_store, retained=True)
            raise

        self._driftlock_last_result = result
        self._driftlock_tokens_consumed += result.tokens_used
        raw = _AccountingSnapshot.capture(context)
        if not same_context:
            _make_phase_accounting(context, self._driftlock_accounting, raw)
        self._driftlock_accounting = raw
        self._driftlock_last_context_id = id(context)
        self._set_result_metadata(context, result)
        retained = self._driftlock_retain_checkpoints
        self._write_phase_record(result, phase_store, retained=retained)
        if not retained:
            shutil.rmtree(phase_store)

    def _ensure_runtime(self, environment: Any, context: Any) -> None:
        if self._driftlock_runtime is not None:
            if environment is not self._driftlock_environment:
                raise RuntimeError(
                    "a driftlock agent cannot switch Harbor environments"
                )
            return
        task_config = getattr(environment, "task_env_config", None)
        workdir = getattr(task_config, "workdir", None)
        workspace = str(workdir or "/app")
        if not workspace.startswith("/") or workspace == "/":
            raise ValueError("Harbor task workdir must be an absolute non-root path")
        logs_dir = Path(self.logs_dir).expanduser().resolve()
        store_root = logs_dir.parent / ".driftlock-checkpoints"
        if store_root == logs_dir or logs_dir in store_root.parents:
            raise ValueError("checkpoint storage must be outside the agent log mount")
        store_root.mkdir(parents=True, exist_ok=True)
        observer = HarborWorkspaceDeltaObserver(
            environment,
            remote_workspace=workspace,
            user=environment.default_user,
        )
        runtime = LHTBTerminusRuntime(
            self,
            environment,
            context,
            remote_workspace=workspace,
            observer=observer,
        )
        self._driftlock_environment = environment
        self._driftlock_workspace = workspace
        self._driftlock_store_root = store_root
        self._driftlock_runtime = runtime
        self._driftlock_step = TerminusStepAdapter(runtime)

    def _set_result_metadata(self, context: Any, result: RunResult) -> None:
        metadata = dict(context.metadata or {})
        summary = {
            "status": result.status.value,
            "steps": len(result.steps),
            "rollbacks": len(result.rollbacks),
            "tokens_used": result.tokens_used,
            "agent_tokens_used": result.agent_tokens_used,
            "judge_tokens_used": result.judge_tokens_used,
            "trial_tokens_used": self._driftlock_tokens_consumed,
            "trial_token_budget": self._driftlock_runner_config.max_tokens,
        }
        metadata["driftlock"] = summary
        metadata["termination_reason"] = (
            "confirmed_task_complete"
            if result.status is RunStatus.COMPLETED
            else f"driftlock_{result.status.value}"
        )
        context.metadata = metadata

    def _write_phase_record(
        self, result: RunResult | None, phase_store: Path, *, retained: bool
    ) -> None:
        record: dict[str, Any] = {
            "phase": len(self._driftlock_phases),
            "checkpoint_dir": str(phase_store),
            "checkpoints_retained": retained,
        }
        if result is None:
            record["status"] = "exception"
        else:
            record.update(
                {
                    "status": result.status.value,
                    "steps": len(result.steps),
                    "rollbacks": len(result.rollbacks),
                    "tokens_used": result.tokens_used,
                    "checkpoint_count": len(result.checkpoints),
                }
            )
        self._driftlock_phases.append(record)
        output = Path(self.logs_dir) / "driftlock-result.json"
        output.write_text(
            json.dumps({"phases": self._driftlock_phases}, indent=2) + "\n",
            encoding="utf-8",
        )


class _AccountingSnapshot:
    def __init__(
        self,
        n_input_tokens: int = 0,
        n_cache_tokens: int = 0,
        n_output_tokens: int = 0,
        cost_usd: float = 0.0,
        rollout_count: int = 0,
        n_episodes: int = 0,
        api_request_count: int = 0,
        terminal_interaction_count: int = 0,
    ) -> None:
        self.n_input_tokens = n_input_tokens
        self.n_cache_tokens = n_cache_tokens
        self.n_output_tokens = n_output_tokens
        self.cost_usd = cost_usd
        self.rollout_count = rollout_count
        self.n_episodes = n_episodes
        self.api_request_count = api_request_count
        self.terminal_interaction_count = terminal_interaction_count

    @classmethod
    def capture(cls, context: Any) -> _AccountingSnapshot:
        metadata = context.metadata or {}
        return cls(
            n_input_tokens=context.n_input_tokens or 0,
            n_cache_tokens=context.n_cache_tokens or 0,
            n_output_tokens=context.n_output_tokens or 0,
            cost_usd=context.cost_usd or 0.0,
            rollout_count=len(context.rollout_details or []),
            n_episodes=int(metadata.get("n_episodes") or 0),
            api_request_count=len(metadata.get("api_request_times_msec") or []),
            terminal_interaction_count=len(
                metadata.get("terminal_interaction_times_msec") or []
            ),
        )


def _make_phase_accounting(
    context: Any, before: _AccountingSnapshot, after: _AccountingSnapshot
) -> None:
    context.n_input_tokens = after.n_input_tokens - before.n_input_tokens
    context.n_cache_tokens = after.n_cache_tokens - before.n_cache_tokens
    context.n_output_tokens = after.n_output_tokens - before.n_output_tokens
    phase_cost = after.cost_usd - before.cost_usd
    context.cost_usd = phase_cost if phase_cost > 0 else None
    context.rollout_details = (context.rollout_details or [])[before.rollout_count :]
    metadata = dict(context.metadata or {})
    metadata["n_episodes"] = after.n_episodes - before.n_episodes
    for key, start in (
        ("api_request_times_msec", before.api_request_count),
        ("terminal_interaction_times_msec", before.terminal_interaction_count),
    ):
        metadata[key] = list(metadata.get(key) or [])[start:]
    metadata["llm_time_sec"] = sum(metadata["api_request_times_msec"]) / 1000.0
    metadata["terminal_interaction_time_sec"] = (
        sum(metadata["terminal_interaction_times_msec"]) / 1000.0
    )
    context.metadata = metadata
