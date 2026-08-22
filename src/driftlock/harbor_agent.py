"""Harbor agent plugin that runs Terminus-2 through driftlock.

This module intentionally depends on Harbor and is imported only through Harbor's
``AgentConfig.import_path``.  Importing the rest of :mod:`driftlock` remains
dependency-free.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import UUID

from harbor.agents.base import BaseAgent
from harbor.agents.terminus_2 import Terminus2
from harbor.llms.lite_llm import LiteLLM

from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.judges import CallableLLMJudge
from driftlock.lhtb import (
    HarborWorkspaceDeltaObserver,
    LHTBTerminusRuntime,
    openrouter_provider_from_call_kwargs,
)
from driftlock.models import (
    DriftContext,
    JudgeCompletion,
    JudgeVerdict,
    RunResult,
    RunStatus,
)
from driftlock.oracle import (
    ReplayUsage,
    load_remote_checkpoint_bundle,
    load_source_trial_provenance,
    validate_checkpoint_source_audit,
)
from driftlock.remote import RemoteArchiveCheckpointStore
from driftlock.runner import DriftlockRunner, RunnerConfig
from driftlock.terminus import TerminusConversationCodec, TerminusStepAdapter

# The fine judge must not be the same model as the agent. Its job is to notice
# that the agent has drifted, and a judge sharing the agent's weights shares its
# blind spots: if the agent talked itself into a wrong path, the same model
# reading the same trajectory is disposed to agree. The judge also runs only when
# the coarse tier fires, so the stronger model sits on the low-call-count side.
PINNED_LHTB_JUDGE_MODEL = "openrouter/deepseek/deepseek-v4-pro-0813"


@dataclass(frozen=True, slots=True)
class _JudgeProviderPricing:
    input_cost_per_token: float
    cache_cost_per_token: float
    output_cost_per_token: float


# Pricing is keyed by the exact routable provider slug so changing the provider
# without adding its audited rates fails instead of silently billing at stale rates.
_JUDGE_PROVIDER_PRICING = {
    "alibaba": _JudgeProviderPricing(
        input_cost_per_token=0.000001162,
        cache_cost_per_token=0.0000001162,
        output_cost_per_token=0.000003485,
    )
}


class LHTBCheckpointReplayOracle(BaseAgent):
    """Restore one retained checkpoint for Harbor's isolated hidden verifier.

    The class never calls a model and never sees verifier output. Harbor creates a
    fresh task environment, calls this agent once to restore a predeclared bundle,
    and then runs the task's ordinary verifier after the agent returns.
    """

    def __init__(
        self,
        *args: Any,
        driftlock_oracle_mode: str,
        driftlock_checkpoint_dir: str,
        driftlock_checkpoint_digest: str,
        driftlock_expected_workspace: str,
        driftlock_source_trial_id: str,
        driftlock_source_task_name: str,
        driftlock_source_result: str,
        driftlock_source_result_sha256: str,
        driftlock_source_audit: str,
        driftlock_source_audit_sha256: str,
        driftlock_source_usage: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if driftlock_oracle_mode != "isolated-checkpoint-replay":
            raise ValueError("oracle mode must be isolated-checkpoint-replay")
        try:
            UUID(driftlock_source_trial_id)
        except (TypeError, ValueError) as error:
            raise ValueError("source trial id must be a UUID") from error
        if len(driftlock_checkpoint_digest) != 64:
            raise ValueError("checkpoint digest must be a SHA-256")
        if len(driftlock_source_result_sha256) != 64:
            raise ValueError("source result digest must be a SHA-256")
        super().__init__(*args, **kwargs)
        self._oracle_checkpoint_dir = Path(driftlock_checkpoint_dir)
        self._oracle_checkpoint_digest = driftlock_checkpoint_digest
        self._oracle_expected_workspace = driftlock_expected_workspace
        self._oracle_source_trial_id = driftlock_source_trial_id
        self._oracle_source_task_name = driftlock_source_task_name
        self._oracle_source_result = Path(driftlock_source_result)
        self._oracle_source_result_sha256 = driftlock_source_result_sha256
        self._oracle_source_audit = Path(driftlock_source_audit)
        self._oracle_source_audit_sha256 = driftlock_source_audit_sha256
        self._oracle_source_usage = ReplayUsage.from_mapping(driftlock_source_usage)

    @staticmethod
    def name() -> str:
        return "driftlock-checkpoint-replay-oracle"

    def version(self) -> str | None:
        return version("driftlock")

    async def setup(self, environment: Any) -> None:
        del environment

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        del instruction
        source = load_source_trial_provenance(
            self._oracle_source_result,
            expected_sha256=self._oracle_source_result_sha256,
        )
        if (
            source.trial_id != self._oracle_source_trial_id
            or source.task_name != self._oracle_source_task_name
            or source.model_name != self.model_name
            or source.usage != self._oracle_source_usage
        ):
            raise ValueError("replay parameters differ from hashed source result")
        bundle = load_remote_checkpoint_bundle(
            self._oracle_checkpoint_dir,
            expected_digest=self._oracle_checkpoint_digest,
            expected_workspace=self._oracle_expected_workspace,
        )
        phase_audit = validate_checkpoint_source_audit(
            bundle.checkpoint.path,
            source_result=source.result_path,
            source_audit=self._oracle_source_audit,
            expected_audit_sha256=self._oracle_source_audit_sha256,
        )
        task_config = getattr(environment, "task_env_config", None)
        workspace = str(getattr(task_config, "workdir", None) or "/app")
        if workspace != bundle.remote_workspace:
            raise ValueError("fresh verifier workspace differs from checkpoint")

        store = RemoteArchiveCheckpointStore(
            environment,
            remote_workspace=workspace,
            store_dir=bundle.checkpoint.path.parent.parent,
            user=environment.default_user,
        )
        await store.restore(bundle.checkpoint)

        usage = self._oracle_source_usage
        context.n_input_tokens = usage.input_tokens
        context.n_cache_tokens = usage.cache_tokens
        context.n_output_tokens = usage.output_tokens
        context.cost_usd = float(usage.cost_usd)
        metadata = dict(context.metadata or {})
        metadata["termination_reason"] = "oracle_checkpoint_replay"
        metadata["oracle"] = {
            "mode": "isolated-checkpoint-replay",
            "source_trial_id": self._oracle_source_trial_id,
            "source_result": str(self._oracle_source_result.resolve()),
            "source_result_sha256": self._oracle_source_result_sha256,
            "source_audit": str(self._oracle_source_audit.resolve()),
            "source_audit_sha256": self._oracle_source_audit_sha256,
            "source_task_name": source.task_name,
            "checkpoint_id": bundle.checkpoint.checkpoint_id,
            "checkpoint_step": bundle.checkpoint.step,
            "checkpoint_digest": bundle.checkpoint.digest,
            "archive_sha256": bundle.archive_sha256,
            "state_sha256": bundle.state_sha256,
            "workspace": workspace,
            "usage_policy": "full-source-trial-conservative",
            "source_usage": usage.as_dict(),
            "source_phase": phase_audit,
        }
        context.metadata = metadata
        output = Path(self.logs_dir) / "oracle-replay.json"
        output.write_text(
            json.dumps(metadata["oracle"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    async def resume_after_verifier_rejection(
        self, user_prompt: str, context: Any
    ) -> None:
        del user_prompt, context
        raise RuntimeError("oracle replay is a single terminal phase")


class LHTBDriftlockAgent(Terminus2):
    """Pinned Terminus-2 with checkpointed two-tier rollback.

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
        driftlock_reward_stall_steps: int = 5,
        driftlock_judge_model: str | None = None,
        driftlock_judge_api_base: str | None = None,
        driftlock_judge_max_output_tokens: int = 512,
        driftlock_judge_timeout_sec: float = 120.0,
        driftlock_judge_llm_call_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if enable_summarize:
            raise ValueError("driftlock requires enable_summarize=false")
        call_kwargs = kwargs.get("llm_call_kwargs")
        if not isinstance(call_kwargs, dict):
            call_kwargs = {}
        openrouter_provider_from_call_kwargs(call_kwargs, source="llm_call_kwargs")
        super().__init__(*args, enable_summarize=False, **kwargs)
        self._driftlock_runner_config = RunnerConfig(
            max_steps=driftlock_max_steps,
            max_rollbacks=driftlock_max_rollbacks,
            checkpoint_interval=driftlock_checkpoint_interval,
            max_tokens=driftlock_max_tokens,
            checkpoint_on_exit=driftlock_retain_checkpoints,
        )
        self._driftlock_heuristic_config = HeuristicConfig(
            no_change_steps=driftlock_no_change_steps,
            loop_window=driftlock_loop_window,
            loop_repetitions=driftlock_loop_repetitions,
            error_window=driftlock_error_window,
            error_rate=driftlock_error_rate,
            reward_stall_steps=driftlock_reward_stall_steps,
        )
        self._driftlock_judge_client = (
            None
            if driftlock_judge_model is None
            else _LHTBJudgeClient(
                model=driftlock_judge_model,
                api_base=driftlock_judge_api_base,
                max_output_tokens=driftlock_judge_max_output_tokens,
                timeout_sec=driftlock_judge_timeout_sec,
                llm_call_kwargs=driftlock_judge_llm_call_kwargs,
            )
        )
        self._driftlock_fine_judge = (
            None
            if self._driftlock_judge_client is None
            else _LHTBFineJudge(self._driftlock_judge_client)
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
        assert self._driftlock_store_root is not None
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
        if self._driftlock_judge_client is not None:
            self._driftlock_judge_client.prepare_accounting(context)
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
            fine_judge=self._driftlock_fine_judge,
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
            if self._driftlock_judge_client is not None:
                self._driftlock_judge_client.apply_accounting(context)
            self._write_phase_record(None, phase_store, retained=True)
            raise

        self._driftlock_last_result = result
        self._driftlock_tokens_consumed += result.tokens_used
        if self._driftlock_judge_client is not None:
            self._driftlock_judge_client.apply_accounting(context)
        raw = _AccountingSnapshot.capture(context)
        if not same_context:
            _make_phase_accounting(context, self._driftlock_accounting, raw)
        self._driftlock_accounting = raw
        self._driftlock_last_context_id = id(context)
        self._set_result_metadata(context, result)
        retained = self._retain_phase_checkpoints(len(self._driftlock_phases))
        self._write_phase_record(result, phase_store, retained=retained)
        if not retained:
            shutil.rmtree(phase_store)

    def _retain_phase_checkpoints(self, phase: int) -> bool:
        del phase
        return self._driftlock_retain_checkpoints

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


class LHTBBlindRetryAgent(LHTBDriftlockAgent):
    """Compute-matched control that restarts blindly after verifier rejection.

    The binary rejection is used only as the retry trigger.  Its textual verifier
    feedback is deliberately discarded, and every retry restores the original
    workspace plus a fresh Terminus conversation while preserving physical token
    accounting across attempts.
    """

    def __init__(
        self,
        *args: Any,
        driftlock_max_steps: int = 500,
        **kwargs: Any,
    ) -> None:
        horizon = driftlock_max_steps + 1
        kwargs.update(
            {
                "driftlock_max_rollbacks": 0,
                "driftlock_checkpoint_interval": horizon,
                "driftlock_no_change_steps": horizon,
                "driftlock_loop_window": horizon,
                "driftlock_loop_repetitions": horizon,
                "driftlock_error_window": horizon,
                "driftlock_reward_stall_steps": horizon,
                "driftlock_judge_model": None,
                "driftlock_retain_checkpoints": False,
            }
        )
        super().__init__(
            *args,
            driftlock_max_steps=driftlock_max_steps,
            **kwargs,
        )
        self._driftlock_retry_checkpoint: Any | None = None
        self._driftlock_retry_count = 0

    @staticmethod
    def name() -> str:
        return "compute-matched-blind-retry-terminus-2"

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        await super().run(instruction, environment, context)
        result = self._driftlock_last_result
        if result is None or not result.checkpoints:
            raise RuntimeError("blind retry initial run did not create a checkpoint")
        self._driftlock_retry_checkpoint = result.checkpoints[0]
        self._set_retry_metadata(context)

    async def resume_after_verifier_rejection(
        self, user_prompt: str, context: Any
    ) -> None:
        del user_prompt
        checkpoint = self._driftlock_retry_checkpoint
        if checkpoint is None or self._driftlock_environment is None:
            raise RuntimeError("cannot retry before the initial run")
        if id(context) != self._driftlock_last_context_id:
            raise RuntimeError("blind retry must reuse AgentContext")
        assert self._driftlock_step is not None
        configured_budget = self._driftlock_runner_config.max_tokens
        if (
            configured_budget is not None
            and self._driftlock_tokens_consumed >= configured_budget
        ):
            await self._run_driftlock_phase(
                instruction=self._original_instruction,
                environment=self._driftlock_environment,
                context=context,
                initial_state=TerminusConversationCodec().initial_state(),
            )
            self._set_retry_metadata(context)
            return
        assert self._driftlock_runtime is not None
        previous_result = self._driftlock_last_result
        if previous_result is None:
            raise RuntimeError("cannot retry without a completed prior attempt")
        guard_root = self._driftlock_store_root / (
            f"retry-guard-{self._driftlock_retry_count}"
        )
        guard_store = RemoteArchiveCheckpointStore(
            self._driftlock_environment,
            remote_workspace=self._driftlock_workspace or "",
            store_dir=guard_root,
            user=self._driftlock_environment.default_user,
            before_restore=self._driftlock_step.before_workspace_restore,
        )
        guard_checkpoint = await guard_store.create(
            previous_result.state,
            step=0,
            label="pre-retry",
        )
        phase_root = checkpoint.path.parent.parent
        store = RemoteArchiveCheckpointStore(
            self._driftlock_environment,
            remote_workspace=self._driftlock_workspace or "",
            store_dir=phase_root,
            user=self._driftlock_environment.default_user,
            before_restore=self._driftlock_step.before_workspace_restore,
        )
        provider_calls_before = self._driftlock_runtime.provider_call_count
        try:
            initial_state = await store.restore(checkpoint)
            await self._run_driftlock_phase(
                instruction=self._original_instruction,
                environment=self._driftlock_environment,
                context=context,
                initial_state=initial_state,
            )
        finally:
            provider_calls_after = self._driftlock_runtime.provider_call_count
            if provider_calls_after == provider_calls_before:
                await guard_store.restore(guard_checkpoint)
            else:
                self._driftlock_retry_count += 1
            shutil.rmtree(guard_root, ignore_errors=True)
        self._set_retry_metadata(context)

    def _retain_phase_checkpoints(self, phase: int) -> bool:
        return phase == 0

    def _set_retry_metadata(self, context: Any) -> None:
        metadata = dict(context.metadata or {})
        metadata["driftlock_blind_retry"] = {
            "retries_started": self._driftlock_retry_count,
            "verifier_feedback_used": False,
            "restart_checkpoint": "initial",
        }
        context.metadata = metadata

    async def _driftlock_finalize_after_agent_run(self) -> None:
        """Release the retry-only checkpoint after Harbor's continuation loop."""
        if self._driftlock_store_root is not None:
            shutil.rmtree(self._driftlock_store_root, ignore_errors=True)
        self._driftlock_retry_checkpoint = None


class _LHTBFineJudge:
    def __init__(self, client: _LHTBJudgeClient) -> None:
        self.client = client

    async def judge(self, context: DriftContext) -> JudgeVerdict:
        async def complete(prompt: str) -> JudgeCompletion:
            return await self.client.complete(
                prompt, tokens_remaining=context.tokens_remaining
            )

        return await CallableLLMJudge(complete).judge(context)


class _LHTBJudgeClient:
    """Single-attempt LiteLLM judge with conservative budget and cost accounting."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None,
        max_output_tokens: int,
        timeout_sec: float,
        llm_call_kwargs: dict[str, Any] | None,
    ) -> None:
        if model != PINNED_LHTB_JUDGE_MODEL:
            raise ValueError(
                "driftlock_judge_model must use the pinned model with audited pricing: "
                + PINNED_LHTB_JUDGE_MODEL
            )
        if max_output_tokens <= 0 or timeout_sec <= 0:
            raise ValueError("judge output limit and timeout must be positive")
        call_kwargs = dict(llm_call_kwargs or {})
        if set(call_kwargs) != {"extra_body"}:
            raise ValueError(
                "driftlock_judge_llm_call_kwargs must contain exactly extra_body"
            )
        provider = openrouter_provider_from_call_kwargs(
            call_kwargs, source="driftlock_judge_llm_call_kwargs"
        )
        pricing = _JUDGE_PROVIDER_PRICING.get(provider)
        if pricing is None:
            raise ValueError(
                f"judge provider {provider!r} has no audited pricing; add its pinned "
                "rates before changing driftlock_judge_llm_call_kwargs"
            )
        self.model = model
        self.provider = provider
        self.pricing = pricing
        self.max_output_tokens = max_output_tokens
        self.timeout_sec = timeout_sec
        self.llm = LiteLLM(
            model_name=model,
            api_base=api_base,
            temperature=0.0,
            model_info={
                "max_input_tokens": 1_000_000,
                "max_output_tokens": 8_192,
                "input_cost_per_token": pricing.input_cost_per_token,
                "cache_read_input_token_cost": pricing.cache_cost_per_token,
                "output_cost_per_token": pricing.output_cost_per_token,
            },
            **call_kwargs,
        )
        self.llm._driftlock_single_attempt = True
        self.n_input_tokens = 0
        self.n_cache_tokens = 0
        self.n_output_tokens = 0
        self.cost_usd = 0.0
        self.request_times_msec: list[float] = []
        self.usage_fallbacks = 0
        self._applied_context_id: int | None = None
        self._applied_input_tokens = 0
        self._applied_cache_tokens = 0
        self._applied_output_tokens = 0
        self._applied_cost_usd = 0.0
        self._applied_request_count = 0

    async def complete(
        self, prompt: str, *, tokens_remaining: int | None
    ) -> JudgeCompletion:
        input_bound = len(prompt.encode("utf-8")) + 256
        ceiling = self.max_output_tokens
        if tokens_remaining is not None:
            ceiling = min(ceiling, max(0, tokens_remaining - input_bound))
        if ceiling <= 0:
            return JudgeCompletion(text="", tokens=0)

        started = time.monotonic()
        response: Any | None = None
        fatal_error: BaseException | None = None
        try:
            call = getattr(self.llm.call, "__wrapped__", None)
            if call is None:
                raise RuntimeError("pinned LiteLLM.call lacks single-attempt access")
            response = await call(
                self.llm,
                prompt=prompt,
                max_tokens=ceiling,
                num_retries=0,
                max_retries=0,
                timeout=self.timeout_sec,
            )
            content = response.content
        except BaseException as error:
            response = getattr(error, "response", None)
            content = getattr(error, "truncated_response", None) or ""
            if not isinstance(error, Exception):
                fatal_error = error
        finally:
            self.request_times_msec.append((time.monotonic() - started) * 1000)

        usage = getattr(response, "usage", None)
        if self._valid_usage(usage):
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            cache_tokens = usage.cache_tokens
            cost_usd = float(usage.cost_usd)
        else:
            prompt_tokens = input_bound
            completion_tokens = ceiling
            cache_tokens = 0
            cost_usd = (
                prompt_tokens * self.pricing.input_cost_per_token
                + completion_tokens * self.pricing.output_cost_per_token
            )
            self.usage_fallbacks += 1
        self.n_input_tokens += prompt_tokens
        self.n_cache_tokens += cache_tokens
        self.n_output_tokens += completion_tokens
        self.cost_usd += cost_usd
        if fatal_error is not None:
            raise fatal_error
        return JudgeCompletion(
            text=content,
            tokens=prompt_tokens + completion_tokens,
        )

    @staticmethod
    def _valid_usage(usage: Any) -> bool:
        if usage is None:
            return False
        integer_values = (
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "cache_tokens", None),
        )
        cost = getattr(usage, "cost_usd", None)
        return (
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in integer_values
            )
            and isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and cost >= 0
        )

    def apply_accounting(self, context: Any) -> None:
        context.n_input_tokens = (context.n_input_tokens or 0) + self.n_input_tokens
        context.n_cache_tokens = (context.n_cache_tokens or 0) + self.n_cache_tokens
        context.n_output_tokens = (context.n_output_tokens or 0) + self.n_output_tokens
        total_cost = (context.cost_usd or 0.0) + self.cost_usd
        context.cost_usd = total_cost if total_cost > 0 else None
        metadata = dict(context.metadata or {})
        request_times = list(metadata.get("api_request_times_msec") or [])
        request_times.extend(self.request_times_msec)
        metadata["api_request_times_msec"] = request_times
        metadata["llm_time_sec"] = sum(request_times) / 1000.0
        metadata["driftlock_judge_usage"] = {
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.n_input_tokens,
            "cache_tokens": self.n_cache_tokens,
            "output_tokens": self.n_output_tokens,
            "cost_usd": self.cost_usd,
            "request_count": len(self.request_times_msec),
            "conservative_usage_fallbacks": self.usage_fallbacks,
        }
        context.metadata = metadata
        self._applied_context_id = id(context)
        self._applied_input_tokens = self.n_input_tokens
        self._applied_cache_tokens = self.n_cache_tokens
        self._applied_output_tokens = self.n_output_tokens
        self._applied_cost_usd = self.cost_usd
        self._applied_request_count = len(self.request_times_msec)

    def prepare_accounting(self, context: Any) -> None:
        """Remove the prior phase's reporting overlay before reusing a context."""
        if id(context) != self._applied_context_id:
            return
        context.n_input_tokens = max(
            0, (context.n_input_tokens or 0) - self._applied_input_tokens
        )
        context.n_cache_tokens = max(
            0, (context.n_cache_tokens or 0) - self._applied_cache_tokens
        )
        context.n_output_tokens = max(
            0, (context.n_output_tokens or 0) - self._applied_output_tokens
        )
        base_cost = (context.cost_usd or 0.0) - self._applied_cost_usd
        context.cost_usd = base_cost if base_cost > 0 else None
        metadata = dict(context.metadata or {})
        request_times = list(metadata.get("api_request_times_msec") or [])
        count = self._applied_request_count
        expected = self.request_times_msec[:count]
        if count:
            if len(request_times) < count or request_times[-count:] != expected:
                raise RuntimeError("judge request accounting overlay was modified")
            request_times = request_times[:-count]
        metadata["api_request_times_msec"] = request_times
        metadata["llm_time_sec"] = sum(request_times) / 1000.0
        metadata.pop("driftlock_judge_usage", None)
        context.metadata = metadata
        self._applied_context_id = None


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
