"""Thin Harbor plugin for driftlock's native LHTB tool-calling agent."""

from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.llms.lite_llm import LiteLLM

from driftlock.harbor_agent import _LHTBFineJudge, _LHTBJudgeClient
from driftlock.heuristics import HeuristicConfig
from driftlock.lhtb import (
    HarborWorkspaceDeltaObserver,
    LHTBRuntimeCompatibilityError,
    _validate_pinned_harbor,
    _validate_single_attempt_configuration,
    openrouter_provider_from_call_kwargs,
)
from driftlock.models import RunResult, StepTokenBudgetExhausted
from driftlock.native_lhtb import (
    BilledProviderResponse,
    ContextUsageRecorder,
    LHTBNativeAgentRuntime,
    SingleAttemptJSONProvider,
    append_verifier_feedback,
    apply_native_accounting,
    billed_provider_exception,
    billed_provider_response,
    native_checkpoint_store_root,
    set_native_result_metadata,
    set_native_token_limit_metadata,
)
from driftlock.runner import RunnerConfig


class _HarborLiteLLMSingleAttempt:
    """Expose one unwrapped, exact-usage Harbor LiteLLM request."""

    def __init__(
        self,
        *,
        model_name: str,
        api_base: str | None,
        temperature: float,
        model_info: dict[str, Any],
        timeout_sec: float,
        extra_body: dict[str, Any],
    ) -> None:
        _validate_pinned_harbor()
        if timeout_sec <= 0:
            raise ValueError("provider timeout must be positive")
        self.timeout_sec = timeout_sec
        self.llm = LiteLLM(
            model_name=model_name,
            api_base=api_base,
            temperature=temperature,
            model_info=model_info,
            extra_body=extra_body,
        )
        self.llm._driftlock_single_attempt = True
        _validate_single_attempt_configuration(self, self.llm)
        call = getattr(self.llm.call, "__wrapped__", None)
        if call is None:
            raise LHTBRuntimeCompatibilityError(
                "pinned LiteLLM.call must expose its unwrapped single attempt"
            )
        self._unwrapped_call = call
        self._llm_kwargs: dict[str, Any] = {}
        self._llm_call_kwargs: dict[str, Any] = {}
        self.request_times_msec: list[float] = []

    @property
    def physical_call_count(self) -> int:
        count = getattr(self.llm, "_driftlock_provider_call_count", None)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise LHTBRuntimeCompatibilityError(
                "pinned LiteLLM lacks physical provider-call accounting"
            )
        return count

    async def __call__(
        self, prompt: str, *, max_output_tokens: int
    ) -> BilledProviderResponse:
        started = time.monotonic()
        try:
            response = await self._unwrapped_call(
                self.llm,
                prompt=prompt,
                max_tokens=max_output_tokens,
                num_retries=0,
                max_retries=0,
                timeout=self.timeout_sec,
            )
        except Exception as error:
            return billed_provider_exception(error)
        finally:
            self.request_times_msec.append((time.monotonic() - started) * 1000)

        return billed_provider_response(response)


class LHTBNativeDriftlockAgent(BaseAgent):
    """Harbor-loadable plugin that runs driftlock's own tool-calling agent."""

    def __init__(
        self,
        *args: Any,
        api_base: str | None = None,
        parser_name: str = "json",
        temperature: float = 0.7,
        record_terminal_session: bool = True,
        llm_call_kwargs: dict[str, Any] | None = None,
        model_info: dict[str, Any] | None = None,
        enable_summarize: bool = False,
        driftlock_max_steps: int = 500,
        driftlock_max_rollbacks: int = 3,
        driftlock_checkpoint_interval: int = 5,
        driftlock_max_tokens: int = 10_000_000,
        driftlock_plan: str = "inspect, implement, verify",
        driftlock_retain_checkpoints: bool = False,
        driftlock_no_change_steps: int = 4,
        driftlock_loop_window: int = 6,
        driftlock_loop_repetitions: int = 3,
        driftlock_error_window: int = 5,
        driftlock_error_rate: float = 0.6,
        driftlock_command_failure_window: int = 8,
        driftlock_command_failure_rate: float = 1.0,
        driftlock_reward_stall_steps: int = 5,
        driftlock_reward_epsilon: float = 1e-6,
        driftlock_judge_model: str | None = None,
        driftlock_judge_api_base: str | None = None,
        driftlock_judge_max_output_tokens: int = 512,
        driftlock_judge_timeout_sec: float = 120.0,
        driftlock_judge_llm_call_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if enable_summarize:
            raise ValueError("native driftlock does not support context compression")
        if parser_name != "json":
            raise ValueError("native driftlock requires parser_name='json'")
        if not record_terminal_session:
            raise ValueError("the frozen LHTB native arm records terminal activity")
        super().__init__(*args, **kwargs)
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("native driftlock requires model_name")
        call_kwargs = dict(llm_call_kwargs or {})
        if set(call_kwargs) != {
            "temperature",
            "max_tokens",
            "timeout",
            "extra_body",
        }:
            raise ValueError(
                "llm_call_kwargs must contain exactly temperature, max_tokens, "
                "timeout, extra_body"
            )
        openrouter_provider_from_call_kwargs(call_kwargs, source="llm_call_kwargs")
        if call_kwargs["temperature"] != temperature:
            raise ValueError("top-level and call-level temperatures must match")
        max_output_tokens = call_kwargs["max_tokens"]
        timeout_sec = call_kwargs["timeout"]
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(model_info, dict):
            raise ValueError("model_info is required")
        low_level = _HarborLiteLLMSingleAttempt(
            model_name=self.model_name,
            api_base=api_base,
            temperature=temperature,
            model_info=model_info,
            timeout_sec=timeout_sec,
            extra_body=call_kwargs["extra_body"],
        )
        self._native_low_level = low_level
        self._native_provider = SingleAttemptJSONProvider(low_level)
        self._native_runner_config = RunnerConfig(
            max_steps=driftlock_max_steps,
            max_rollbacks=driftlock_max_rollbacks,
            checkpoint_interval=driftlock_checkpoint_interval,
            max_tokens=driftlock_max_tokens,
            checkpoint_on_exit=driftlock_retain_checkpoints,
        )
        self._native_heuristic_config = HeuristicConfig(
            no_change_steps=driftlock_no_change_steps,
            loop_window=driftlock_loop_window,
            loop_repetitions=driftlock_loop_repetitions,
            error_window=driftlock_error_window,
            error_rate=driftlock_error_rate,
            command_failure_window=driftlock_command_failure_window,
            command_failure_rate=driftlock_command_failure_rate,
            reward_stall_steps=driftlock_reward_stall_steps,
            reward_epsilon=driftlock_reward_epsilon,
        )
        self._native_judge_client = (
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
        self._native_fine_judge = (
            None
            if self._native_judge_client is None
            else _LHTBFineJudge(self._native_judge_client)
        )
        self._native_plan = driftlock_plan
        self._native_retain_checkpoints = driftlock_retain_checkpoints
        self._native_max_output_tokens = max_output_tokens
        self._native_runtime: LHTBNativeAgentRuntime | None = None
        self._native_environment: Any | None = None
        self._native_context_id: int | None = None
        self._native_usage_recorder: ContextUsageRecorder | None = None
        self._native_instruction = ""
        self._native_last_result: RunResult | None = None
        self._native_phases: list[dict[str, Any]] = []

    @staticmethod
    def name() -> str:
        return "driftlock-native-tool-agent"

    def version(self) -> str | None:
        return version("driftlock")

    async def setup(self, environment: Any) -> None:
        del environment

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        self._native_instruction = instruction
        await self._run_phase(
            instruction=instruction,
            environment=environment,
            context=context,
            initial_state=None,
        )

    async def resume_after_verifier_rejection(
        self, user_prompt: str, context: Any
    ) -> None:
        if self._native_runtime is None or self._native_last_result is None:
            raise RuntimeError("cannot resume native driftlock before its initial run")
        if id(context) != self._native_context_id:
            raise RuntimeError("same-conversation resume must reuse AgentContext")
        state = append_verifier_feedback(self._native_last_result.state, user_prompt)
        assert self._native_environment is not None
        await self._run_phase(
            instruction=self._native_instruction,
            environment=self._native_environment,
            context=context,
            initial_state=state,
        )

    async def _run_phase(
        self,
        *,
        instruction: str,
        environment: Any,
        context: Any,
        initial_state: dict[str, Any] | None,
    ) -> None:
        runtime = self._ensure_runtime(environment, context)
        if runtime.tokens_remaining == 0:
            self._set_token_limit_metadata(context)
            return
        try:
            result = await runtime.run(goal=instruction, initial_state=initial_state)
        except StepTokenBudgetExhausted:
            self._apply_accounting(context, reconcile=True)
            self._set_token_limit_metadata(context)
            return
        except BaseException:
            self._apply_accounting(context, reconcile=False)
            raise
        self._native_last_result = result
        self._apply_accounting(context, reconcile=True)
        set_native_result_metadata(
            context,
            result=result,
            runtime=runtime,
            trial_token_budget=self._native_runner_config.max_tokens,
        )
        self._write_phase_record(result)

    def _write_phase_record(self, result: RunResult) -> None:
        record = {
            "phase": len(self._native_phases),
            "status": result.status.value,
            "steps": len(result.steps),
            "rollbacks": len(result.rollbacks),
            "tokens_used": result.tokens_used,
            "checkpoint_count": len(result.checkpoints),
            "unstable_checkpoint_count": sum(
                bool(checkpoint.unstable_paths) for checkpoint in result.checkpoints
            ),
            "checkpoints_retained": self._native_retain_checkpoints,
            "coarse_triggers": [
                trigger.to_dict() for trigger in result.coarse_triggers
            ],
            "signal_counts": result.signal_counts,
        }
        self._native_phases.append(record)
        output = Path(self.logs_dir) / "driftlock-native-result.json"
        output.write_text(
            json.dumps({"phases": self._native_phases}, indent=2) + "\n",
            encoding="utf-8",
        )

    def _ensure_runtime(self, environment: Any, context: Any) -> LHTBNativeAgentRuntime:
        if self._native_runtime is not None:
            if environment is not self._native_environment:
                raise RuntimeError(
                    "a native driftlock agent cannot switch environments"
                )
            if id(context) != self._native_context_id:
                raise RuntimeError("native driftlock must reuse its AgentContext")
            return self._native_runtime
        task_config = getattr(environment, "task_env_config", None)
        workspace = str(getattr(task_config, "workdir", None) or "/app")
        store_root = native_checkpoint_store_root(self.logs_dir)
        store_root.mkdir(parents=True, exist_ok=True)
        observer = HarborWorkspaceDeltaObserver(
            environment,
            remote_workspace=workspace,
            user=environment.default_user,
        )
        runtime = LHTBNativeAgentRuntime(
            environment,
            observer,
            self._native_provider,
            remote_workspace=workspace,
            store_dir=store_root,
            user=environment.default_user,
            runner_config=self._native_runner_config,
            heuristic_config=self._native_heuristic_config,
            fine_judge=self._native_fine_judge,
            plan=self._native_plan,
            retain_checkpoints=self._native_retain_checkpoints,
            agent_max_output_tokens=self._native_max_output_tokens,
        )
        self._native_runtime = runtime
        self._native_environment = environment
        self._native_context_id = id(context)
        self._native_usage_recorder = ContextUsageRecorder(context)
        return runtime

    def _apply_accounting(self, context: Any, *, reconcile: bool) -> None:
        runtime = self._native_runtime
        recorder = self._native_usage_recorder
        if runtime is None or recorder is None:
            return
        apply_native_accounting(
            context,
            recorder=recorder,
            runtime=runtime,
            provider=self._native_provider,
            agent_request_times_msec=tuple(self._native_low_level.request_times_msec),
            judge=self._native_judge_client,
            reconcile=reconcile,
        )

    def _set_token_limit_metadata(self, context: Any) -> None:
        runtime = self._native_runtime
        assert runtime is not None
        set_native_token_limit_metadata(
            context,
            runtime=runtime,
            trial_token_budget=self._native_runner_config.max_tokens,
        )
