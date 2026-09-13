"""Thin Harbor plugin for driftlock's native LHTB tool-calling agent."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from collections.abc import Sequence
from importlib.metadata import version
from numbers import Real
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.llms.lite_llm import LiteLLM

from driftlock.agent import (
    DEFAULT_MAX_HISTORY_CHARACTERS,
    DEFAULT_MAX_TOOL_CALLS_PER_STEP,
    DEFAULT_MAX_TOOL_OUTPUT_CHARACTERS,
    MIN_MAX_HISTORY_CHARACTERS,
    ToolCallingSubagentExecutor,
)
from driftlock.delegation import DelegationConfig, DelegationTool
from driftlock.harbor_agent import _LHTBFineJudge, _LHTBJudgeClient
from driftlock.heuristics import HeuristicConfig
from driftlock.judges import DEFAULT_JUDGE_MAX_OUTPUT_TOKENS
from driftlock.lhtb import (
    HarborWorkspaceDeltaObserver,
    LHTBRuntimeCompatibilityError,
    _validate_pinned_harbor,
    _validate_single_attempt_configuration,
    openrouter_provider_from_call_kwargs,
)
from driftlock.memory import MemoryStore
from driftlock.models import RunResult, StepTokenBudgetExhausted
from driftlock.native_lhtb import (
    BilledProviderResponse,
    ContextUsageRecorder,
    LHTBNativeAgentRuntime,
    NativeComponentConfigurationError,
    SingleAttemptJSONProvider,
    append_verifier_feedback,
    apply_native_accounting,
    billed_provider_exception,
    billed_provider_response,
    build_remote_agentic_retrieval_tool,
    native_checkpoint_store_root,
    set_native_result_metadata,
    set_native_token_limit_metadata,
    validate_parallel_compaction_bounds,
)
from driftlock.prompt_cache import PromptCacheConfig
from driftlock.runner import RunnerConfig
from driftlock.skill_admission import SkillLibrary
from driftlock.verification import SelfVerificationConfig, VerificationStatus


def _pinned_retrieval_embedder() -> Any:
    """Load the optional pinned local embedder before any trial can start."""

    if importlib.util.find_spec("sentence_transformers") is None:
        raise NativeComponentConfigurationError(
            "driftlock_agentic_retrieval requires the pinned optional embedder "
            "driftlock.st_embedder (all-MiniLM-L6-v2) to be available"
        )
    from driftlock.st_embedder import embed

    try:
        vectors = list(embed(("driftlock retrieval configuration probe",)))
        if len(vectors) != 1:
            raise ValueError("embedder returned the wrong vector count")
        raw_vector = list(vectors[0])
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in raw_vector
        ):
            raise ValueError("embedder returned a non-numeric vector")
        vector = [float(value) for value in raw_vector]
        if (
            not vector
            or any(not math.isfinite(value) for value in vector)
            or math.fsum(value * value for value in vector) == 0.0
        ):
            raise ValueError("embedder returned a malformed vector")
    except Exception as error:
        raise NativeComponentConfigurationError(
            "driftlock_agentic_retrieval cannot initialize the pinned optional "
            f"embedder all-MiniLM-L6-v2: {type(error).__name__}: {error}"
        ) from error
    return embed


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
        explicit_prompt_cache_control: bool,
    ) -> None:
        _validate_pinned_harbor()
        if timeout_sec <= 0:
            raise ValueError("provider timeout must be positive")
        if not isinstance(explicit_prompt_cache_control, bool):
            raise TypeError("explicit_prompt_cache_control must be a boolean")
        self.timeout_sec = timeout_sec
        self.explicit_prompt_cache_control = explicit_prompt_cache_control
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
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        cacheable_prefix_characters: int | None,
    ) -> BilledProviderResponse:
        started = time.monotonic()
        try:
            provider_prompt: str | list[dict[str, Any]] = prompt
            if (
                cacheable_prefix_characters is not None
                and self.explicit_prompt_cache_control
            ):
                if not 0 <= cacheable_prefix_characters <= len(prompt):
                    raise ValueError(
                        "cacheable prefix character offset is out of range"
                    )
                provider_prompt = [
                    {
                        "type": "text",
                        "text": prompt[:cacheable_prefix_characters],
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": prompt[cacheable_prefix_characters:],
                    },
                ]
            response = await self._unwrapped_call(
                self.llm,
                prompt=provider_prompt,
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
        # False preserves the archived five-tool request and once-per-task injection.
        driftlock_agentic_retrieval: bool = False,
        driftlock_retrieval_skill_library_dir: str | None = None,
        # False leaves the caller plan read-only, as in the archived experiment.
        driftlock_planning: bool = False,
        # False prevents durable cross-task state from entering historical trials.
        driftlock_memory: bool = False,
        # False preserves one provider request per historical parent-agent step.
        driftlock_delegation: bool = False,
        # False preserves the historical serial tool execution order.
        driftlock_parallel_reads: bool = False,
        # Historical defaults are coupled to retain one worst-case parallel turn.
        driftlock_max_tool_output_characters: int = (
            DEFAULT_MAX_TOOL_OUTPUT_CHARACTERS
        ),
        driftlock_max_tool_calls_per_step: int = DEFAULT_MAX_TOOL_CALLS_PER_STEP,
        driftlock_max_history_characters: int = DEFAULT_MAX_HISTORY_CHARACTERS,
        # False preserves completion as the historical terminal condition.
        driftlock_self_verification: bool = False,
        # False preserves the historical provider request type and prefix handling.
        driftlock_prompt_cache: bool = False,
        driftlock_explicit_prompt_cache_control: bool = False,
        driftlock_corroborating_signals: Sequence[str] = ("no_file_change",),
        driftlock_judge_model: str | None = None,
        driftlock_judge_api_base: str | None = None,
        driftlock_judge_max_output_tokens: int = DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
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
        component_flags = {
            "driftlock_agentic_retrieval": driftlock_agentic_retrieval,
            "driftlock_planning": driftlock_planning,
            "driftlock_memory": driftlock_memory,
            "driftlock_delegation": driftlock_delegation,
            "driftlock_parallel_reads": driftlock_parallel_reads,
            "driftlock_self_verification": driftlock_self_verification,
        }
        for name, value in component_flags.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        if not isinstance(driftlock_prompt_cache, bool):
            raise TypeError("driftlock_prompt_cache must be a boolean")
        if not isinstance(driftlock_explicit_prompt_cache_control, bool):
            raise TypeError("driftlock_explicit_prompt_cache_control must be a boolean")
        if driftlock_explicit_prompt_cache_control and not driftlock_prompt_cache:
            raise ValueError(
                "explicit prompt cache control requires driftlock_prompt_cache"
            )
        if driftlock_agentic_retrieval:
            if not isinstance(driftlock_retrieval_skill_library_dir, str) or not (
                driftlock_retrieval_skill_library_dir
            ):
                raise NativeComponentConfigurationError(
                    "driftlock_agentic_retrieval requires "
                    "driftlock_retrieval_skill_library_dir"
                )
            library_dir = Path(driftlock_retrieval_skill_library_dir).expanduser()
            if not library_dir.is_dir():
                raise NativeComponentConfigurationError(
                    "driftlock retrieval skill library directory does not exist: "
                    f"{library_dir}"
                )
            retrieval_embedder = _pinned_retrieval_embedder()
            retrieval_library = SkillLibrary(library_dir)
            try:
                admitted_skill_ids = retrieval_library.admitted_skill_ids()
            except Exception as error:
                raise NativeComponentConfigurationError(
                    "driftlock retrieval skill library cannot be read: "
                    f"{type(error).__name__}: {error}"
                ) from error
            if not admitted_skill_ids:
                raise NativeComponentConfigurationError(
                    "driftlock_agentic_retrieval requires at least one admitted "
                    "skill in driftlock_retrieval_skill_library_dir"
                )
        else:
            if driftlock_retrieval_skill_library_dir is not None:
                raise NativeComponentConfigurationError(
                    "driftlock_retrieval_skill_library_dir requires "
                    "driftlock_agentic_retrieval"
                )
            library_dir = None
            retrieval_embedder = None
            retrieval_library = None
        if (
            not isinstance(driftlock_max_tool_output_characters, int)
            or isinstance(driftlock_max_tool_output_characters, bool)
            or driftlock_max_tool_output_characters < 128
        ):
            raise ValueError(
                "driftlock_max_tool_output_characters must be at least 128"
            )
        if (
            not isinstance(driftlock_max_tool_calls_per_step, int)
            or isinstance(driftlock_max_tool_calls_per_step, bool)
            or driftlock_max_tool_calls_per_step <= 0
        ):
            raise ValueError(
                "driftlock_max_tool_calls_per_step must be a positive integer"
            )
        if (
            not isinstance(driftlock_max_history_characters, int)
            or isinstance(driftlock_max_history_characters, bool)
            or driftlock_max_history_characters < MIN_MAX_HISTORY_CHARACTERS
        ):
            raise ValueError(
                "driftlock_max_history_characters must be at least "
                f"{MIN_MAX_HISTORY_CHARACTERS}"
            )
        validate_parallel_compaction_bounds(
            parallel_tool_calls=driftlock_parallel_reads,
            max_tool_calls_per_step=driftlock_max_tool_calls_per_step,
            max_tool_output_characters=driftlock_max_tool_output_characters,
            max_history_characters=driftlock_max_history_characters,
        )
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
            explicit_prompt_cache_control=(driftlock_explicit_prompt_cache_control),
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
            corroborating_signals=frozenset(driftlock_corroborating_signals),
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
        self._native_prompt_cache = driftlock_prompt_cache
        self._native_agentic_retrieval = driftlock_agentic_retrieval
        self._native_retrieval_library = retrieval_library
        self._native_retrieval_embedder = retrieval_embedder
        self._native_planning = driftlock_planning
        self._native_delegation = driftlock_delegation
        self._native_parallel_reads = driftlock_parallel_reads
        self._native_max_tool_output_characters = driftlock_max_tool_output_characters
        self._native_max_tool_calls_per_step = driftlock_max_tool_calls_per_step
        self._native_max_history_characters = driftlock_max_history_characters
        self._native_self_verification = driftlock_self_verification
        # Memory is intentionally scoped to one Harbor trial, whose agent log
        # directory is unique. Treatment and paired control trials therefore
        # cannot read each other's writes, while verifier-resume phases of the
        # same trial retain the designed persistence.
        memory_root = (Path(self.logs_dir) / "driftlock-memory").resolve()
        try:
            memory_root_in_use = (
                driftlock_memory
                and memory_root.exists()
                and (not memory_root.is_dir() or any(memory_root.iterdir()))
            )
        except OSError as error:
            raise NativeComponentConfigurationError(
                "driftlock memory root cannot be inspected at trial construction: "
                f"{memory_root}: {type(error).__name__}: {error}"
            ) from error
        if memory_root_in_use:
            raise NativeComponentConfigurationError(
                "driftlock memory root must be empty at trial construction: "
                f"{memory_root}"
            )
        self._native_memory_store = (
            MemoryStore(memory_root) if driftlock_memory else None
        )
        self._native_memory_run_id = hashlib.sha256(
            str(Path(self.logs_dir).resolve()).encode()
        ).hexdigest()

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
        runtime = await self._ensure_runtime(environment, context, instruction)
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
            components=runtime.component_report(),
        )
        self._write_phase_record(result)

    def _write_phase_record(self, result: RunResult) -> None:
        record = {
            "phase": len(self._native_phases),
            "status": result.status.value,
            "judge_reliability": result.judge_reliability.value,
            "judge_attempts": result.judge_attempts,
            "judge_failures": result.judge_failures,
            "steps": len(result.steps),
            "rollbacks": len(result.rollbacks),
            "tokens_used": result.tokens_used,
            "checkpoint_count": len(result.checkpoints),
            "unstable_checkpoint_count": sum(
                bool(checkpoint.unstable_paths) for checkpoint in result.checkpoints
            ),
            "non_restorable_checkpoint_count": sum(
                not checkpoint.restorable for checkpoint in result.checkpoints
            ),
            "checkpoints_retained": self._native_retain_checkpoints,
            "coarse_triggers": [
                trigger.to_dict() for trigger in result.coarse_triggers
            ],
            "signal_counts": result.signal_counts,
        }
        tool_audits = [
            {
                "sequence": step.sequence,
                "logical_step": step.logical_step,
                "attempt": step.attempt,
                "audits": [dict(audit) for audit in step.outcome.tool_audits],
            }
            for step in result.steps
            if step.outcome.tool_audits
        ]
        if tool_audits:
            record["tool_audits"] = tool_audits
        context_compactions = [
            {
                "sequence": step.sequence,
                "logical_step": step.logical_step,
                "attempt": step.attempt,
                "audits": [dict(audit) for audit in step.outcome.context_compactions],
            }
            for step in result.steps
            if step.outcome.context_compactions
        ]
        if context_compactions:
            record["context_compactions"] = context_compactions
        if result.prompt_cache_summary is not None:
            record["prompt_cache"] = result.prompt_cache_summary.to_dict()
        if result.verification_records:
            record["self_verification"] = {
                "verification_ran": True,
                "affected_outcome": any(
                    record.status is not VerificationStatus.VERIFIED
                    for record in result.verification_records
                ),
                "run_status": result.status.value,
                "tokens_used": result.verification_tokens_used,
                "status_counts": result.verification_status_counts,
                "records": [item.to_dict() for item in result.verification_records],
            }
        runtime = getattr(self, "_native_runtime", None)
        reconciliation = getattr(runtime, "last_provider_call_reconciliation", None)
        if reconciliation is not None:
            record["provider_call_reconciliation"] = dict(reconciliation)
        self._native_phases.append(record)

        self._write_run_record()

    def _write_run_record(self) -> None:
        """Persist configuration even when no phase reaches a terminal result."""

        output = Path(self.logs_dir) / "driftlock-native-result.json"
        runtime = getattr(self, "_native_runtime", None)
        if runtime is None:
            # Preserve the narrow unit seam used by historical phase-record tests;
            # real configured runs create the runtime before writing this file.
            payload = {"phases": self._native_phases}
        else:
            component_report = runtime.component_report()
            payload = {
                "schema_version": 2,
                "active_components": component_report["active"],
                "components": component_report["components"],
                "phases": self._native_phases,
            }
        output.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    async def _ensure_runtime(
        self, environment: Any, context: Any, instruction: str
    ) -> LHTBNativeAgentRuntime:
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
        retrieval_tool = None
        if self._native_agentic_retrieval:
            assert self._native_retrieval_library is not None
            assert self._native_retrieval_embedder is not None
            retrieval_tool = await build_remote_agentic_retrieval_tool(
                environment,
                remote_workspace=workspace,
                store_dir=store_root,
                remote_tmp_dir="/tmp",
                user=environment.default_user,
                skill_library=self._native_retrieval_library,
                embed=self._native_retrieval_embedder,
                memory_store=self._native_memory_store,
            )
        delegation_tool = None
        if self._native_delegation:
            child = ToolCallingSubagentExecutor(
                environment,
                observer,
                self._native_provider,
                max_output_tokens=self._native_max_output_tokens,
                prefill_estimator=self._native_provider.prefill_estimate,
                max_tool_output_chars=self._native_max_tool_output_characters,
                max_tool_calls_per_step=self._native_max_tool_calls_per_step,
                max_history_characters=self._native_max_history_characters,
                user=environment.default_user,
            )
            delegation_tool = DelegationTool(child, config=DelegationConfig())
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
            agent_max_tool_output_characters=(self._native_max_tool_output_characters),
            agent_max_tool_calls_per_step=self._native_max_tool_calls_per_step,
            agent_max_history_characters=self._native_max_history_characters,
            retrieval_tool=retrieval_tool,
            retrieval_embedder_identity=(
                {
                    "import_path": "driftlock.st_embedder:embed",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
                }
                if retrieval_tool is not None
                else None
            ),
            memory_store=self._native_memory_store,
            memory_task_id=(
                hashlib.sha256(instruction.encode()).hexdigest()
                if self._native_memory_store is not None
                else None
            ),
            memory_run_id=(
                self._native_memory_run_id
                if self._native_memory_store is not None
                else None
            ),
            delegation_tool=delegation_tool,
            planning=self._native_planning,
            parallel_tool_calls=self._native_parallel_reads,
            prompt_cache=(PromptCacheConfig() if self._native_prompt_cache else None),
            explicit_prompt_cache_control=(
                self._native_low_level.explicit_prompt_cache_control
            ),
            self_verification=(
                SelfVerificationConfig() if self._native_self_verification else None
            ),
        )
        self._native_runtime = runtime
        self._native_environment = environment
        self._native_context_id = id(context)
        self._native_usage_recorder = ContextUsageRecorder(context)
        self._write_run_record()
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
            components=runtime.component_report(),
        )
