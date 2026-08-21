"""Harbor-free wiring for driftlock's native tool-calling LHTB agent."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from driftlock.agent import (
    AgentCompletion,
    AgentCompletionRequest,
    AgentProviderError,
    ToolCall,
    ToolCallingAgent,
)
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.judges import FineJudge
from driftlock.lhtb import WorkspaceDeltaObserver
from driftlock.models import RunResult, StepOutcome, StepTokenBudgetExhausted
from driftlock.remote import RemoteArchiveCheckpointStore, RemoteEnvironment
from driftlock.runner import DriftlockRunner, RunnerConfig


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    """Exact usage returned by a single physical provider response."""

    input_tokens: int = 0
    cache_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "cache_tokens", "output_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cache_tokens > self.input_tokens:
            raise ValueError("cache_tokens cannot exceed input_tokens")
        if (
            isinstance(self.cost_usd, bool)
            or not isinstance(self.cost_usd, (int, float))
            or not math.isfinite(float(self.cost_usd))
            or self.cost_usd < 0
        ):
            raise ValueError("cost_usd must be a finite non-negative number")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: ProviderUsage) -> ProviderUsage:
        if not isinstance(other, ProviderUsage):
            return NotImplemented
        return ProviderUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_tokens=self.cache_tokens + other.cache_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def delta_from(self, before: ProviderUsage) -> ProviderUsage:
        values = {
            "input_tokens": self.input_tokens - before.input_tokens,
            "cache_tokens": self.cache_tokens - before.cache_tokens,
            "output_tokens": self.output_tokens - before.output_tokens,
            "cost_usd": self.cost_usd - before.cost_usd,
        }
        if any(value < 0 for value in values.values()):
            raise RuntimeError("provider accounting moved backwards")
        return ProviderUsage(**values)


class ContextUsageRecorder:
    """Apply monotonic native-agent and judge usage to a Harbor-like context."""

    def __init__(self, context: Any) -> None:
        metadata = context.metadata or {}
        self._context_id = id(context)
        self._base = ProviderUsage(
            input_tokens=context.n_input_tokens or 0,
            cache_tokens=context.n_cache_tokens or 0,
            output_tokens=context.n_output_tokens or 0,
            cost_usd=context.cost_usd or 0.0,
        )
        self._base_request_times = tuple(metadata.get("api_request_times_msec") or ())

    def apply(
        self,
        context: Any,
        *,
        agent_usage: ProviderUsage,
        judge_usage: ProviderUsage | None = None,
        agent_request_times_msec: tuple[float, ...] = (),
        judge_request_times_msec: tuple[float, ...] = (),
        provider_request_count: int,
    ) -> None:
        if id(context) != self._context_id:
            raise RuntimeError("usage recorder cannot switch contexts")
        if (
            not isinstance(provider_request_count, int)
            or isinstance(provider_request_count, bool)
            or provider_request_count < 0
        ):
            raise ValueError("provider_request_count must be non-negative")
        resolved_judge_usage = judge_usage or ProviderUsage()
        combined = self._base + agent_usage + resolved_judge_usage
        context.n_input_tokens = combined.input_tokens
        context.n_cache_tokens = combined.cache_tokens
        context.n_output_tokens = combined.output_tokens
        context.cost_usd = combined.cost_usd if combined.cost_usd > 0 else None
        request_times = [
            *self._base_request_times,
            *agent_request_times_msec,
            *judge_request_times_msec,
        ]
        metadata = dict(context.metadata or {})
        metadata["api_request_times_msec"] = request_times
        metadata["llm_time_sec"] = sum(request_times) / 1000.0
        metadata["n_episodes"] = provider_request_count
        metadata["driftlock_native_usage"] = {
            "input_tokens": agent_usage.input_tokens,
            "cache_tokens": agent_usage.cache_tokens,
            "output_tokens": agent_usage.output_tokens,
            "cost_usd": agent_usage.cost_usd,
            "provider_request_count": provider_request_count,
            "judge_request_count": len(judge_request_times_msec),
        }
        context.metadata = metadata


@dataclass(frozen=True, slots=True)
class BilledProviderResponse:
    """Text and usage returned by one low-level provider attempt."""

    text: str
    usage: ProviderUsage
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("provider response text must be a string")
        if not isinstance(self.usage, ProviderUsage):
            raise TypeError("provider response usage must be ProviderUsage")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be a boolean")


class BilledProviderFailure(RuntimeError):
    """A failed physical provider attempt whose exact usage is known."""

    def __init__(self, message: str, *, usage: ProviderUsage) -> None:
        super().__init__(message)
        self.usage = usage


class PhysicalProviderBoundaryError(RuntimeError):
    """Raised when one logical agent step made other than one physical request."""


class SingleAttemptCall(Protocol):
    """The Harbor-facing sliver needed by the provider-neutral adapter."""

    @property
    def physical_call_count(self) -> int: ...

    async def __call__(
        self, prompt: str, *, max_output_tokens: int
    ) -> BilledProviderResponse: ...


class AuditedCompletionProvider(Protocol):
    """Completion callable with monotonic physical-call and usage accounting."""

    @property
    def provider_call_count(self) -> int: ...

    @property
    def usage(self) -> ProviderUsage: ...

    def prefill_estimate(self, request: AgentCompletionRequest) -> int: ...

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion: ...


class SingleAttemptJSONProvider:
    """Translate native agent requests to one JSON-producing provider attempt."""

    def __init__(self, call: SingleAttemptCall) -> None:
        if not callable(call):
            raise TypeError("call must be callable")
        count = call.physical_call_count
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise TypeError("physical_call_count must be a non-negative integer")
        self._call = call
        self._usage = ProviderUsage()

    @property
    def provider_call_count(self) -> int:
        count = self._call.physical_call_count
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise TypeError("physical_call_count must be a non-negative integer")
        return count

    @property
    def usage(self) -> ProviderUsage:
        return self._usage

    def prefill_estimate(self, request: AgentCompletionRequest) -> int:
        prompt = _provider_prompt(request)
        wire_messages = [{"role": "user", "content": prompt}]
        encoded = json.dumps(
            wire_messages,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return len(encoded) + 256

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
        calls_before = self.provider_call_count
        try:
            response = await self._call(
                _provider_prompt(request),
                max_output_tokens=request.max_output_tokens,
            )
        except BilledProviderFailure as error:
            self._require_one_call(calls_before)
            self._usage = self._usage + error.usage
            raise AgentProviderError(
                str(error), tokens=error.usage.total_tokens
            ) from error
        except Exception:
            self._require_one_call(calls_before)
            raise

        self._require_one_call(calls_before)
        self._usage = self._usage + response.usage
        if response.truncated:
            return AgentCompletion(
                text=response.text,
                tokens=response.usage.total_tokens,
                truncated=True,
            )
        try:
            text, calls = _decode_provider_response(response.text)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise AgentProviderError(
                f"provider returned invalid tool JSON: {error}",
                tokens=response.usage.total_tokens,
            ) from error
        return AgentCompletion(
            text=text,
            tool_calls=calls,
            tokens=response.usage.total_tokens,
        )

    def _require_one_call(self, calls_before: int) -> None:
        calls = self.provider_call_count - calls_before
        if calls != 1:
            raise PhysicalProviderBoundaryError(
                "one native-agent step must make exactly one physical provider "
                f"request; observed {calls}"
            )


class LHTBNativeAgentRuntime:
    """Run ``ToolCallingAgent`` with remote checkpoints and one shared budget."""

    def __init__(
        self,
        environment: RemoteEnvironment,
        observer: WorkspaceDeltaObserver,
        provider: AuditedCompletionProvider,
        *,
        remote_workspace: str,
        store_dir: Path | str,
        user: str | int | None,
        runner_config: RunnerConfig,
        heuristic_config: HeuristicConfig | None = None,
        fine_judge: FineJudge | None = None,
        plan: str = "inspect, implement, verify",
        retain_checkpoints: bool = False,
        remote_tmp_dir: str = "/tmp",
        agent_max_output_tokens: int = 8192,
        agent_min_output_tokens: int = 64,
        shell_timeout_sec: int = 60,
    ) -> None:
        workspace = PurePosixPath(remote_workspace)
        if not workspace.is_absolute() or workspace == PurePosixPath("/"):
            raise ValueError("remote_workspace must be an absolute non-root path")
        if runner_config.max_tokens is None:
            raise ValueError("native LHTB runs require a finite total token budget")
        if not isinstance(plan, str):
            raise TypeError("plan must be a string")
        self.environment = environment
        self.observer = observer
        self.provider = provider
        self.remote_workspace = remote_workspace
        self.store_dir = Path(store_dir).expanduser().resolve()
        self.user = user
        self.runner_config = runner_config
        self.heuristic_config = heuristic_config or HeuristicConfig()
        self.fine_judge = fine_judge
        self.plan = plan
        self.retain_checkpoints = retain_checkpoints
        self.remote_tmp_dir = remote_tmp_dir
        self.agent = ToolCallingAgent(
            environment,
            observer,
            provider,
            max_output_tokens=agent_max_output_tokens,
            min_output_tokens=agent_min_output_tokens,
            prefill_estimator=provider.prefill_estimate,
            shell_timeout_sec=shell_timeout_sec,
            user=user,
        )
        self.tokens_consumed = 0
        self.agent_tokens_consumed = 0
        self.judge_tokens_consumed = 0
        self.phase_count = 0
        self.last_result: RunResult | None = None

    @property
    def tokens_remaining(self) -> int:
        assert self.runner_config.max_tokens is not None
        return max(0, self.runner_config.max_tokens - self.tokens_consumed)

    async def run(
        self,
        *,
        goal: str,
        initial_state: Mapping[str, Any] | None = None,
    ) -> RunResult:
        if self.tokens_remaining == 0:
            raise StepTokenBudgetExhausted("native LHTB token budget is exhausted")
        phase_dir = self.store_dir / f"phase-{self.phase_count}"
        self.phase_count += 1
        store = RemoteArchiveCheckpointStore(
            self.environment,
            remote_workspace=self.remote_workspace,
            store_dir=phase_dir,
            remote_tmp_dir=self.remote_tmp_dir,
            user=self.user,
        )
        runner = DriftlockRunner(
            store,
            HeuristicJudge(self.heuristic_config),
            fine_judge=self.fine_judge,
            config=replace(self.runner_config, max_tokens=self.tokens_remaining),
        )
        usage_before = self.provider.usage
        calls_before = self.provider.provider_call_count

        async def audited_step(context: Any) -> StepOutcome:
            step_calls_before = self.provider.provider_call_count
            step_usage_before = self.provider.usage
            outcome = await self.agent(context)
            calls = self.provider.provider_call_count - step_calls_before
            usage = self.provider.usage.delta_from(step_usage_before)
            if calls != 1:
                raise PhysicalProviderBoundaryError(
                    "one native-agent step must make exactly one physical provider "
                    f"request; observed {calls}"
                )
            if usage.total_tokens != outcome.tokens:
                raise RuntimeError(
                    "provider usage does not reconcile with StepOutcome.tokens"
                )
            return outcome

        try:
            result = await runner.run(
                goal=goal,
                plan=self.plan,
                step=audited_step,
                initial_state=(
                    dict(initial_state)
                    if initial_state is not None
                    else self.agent.initial_state()
                ),
            )
            phase_usage = self.provider.usage.delta_from(usage_before)
            phase_calls = self.provider.provider_call_count - calls_before
            if phase_calls != len(result.steps):
                raise RuntimeError(
                    "physical provider calls do not reconcile with recorded steps"
                )
            if phase_usage.total_tokens != result.agent_tokens_used:
                raise RuntimeError(
                    "provider usage does not reconcile with agent token accounting"
                )
            self.tokens_consumed += result.tokens_used
            self.agent_tokens_consumed += result.agent_tokens_used
            self.judge_tokens_consumed += result.judge_tokens_used
            self.last_result = result
            return result
        finally:
            if not self.retain_checkpoints:
                shutil.rmtree(phase_dir, ignore_errors=True)


def append_verifier_feedback(state: Mapping[str, Any], feedback: str) -> dict[str, Any]:
    """Append Harbor verifier feedback to a completed native conversation."""
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("verifier feedback must be a non-empty string")
    from driftlock.agent import AgentConversationCodec

    codec = AgentConversationCodec()
    messages, steps = codec.decode(state)
    messages.append(
        {
            "role": "user",
            "content": (
                "The task verifier rejected the previous completion. Continue from "
                f"the current workspace and address this feedback:\n{feedback}"
            ),
        }
    )
    return codec.encode(messages, steps=steps)


def _provider_prompt(request: AgentCompletionRequest) -> str:
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in request.tools
    ]
    conversation = json.dumps(
        request.messages,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    tool_spec = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return (
        "Continue the tool-agent conversation below. Return exactly one JSON object "
        "with keys 'text' (string) and 'tool_calls' (array). Each tool call must "
        "contain 'name', 'arguments', and optional 'call_id'. Do not wrap the JSON "
        "in Markdown.\n\nConversation JSON:\n"
        + conversation
        + "\n\nAvailable tools JSON:\n"
        + tool_spec
    )


def _decode_provider_response(text: str) -> tuple[str, tuple[ToolCall, ...]]:
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise TypeError("response must be a JSON object")
    if set(payload) != {"text", "tool_calls"}:
        raise ValueError("response must contain exactly text and tool_calls")
    response_text = payload["text"]
    raw_calls = payload["tool_calls"]
    if not isinstance(response_text, str):
        raise TypeError("text must be a string")
    if not isinstance(raw_calls, list):
        raise TypeError("tool_calls must be an array")
    calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            raise TypeError("each tool call must be an object")
        if not {"name", "arguments"} <= raw.keys() or not set(raw) <= {
            "name",
            "arguments",
            "call_id",
        }:
            raise ValueError("tool call keys are invalid")
        calls.append(
            ToolCall(
                name=raw["name"],
                arguments=raw["arguments"],
                call_id=raw.get("call_id", ""),
            )
        )
    return response_text, tuple(calls)
