"""Harbor-free wiring for driftlock's native tool-calling LHTB agent."""

from __future__ import annotations

import asyncio
import json
import math
import shlex
import shutil
import tarfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from driftlock.agent import (
    DEFAULT_MAX_HISTORY_CHARACTERS,
    DEFAULT_MAX_TOOL_CALLS_PER_STEP,
    DEFAULT_MAX_TOOL_OUTPUT_CHARACTERS,
    MAX_PLAN_DESCRIPTION_CHARACTERS,
    MAX_PLAN_STEPS,
    PARALLEL_HISTORY_RESERVE_CHARACTERS,
    AgentCompletion,
    AgentCompletionRequest,
    AgentProviderError,
    ToolCall,
    ToolCallingAgent,
)
from driftlock.agentic_retrieval import AgenticRetrievalTool, RetrievalCorpusStatus
from driftlock.delegation import DelegationTool
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.judges import FineJudge
from driftlock.lhtb import WorkspaceDeltaObserver
from driftlock.memory import MemoryStore
from driftlock.models import (
    RunResult,
    RunStatus,
    StepOutcome,
    StepTokenBudgetExhausted,
    aggregate_run_summary,
)
from driftlock.prompt_cache import PromptCacheConfig
from driftlock.remote import RemoteArchiveCheckpointStore, RemoteEnvironment
from driftlock.runner import DriftlockRunner, RunnerConfig
from driftlock.skill_admission import SkillLibrary
from driftlock.verification import SelfVerificationConfig, VerificationStatus


class NativeComponentConfigurationError(ValueError):
    """An opted-in native component cannot preserve its declared invariants."""


MCP_EXPERIMENT_EXCLUSION_REASON = (
    "LHTB tasks provide no MCP server or credential-free server configuration; "
    "exposing an inert flag would make an enabled treatment indistinguishable "
    "from an empty result. MCP remains available on ToolCallingAgent itself."
)


def validate_parallel_compaction_bounds(
    *,
    parallel_tool_calls: bool,
    max_tool_calls_per_step: int,
    max_tool_output_characters: int,
    max_history_characters: int,
) -> None:
    """Keep a worst-case multi-tool turn retainable by compaction."""

    if not isinstance(parallel_tool_calls, bool):
        raise TypeError("parallel_tool_calls must be a boolean")
    required = (
        max_tool_calls_per_step * max_tool_output_characters
        + PARALLEL_HISTORY_RESERVE_CHARACTERS
    )
    if max_history_characters < required:
        raise NativeComponentConfigurationError(
            "parallel reads and serial tool-call batches require "
            "max_history_characters >= "
            "max_tool_calls_per_step * max_tool_output_characters + "
            f"{PARALLEL_HISTORY_RESERVE_CHARACTERS}; got "
            f"{max_history_characters} < {required}"
        )


def _dataclass_config(value: Any) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


@dataclass(frozen=True, slots=True)
class _ProviderCallExpectation:
    exact: int | None
    incomplete_reasons: tuple[str, ...] = ()


def _expected_step_provider_calls(outcome: StepOutcome) -> _ProviderCallExpectation:
    """Derive an exact call count, or preserve why component evidence cannot."""

    expected = 1
    verification = outcome.verification
    # A budget refusal before dispatch records zero tokens. The other
    # BUDGET_EXHAUSTED path is reached after a paid response reports usage above
    # the allowance, so it still consumed one physical call.
    if verification is not None and (
        verification.status is not VerificationStatus.BUDGET_EXHAUSTED
        or verification.tokens > 0
    ):
        expected += 1
    incomplete_reasons: list[str] = []
    for audit in outcome.tool_audits:
        result = audit.get("result") if isinstance(audit, Mapping) else None
        if not isinstance(result, Mapping):
            continue
        if result.get("mode") != "sub-agent-delegation":
            continue
        tokens = result.get("tokens")
        if (
            not isinstance(tokens, Mapping)
            or tokens.get("accounting_known") is not True
        ):
            incomplete_reasons.append("delegation_provider_calls_unknown")
            continue
        steps = result.get("steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise RuntimeError("delegation audit has invalid child step accounting")
        expected += steps
    return _ProviderCallExpectation(
        None if incomplete_reasons else expected,
        tuple(incomplete_reasons),
    )


def _verification_effect_report(result: RunResult) -> dict[str, Any]:
    return {
        "verification_ran": True,
        "affected_outcome": any(
            record.status is not VerificationStatus.VERIFIED
            for record in result.verification_records
        ),
        "run_status": result.status.value,
        "tokens_used": result.verification_tokens_used,
        "status_counts": result.verification_status_counts,
    }


def _component_token_mismatch_reason(outcome: StepOutcome) -> str | None:
    verification = outcome.verification
    if (
        verification is not None
        and verification.status is VerificationStatus.BUDGET_EXHAUSTED
        and verification.tokens > 0
    ):
        return "verification_provider_usage_exceeded_allowance"
    for audit in outcome.tool_audits:
        result = audit.get("result") if isinstance(audit, Mapping) else None
        if not isinstance(result, Mapping):
            continue
        if result.get("mode") != "sub-agent-delegation":
            continue
        tokens = result.get("tokens")
        if (
            not isinstance(tokens, Mapping)
            or tokens.get("accounting_known") is not True
        ):
            return "delegation_token_accounting_unknown"
    return None


_MAX_RETRIEVAL_EXCLUSION_SAMPLES = 16
_MAX_RETRIEVAL_EXCLUSION_VALUE_CHARACTERS = 512


def _bounded_retrieval_corpus_report(tool: AgenticRetrievalTool) -> dict[str, Any]:
    report = tool.corpus.snapshot_report()
    exclusions = report.pop("build_exclusions", [])
    samples = [
        {
            str(name): (
                value[:_MAX_RETRIEVAL_EXCLUSION_VALUE_CHARACTERS]
                if isinstance(value, str)
                else value
            )
            for name, value in exclusion.items()
        }
        for exclusion in exclusions[:_MAX_RETRIEVAL_EXCLUSION_SAMPLES]
        if isinstance(exclusion, Mapping)
    ]
    report["build_exclusion_sample"] = samples
    report["unreported_build_exclusion_count"] = max(
        0, report["build_exclusion_count"] - len(samples)
    )
    return report


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
        physical_provider_request_count: int | None = None,
        unknown_billed_request_count: int = 0,
    ) -> None:
        if id(context) != self._context_id:
            raise RuntimeError("usage recorder cannot switch contexts")
        if (
            not isinstance(provider_request_count, int)
            or isinstance(provider_request_count, bool)
            or provider_request_count < 0
        ):
            raise ValueError("provider_request_count must be non-negative")
        physical_count = (
            provider_request_count
            if physical_provider_request_count is None
            else physical_provider_request_count
        )
        for name, value in (
            ("physical_provider_request_count", physical_count),
            ("unknown_billed_request_count", unknown_billed_request_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if physical_count != provider_request_count + unknown_billed_request_count:
            raise ValueError(
                "physical provider requests must equal exact-usage plus "
                "unknown-billed requests"
            )
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
            "physical_provider_request_count": physical_count,
            "unknown_billed_request_count": unknown_billed_request_count,
            "usage_complete": unknown_billed_request_count == 0,
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
    """A failed physical attempt, with exact usage when the provider supplied it."""

    def __init__(self, message: str, *, usage: ProviderUsage | None) -> None:
        super().__init__(message)
        self.usage = usage


class UnknownBilledProviderUsage(RuntimeError):
    """A physical request was billed but its exact usage is unavailable."""


class PhysicalProviderBoundaryError(RuntimeError):
    """Raised when one logical agent step made other than one physical request."""


class SingleAttemptCall(Protocol):
    """The Harbor-facing sliver needed by the provider-neutral adapter."""

    @property
    def physical_call_count(self) -> int: ...

    async def __call__(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        cacheable_prefix_characters: int | None,
    ) -> BilledProviderResponse: ...


class AuditedCompletionProvider(Protocol):
    """Completion callable with monotonic physical-call and usage accounting."""

    @property
    def provider_call_count(self) -> int: ...

    @property
    def usage(self) -> ProviderUsage: ...

    @property
    def exact_usage_request_count(self) -> int: ...

    @property
    def unknown_billed_request_count(self) -> int: ...

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
        self._exact_usage_request_count = 0
        self._unknown_billed_request_count = 0

    @property
    def provider_call_count(self) -> int:
        count = self._call.physical_call_count
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise TypeError("physical_call_count must be a non-negative integer")
        return count

    @property
    def usage(self) -> ProviderUsage:
        return self._usage

    @property
    def exact_usage_request_count(self) -> int:
        return self._exact_usage_request_count

    @property
    def unknown_billed_request_count(self) -> int:
        return self._unknown_billed_request_count

    def prefill_estimate(self, request: AgentCompletionRequest) -> int:
        prompt = _provider_prompt(request)
        wire_messages = [{"role": "user", "content": prompt.text}]
        encoded = json.dumps(
            wire_messages,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return len(encoded) + 256

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
        calls_before = self.provider_call_count
        prompt = _provider_prompt(request)
        try:
            response = await self._call(
                prompt.text,
                max_output_tokens=request.max_output_tokens,
                cacheable_prefix_characters=prompt.cacheable_prefix_characters,
            )
        except asyncio.CancelledError:
            # A delegation timeout can cancel an in-flight paid request before
            # usage is returned. Preserve that physical request as explicitly
            # unknown accounting and let the component outcome remain a normal
            # trial observation.
            self._require_one_call(calls_before)
            self._unknown_billed_request_count += 1
            raise
        except BilledProviderFailure as error:
            self._require_one_call(calls_before)
            if error.usage is None:
                self._unknown_billed_request_count += 1
                raise UnknownBilledProviderUsage(
                    "provider request was billed but exact usage is unavailable"
                ) from error
            self._exact_usage_request_count += 1
            self._usage = self._usage + error.usage
            raise AgentProviderError(
                str(error), tokens=error.usage.total_tokens
            ) from error
        except Exception as error:
            self._require_one_call(calls_before)
            self._unknown_billed_request_count += 1
            raise UnknownBilledProviderUsage(
                "provider request failed without auditable billed usage"
            ) from error

        self._require_one_call(calls_before)
        self._exact_usage_request_count += 1
        self._usage = self._usage + response.usage
        if response.truncated:
            return AgentCompletion(
                text=response.text,
                tokens=response.usage.total_tokens,
                truncated=True,
                prompt_tokens=response.usage.input_tokens,
                cached_tokens=response.usage.cache_tokens,
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
            prompt_tokens=response.usage.input_tokens,
            cached_tokens=response.usage.cache_tokens,
        )

    def _require_one_call(self, calls_before: int) -> None:
        calls = self.provider_call_count - calls_before
        if calls != 1:
            if calls > 0:
                self._unknown_billed_request_count += calls
            raise PhysicalProviderBoundaryError(
                "one native-agent step must make exactly one physical provider "
                f"request; observed {calls}"
            )


def exact_provider_usage(response: Any) -> ProviderUsage | None:
    """Extract exact billed usage from a duck-typed provider response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    values = (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "cache_tokens", None),
        getattr(usage, "completion_tokens", None),
    )
    cost = getattr(usage, "cost_usd", None)
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in values
    ):
        return None
    if (
        not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        return None
    return ProviderUsage(
        input_tokens=values[0],
        cache_tokens=values[1],
        output_tokens=values[2],
        cost_usd=float(cost),
    )


def billed_provider_response(response: Any) -> BilledProviderResponse:
    """Map one duck-typed successful provider response to the audited boundary."""
    usage = exact_provider_usage(response)
    if usage is None:
        raise BilledProviderFailure(
            "provider response is missing exact billed usage", usage=None
        )
    content = getattr(response, "content", None)
    if not isinstance(content, str):
        raise BilledProviderFailure(
            "provider response content is not a string", usage=usage
        )
    return BilledProviderResponse(content, usage)


def billed_provider_exception(error: Exception) -> BilledProviderResponse:
    """Triage a duck-typed provider exception without Harbor or LiteLLM imports."""
    usage = exact_provider_usage(getattr(error, "response", None))
    partial = getattr(error, "truncated_response", None)
    if usage is not None and isinstance(partial, str):
        return BilledProviderResponse(partial, usage, truncated=True)
    raise BilledProviderFailure(str(error), usage=usage) from error


def native_checkpoint_store_root(logs_dir: Path | str) -> Path:
    """Derive host-only native checkpoint storage outside the agent log mount."""
    resolved_logs = Path(logs_dir).expanduser().resolve()
    store_root = resolved_logs.parent / ".driftlock-native-checkpoints"
    if store_root == resolved_logs or resolved_logs in store_root.parents:
        raise ValueError("checkpoint storage must be outside the agent log mount")
    return store_root


async def build_remote_agentic_retrieval_tool(
    environment: RemoteEnvironment,
    *,
    remote_workspace: str,
    store_dir: Path | str,
    remote_tmp_dir: str,
    user: str | int | None,
    skill_library: SkillLibrary,
    embed: Any,
    memory_store: MemoryStore | None = None,
) -> AgenticRetrievalTool:
    """Snapshot one remote task, build its immutable corpus, and clean staging."""

    if not callable(embed):
        raise TypeError("retrieval embedder must be callable")
    staging_store = Path(store_dir) / f".retrieval-{uuid.uuid4().hex}"
    extracted = staging_store / "workspace"
    snapshot_store = RemoteArchiveCheckpointStore(
        environment,
        remote_workspace=remote_workspace,
        store_dir=staging_store,
        remote_tmp_dir=remote_tmp_dir,
        user=user,
    )
    try:
        checkpoint = await snapshot_store.create({}, step=0, label="retrieval")
        archive = checkpoint.path / "workspace.tar.gz"
        extracted.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(extracted, filter="data")
        tool = AgenticRetrievalTool.from_workspace(
            extracted,
            skill_library,
            embed,
            memory_store=memory_store,
        )
        corpus = tool.corpus
        if corpus.status is not RetrievalCorpusStatus.READY:
            refusal = dict(corpus.refusal or {})
            reason = refusal.get("reason", "corpus_build_failed")
            raise NativeComponentConfigurationError(
                "driftlock agentic retrieval corpus build failed configuration: "
                f"{reason}"
            )
        if not corpus.documents:
            raise NativeComponentConfigurationError(
                "driftlock agentic retrieval corpus is empty; no documents were indexed"
            )
        return tool
    finally:
        shutil.rmtree(staging_store, ignore_errors=True)


class NativeProcessQuiescer:
    """Stop processes created by the native agent before restoring its workspace."""

    def __init__(
        self,
        environment: RemoteEnvironment,
        *,
        user: str | int | None,
        timeout_sec: int = 60,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("process cleanup timeout must be positive")
        self.environment = environment
        self.user = user
        self.timeout_sec = timeout_sec
        self._baseline: tuple[str, ...] | None = None

    async def prepare(self) -> None:
        """Capture the pre-agent PID/start-time identities once per trial."""
        if self._baseline is not None:
            return
        result = await self.environment.exec(
            _process_identity_snapshot_script(),
            timeout_sec=self.timeout_sec,
            user=self.user,
        )
        _require_exec_success(result, "capture pre-agent process baseline")
        identities = tuple((getattr(result, "stdout", None) or "").splitlines())
        if any(not _valid_process_identity(value) for value in identities) or len(
            identities
        ) != len(set(identities)):
            raise RuntimeError("remote process baseline is malformed")
        self._baseline = identities

    async def before_restore(self, remote_workspace: str) -> None:
        """Freeze and kill every non-baseline process, or abort the restore."""
        if self._baseline is None:
            raise RuntimeError("process baseline must be captured before restore")
        result = await self.environment.exec(
            _quiesce_native_processes_script(
                remote_workspace, process_baseline=self._baseline
            ),
            timeout_sec=self.timeout_sec,
            user=self.user,
        )
        _require_exec_success(result, "quiesce rejected native-agent processes")


def apply_native_accounting(
    context: Any,
    *,
    recorder: ContextUsageRecorder,
    runtime: Any,
    provider: AuditedCompletionProvider,
    agent_request_times_msec: tuple[float, ...],
    judge: Any | None = None,
    reconcile: bool,
) -> None:
    """Reconcile and publish native provider/judge accounting."""
    provider_usage = provider.usage
    judge_input = judge.n_input_tokens if judge is not None else 0
    judge_cache = judge.n_cache_tokens if judge is not None else 0
    judge_output = judge.n_output_tokens if judge is not None else 0
    judge_cost = judge.cost_usd if judge is not None else 0.0
    if reconcile:
        reconciliation = getattr(runtime, "last_provider_call_reconciliation", None)
        tolerated_unknown_calls = (
            reconciliation.get("cumulative_unknown_billed_calls", 0)
            if isinstance(reconciliation, Mapping)
            else 0
        )
        if provider.unknown_billed_request_count != tolerated_unknown_calls:
            raise RuntimeError("native provider has unexplained unknown billed usage")
        tolerated_token_overage = (
            reconciliation.get("cumulative_provider_tokens_outside_runner_budget", 0)
            if isinstance(reconciliation, Mapping)
            else 0
        )
        if provider_usage.total_tokens != (
            runtime.agent_tokens_consumed + tolerated_token_overage
        ):
            raise RuntimeError("native provider usage does not reconcile with steps")
        if judge_input + judge_output != runtime.judge_tokens_consumed:
            raise RuntimeError("native judge usage does not reconcile with verdicts")
    recorder.apply(
        context,
        agent_usage=provider_usage,
        judge_usage=ProviderUsage(
            input_tokens=judge_input,
            cache_tokens=judge_cache,
            output_tokens=judge_output,
            cost_usd=judge_cost,
        ),
        agent_request_times_msec=agent_request_times_msec,
        judge_request_times_msec=(
            tuple(judge.request_times_msec) if judge is not None else ()
        ),
        provider_request_count=provider.exact_usage_request_count,
        physical_provider_request_count=provider.provider_call_count,
        unknown_billed_request_count=provider.unknown_billed_request_count,
    )


def set_native_result_metadata(
    context: Any,
    *,
    result: RunResult,
    runtime: Any,
    trial_token_budget: int,
    components: Mapping[str, Any] | None = None,
) -> None:
    """Publish one successful native phase's terminal metadata."""
    metadata = dict(context.metadata or {})
    previous = metadata.get("driftlock")
    summary = aggregate_run_summary(
        previous if isinstance(previous, dict) else None,
        result,
    )
    summary.update(
        {
            "trial_tokens_used": runtime.tokens_consumed,
            "trial_token_budget": trial_token_budget,
        }
    )
    if components is not None:
        summary["components"] = dict(components)
    if result.verification_records:
        summary["self_verification"] = _verification_effect_report(result)
    provider_call_reconciliation = getattr(
        runtime, "last_provider_call_reconciliation", None
    )
    if provider_call_reconciliation is not None:
        summary["provider_call_reconciliation"] = dict(provider_call_reconciliation)
    metadata["driftlock"] = summary
    metadata["termination_reason"] = (
        "confirmed_task_complete"
        if result.status is RunStatus.COMPLETED
        else f"driftlock_{result.status.value}"
    )
    context.metadata = metadata


def set_native_token_limit_metadata(
    context: Any,
    *,
    runtime: Any,
    trial_token_budget: int,
    components: Mapping[str, Any] | None = None,
) -> None:
    """Publish native trial budget exhaustion metadata."""
    metadata = dict(context.metadata or {})
    metadata["termination_reason"] = "driftlock_token_limit"
    previous = metadata.get("driftlock")
    summary = dict(previous) if isinstance(previous, dict) else {}
    summary.update(
        {
            "status": RunStatus.TOKEN_LIMIT.value,
            "tokens_used": runtime.tokens_consumed,
            "trial_tokens_used": runtime.tokens_consumed,
            "trial_token_budget": trial_token_budget,
        }
    )
    if components is not None:
        summary["components"] = dict(components)
    metadata["driftlock"] = summary
    context.metadata = metadata


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
        agent_max_tool_output_characters: int = DEFAULT_MAX_TOOL_OUTPUT_CHARACTERS,
        agent_max_tool_calls_per_step: int = DEFAULT_MAX_TOOL_CALLS_PER_STEP,
        agent_max_history_characters: int = DEFAULT_MAX_HISTORY_CHARACTERS,
        shell_timeout_sec: int = 60,
        retrieval_tool: AgenticRetrievalTool | None = None,
        retrieval_embedder_identity: Mapping[str, str] | None = None,
        memory_store: MemoryStore | None = None,
        memory_task_id: str | None = None,
        memory_run_id: str | None = None,
        delegation_tool: DelegationTool | None = None,
        planning: bool = False,
        parallel_tool_calls: bool = False,
        prompt_cache: PromptCacheConfig | None = None,
        explicit_prompt_cache_control: bool = False,
        self_verification: SelfVerificationConfig | None = None,
    ) -> None:
        workspace = PurePosixPath(remote_workspace)
        if not workspace.is_absolute() or workspace == PurePosixPath("/"):
            raise ValueError("remote_workspace must be an absolute non-root path")
        if runner_config.max_tokens is None:
            raise ValueError("native LHTB runs require a finite total token budget")
        if not isinstance(plan, str):
            raise TypeError("plan must be a string")
        if not isinstance(explicit_prompt_cache_control, bool):
            raise TypeError("explicit_prompt_cache_control must be a boolean")
        if explicit_prompt_cache_control and prompt_cache is None:
            raise ValueError("explicit prompt cache control requires prompt_cache")
        if (retrieval_tool is None) != (retrieval_embedder_identity is None):
            raise ValueError(
                "retrieval_tool and retrieval_embedder_identity must be configured "
                "together"
            )
        if retrieval_embedder_identity is not None and (
            not isinstance(retrieval_embedder_identity, Mapping)
            or not retrieval_embedder_identity
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or not value
                for name, value in retrieval_embedder_identity.items()
            )
        ):
            raise TypeError(
                "retrieval_embedder_identity must be a non-empty text mapping"
            )
        validate_parallel_compaction_bounds(
            parallel_tool_calls=parallel_tool_calls,
            max_tool_calls_per_step=agent_max_tool_calls_per_step,
            max_tool_output_characters=agent_max_tool_output_characters,
            max_history_characters=agent_max_history_characters,
        )
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
        self.explicit_prompt_cache_control = explicit_prompt_cache_control
        self.retrieval_embedder_identity = (
            dict(retrieval_embedder_identity)
            if retrieval_embedder_identity is not None
            else None
        )
        self.agent = ToolCallingAgent(
            environment,
            observer,
            provider,
            max_output_tokens=agent_max_output_tokens,
            min_output_tokens=agent_min_output_tokens,
            prefill_estimator=provider.prefill_estimate,
            max_tool_output_chars=agent_max_tool_output_characters,
            max_tool_calls_per_step=agent_max_tool_calls_per_step,
            max_history_characters=agent_max_history_characters,
            shell_timeout_sec=shell_timeout_sec,
            user=user,
            retrieval_tool=retrieval_tool,
            memory_store=memory_store,
            memory_task_id=memory_task_id,
            memory_run_id=memory_run_id,
            delegation_tool=delegation_tool,
            planning=planning,
            parallel_tool_calls=parallel_tool_calls,
            prompt_cache=prompt_cache,
            self_verification=self_verification,
        )
        self._process_quiescer = NativeProcessQuiescer(
            environment,
            user=user,
            timeout_sec=shell_timeout_sec,
        )
        self.tokens_consumed = 0
        self.agent_tokens_consumed = 0
        self.judge_tokens_consumed = 0
        self.phase_count = 0
        self.last_result: RunResult | None = None
        self.last_provider_call_reconciliation: dict[str, Any] | None = None
        self._component_unknown_billed_calls = 0
        self._component_provider_token_overage = 0

    def component_report(self) -> dict[str, Any]:
        """Return the complete effective component configuration for attribution."""

        retrieval = self.agent.retrieval_tool
        memory = self.agent.memory_store
        delegation = self.agent.delegation_tool
        verification = self.agent.self_verification
        components: dict[str, Any] = {
            "agentic_retrieval": {
                "enabled": retrieval is not None,
                "configuration": (
                    {
                        **retrieval.corpus.config.to_report(),
                        "fingerprint": retrieval.corpus.config.fingerprint,
                        "embedder": self.retrieval_embedder_identity,
                    }
                    if retrieval is not None
                    else None
                ),
                "corpus": (
                    _bounded_retrieval_corpus_report(retrieval)
                    if retrieval is not None
                    else None
                ),
            },
            "compaction": {
                "enabled": True,
                "max_history_characters": self.agent.max_history_characters,
                "max_tool_output_characters": self.agent.max_tool_output_chars,
                "max_tool_calls_per_step": self.agent.max_tool_calls_per_step,
                "parallel_history_reserve_characters": (
                    PARALLEL_HISTORY_RESERVE_CHARACTERS
                ),
            },
            "planning": {
                "enabled": self.agent.planning,
                "max_steps": MAX_PLAN_STEPS,
                "max_description_characters": MAX_PLAN_DESCRIPTION_CHARACTERS,
            },
            "memory": {
                "enabled": memory is not None,
                "scope": "harbor_trial" if memory is not None else None,
                "limitation": (
                    "memory persists across verifier-resume phases within one "
                    "trial, but never across Harbor tasks or paired trials"
                    if memory is not None
                    else None
                ),
                "root_policy": (
                    "agent_logs/driftlock-memory" if memory is not None else None
                ),
                "configuration": (
                    _dataclass_config(memory.config) if memory is not None else None
                ),
            },
            "delegation": {
                "enabled": delegation is not None,
                "configuration": (
                    _dataclass_config(delegation.config)
                    if delegation is not None
                    else None
                ),
                "child_can_delegate": False if delegation is not None else None,
            },
            "mcp": {
                "enabled": False,
                "availability": "excluded_from_lhtb_experiment",
                "reason": MCP_EXPERIMENT_EXCLUSION_REASON,
            },
            "parallel_reads": {
                "enabled": self.agent.parallel_tool_calls,
                "eligible_tools": ["read_file", "search_files"],
                "required_history_characters": (
                    self.agent.max_tool_calls_per_step
                    * self.agent.max_tool_output_chars
                    + PARALLEL_HISTORY_RESERVE_CHARACTERS
                ),
            },
            "prompt_cache": {
                "enabled": self.agent.prompt_cache is not None,
                "explicit_provider_control": self.explicit_prompt_cache_control,
            },
            "self_verification": {
                "enabled": verification is not None,
                "configuration": (
                    _dataclass_config(verification)
                    if verification is not None
                    else None
                ),
            },
        }
        return {
            "schema_version": 1,
            "active": [
                name for name, report in components.items() if report["enabled"]
            ],
            "components": components,
        }

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
            before_restore=self._process_quiescer.before_restore,
        )
        runner = DriftlockRunner(
            store,
            HeuristicJudge(self.heuristic_config),
            fine_judge=self.fine_judge,
            config=replace(
                self.runner_config,
                max_tokens=self.tokens_remaining,
                checkpoint_on_exit=(
                    self.runner_config.checkpoint_on_exit or self.retain_checkpoints
                ),
            ),
        )
        usage_before = self.provider.usage
        calls_before = self.provider.provider_call_count
        exact_calls_before = self.provider.exact_usage_request_count
        unknown_calls_before = self.provider.unknown_billed_request_count
        step_reconciliations: list[dict[str, Any]] = []

        async def audited_step(context: Any) -> StepOutcome:
            step_calls_before = self.provider.provider_call_count
            step_exact_calls_before = self.provider.exact_usage_request_count
            step_unknown_calls_before = self.provider.unknown_billed_request_count
            step_usage_before = self.provider.usage
            outcome = await self.agent(context)
            delegation = self.agent.delegation_tool
            if delegation is not None:
                # Interrupted children update cancellation-time physical-call
                # accounting asynchronously. Settle them before reconciling this
                # otherwise-successful parent step.
                await delegation.drain_cancelled()
            calls = self.provider.provider_call_count - step_calls_before
            exact_calls = (
                self.provider.exact_usage_request_count - step_exact_calls_before
            )
            unknown_calls = (
                self.provider.unknown_billed_request_count - step_unknown_calls_before
            )
            usage = self.provider.usage.delta_from(step_usage_before)
            expectation = _expected_step_provider_calls(outcome)
            if expectation.exact is not None and calls != expectation.exact:
                if expectation.exact == 1:
                    raise PhysicalProviderBoundaryError(
                        "one native-agent step must make exactly one physical "
                        f"provider request; observed {calls}"
                    )
                raise PhysicalProviderBoundaryError(
                    "native-agent step physical provider requests do not match "
                    "component evidence; expected "
                    f"{expectation.exact}, observed {calls}"
                )
            if calls != exact_calls + unknown_calls:
                raise PhysicalProviderBoundaryError(
                    "native-agent step physical provider requests do not reconcile "
                    "with exact and unknown billed-call accounting"
                )
            step_reconciliations.append(
                {
                    "sequence": context.sequence,
                    "status": (
                        "complete"
                        if expectation.exact is not None
                        else "component_evidence_incomplete"
                    ),
                    "expected_physical_calls": expectation.exact,
                    "observed_physical_calls": calls,
                    "exact_usage_calls": exact_calls,
                    "unknown_billed_calls": unknown_calls,
                    "incomplete_reasons": list(expectation.incomplete_reasons),
                    "runner_accounted_tokens": outcome.tokens,
                    "provider_reported_tokens": usage.total_tokens,
                }
            )
            if usage.total_tokens != outcome.tokens:
                mismatch_reason = _component_token_mismatch_reason(outcome)
                if usage.total_tokens < outcome.tokens or mismatch_reason is None:
                    raise RuntimeError(
                        "provider usage does not reconcile with StepOutcome.tokens"
                    )
                step_reconciliations[-1]["status"] = "component_accounting_incomplete"
                step_reconciliations[-1]["incomplete_reasons"].append(mismatch_reason)
            return outcome

        phase_succeeded = False
        try:
            await self._process_quiescer.prepare()
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
            phase_exact_calls = (
                self.provider.exact_usage_request_count - exact_calls_before
            )
            phase_unknown_calls = (
                self.provider.unknown_billed_request_count - unknown_calls_before
            )
            expectations = [
                _expected_step_provider_calls(step.outcome) for step in result.steps
            ]
            evidence_complete = all(item.exact is not None for item in expectations)
            if evidence_complete:
                expected_phase_calls = sum(
                    item.exact for item in expectations if item.exact is not None
                )
                if phase_calls != expected_phase_calls:
                    raise RuntimeError(
                        "physical provider calls do not reconcile with recorded steps"
                    )
                if phase_exact_calls != expected_phase_calls or phase_unknown_calls:
                    raise RuntimeError(
                        "exact-usage provider calls do not reconcile with recorded "
                        "steps"
                    )
            elif phase_calls != phase_exact_calls + phase_unknown_calls:
                raise RuntimeError(
                    "physical provider calls do not reconcile with exact and "
                    "unknown billed-call accounting"
                )
            phase_token_overage = phase_usage.total_tokens - result.agent_tokens_used
            recorded_step_overage = sum(
                step["provider_reported_tokens"] - step["runner_accounted_tokens"]
                for step in step_reconciliations
            )
            if phase_token_overage < 0 or phase_token_overage != recorded_step_overage:
                raise RuntimeError(
                    "provider usage does not reconcile with agent token accounting"
                )
            self.tokens_consumed += result.tokens_used
            self.agent_tokens_consumed += result.agent_tokens_used
            self.judge_tokens_consumed += result.judge_tokens_used
            self.last_result = result
            if not evidence_complete:
                self._component_unknown_billed_calls += phase_unknown_calls
            self._component_provider_token_overage += phase_token_overage
            reconciliation_complete = (
                evidence_complete
                and phase_unknown_calls == 0
                and phase_token_overage == 0
            )
            self.last_provider_call_reconciliation = {
                "status": (
                    "complete"
                    if reconciliation_complete
                    else "component_accounting_incomplete"
                ),
                "physical_calls": phase_calls,
                "exact_usage_calls": phase_exact_calls,
                "unknown_billed_calls": phase_unknown_calls,
                "cumulative_unknown_billed_calls": (
                    self._component_unknown_billed_calls
                ),
                "provider_tokens_outside_runner_budget": phase_token_overage,
                "cumulative_provider_tokens_outside_runner_budget": (
                    self._component_provider_token_overage
                ),
                "steps": step_reconciliations,
            }
            phase_succeeded = True
            return result
        finally:
            delegation = self.agent.delegation_tool
            if delegation is not None:
                await delegation.drain_cancelled()
            if phase_succeeded and not self.retain_checkpoints:
                shutil.rmtree(phase_dir, ignore_errors=True)


def append_verifier_feedback(state: Mapping[str, Any], feedback: str) -> dict[str, Any]:
    """Append Harbor verifier feedback to a completed native conversation."""
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("verifier feedback must be a non-empty string")
    from driftlock.agent import AgentConversationCodec

    codec = AgentConversationCodec()
    messages, steps, plan = codec.decode_with_plan(state)
    messages.append(
        {
            "role": "user",
            "content": (
                "The task verifier rejected the previous completion. Continue from "
                f"the current workspace and address this feedback:\n{feedback}"
            ),
        }
    )
    payload = state.get(codec.state_key)
    if isinstance(payload, Mapping) and payload.get("schema_version") == 1:
        # Preserve historical verifier-continuation bytes for completed runs;
        # version one has no plan field to carry forward.
        return {
            codec.state_key: {
                "schema_version": 1,
                "messages": messages,
                "steps": steps,
            }
        }
    return codec.encode(messages, steps=steps, plan=plan)


@dataclass(frozen=True, slots=True)
class _ProviderPrompt:
    text: str
    cacheable_prefix_characters: int | None


def _provider_prompt(request: AgentCompletionRequest) -> _ProviderPrompt:
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in request.tools
    ]
    tool_spec = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    instruction = (
        "Continue the tool-agent conversation below. Return exactly one JSON object "
        "with keys 'text' (string) and 'tool_calls' (array). Each tool call must "
        "contain 'name', 'arguments', and optional 'call_id'. Do not wrap the JSON "
        "in Markdown."
    )
    conversation = json.dumps(
        request.messages,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    conversation_prefix = instruction + "\n\nConversation JSON:\n"
    text = (
        conversation_prefix + conversation + "\n\nAvailable tools JSON:\n" + tool_spec
    )
    breakpoint = request.cache_breakpoint
    if breakpoint is None:
        return _ProviderPrompt(
            text=text,
            cacheable_prefix_characters=None,
        )

    # The prompt text is byte-identical in both arms. The marker stops before
    # the prefix array's closing bracket so a later request can append a message
    # without changing any cached byte.
    cacheable_messages = json.dumps(
        request.messages[: breakpoint.message_count],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _ProviderPrompt(
        text=text,
        cacheable_prefix_characters=(
            len(conversation_prefix) + len(cacheable_messages) - 1
        ),
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


def _require_exec_success(result: Any, operation: str) -> None:
    return_code = getattr(result, "return_code", None)
    if return_code != 0:
        stderr = (getattr(result, "stderr", None) or "").strip()
        raise RuntimeError(f"failed to {operation}: {stderr or f'exit {return_code}'}")


def _valid_process_identity(value: str) -> bool:
    pid, separator, started = value.partition(":")
    return (
        separator == ":"
        and pid.isascii()
        and pid.isdigit()
        and int(pid) > 1
        and started.isascii()
        and started.isdigit()
        and int(started) >= 0
    )


def _process_identity_snapshot_script() -> str:
    return r"""
set -eu
for stat in /proc/[0-9]*/stat; do
  [ -r "$stat" ] || continue
  pid=${stat#/proc/}
  pid=${pid%/stat}
  IFS= read -r value < "$stat" || continue
  rest=${value##*) }
  set -- $rest
  start=${20:-}
  case "$start" in ''|*[!0-9]*) continue;; esac
  printf '%s:%s\n' "$pid" "$start"
done
""".strip()


def _quiesce_native_processes_script(
    workspace: str, *, process_baseline: tuple[str, ...]
) -> str:
    if any(not _valid_process_identity(value) for value in process_baseline):
        raise ValueError("process_baseline contains an invalid identity")
    root = shlex.quote(workspace)
    baseline = shlex.quote(" ".join(process_baseline))
    return f"""
set -eu
workspace=$(realpath -- {root})
[ "$workspace" != / ]
[ -d "$workspace" ]
baseline={baseline}
protected=$$
cursor=$$
while [ "$cursor" -gt 1 ]; do
  status=/proc/$cursor/status
  [ -r "$status" ] || break
  parent=
  while IFS= read -r line; do
    case "$line" in PPid:*) set -- $line; parent=$2; break;; esac
  done < "$status"
  case "$parent" in ''|*[!0-9]*) break;; esac
  protected="$protected $parent"
  cursor=$parent
done

candidates=
changed=1
while [ "$changed" -eq 1 ]; do
  changed=0
  for stat in /proc/[0-9]*/stat; do
    [ -r "$stat" ] || continue
    pid=${{stat#/proc/}}
    pid=${{pid%/stat}}
    case " $protected " in *" $pid "*) continue;; esac
    IFS= read -r value < "$stat" || continue
    rest=${{value##*) }}
    set -- $rest
    start=${{20:-}}
    case "$start" in ''|*[!0-9]*) continue;; esac
    case " $baseline " in *" $pid:$start "*) continue;; esac
    case " $candidates " in *" $pid:$start "*) continue;; esac
    candidates="$candidates $pid:$start"
    kill -STOP "$pid" 2>/dev/null || true
    changed=1
  done
done
for identity in $candidates; do
  pid=${{identity%%:*}}
  kill -KILL "$pid" 2>/dev/null || true
done

attempt=0
while [ "$attempt" -lt 50 ]; do
  survivors=
  for identity in $candidates; do
    pid=${{identity%%:*}}
    expected=${{identity#*:}}
    stat=/proc/$pid/stat
    [ -r "$stat" ] || continue
    IFS= read -r value < "$stat" || continue
    rest=${{value##*) }}
    set -- $rest
    state=${{1:-}}
    actual=${{20:-}}
    if [ "$actual" = "$expected" ] && [ "$state" != Z ]; then
      survivors="$survivors $identity"
    fi
  done
  [ -z "$survivors" ] && exit 0
  attempt=$((attempt + 1))
  sleep 0.1
done
printf 'rejected processes survived cleanup:%s\n' "$survivors" >&2
exit 42
""".strip()
