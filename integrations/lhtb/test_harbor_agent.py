from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from uuid import uuid4

import litellm
import pytest
from harbor.models.agent.context import AgentContext

import driftlock.harbor_agent as plugin
from driftlock.judges import JudgeTokenBudgetExhausted, judge_input_token_bound
from driftlock.lhtb_experiment import (
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_PROVIDER,
    openrouter_provider_call_kwargs,
)
from driftlock.models import (
    Checkpoint,
    DriftContext,
    FineJudgeStatus,
    RunResult,
    RunStatus,
)
from driftlock.terminus import TerminusConversationCodec, TerminusConversationState


def _pinned_judge_call_kwargs() -> dict[str, Any]:
    """The routing the experiment generator emits for the pinned judge provider."""
    return openrouter_provider_call_kwargs(DEFAULT_JUDGE_PROVIDER)


class FakeRuntime:
    summarization_enabled = False
    internal_retries_enabled = False
    provider_call_count = 0

    def __init__(self, agent: Any, environment: Any, context: Any, **kwargs: Any):
        self.context = context


class FakeStep:
    def __init__(self, runtime: FakeRuntime):
        self.runtime = runtime

    def initial_state(self) -> dict[str, Any]:
        return TerminusConversationCodec().initial_state()

    async def before_workspace_restore(self, workspace: str) -> None:
        return None


class FakeStore:
    restores: ClassVar[list[Any]] = []
    creates: ClassVar[list[Any]] = []

    def __init__(self, *args: Any, store_dir: Path, **kwargs: Any):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    async def restore(self, checkpoint: Any) -> dict[str, Any]:
        type(self).restores.append(checkpoint)
        return TerminusConversationCodec().initial_state()

    async def create(self, state: Any, **kwargs: Any) -> Any:
        checkpoint = SimpleNamespace(
            path=self.store_dir / "checkpoints" / f"created-{len(self.creates)}",
            unstable_paths=(),
        )
        type(self).creates.append(checkpoint)
        return checkpoint


class FakeRunner:
    calls = 0
    budgets: ClassVar[list[int | None]] = []
    fine_judges: ClassVar[list[Any]] = []
    raise_on_call: ClassVar[int | None] = None

    def __init__(self, store: Any, judge: Any, *, fine_judge: Any = None, config: Any):
        self.store = store
        self.config = config
        type(self).budgets.append(config.max_tokens)
        type(self).fine_judges.append(fine_judge)

    async def run(
        self, *, step: FakeStep, initial_state: Any, **kwargs: Any
    ) -> RunResult:
        type(self).calls += 1
        count = type(self).calls
        context = step.runtime.context
        step.runtime.provider_call_count += 1
        context.n_input_tokens = count * 10
        context.n_cache_tokens = count * 2
        context.n_output_tokens = count * 3
        context.cost_usd = count * 0.5
        context.rollout_details = [SimpleNamespace(index=i) for i in range(count)]
        context.metadata = {
            "n_episodes": count,
            "api_request_times_msec": [100] * count,
            "terminal_interaction_times_msec": [20] * count,
        }
        if count == type(self).raise_on_call:
            raise TimeoutError("simulated later-phase timeout")
        state = TerminusConversationState(
            messages=(
                {"role": "user", "content": f"phase {count}"},
                {"role": "assistant", "content": "done"},
            ),
            next_prompt="terminal",
            pending_completion=True,
            episode=count,
        )
        tokens_used = min(13, self.config.max_tokens or 13)
        return RunResult(
            status=(
                RunStatus.TOKEN_LIMIT
                if self.config.max_tokens is not None
                and tokens_used >= self.config.max_tokens
                else RunStatus.COMPLETED
            ),
            state=TerminusConversationCodec().encode(state),
            steps=(),
            rollbacks=(),
            checkpoints=(
                SimpleNamespace(
                    path=self.store.store_dir / "checkpoints" / f"initial-{count}",
                    unstable_paths=(),
                ),
            ),
            tokens_used=tokens_used,
            agent_tokens_used=tokens_used,
            judge_tokens_used=0,
        )


@pytest.fixture(autouse=True)
def fake_components(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRunner.calls = 0
    FakeRunner.budgets = []
    FakeRunner.fine_judges = []
    FakeRunner.raise_on_call = None
    FakeStore.restores = []
    FakeStore.creates = []
    monkeypatch.setattr(plugin, "HarborWorkspaceDeltaObserver", lambda *a, **k: None)
    monkeypatch.setattr(plugin, "LHTBTerminusRuntime", FakeRuntime)
    monkeypatch.setattr(plugin, "TerminusStepAdapter", FakeStep)
    monkeypatch.setattr(plugin, "RemoteArchiveCheckpointStore", FakeStore)
    monkeypatch.setattr(plugin, "DriftlockRunner", FakeRunner)


def _agent(
    tmp_path: Path, *, max_tokens: int | None = None
) -> plugin.LHTBDriftlockAgent:
    return plugin.LHTBDriftlockAgent(
        logs_dir=tmp_path / "agent",
        model_name="fake-provider/fake-model",
        enable_summarize=False,
        record_terminal_session=False,
        driftlock_retain_checkpoints=True,
        driftlock_max_tokens=max_tokens,
        llm_call_kwargs=openrouter_provider_call_kwargs(DEFAULT_PROVIDER),
    )


@pytest.mark.asyncio
async def test_oracle_restores_one_verified_checkpoint_without_model_access(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "agent"
    logs.mkdir()
    trial_id = str(uuid4())
    source_result = tmp_path / "result.json"
    checkpoint_id = "d" * 32
    checkpoint = (
        tmp_path / ".driftlock-checkpoints" / "phase-0" / "checkpoints" / checkpoint_id
    )
    checkpoint.mkdir(parents=True)
    archive = b"archive"
    state_text = "{}"
    digest = hashlib.sha256(archive)
    digest.update(b"\0state\0")
    digest.update(state_text.encode())
    (checkpoint / "workspace.tar.gz").write_bytes(archive)
    (checkpoint / "state.json").write_text(state_text, encoding="utf-8")
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "step": 10,
                "created_at": datetime.now(UTC).isoformat(),
                "digest": digest.hexdigest(),
                "parent_id": None,
                "label": "step-10",
                "remote_workspace": "/app",
            }
        ),
        encoding="utf-8",
    )
    source_result.write_text(
        json.dumps(
            {
                "id": trial_id,
                "task_name": "long-horizon-terminal-bench/task-a",
                "agent_info": {
                    "name": "driftlock-terminus-2",
                    "version": "0.1.0",
                    "model_info": {
                        "provider": "openrouter",
                        "name": "source-model",
                    },
                },
                "config": {
                    "agent": {
                        "import_path": ("driftlock.harbor_agent:LHTBDriftlockAgent"),
                        "model_name": "openrouter/source-model",
                        "kwargs": {"driftlock_retain_checkpoints": True},
                    }
                },
                "agent_result": {
                    "n_input_tokens": 100,
                    "n_cache_tokens": 20,
                    "n_output_tokens": 30,
                    "cost_usd": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )
    source_digest = hashlib.sha256(source_result.read_bytes()).hexdigest()
    source_audit = logs / "driftlock-result.json"
    source_audit.write_text(
        json.dumps(
            {
                "phases": [
                    {
                        "phase": 0,
                        "checkpoint_dir": str(checkpoint.parent.parent.resolve()),
                        "checkpoints_retained": True,
                        "status": "completed",
                        "checkpoint_count": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source_audit_digest = hashlib.sha256(source_audit.read_bytes()).hexdigest()
    agent = plugin.LHTBCheckpointReplayOracle(
        logs_dir=logs,
        model_name="openrouter/source-model",
        driftlock_oracle_mode="isolated-checkpoint-replay",
        driftlock_checkpoint_dir=str(checkpoint),
        driftlock_checkpoint_digest=digest.hexdigest(),
        driftlock_expected_workspace="/app",
        driftlock_source_trial_id=trial_id,
        driftlock_source_task_name="long-horizon-terminal-bench/task-a",
        driftlock_source_result=str(source_result),
        driftlock_source_result_sha256=source_digest,
        driftlock_source_audit=str(source_audit),
        driftlock_source_audit_sha256=source_audit_digest,
        driftlock_source_usage={
            "input_tokens": 100,
            "cache_tokens": 20,
            "output_tokens": 30,
            "cost_usd": 0.75,
        },
    )
    context = AgentContext()
    environment = SimpleNamespace(
        default_user="root", task_env_config=SimpleNamespace(workdir="/app")
    )

    await agent.run("hidden task instruction", environment, context)

    assert [item.checkpoint_id for item in FakeStore.restores] == [checkpoint_id]
    assert context.n_input_tokens == 100
    assert context.n_cache_tokens == 20
    assert context.n_output_tokens == 30
    assert context.cost_usd == pytest.approx(0.75)
    assert context.metadata["termination_reason"] == "oracle_checkpoint_replay"
    assert context.metadata["oracle"]["source_trial_id"] == trial_id
    assert context.metadata["oracle"]["usage_policy"] == (
        "full-source-trial-conservative"
    )
    assert (logs / "oracle-replay.json").is_file()


@pytest.mark.asyncio
async def test_checkpoint_scoring_restores_verified_bundle_with_zero_provider_usage(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "agent"
    logs.mkdir()
    checkpoint_id = "e" * 32
    checkpoint = (
        tmp_path / ".driftlock-checkpoints" / "phase-2" / "checkpoints" / checkpoint_id
    )
    checkpoint.mkdir(parents=True)
    archive = b"scoring archive"
    state_text = "{}"
    digest = hashlib.sha256(archive)
    digest.update(b"\0state\0")
    digest.update(state_text.encode())
    (checkpoint / "workspace.tar.gz").write_bytes(archive)
    (checkpoint / "state.json").write_text(state_text, encoding="utf-8")
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "step": 25,
                "created_at": datetime.now(UTC).isoformat(),
                "digest": digest.hexdigest(),
                "parent_id": None,
                "label": "step-25",
                "remote_workspace": "/app",
            }
        ),
        encoding="utf-8",
    )
    agent = plugin.LHTBCheckpointScoringAgent(
        logs_dir=logs,
        model_name="openrouter/source-model",
        driftlock_scoring_checkpoint_dir=str(checkpoint),
        driftlock_scoring_checkpoint_digest=digest.hexdigest(),
        driftlock_scoring_expected_workspace="/app",
    )
    context = AgentContext()
    environment = SimpleNamespace(
        default_user="root", task_env_config=SimpleNamespace(workdir="/app")
    )

    await agent.run("hidden task instruction", environment, context)

    assert [item.checkpoint_id for item in FakeStore.restores] == [checkpoint_id]
    assert context.n_input_tokens == 0
    assert context.n_cache_tokens == 0
    assert context.n_output_tokens == 0
    assert context.cost_usd == 0.0
    assert context.metadata["termination_reason"] == "checkpoint_scoring_replay"
    assert context.metadata["checkpoint_scoring"]["provider_tokens"] == 0


@pytest.mark.asyncio
async def test_fresh_harbor_phases_report_incremental_accounting(
    tmp_path: Path,
) -> None:
    (tmp_path / "agent").mkdir()
    agent = _agent(tmp_path)
    environment = SimpleNamespace(
        default_user="root", task_env_config=SimpleNamespace(workdir="/app")
    )
    first = AgentContext()
    second = AgentContext()

    await agent.run("task", environment, first)
    await agent.run("task, keep going", environment, second)

    assert first.n_input_tokens == 10
    assert second.n_input_tokens == 10
    assert second.n_output_tokens == 3
    assert second.metadata["n_episodes"] == 1
    assert second.metadata["termination_reason"] == "confirmed_task_complete"


@pytest.mark.asyncio
async def test_same_conversation_resume_keeps_cumulative_accounting(
    tmp_path: Path,
) -> None:
    (tmp_path / "agent").mkdir()
    agent = _agent(tmp_path)
    environment = SimpleNamespace(
        default_user="root", task_env_config=SimpleNamespace(workdir="/app")
    )
    context = AgentContext()

    await agent.run("task", environment, context)
    await agent.resume_after_verifier_rejection("verifier rejected", context)

    assert context.n_input_tokens == 20
    assert context.n_output_tokens == 6
    assert context.metadata["n_episodes"] == 2
    assert len(context.rollout_details) == 2


@pytest.mark.asyncio
async def test_later_phase_exception_restores_cumulative_judge_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLiteLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def call(self, prompt: str, **kwargs: Any) -> Any:
            raise AssertionError("judge provider is not called by FakeRunner")

    FakeLiteLLM.call.__wrapped__ = FakeLiteLLM.call  # type: ignore[attr-defined]
    monkeypatch.setattr(plugin, "LiteLLM", FakeLiteLLM)
    (tmp_path / "agent").mkdir()
    agent = plugin.LHTBDriftlockAgent(
        logs_dir=tmp_path / "agent",
        model_name="fake-provider/fake-model",
        enable_summarize=False,
        record_terminal_session=False,
        driftlock_retain_checkpoints=True,
        driftlock_judge_model=plugin.PINNED_LHTB_JUDGE_MODEL,
        driftlock_judge_llm_call_kwargs=_pinned_judge_call_kwargs(),
        llm_call_kwargs=openrouter_provider_call_kwargs(DEFAULT_PROVIDER),
    )
    environment = SimpleNamespace(
        default_user="root", task_env_config=SimpleNamespace(workdir="/app")
    )
    context = AgentContext()
    client = agent._driftlock_judge_client
    assert client is not None
    client.n_input_tokens = 10
    client.n_cache_tokens = 3
    client.n_output_tokens = 2
    client.cost_usd = 0.01
    client.request_times_msec = [5.0]

    await agent.run("task", environment, context)
    assert context.n_input_tokens == 20
    client.n_input_tokens = 20
    client.n_cache_tokens = 6
    client.n_output_tokens = 4
    client.cost_usd = 0.02
    client.request_times_msec = [5.0, 7.0]
    FakeRunner.raise_on_call = 2

    with pytest.raises(TimeoutError, match="later-phase timeout"):
        await agent.resume_after_verifier_rejection("rejected", context)

    assert context.n_input_tokens == 40
    assert context.n_cache_tokens == 10
    assert context.n_output_tokens == 10
    assert context.cost_usd == pytest.approx(1.02)
    assert context.metadata["api_request_times_msec"] == [100, 100, 5.0, 7.0]
    assert context.metadata["driftlock_judge_usage"]["input_tokens"] == 20


@pytest.mark.asyncio
async def test_total_token_budget_is_shared_across_harbor_phases(
    tmp_path: Path,
) -> None:
    (tmp_path / "agent").mkdir()
    agent = _agent(tmp_path, max_tokens=20)
    environment = SimpleNamespace(
        default_user="root", task_env_config=SimpleNamespace(workdir="/app")
    )

    await agent.run("first", environment, AgentContext())
    second = AgentContext()
    await agent.run("second", environment, second)
    third = AgentContext()
    await agent.run("third", environment, third)

    assert FakeRunner.budgets == [20, 7]
    assert second.metadata["driftlock"]["trial_tokens_used"] == 20
    assert third.metadata["termination_reason"] == "driftlock_token_limit"
    assert third.metadata["driftlock"]["tokens_used"] == 20


@pytest.mark.asyncio
async def test_blind_retry_restores_initial_state_and_discards_feedback(
    tmp_path: Path,
) -> None:
    (tmp_path / "agent").mkdir()
    agent = plugin.LHTBBlindRetryAgent(
        logs_dir=tmp_path / "agent",
        model_name="fake-provider/fake-model",
        enable_summarize=False,
        record_terminal_session=False,
        driftlock_max_tokens=100,
        llm_call_kwargs=openrouter_provider_call_kwargs(DEFAULT_PROVIDER),
    )
    environment = SimpleNamespace(
        default_user="root", task_env_config=SimpleNamespace(workdir="/app")
    )
    context = AgentContext()

    await agent.run("original task", environment, context)
    initial_checkpoint = agent._driftlock_retry_checkpoint
    await agent.resume_after_verifier_rejection(
        "SECRET VERIFIER FEEDBACK MUST NOT BE USED", context
    )

    assert FakeStore.restores == [initial_checkpoint]
    assert FakeRunner.calls == 2
    assert FakeRunner.budgets == [100, 87]
    assert agent._driftlock_heuristic_config.no_change_steps == 501
    assert context.metadata["driftlock_blind_retry"] == {
        "retries_started": 1,
        "verifier_feedback_used": False,
        "restart_checkpoint": "initial",
    }


@pytest.mark.asyncio
async def test_blind_retry_does_not_restore_when_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    (tmp_path / "agent").mkdir()
    agent = plugin.LHTBBlindRetryAgent(
        logs_dir=tmp_path / "agent",
        model_name="fake-provider/fake-model",
        enable_summarize=False,
        record_terminal_session=False,
        driftlock_max_tokens=13,
        llm_call_kwargs=openrouter_provider_call_kwargs(DEFAULT_PROVIDER),
    )
    environment = SimpleNamespace(
        default_user="root", task_env_config=SimpleNamespace(workdir="/app")
    )
    context = AgentContext()

    await agent.run("original task", environment, context)
    await agent.resume_after_verifier_rejection("rejected", context)

    assert FakeStore.restores == []
    assert FakeStore.creates == []
    assert FakeRunner.calls == 1
    assert context.metadata["termination_reason"] == "driftlock_token_limit"
    assert context.metadata["driftlock_blind_retry"]["retries_started"] == 0


@pytest.mark.asyncio
async def test_blind_retry_finalizer_removes_retained_checkpoint(
    tmp_path: Path,
) -> None:
    (tmp_path / "agent").mkdir()
    agent = plugin.LHTBBlindRetryAgent(
        logs_dir=tmp_path / "agent",
        model_name="fake-provider/fake-model",
        enable_summarize=False,
        record_terminal_session=False,
        llm_call_kwargs=openrouter_provider_call_kwargs(DEFAULT_PROVIDER),
    )
    environment = SimpleNamespace(
        default_user="root", task_env_config=SimpleNamespace(workdir="/app")
    )

    await agent.run("original task", environment, AgentContext())
    assert agent._driftlock_store_root is not None
    assert agent._driftlock_store_root.is_dir()

    await agent._driftlock_finalize_after_agent_run()

    assert not agent._driftlock_store_root.exists()
    assert agent._driftlock_retry_checkpoint is None


@pytest.mark.asyncio
async def test_judge_client_caps_one_call_and_merges_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeLiteLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def call(self, prompt: str, **kwargs: Any) -> Any:
            raise AssertionError("decorated call must be bypassed")

    async def unwrapped(llm: FakeLiteLLM, *, prompt: str, **kwargs: Any) -> Any:
        calls.append({"llm": llm, "prompt": prompt, **kwargs})
        return SimpleNamespace(
            content='{"verdict":"healthy","reason":"progress"}',
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=7,
                cache_tokens=12,
                cost_usd=0.004,
            ),
        )

    FakeLiteLLM.call.__wrapped__ = unwrapped  # type: ignore[attr-defined]
    monkeypatch.setattr(plugin, "LiteLLM", FakeLiteLLM)
    client = plugin._LHTBJudgeClient(
        model=plugin.PINNED_LHTB_JUDGE_MODEL,
        api_base="https://example.invalid/v1",
        max_output_tokens=50,
        timeout_sec=10,
        llm_call_kwargs=_pinned_judge_call_kwargs(),
    )

    completion = await client.complete("judge this", tokens_remaining=1_000)
    context = AgentContext(
        n_input_tokens=100,
        n_cache_tokens=40,
        n_output_tokens=10,
        cost_usd=1.0,
        metadata={"api_request_times_msec": [5.0]},
    )
    client.apply_accounting(context)

    assert completion.tokens == 27
    assert calls[0]["max_tokens"] == 50
    assert calls[0]["num_retries"] == 0
    assert calls[0]["max_retries"] == 0
    assert context.n_input_tokens == 120
    assert context.n_cache_tokens == 52
    assert context.n_output_tokens == 17
    assert context.cost_usd == pytest.approx(1.004)
    assert context.metadata["driftlock_judge_usage"]["request_count"] == 1

    client.prepare_accounting(context)
    assert context.n_input_tokens == 100
    assert context.n_cache_tokens == 40
    assert context.n_output_tokens == 10
    assert context.cost_usd == pytest.approx(1.0)
    assert context.metadata["api_request_times_msec"] == [5.0]
    client.apply_accounting(context)
    assert context.n_input_tokens == 120
    assert context.cost_usd == pytest.approx(1.004)


@pytest.mark.asyncio
async def test_judge_client_skips_request_when_budget_cannot_cover_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLiteLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def call(self, prompt: str, **kwargs: Any) -> Any:
            raise AssertionError("judge provider must not be called")

    FakeLiteLLM.call.__wrapped__ = FakeLiteLLM.call  # type: ignore[attr-defined]
    monkeypatch.setattr(plugin, "LiteLLM", FakeLiteLLM)
    client = plugin._LHTBJudgeClient(
        model=plugin.PINNED_LHTB_JUDGE_MODEL,
        api_base=None,
        max_output_tokens=50,
        timeout_sec=10,
        llm_call_kwargs=_pinned_judge_call_kwargs(),
    )

    prompt = "large enough prompt"
    input_bound = judge_input_token_bound(prompt)

    with pytest.raises(JudgeTokenBudgetExhausted) as exc_info:
        await client.complete(prompt, tokens_remaining=10)

    assert "10 tokens remain" in str(exc_info.value)
    assert f"reserves {input_bound} input tokens" in str(exc_info.value)
    assert client.request_times_msec == []


@pytest.mark.asyncio
async def test_lhtb_fine_judge_classifies_preflight_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLiteLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def call(self, prompt: str, **kwargs: Any) -> Any:
            raise AssertionError("judge provider must not be called")

    FakeLiteLLM.call.__wrapped__ = FakeLiteLLM.call  # type: ignore[attr-defined]
    monkeypatch.setattr(plugin, "LiteLLM", FakeLiteLLM)
    client = plugin._LHTBJudgeClient(
        model=plugin.PINNED_LHTB_JUDGE_MODEL,
        api_base=None,
        max_output_tokens=50,
        timeout_sec=10,
        llm_call_kwargs=_pinned_judge_call_kwargs(),
    )
    context = DriftContext(
        goal="fix the parser",
        plan="add a regression test",
        checkpoint=Checkpoint(
            checkpoint_id="abc",
            step=2,
            created_at=datetime.now(UTC),
            digest="digest",
            path=Path("/tmp/checkpoint"),
        ),
        signals=(),
        recent_steps=(),
        diff="",
        tokens_remaining=10,
    )

    verdict = await plugin._LHTBFineJudge(client).judge(context)

    assert verdict.status is FineJudgeStatus.BUDGET_EXHAUSTED
    assert verdict.verdict is None
    assert client.request_times_msec == []


def test_judge_client_rejects_unaudited_model_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plugin, "LiteLLM", lambda **kwargs: None)

    with pytest.raises(ValueError, match="pinned model with audited pricing"):
        plugin._LHTBJudgeClient(
            model="openrouter/other/model",
            api_base=None,
            max_output_tokens=50,
            timeout_sec=10,
            llm_call_kwargs=_pinned_judge_call_kwargs(),
        )


@pytest.mark.asyncio
async def test_judge_client_conservatively_prices_missing_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingLiteLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def call(self, prompt: str, **kwargs: Any) -> Any:
            raise AssertionError("decorated call must be bypassed")

    async def unwrapped(llm: FailingLiteLLM, *, prompt: str, **kwargs: Any) -> Any:
        raise TimeoutError("provider timed out without usage")

    FailingLiteLLM.call.__wrapped__ = unwrapped  # type: ignore[attr-defined]
    monkeypatch.setattr(plugin, "LiteLLM", FailingLiteLLM)
    client = plugin._LHTBJudgeClient(
        model=plugin.PINNED_LHTB_JUDGE_MODEL,
        api_base=None,
        max_output_tokens=50,
        timeout_sec=10,
        llm_call_kwargs=_pinned_judge_call_kwargs(),
    )

    completion = await client.complete("judge this", tokens_remaining=1_000)

    expected_input = len(b"judge this") + 256
    expected_cost = expected_input * 0.000001162 + 50 * 0.000003485
    assert completion.tokens == expected_input + 50
    assert client.cost_usd == pytest.approx(expected_cost)
    assert client.usage_fallbacks == 1


@pytest.mark.asyncio
async def test_cancelled_judge_call_is_accounted_then_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CancelledLiteLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def call(self, prompt: str, **kwargs: Any) -> Any:
            raise AssertionError("decorated call must be bypassed")

    async def unwrapped(llm: CancelledLiteLLM, *, prompt: str, **kwargs: Any) -> Any:
        raise asyncio.CancelledError

    CancelledLiteLLM.call.__wrapped__ = unwrapped  # type: ignore[attr-defined]
    monkeypatch.setattr(plugin, "LiteLLM", CancelledLiteLLM)
    client = plugin._LHTBJudgeClient(
        model=plugin.PINNED_LHTB_JUDGE_MODEL,
        api_base=None,
        max_output_tokens=50,
        timeout_sec=10,
        llm_call_kwargs=_pinned_judge_call_kwargs(),
    )

    with pytest.raises(asyncio.CancelledError):
        await client.complete("judge this", tokens_remaining=1_000)

    assert client.n_input_tokens == len(b"judge this") + 256
    assert client.n_output_tokens == 50
    assert client.cost_usd > 0
    assert client.usage_fallbacks == 1


@pytest.mark.asyncio
async def test_real_pinned_judge_litellm_path_is_one_physical_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_completion(**kwargs: Any) -> Any:
        calls.append(kwargs)

        class Completion(dict):
            pass

        response = Completion(
            {
                "model": "fake-judge",
                "choices": [
                    {
                        "message": {
                            "content": '{"verdict":"healthy","reason":"ok"}',
                            "reasoning_content": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        response.usage = SimpleNamespace(
            prompt_tokens=30,
            completion_tokens=8,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
        )
        response._hidden_params = {"response_cost": 0.002}
        return response

    monkeypatch.setattr(litellm, "acompletion", fake_completion)
    client = plugin._LHTBJudgeClient(
        model=plugin.PINNED_LHTB_JUDGE_MODEL,
        api_base="https://example.invalid/v1",
        max_output_tokens=50,
        timeout_sec=10,
        llm_call_kwargs=_pinned_judge_call_kwargs(),
    )

    completion = await client.complete("judge this", tokens_remaining=1_000)

    assert len(calls) == 1
    assert client.llm._driftlock_provider_call_count == 1
    assert calls[0]["num_retries"] == 0
    assert calls[0]["max_retries"] == 0
    assert completion.tokens == 38
    assert client.n_cache_tokens == 20
    assert client.cost_usd == pytest.approx(0.002)
