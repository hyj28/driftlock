from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import litellm
import pytest
from harbor.models.agent.context import AgentContext

import driftlock.harbor_agent as plugin
from driftlock.models import RunResult, RunStatus
from driftlock.terminus import TerminusConversationCodec, TerminusConversationState


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
            path=self.store_dir / "checkpoints" / f"created-{len(self.creates)}"
        )
        type(self).creates.append(checkpoint)
        return checkpoint


class FakeRunner:
    calls = 0
    budgets: ClassVar[list[int | None]] = []
    fine_judges: ClassVar[list[Any]] = []

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
                    path=self.store.store_dir / "checkpoints" / f"initial-{count}"
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
    )


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
    )

    completion = await client.complete("large enough prompt", tokens_remaining=10)

    assert completion.tokens == 0
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
    )

    completion = await client.complete("judge this", tokens_remaining=1_000)

    expected_input = len(b"judge this") + 256
    expected_cost = (
        expected_input * plugin._JUDGE_INPUT_COST_PER_TOKEN
        + 50 * plugin._JUDGE_OUTPUT_COST_PER_TOKEN
    )
    assert completion.tokens == expected_input + 50
    assert client.cost_usd == pytest.approx(expected_cost)
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
    )

    completion = await client.complete("judge this", tokens_remaining=1_000)

    assert len(calls) == 1
    assert client.llm._driftlock_provider_call_count == 1
    assert calls[0]["num_retries"] == 0
    assert calls[0]["max_retries"] == 0
    assert completion.tokens == 38
    assert client.n_cache_tokens == 20
    assert client.cost_usd == pytest.approx(0.002)
