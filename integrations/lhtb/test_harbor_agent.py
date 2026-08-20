from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

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
    def __init__(self, *args: Any, store_dir: Path, **kwargs: Any):
        Path(store_dir).mkdir(parents=True)


class FakeRunner:
    calls = 0
    budgets: ClassVar[list[int | None]] = []

    def __init__(self, store: Any, judge: Any, *, config: Any):
        self.config = config
        type(self).budgets.append(config.max_tokens)

    async def run(
        self, *, step: FakeStep, initial_state: Any, **kwargs: Any
    ) -> RunResult:
        type(self).calls += 1
        count = type(self).calls
        context = step.runtime.context
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
            checkpoints=(),
            tokens_used=tokens_used,
            agent_tokens_used=tokens_used,
            judge_tokens_used=0,
        )


@pytest.fixture(autouse=True)
def fake_components(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRunner.calls = 0
    FakeRunner.budgets = []
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
