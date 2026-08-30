from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.judges import JudgeTokenBudgetExhausted
from driftlock.lhtb import openrouter_provider_from_call_kwargs
from driftlock.lhtb_experiment import build_job_config
from driftlock.models import (
    Checkpoint,
    DriftSignal,
    DriftTriggerOutcome,
    DriftTriggerRecord,
    FineJudgeStatus,
    JudgeReliabilityStatus,
    RollbackRecord,
    RunResult,
    RunStatus,
    StepOutcome,
    StepRecord,
    Verdict,
)
from driftlock.runner import RunnerConfig


class _FakeBaseAgent:
    def __init__(
        self,
        *args: Any,
        model_name: str | None = None,
        logs_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        self.model_name = model_name
        self.logs_dir = logs_dir or Path("logs")


class _FakeTerminus2(_FakeBaseAgent):
    pass


class _FakeLiteLLM:
    instances: ClassVar[list[_FakeLiteLLM]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._llm_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"model_name", "api_base", "temperature", "model_info"}
        }
        self._driftlock_provider_call_count = 0
        self._use_responses_api = False
        self.instances.append(self)

    async def call(self, **kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("provider call was not expected")


_FakeLiteLLM.call.__wrapped__ = _FakeLiteLLM.call  # type: ignore[attr-defined]


@pytest.fixture
def harbor_agent_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    harbor = ModuleType("harbor")
    harbor.__path__ = []  # type: ignore[attr-defined]
    agents = ModuleType("harbor.agents")
    agents.__path__ = []  # type: ignore[attr-defined]
    base = ModuleType("harbor.agents.base")
    terminus = ModuleType("harbor.agents.terminus_2")
    llms = ModuleType("harbor.llms")
    llms.__path__ = []  # type: ignore[attr-defined]
    lite_llm = ModuleType("harbor.llms.lite_llm")
    base.BaseAgent = _FakeBaseAgent  # type: ignore[attr-defined]
    terminus.Terminus2 = _FakeTerminus2  # type: ignore[attr-defined]
    lite_llm.LiteLLM = _FakeLiteLLM  # type: ignore[attr-defined]
    fake_modules = {
        "harbor": harbor,
        "harbor.agents": agents,
        "harbor.agents.base": base,
        "harbor.agents.terminus_2": terminus,
        "harbor.llms": llms,
        "harbor.llms.lite_llm": lite_llm,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("driftlock.harbor_native_agent", None)
    sys.modules.pop("driftlock.harbor_agent", None)
    _FakeLiteLLM.instances.clear()
    harbor_agent = importlib.import_module("driftlock.harbor_agent")
    native_agent = importlib.import_module("driftlock.harbor_native_agent")
    monkeypatch.setattr(native_agent, "_validate_pinned_harbor", lambda: None)
    yield harbor_agent, native_agent
    sys.modules.pop("driftlock.harbor_native_agent", None)
    sys.modules.pop("driftlock.harbor_agent", None)


def _agent_call_kwargs() -> dict[str, Any]:
    return {
        "temperature": 0.7,
        "max_tokens": 8192,
        "timeout": 240,
        "extra_body": {
            "provider": {
                "only": ["baidu/fp8"],
                "allow_fallbacks": False,
            }
        },
    }


def _judge_call_kwargs(provider: str = "alibaba") -> dict[str, Any]:
    return {"extra_body": {"provider": {"only": [provider], "allow_fallbacks": False}}}


def _assert_routing_rejected(call_kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError) as raised:
        openrouter_provider_from_call_kwargs(call_kwargs, source="candidate")

    assert str(raised.value) == message


def test_strict_routing_rejects_order_instead_of_only() -> None:
    _assert_routing_rejected(
        {
            "extra_body": {
                "provider": {
                    "order": ["baidu/fp8"],
                    "allow_fallbacks": False,
                }
            }
        },
        "candidate.extra_body.provider must contain exactly only and allow_fallbacks",
    )


@pytest.mark.parametrize("only", [[], ["baidu/fp8", "sail-research/fp4"]])
def test_strict_routing_rejects_non_singleton_only(only: list[str]) -> None:
    _assert_routing_rejected(
        {"extra_body": {"provider": {"only": only, "allow_fallbacks": False}}},
        "candidate.extra_body.provider.only must contain exactly one provider slug",
    )


def test_strict_routing_rejects_only_that_is_not_a_list() -> None:
    _assert_routing_rejected(
        {
            "extra_body": {
                "provider": {"only": ("baidu/fp8",), "allow_fallbacks": False}
            }
        },
        "candidate.extra_body.provider.only must be a list",
    )


@pytest.mark.parametrize("slug", ["", 7])
def test_strict_routing_rejects_invalid_provider_slug(slug: object) -> None:
    _assert_routing_rejected(
        {"extra_body": {"provider": {"only": [slug], "allow_fallbacks": False}}},
        "candidate.extra_body.provider.only provider slug must be a non-empty string",
    )


@pytest.mark.parametrize("allow_fallbacks", [0, 1, "false", None])
def test_strict_routing_requires_literal_false(allow_fallbacks: object) -> None:
    _assert_routing_rejected(
        {
            "extra_body": {
                "provider": {
                    "only": ["baidu/fp8"],
                    "allow_fallbacks": allow_fallbacks,
                }
            }
        },
        "candidate.extra_body.provider.allow_fallbacks must be false",
    )


def test_strict_routing_rejects_extra_body_keys() -> None:
    _assert_routing_rejected(
        {
            "extra_body": {
                "provider": {
                    "only": ["baidu/fp8"],
                    "allow_fallbacks": False,
                },
                "models": [],
            }
        },
        "candidate.extra_body must contain exactly provider",
    )


def test_strict_routing_rejects_extra_provider_keys() -> None:
    _assert_routing_rejected(
        {
            "extra_body": {
                "provider": {
                    "only": ["baidu/fp8"],
                    "allow_fallbacks": False,
                    "sort": "price",
                }
            }
        },
        "candidate.extra_body.provider must contain exactly only and allow_fallbacks",
    )


def test_strict_routing_rejects_missing_extra_body() -> None:
    _assert_routing_rejected({}, "candidate must contain extra_body")


def test_strict_routing_rejects_non_mapping_extra_body() -> None:
    _assert_routing_rejected(
        {"extra_body": ["provider"]},
        "candidate.extra_body must be a mapping",
    )


def test_strict_routing_rejects_non_mapping_provider() -> None:
    _assert_routing_rejected(
        {"extra_body": {"provider": ["baidu/fp8"]}},
        "candidate.extra_body.provider must be a mapping",
    )


@pytest.mark.parametrize(
    ("arm", "class_name", "fine_judge"),
    [
        ("driftlock", "LHTBDriftlockAgent", True),
        ("driftlock-heuristic", "LHTBDriftlockAgent", False),
        ("retry", "LHTBBlindRetryAgent", False),
    ],
)
@pytest.mark.parametrize(
    ("llm_call_kwargs", "message"),
    [
        (
            {"temperature": 0.7, "max_tokens": 8192, "timeout": 240},
            "llm_call_kwargs must contain extra_body",
        ),
        (
            {
                "temperature": 0.7,
                "max_tokens": 8192,
                "timeout": 240,
                "extra_body": {
                    "provider": {
                        "order": ["baidu/fp8"],
                        "allow_fallbacks": False,
                    }
                },
            },
            "llm_call_kwargs.extra_body.provider must contain exactly only and "
            "allow_fallbacks",
        ),
    ],
)
def test_terminus_arms_reject_invalid_routing_before_provider_initialization(
    harbor_agent_modules: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    class_name: str,
    fine_judge: bool,
    llm_call_kwargs: dict[str, Any],
    message: str,
) -> None:
    harbor_agent, _ = harbor_agent_modules
    terminus_initializations = 0

    def fail_if_terminus_initializes(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        nonlocal terminus_initializations
        terminus_initializations += 1

    monkeypatch.setattr(_FakeTerminus2, "__init__", fail_if_terminus_initializes)
    agent_kwargs: dict[str, Any] = {
        "model_name": "openrouter/deepseek/deepseek-v4-flash-0731",
        "llm_call_kwargs": llm_call_kwargs,
    }
    if fine_judge:
        agent_kwargs.update(
            {
                "driftlock_judge_model": ("openrouter/deepseek/deepseek-v4-pro-0813"),
                "driftlock_judge_llm_call_kwargs": {
                    "extra_body": {
                        "provider": {
                            "only": ["alibaba"],
                            "allow_fallbacks": False,
                        }
                    }
                },
            }
        )

    with pytest.raises(ValueError) as raised:
        getattr(harbor_agent, class_name)(**agent_kwargs)

    assert str(raised.value) == message
    assert terminus_initializations == 0, arm
    assert _FakeLiteLLM.instances == []


def test_judge_binds_routing_and_audited_prices_in_litellm_construction(
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, _ = harbor_agent_modules

    client = harbor_agent._LHTBJudgeClient(
        model="openrouter/deepseek/deepseek-v4-pro-0813",
        api_base="https://openrouter.ai/api/v1",
        max_output_tokens=8192,
        timeout_sec=120,
        llm_call_kwargs=_judge_call_kwargs(),
    )

    assert client.provider == "alibaba"
    assert client.llm.kwargs["extra_body"] == {
        "provider": {"only": ["alibaba"], "allow_fallbacks": False}
    }
    assert client.llm.kwargs["model_info"] == {
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 8_192,
        "input_cost_per_token": 0.000001162,
        "cache_read_input_token_cost": 0.0000001162,
        "output_cost_per_token": 0.000003485,
    }


def test_judge_provider_without_matching_prices_fails_loudly(
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, _ = harbor_agent_modules

    with pytest.raises(ValueError, match=r"baidu/fp8.*no audited pricing"):
        harbor_agent._LHTBJudgeClient(
            model="openrouter/deepseek/deepseek-v4-pro-0813",
            api_base="https://openrouter.ai/api/v1",
            max_output_tokens=8192,
            timeout_sec=120,
            llm_call_kwargs=_judge_call_kwargs("baidu/fp8"),
        )


async def test_judge_client_reports_preflight_output_budget_exhaustion(
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, _ = harbor_agent_modules
    client = harbor_agent._LHTBJudgeClient(
        model="openrouter/deepseek/deepseek-v4-pro-0813",
        api_base="https://openrouter.ai/api/v1",
        max_output_tokens=8192,
        timeout_sec=120,
        llm_call_kwargs=_judge_call_kwargs(),
    )

    with pytest.raises(
        JudgeTokenBudgetExhausted,
        match="258 tokens remain but the judge prompt reserves 258 input tokens",
    ):
        await client.complete("é", tokens_remaining=258)

    assert client.request_times_msec == []
    assert client.n_input_tokens == 0
    assert client.n_output_tokens == 0


async def test_judge_client_prices_reported_tokens_when_response_omits_cost(
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, _ = harbor_agent_modules
    client = harbor_agent._LHTBJudgeClient(
        model="openrouter/deepseek/deepseek-v4-pro-0813",
        api_base="https://openrouter.ai/api/v1",
        max_output_tokens=8192,
        timeout_sec=120,
        llm_call_kwargs=_judge_call_kwargs(),
    )

    async def respond(_llm: object, **kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(
            content="complete",
            usage=SimpleNamespace(
                prompt_tokens=120,
                cache_tokens=20,
                completion_tokens=10,
                cost_usd=None,
            ),
        )

    client.llm.call = SimpleNamespace(__wrapped__=respond)

    completion = await client.complete("short prompt", tokens_remaining=None)

    assert completion.tokens == 130
    assert client.n_input_tokens == 120
    assert client.n_cache_tokens == 20
    assert client.n_output_tokens == 10
    assert client.cost_usd == pytest.approx(0.000153374)
    assert client.usage_fallbacks == 1
    assert client.accounting_snapshot() == {
        "physical_request_count": 1,
        "provider_reported_token_count": 1,
        "provider_reported_cost_count": 0,
        "reported_tokens_priced_count": 1,
        "successful_response_fallback_count": 0,
        "missing_reported_usage_count": 0,
        "invalid_reported_usage_count": 0,
        "error_without_usage_count": 0,
        "conservative_usage_fallbacks": 1,
    }


async def test_raised_provider_rejection_without_usage_is_not_imputed_as_billed(
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, _ = harbor_agent_modules
    client = harbor_agent._LHTBJudgeClient(
        model="openrouter/deepseek/deepseek-v4-pro-0813",
        api_base="https://openrouter.ai/api/v1",
        max_output_tokens=8192,
        timeout_sec=120,
        llm_call_kwargs=_judge_call_kwargs(),
        raise_provider_errors=True,
    )

    async def reject(_llm: object, **kwargs: object) -> object:
        del kwargs
        raise ValueError("Range of input length should be [1, 1000000]")

    client.llm.call = SimpleNamespace(__wrapped__=reject)

    with pytest.raises(ValueError, match="Range of input length"):
        await client.complete("oversized", tokens_remaining=None)

    assert client.n_input_tokens == 0
    assert client.n_cache_tokens == 0
    assert client.n_output_tokens == 0
    assert client.cost_usd == 0.0
    assert client.usage_fallbacks == 0


async def test_fine_judge_rejection_without_usage_is_not_imputed_as_billed(
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, _ = harbor_agent_modules
    client = harbor_agent._LHTBJudgeClient(
        model="openrouter/deepseek/deepseek-v4-pro-0813",
        api_base="https://openrouter.ai/api/v1",
        max_output_tokens=8192,
        timeout_sec=120,
        llm_call_kwargs=_judge_call_kwargs(),
    )

    async def reject(_llm: object, **kwargs: object) -> object:
        del kwargs
        raise ValueError("Range of input length should be [1, 1000000]")

    client.llm.call = SimpleNamespace(__wrapped__=reject)

    completion = await client.complete("oversized", tokens_remaining=None)

    assert completion.tokens == 0
    assert completion.text == ""
    assert client.n_input_tokens == 0
    assert client.n_cache_tokens == 0
    assert client.n_output_tokens == 0
    assert client.cost_usd == 0.0
    assert client.accounting_snapshot() == {
        "physical_request_count": 1,
        "provider_reported_token_count": 0,
        "provider_reported_cost_count": 0,
        "reported_tokens_priced_count": 0,
        "successful_response_fallback_count": 0,
        "missing_reported_usage_count": 0,
        "invalid_reported_usage_count": 0,
        "error_without_usage_count": 1,
        "conservative_usage_fallbacks": 0,
    }


async def test_impossible_reported_tokens_use_the_stated_fallback_ceiling(
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, _ = harbor_agent_modules
    client = harbor_agent._LHTBJudgeClient(
        model="openrouter/deepseek/deepseek-v4-pro-0813",
        api_base="https://openrouter.ai/api/v1",
        max_output_tokens=10,
        timeout_sec=120,
        llm_call_kwargs=_judge_call_kwargs(),
    )

    async def respond(_llm: object, **kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(
            content="complete",
            usage=SimpleNamespace(
                prompt_tokens=500,
                cache_tokens=0,
                completion_tokens=3,
                cost_usd=0.5,
            ),
        )

    client.llm.call = SimpleNamespace(__wrapped__=respond)

    completion = await client.complete("x", tokens_remaining=None)

    assert completion.tokens == 267
    assert client.n_input_tokens == 257
    assert client.n_output_tokens == 10
    assert client.accounting_snapshot() == {
        "physical_request_count": 1,
        "provider_reported_token_count": 0,
        "provider_reported_cost_count": 0,
        "reported_tokens_priced_count": 0,
        "successful_response_fallback_count": 1,
        "missing_reported_usage_count": 0,
        "invalid_reported_usage_count": 1,
        "error_without_usage_count": 0,
        "conservative_usage_fallbacks": 1,
    }


def test_native_agent_forwards_only_audited_call_kwargs_to_litellm(
    tmp_path: Path,
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    _, native_agent = harbor_agent_modules

    agent = native_agent.LHTBNativeDriftlockAgent(
        logs_dir=tmp_path,
        model_name="openrouter/deepseek/deepseek-v4-flash-0731",
        llm_call_kwargs=_agent_call_kwargs(),
        model_info={
            "max_input_tokens": 128000,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    )

    assert agent._native_low_level.llm.kwargs["extra_body"] == {
        "provider": {"only": ["baidu/fp8"], "allow_fallbacks": False}
    }
    unexpected = _agent_call_kwargs()
    unexpected["top_p"] = 0.9
    with pytest.raises(ValueError, match="must contain exactly"):
        native_agent.LHTBNativeDriftlockAgent(
            logs_dir=tmp_path,
            model_name="openrouter/deepseek/deepseek-v4-flash-0731",
            llm_call_kwargs=unexpected,
            model_info={
                "max_input_tokens": 128000,
                "max_output_tokens": 8192,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
        )


def test_generated_config_constructs_unchanged_terminus_detector_behavior(
    tmp_path: Path,
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, _ = harbor_agent_modules
    task = tmp_path / "LHTB" / "tasks" / "task-a"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        "[task]\nname = 'long-horizon-terminal-bench/task-a'\n",
        encoding="utf-8",
    )
    config = build_job_config(
        lhtb_dir=tmp_path / "LHTB",
        jobs_dir=tmp_path / "jobs",
        job_name="detector-construction",
        arm="driftlock-heuristic",
        tasks=["task-a"],
    )
    agent_config = config["agents"][0]
    agent = harbor_agent.LHTBDriftlockAgent(
        model_name=agent_config["model_name"], **agent_config["kwargs"]
    )

    assert agent._driftlock_heuristic_config == HeuristicConfig(
        no_change_steps=4,
        loop_window=6,
        loop_repetitions=3,
        error_window=5,
        error_rate=0.6,
        command_failure_window=8,
        command_failure_rate=1.0,
        reward_stall_steps=5,
        reward_epsilon=0.000001,
    )

    def record(
        sequence: int,
        *,
        action: str,
        changed: bool = True,
        error: str | None = None,
        commands_run: int = 0,
        commands_failed: int = 0,
        reward: float | None = None,
    ) -> StepRecord:
        return StepRecord(
            sequence=sequence,
            logical_step=sequence,
            attempt=1,
            outcome=StepOutcome(
                action=action,
                state={},
                changed_paths=(f"file-{sequence}",) if changed else (),
                error=error,
                commands_run=commands_run,
                commands_failed=commands_failed,
                reward=reward,
            ),
        )

    judge = HeuristicJudge(agent._driftlock_heuristic_config)
    no_change = [
        record(sequence, action=f"inspect-{sequence}", changed=False)
        for sequence in range(1, 5)
    ]
    loop = [
        record(sequence, action=action)
        for sequence, action in enumerate(
            ("repeat", "edit-a", "repeat", "edit-b", "repeat", "edit-c"),
            start=1,
        )
    ]
    errors = [
        record(
            sequence,
            action=f"error-{sequence}",
            error="failed" if sequence <= 3 else None,
        )
        for sequence in range(1, 6)
    ]
    commands = [
        record(
            sequence,
            action=f"command-{sequence}",
            commands_run=1,
            commands_failed=1,
        )
        for sequence in range(1, 9)
    ]
    rewards = [
        record(sequence, action=f"reward-{sequence}", reward=reward)
        for sequence, reward in enumerate((0.0, 0.000001, 0.0, 0.000001, 0.0), start=1)
    ]

    for steps, expected in (
        (no_change, ("no_file_change", 4)),
        (loop, ("action_loop", 6)),
        (errors, ("error_spike", 5)),
        (commands, ("sustained_command_failure", 8)),
        (rewards, ("reward_stall", 5)),
    ):
        assert judge.evaluate(steps[:-1]) == ()
        assert [(signal.kind, signal.lookback) for signal in judge.evaluate(steps)] == [
            expected
        ]

    commands[3] = record(
        4,
        action="command-4",
        commands_run=2,
        commands_failed=1,
    )
    assert judge.evaluate(commands) == ()
    rewards[-1] = record(5, action="reward-5", reward=0.0000011)
    assert judge.evaluate(rewards) == ()


def test_terminus_agent_accepts_command_failure_detector_settings(
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, _ = harbor_agent_modules

    agent = harbor_agent.LHTBDriftlockAgent(
        model_name="openrouter/deepseek/deepseek-v4-flash-0731",
        llm_call_kwargs=_agent_call_kwargs(),
        driftlock_command_failure_window=3,
        driftlock_command_failure_rate=0.5,
    )

    assert agent._driftlock_heuristic_config.command_failure_window == 3
    assert agent._driftlock_heuristic_config.command_failure_rate == 0.5


def test_harbor_phase_record_writes_full_triggers_and_metadata_counts(
    tmp_path: Path,
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, native_agent = harbor_agent_modules
    no_change = DriftSignal(
        "no_file_change",
        "no files changed in the last 2 steps",
        lookback=2,
    )
    action_loop = DriftSignal(
        "action_loop",
        "the same action appeared 2 times in the last 2 steps",
        lookback=2,
    )
    result = RunResult(
        status=RunStatus.ROLLBACK_LIMIT,
        state={"turn": 3},
        steps=(),
        rollbacks=(
            RollbackRecord(
                sequence=3,
                checkpoint_id="checkpoint-1",
                signals=(no_change, action_loop),
                reason="confirmed drift",
            ),
        ),
        checkpoints=(),
        tokens_used=19,
        agent_tokens_used=14,
        judge_tokens_used=5,
        coarse_triggers=(
            DriftTriggerRecord(
                sequence=2,
                logical_step=2,
                signals=(no_change,),
                judge_status=FineJudgeStatus.VERDICT,
                judge_verdict=Verdict.HEALTHY,
                judge_reason="useful exploration",
                outcome=DriftTriggerOutcome.VETOED,
            ),
            DriftTriggerRecord(
                sequence=3,
                logical_step=3,
                signals=(no_change, action_loop),
                judge_status=FineJudgeStatus.VERDICT,
                judge_verdict=Verdict.DRIFTED,
                judge_reason="confirmed drift",
                outcome=DriftTriggerOutcome.ROLLED_BACK,
                rollback_checkpoint_id="checkpoint-1",
                rollback_checkpoint_step=1,
            ),
            DriftTriggerRecord(
                sequence=4,
                logical_step=2,
                signals=(no_change, action_loop),
                judge_status=FineJudgeStatus.VERDICT,
                judge_verdict=Verdict.DRIFTED,
                judge_reason="rollback budget exhausted",
                outcome=DriftTriggerOutcome.ROLLBACK_LIMIT_REFUSED,
            ),
            DriftTriggerRecord(
                sequence=5,
                logical_step=3,
                signals=(action_loop,),
                judge_status=FineJudgeStatus.FAILED,
                judge_verdict=None,
                judge_reason="fine judge returned invalid JSON",
                outcome=DriftTriggerOutcome.JUDGE_FAILED,
            ),
        ),
    )
    agent = object.__new__(harbor_agent.LHTBDriftlockAgent)
    agent.logs_dir = tmp_path
    agent._driftlock_phases = []
    agent._driftlock_tokens_consumed = 23
    agent._driftlock_runner_config = RunnerConfig(max_tokens=100)
    phase_store = tmp_path / "phase-0"

    agent._write_phase_record(result, phase_store, retained=False)
    context = SimpleNamespace(metadata={"preserved": "yes"})
    agent._set_result_metadata(context, result)

    payload = json.loads((tmp_path / "driftlock-result.json").read_text())
    phase = payload["phases"][0]
    assert phase["phase"] == 0
    assert phase["checkpoint_dir"] == str(phase_store)
    assert phase["checkpoints_retained"] is False
    assert phase["status"] == "rollback_limit"
    assert phase["judge_reliability"] == "reliable"
    assert phase["judge_attempts"] == 4
    assert phase["judge_failures"] == 1
    assert phase["steps"] == 0
    assert phase["rollbacks"] == 1
    assert phase["tokens_used"] == 19
    assert phase["checkpoint_count"] == 0
    assert phase["unstable_checkpoint_count"] == 0
    assert phase["coarse_triggers"] == [
        {
            "sequence": 2,
            "logical_step": 2,
            "signals": [
                {
                    "kind": "no_file_change",
                    "detail": "no files changed in the last 2 steps",
                    "lookback": 2,
                }
            ],
            "judge": {
                "status": "verdict",
                "verdict": "healthy",
                "reason": "useful exploration",
            },
            "outcome": "vetoed",
            "rollback_checkpoint": None,
        },
        {
            "sequence": 3,
            "logical_step": 3,
            "signals": [
                {
                    "kind": "no_file_change",
                    "detail": "no files changed in the last 2 steps",
                    "lookback": 2,
                },
                {
                    "kind": "action_loop",
                    "detail": ("the same action appeared 2 times in the last 2 steps"),
                    "lookback": 2,
                },
            ],
            "judge": {
                "status": "verdict",
                "verdict": "drifted",
                "reason": "confirmed drift",
            },
            "outcome": "rolled_back",
            "rollback_checkpoint": {
                "checkpoint_id": "checkpoint-1",
                "step": 1,
            },
        },
        {
            "sequence": 4,
            "logical_step": 2,
            "signals": [
                {
                    "kind": "no_file_change",
                    "detail": "no files changed in the last 2 steps",
                    "lookback": 2,
                },
                {
                    "kind": "action_loop",
                    "detail": ("the same action appeared 2 times in the last 2 steps"),
                    "lookback": 2,
                },
            ],
            "judge": {
                "status": "verdict",
                "verdict": "drifted",
                "reason": "rollback budget exhausted",
            },
            "outcome": "rollback_limit_refused",
            "rollback_checkpoint": None,
        },
        {
            "sequence": 5,
            "logical_step": 3,
            "signals": [
                {
                    "kind": "action_loop",
                    "detail": "the same action appeared 2 times in the last 2 steps",
                    "lookback": 2,
                }
            ],
            "judge": {
                "status": "failed",
                "verdict": None,
                "reason": "fine judge returned invalid JSON",
            },
            "outcome": "judge_failed",
            "rollback_checkpoint": None,
        },
    ]
    expected_counts = {
        "action_loop": {
            "upheld": 2,
            "vetoed": 0,
            "suppressed": 0,
            "judge_failed": 1,
            "judge_budget_exhausted": 0,
        },
        "no_file_change": {
            "upheld": 2,
            "vetoed": 1,
            "suppressed": 0,
            "judge_failed": 0,
            "judge_budget_exhausted": 0,
        },
    }
    assert phase["signal_counts"] == expected_counts
    assert context.metadata["preserved"] == "yes"
    assert context.metadata["driftlock"]["signal_counts"] == expected_counts
    assert context.metadata["driftlock"]["judge_reliability"] == "reliable"
    assert context.metadata["driftlock"]["judge_attempts"] == 4
    assert context.metadata["driftlock"]["judge_failures"] == 1
    assert context.metadata["termination_reason"] == "driftlock_rollback_limit"

    native = object.__new__(native_agent.LHTBNativeDriftlockAgent)
    native.logs_dir = tmp_path
    native._native_phases = []
    native._native_retain_checkpoints = False
    native._write_phase_record(result)

    native_payload = json.loads((tmp_path / "driftlock-native-result.json").read_text())
    native_phase = native_payload["phases"][0]
    assert native_phase["phase"] == 0
    assert native_phase["status"] == "rollback_limit"
    assert native_phase["judge_reliability"] == "reliable"
    assert native_phase["judge_attempts"] == 4
    assert native_phase["judge_failures"] == 1
    assert native_phase["coarse_triggers"] == phase["coarse_triggers"]
    assert native_phase["signal_counts"] == expected_counts
    assert native_phase["unstable_checkpoint_count"] == 0

    empty_result = RunResult(
        status=RunStatus.COMPLETED,
        state={"done": True},
        steps=(),
        rollbacks=(),
        checkpoints=(),
        tokens_used=0,
        agent_tokens_used=0,
        judge_tokens_used=0,
    )
    agent._write_phase_record(empty_result, tmp_path / "phase-1", retained=False)
    empty_phase = json.loads((tmp_path / "driftlock-result.json").read_text())[
        "phases"
    ][1]
    assert empty_phase["coarse_triggers"] == []
    assert empty_phase["signal_counts"] == {}
    assert empty_phase["unstable_checkpoint_count"] == 0


@pytest.mark.parametrize(
    ("failure_count", "expected_reliability"),
    [
        (1, JudgeReliabilityStatus.INCONCLUSIVE),
        (4, JudgeReliabilityStatus.FAILED),
    ],
)
def test_harbor_metadata_keeps_terminal_status_and_judge_reliability_separate(
    failure_count: int,
    expected_reliability: JudgeReliabilityStatus,
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, native_agent = harbor_agent_modules
    signal = DriftSignal("no_file_change", "stalled")
    result = RunResult(
        status=RunStatus.COMPLETED,
        state={},
        steps=(),
        rollbacks=(),
        checkpoints=(),
        tokens_used=0,
        agent_tokens_used=0,
        judge_tokens_used=0,
        coarse_triggers=tuple(
            DriftTriggerRecord(
                sequence=index,
                logical_step=index,
                signals=(signal,),
                judge_status=FineJudgeStatus.FAILED,
                judge_verdict=None,
                judge_reason="invalid response",
                outcome=DriftTriggerOutcome.JUDGE_FAILED,
            )
            for index in range(1, failure_count + 1)
        ),
    )

    agent = object.__new__(harbor_agent.LHTBDriftlockAgent)
    agent._driftlock_runtime = None
    agent._driftlock_tokens_consumed = 0
    agent._driftlock_runner_config = RunnerConfig(max_tokens=100)
    harbor_context = SimpleNamespace(metadata={})
    agent._set_result_metadata(harbor_context, result)

    native_context = SimpleNamespace(metadata={})
    native_agent.set_native_result_metadata(
        native_context,
        result=result,
        runtime=SimpleNamespace(tokens_consumed=0),
        trial_token_budget=100,
    )

    assert harbor_context.metadata["termination_reason"] == "confirmed_task_complete"
    assert native_context.metadata["termination_reason"] == "confirmed_task_complete"
    for metadata in (harbor_context.metadata, native_context.metadata):
        assert metadata["driftlock"]["status"] == "completed"
        assert metadata["driftlock"]["judge_reliability"] == expected_reliability.value


def test_trial_metadata_accumulates_judge_evidence_across_phases(
    tmp_path: Path,
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, native_agent = harbor_agent_modules
    signal = DriftSignal("no_file_change", "stalled")

    def trigger(index: int, status: FineJudgeStatus) -> DriftTriggerRecord:
        failed = status is FineJudgeStatus.FAILED
        return DriftTriggerRecord(
            sequence=index,
            logical_step=index,
            signals=(signal,),
            judge_status=status,
            judge_verdict=None if failed else Verdict.HEALTHY,
            judge_reason="invalid response" if failed else "healthy",
            outcome=(
                DriftTriggerOutcome.JUDGE_FAILED
                if failed
                else DriftTriggerOutcome.VETOED
            ),
        )

    phase_zero = RunResult(
        status=RunStatus.TOKEN_LIMIT,
        state={},
        steps=(),
        rollbacks=(),
        checkpoints=(),
        tokens_used=0,
        agent_tokens_used=0,
        judge_tokens_used=0,
        coarse_triggers=tuple(
            trigger(index, FineJudgeStatus.FAILED) for index in range(1, 13)
        ),
    )
    phase_one = RunResult(
        status=RunStatus.COMPLETED,
        state={},
        steps=(),
        rollbacks=(),
        checkpoints=(),
        tokens_used=0,
        agent_tokens_used=0,
        judge_tokens_used=0,
        coarse_triggers=(trigger(1, FineJudgeStatus.VERDICT),),
    )

    agent = object.__new__(harbor_agent.LHTBDriftlockAgent)
    agent.logs_dir = tmp_path
    agent._driftlock_phases = []
    agent._driftlock_runtime = None
    agent._driftlock_tokens_consumed = 0
    agent._driftlock_runner_config = RunnerConfig(max_tokens=100)
    harbor_context = SimpleNamespace(metadata={})
    native_context = SimpleNamespace(metadata={})
    agent._write_phase_record(
        phase_zero,
        tmp_path / "phase-0",
        retained=False,
    )
    phase_record = json.loads(
        (tmp_path / "driftlock-result.json").read_text(encoding="utf-8")
    )["phases"][0]
    assert phase_record["status"] == "token_limit"
    assert phase_record["judge_reliability"] == "failed"
    assert phase_record["judge_attempts"] == 12
    assert phase_record["judge_failures"] == 12

    agent._set_result_metadata(harbor_context, phase_zero)
    native_agent.set_native_result_metadata(
        native_context,
        result=phase_zero,
        runtime=SimpleNamespace(tokens_consumed=0),
        trial_token_budget=100,
    )
    for metadata in (harbor_context.metadata, native_context.metadata):
        assert metadata["driftlock"]["status"] == "token_limit"
        assert metadata["driftlock"]["judge_reliability"] == "failed"
        assert metadata["termination_reason"] == "driftlock_token_limit"

    agent._set_result_metadata(harbor_context, phase_one)
    native_agent.set_native_result_metadata(
        native_context,
        result=phase_one,
        runtime=SimpleNamespace(tokens_consumed=0),
        trial_token_budget=100,
    )

    for metadata in (harbor_context.metadata, native_context.metadata):
        assert metadata["driftlock"]["status"] == "completed"
        assert metadata["driftlock"]["judge_reliability"] == "failed"
        assert metadata["driftlock"]["judge_attempts"] == 13
        assert metadata["driftlock"]["judge_failures"] == 12
        assert metadata["driftlock"]["signal_counts"] == {
            "no_file_change": {
                "upheld": 0,
                "vetoed": 1,
                "suppressed": 0,
                "judge_failed": 12,
                "judge_budget_exhausted": 0,
            }
        }


def test_phase_record_counts_checkpoints_with_unstable_paths(
    tmp_path: Path,
    harbor_agent_modules: tuple[Any, Any],
) -> None:
    harbor_agent, native_agent = harbor_agent_modules
    created_at = datetime(2026, 8, 23, tzinfo=UTC)
    clean = Checkpoint(
        checkpoint_id="clean-checkpoint",
        step=0,
        created_at=created_at,
        digest="clean-digest",
        path=tmp_path / "clean-checkpoint",
    )
    unstable = Checkpoint(
        checkpoint_id="unstable-checkpoint",
        step=1,
        created_at=created_at,
        digest="unstable-digest",
        path=tmp_path / "unstable-checkpoint",
        unstable_paths=("./output/live.log", "./output/server.log"),
    )
    result = RunResult(
        status=RunStatus.COMPLETED,
        state={"done": True},
        steps=(),
        rollbacks=(),
        checkpoints=(clean, unstable),
        tokens_used=0,
        agent_tokens_used=0,
        judge_tokens_used=0,
    )
    agent = object.__new__(harbor_agent.LHTBDriftlockAgent)
    agent.logs_dir = tmp_path
    agent._driftlock_phases = []

    agent._write_phase_record(result, tmp_path / "phase-0", retained=True)

    phase = json.loads((tmp_path / "driftlock-result.json").read_text())["phases"][0]
    assert phase["checkpoint_count"] == 2
    assert phase["unstable_checkpoint_count"] == 1

    native = object.__new__(native_agent.LHTBNativeDriftlockAgent)
    native.logs_dir = tmp_path
    native._native_phases = []
    native._native_retain_checkpoints = False
    native._write_phase_record(result)

    native_phase = json.loads((tmp_path / "driftlock-native-result.json").read_text())[
        "phases"
    ][0]
    assert native_phase["checkpoint_count"] == 2
    assert native_phase["unstable_checkpoint_count"] == 1
