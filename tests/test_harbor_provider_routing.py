from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

from driftlock.lhtb import openrouter_provider_from_call_kwargs
from driftlock.models import (
    DriftSignal,
    DriftTriggerOutcome,
    DriftTriggerRecord,
    FineJudgeStatus,
    RollbackRecord,
    RunResult,
    RunStatus,
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
        max_output_tokens=512,
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
            max_output_tokens=512,
            timeout_sec=120,
            llm_call_kwargs=_judge_call_kwargs("baidu/fp8"),
        )


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
    assert phase["steps"] == 0
    assert phase["rollbacks"] == 1
    assert phase["tokens_used"] == 19
    assert phase["checkpoint_count"] == 0
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
    ]
    expected_counts = {
        "action_loop": {"upheld": 2, "vetoed": 0},
        "no_file_change": {"upheld": 2, "vetoed": 1},
    }
    assert phase["signal_counts"] == expected_counts
    assert context.metadata["preserved"] == "yes"
    assert context.metadata["driftlock"]["signal_counts"] == expected_counts

    native = object.__new__(native_agent.LHTBNativeDriftlockAgent)
    native.logs_dir = tmp_path
    native._native_phases = []
    native._native_retain_checkpoints = False
    native._write_phase_record(result)

    native_payload = json.loads((tmp_path / "driftlock-native-result.json").read_text())
    native_phase = native_payload["phases"][0]
    assert native_phase["phase"] == 0
    assert native_phase["status"] == "rollback_limit"
    assert native_phase["coarse_triggers"] == phase["coarse_triggers"]
    assert native_phase["signal_counts"] == expected_counts

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
