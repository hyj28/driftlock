from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest

from driftlock.lhtb import openrouter_provider_from_call_kwargs


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
