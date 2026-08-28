from __future__ import annotations

import importlib
import json
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

from driftlock.lhtb_experiment import build_job_config
from driftlock.models import RunResult, RunStatus
from driftlock.skill_admission import SkillAdmissionCandidate, SkillLibrary
from driftlock.skill_distillation import Skill
from driftlock.skill_injection import SkillRetrievalFailed
from driftlock.terminus import TerminusConversationCodec, TerminusConversationState

TASK_INSTRUCTION = "Repair the parser without changing accepted syntax."
ACTIVATION = "When a generated parser rejects a valid trailing comma."
EXPECTED_SKILL_BLOCK = """<driftlock-retrieved-skill-context>
<retrieved-skill id="parser-repair">
## activation

When a generated parser rejects a valid trailing comma.

## execution

Preserve the grammar and repair trailing-comma handling.

## termination

Stop after focused parser tests pass.
</retrieved-skill>
</driftlock-retrieved-skill-context>"""
EXPECTED_RETRIEVED_SKILL = """<retrieved-skill id="parser-repair">
## activation

When a generated parser rejects a valid trailing comma.

## execution

Preserve the grammar and repair trailing-comma handling.

## termination

Stop after focused parser tests pass.
</retrieved-skill>"""
EXPECTED_INITIAL_REQUEST = (
    EXPECTED_SKILL_BLOCK + "\n\nRepair the parser without changing accepted syntax."
)
EXPECTED_RESUME_REQUEST = (
    EXPECTED_SKILL_BLOCK + "\n\nThe verifier still rejects the solution.\n\n"
    "Current terminal observation:\nparser test output"
)


class LiteralEmbedder:
    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, texts: Sequence[str]) -> list[Sequence[float]]:
        call = tuple(texts)
        self.calls.append(call)
        return [self.vectors[text] for text in call]


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
    def __init__(self, **kwargs: Any) -> None:
        del kwargs


@pytest.fixture
def harbor_agent(monkeypatch: pytest.MonkeyPatch) -> Any:
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
    for name, module in {
        "harbor": harbor,
        "harbor.agents": agents,
        "harbor.agents.base": base,
        "harbor.agents.terminus_2": terminus,
        "harbor.llms": llms,
        "harbor.llms.lite_llm": lite_llm,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("driftlock.harbor_agent", None)
    module = importlib.import_module("driftlock.harbor_agent")
    yield module
    sys.modules.pop("driftlock.harbor_agent", None)


class _FakeStep:
    async def before_workspace_restore(self, _workspace: str) -> None:
        return None

    def initial_state(self) -> dict[str, Any]:
        return TerminusConversationCodec().initial_state()


class _FakeStore:
    def __init__(self, *_args: Any, store_dir: Path, **_kwargs: Any) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)


class _CapturingRunner:
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def run(self, **kwargs: Any) -> RunResult:
        self.calls.append(dict(kwargs))
        state = TerminusConversationCodec().encode(
            TerminusConversationState(
                messages=({"role": "assistant", "content": "working"},),
                next_prompt="parser test output",
                pending_completion=False,
                episode=len(self.calls),
            )
        )
        return RunResult(
            status=RunStatus.COMPLETED,
            state=state,
            steps=(),
            rollbacks=(),
            checkpoints=(),
            tokens_used=0,
            agent_tokens_used=0,
            judge_tokens_used=0,
        )


def _admit(library: SkillLibrary, activation: str = ACTIVATION) -> None:
    decision = library.submit(
        SkillAdmissionCandidate(
            candidate_id="parser-repair",
            arm="baseline",
            skill=Skill(
                activation=activation,
                execution="Preserve the grammar and repair trailing-comma handling.",
                termination="Stop after focused parser tests pass.",
            ),
            paired_deltas=(0.1,) * 10,
        )
    )
    assert decision["status"] == "admitted"


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        metadata={},
        n_input_tokens=0,
        n_cache_tokens=0,
        n_output_tokens=0,
        cost_usd=None,
        rollout_details=[],
    )


def _agent(
    harbor_agent: Any,
    tmp_path: Path,
    *,
    library: SkillLibrary | None = None,
    embedder: Any | None = None,
    arm: str = "baseline",
    agent_class: str = "LHTBDriftlockAgent",
) -> Any:
    kwargs: dict[str, Any] = {}
    if library is not None:
        kwargs.update(
            {
                "driftlock_skill_library_dir": str(library.root),
                "driftlock_skill_embedder": embedder,
                "driftlock_skill_distillation_arm": arm,
            }
        )
    agent = getattr(harbor_agent, agent_class)(
        model_name="local-test-model",
        logs_dir=tmp_path / f"logs-{arm}-{len(_CapturingRunner.calls)}",
        llm_call_kwargs={
            "extra_body": {
                "provider": {"only": ["local-test"], "allow_fallbacks": False}
            }
        },
        **kwargs,
    )
    Path(agent.logs_dir).mkdir(parents=True)
    environment = SimpleNamespace(
        default_user="root",
        task_env_config=SimpleNamespace(workdir="/app"),
    )
    agent._driftlock_runtime = SimpleNamespace(
        context=None,
        rate_limited_call_count=0,
    )
    agent._driftlock_environment = environment
    agent._driftlock_workspace = "/app"
    agent._driftlock_store_root = tmp_path / (
        f"checkpoints-{arm}-{len(_CapturingRunner.calls)}"
    )
    agent._driftlock_store_root.mkdir()
    agent._driftlock_step = _FakeStep()
    return agent, environment


@pytest.fixture(autouse=True)
def _capture_phase_requests(monkeypatch: pytest.MonkeyPatch, harbor_agent: Any) -> None:
    _CapturingRunner.calls.clear()
    monkeypatch.setattr(harbor_agent, "DriftlockRunner", _CapturingRunner)
    monkeypatch.setattr(harbor_agent, "RemoteArchiveCheckpointStore", _FakeStore)


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline skill wiring must not open a socket")
        ),
    )


@pytest.mark.asyncio
async def test_inert_skill_layer_preserves_the_exact_outgoing_request_bytes(
    tmp_path: Path,
    harbor_agent: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_network(monkeypatch)
    context = _context()
    unconfigured, environment = _agent(harbor_agent, tmp_path)
    await unconfigured.run(TASK_INSTRUCTION, environment, context)
    unconfigured_bytes = _CapturingRunner.calls[-1]["goal"].encode("utf-8")

    empty_library = SkillLibrary(tmp_path / "empty-library")

    def unexpected_embedder(_texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("an empty library must not invoke its embedder")

    empty, empty_environment = _agent(
        harbor_agent,
        tmp_path,
        library=empty_library,
        embedder=unexpected_embedder,
    )
    await empty.run(TASK_INSTRUCTION, empty_environment, _context())
    empty_bytes = _CapturingRunner.calls[-1]["goal"].encode("utf-8")

    inapplicable_library = SkillLibrary(tmp_path / "inapplicable-library")
    inapplicable_activation = "When a database migration deadlocks."
    _admit(inapplicable_library, inapplicable_activation)
    inapplicable_embedder = LiteralEmbedder(
        {
            inapplicable_activation: (1.0, 0.0),
            TASK_INSTRUCTION: (0.0, 1.0),
        }
    )
    inapplicable, inapplicable_environment = _agent(
        harbor_agent,
        tmp_path,
        library=inapplicable_library,
        embedder=inapplicable_embedder,
    )
    await inapplicable.run(TASK_INSTRUCTION, inapplicable_environment, _context())
    inapplicable_bytes = _CapturingRunner.calls[-1]["goal"].encode("utf-8")

    assert unconfigured_bytes == b"Repair the parser without changing accepted syntax."
    assert empty_bytes == unconfigured_bytes
    assert inapplicable_bytes == unconfigured_bytes


@pytest.mark.asyncio
async def test_applicable_skill_is_delimited_in_request_and_trial_record(
    tmp_path: Path,
    harbor_agent: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_network(monkeypatch)
    library = SkillLibrary(tmp_path / "library")
    _admit(library)
    embedder = LiteralEmbedder({ACTIVATION: (1.0, 0.0), TASK_INSTRUCTION: (1.0, 0.0)})
    agent, environment = _agent(
        harbor_agent, tmp_path, library=library, embedder=embedder
    )
    context = _context()

    await agent.run(TASK_INSTRUCTION, environment, context)

    assert _CapturingRunner.calls[0]["goal"] == EXPECTED_INITIAL_REQUEST
    record = json.loads(
        (Path(agent.logs_dir) / "driftlock-result.json").read_text(encoding="utf-8")
    )
    skill_layer = record["skill_layer"]
    assert skill_layer["distillation_arm"] == "baseline"
    assert skill_layer["policy"] == {
        "retrieval_frequency": "once_per_task",
        "query_source": "original_task_instruction",
        "injection_position": "prepended_to_each_phase_entry_prompt",
        "retrieval_failure": "fail_trial_after_recording",
    }
    assert skill_layer["injection"] == {
        "status": "injected",
        "candidate_ids": ["parser-repair"],
        "delimiter_start": "<driftlock-retrieved-skill-context>",
        "delimiter_end": "</driftlock-retrieved-skill-context>",
        "character_count": 326,
        "text": EXPECTED_SKILL_BLOCK,
    }
    selected = skill_layer["retrieval"]["selected_skills"]
    assert selected[0]["candidate_id"] == "parser-repair"
    assert selected[0]["basis"] == (
        "activation cosine similarity met the configured threshold"
    )
    assert selected[0]["injected_text"] == EXPECTED_RETRIEVED_SKILL
    assert record["phases"][0]["skill_injection"] == {
        "status": "injected",
        "candidate_ids": ["parser-repair"],
        "position": "prepended_to_each_phase_entry_prompt",
        "character_count": 326,
    }
    assert context.metadata["driftlock_skill_layer"] == skill_layer


@pytest.mark.asyncio
async def test_verifier_resume_reuses_and_injects_the_same_task_retrieval(
    tmp_path: Path,
    harbor_agent: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_network(monkeypatch)
    library = SkillLibrary(tmp_path / "library")
    _admit(library)
    embedder = LiteralEmbedder({ACTIVATION: (1.0, 0.0), TASK_INSTRUCTION: (1.0, 0.0)})
    agent, environment = _agent(
        harbor_agent, tmp_path, library=library, embedder=embedder
    )
    context = _context()
    await agent.run(TASK_INSTRUCTION, environment, context)

    await agent.resume_after_verifier_rejection(
        "The verifier still rejects the solution.", context
    )

    resumed = TerminusConversationCodec().decode(
        _CapturingRunner.calls[1]["initial_state"]
    )
    assert resumed is not None
    assert resumed.next_prompt == EXPECTED_RESUME_REQUEST
    assert _CapturingRunner.calls[1]["goal"] == EXPECTED_INITIAL_REQUEST
    assert embedder.calls == [(ACTIVATION,), (TASK_INSTRUCTION,)]
    record = json.loads(
        (Path(agent.logs_dir) / "driftlock-result.json").read_text(encoding="utf-8")
    )
    assert [phase["skill_injection"]["status"] for phase in record["phases"]] == [
        "injected",
        "injected",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_class",
    ["LHTBDriftlockAgent", "LHTBBlindRetryAgent"],
)
async def test_embedding_failure_is_recorded_and_fails_before_agent_request(
    tmp_path: Path,
    harbor_agent: Any,
    agent_class: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_network(monkeypatch)
    library = SkillLibrary(tmp_path / "library")
    _admit(library)

    def failing_embedder(_texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("local embedder unavailable")

    agent, environment = _agent(
        harbor_agent,
        tmp_path,
        library=library,
        embedder=failing_embedder,
        agent_class=agent_class,
    )
    context = _context()

    with pytest.raises(
        SkillRetrievalFailed,
        match="skill retrieval failed: embedding_callable_failed",
    ) as caught:
        await agent.run(TASK_INSTRUCTION, environment, context)

    assert caught.traceback[-1].name == "_skill_aware_phase_entry"
    assert _CapturingRunner.calls == []
    record = json.loads(
        (Path(agent.logs_dir) / "driftlock-result.json").read_text(encoding="utf-8")
    )
    assert record["phases"] == []
    assert record["skill_layer"]["status"] == "failed"
    assert record["skill_layer"]["injection"]["status"] == "retrieval_failed"
    assert record["skill_layer"]["retrieval"]["refusal"] == {
        "reason": "embedding_callable_failed",
        "detail": (
            "activation index embedding call failed: RuntimeError: "
            "local embedder unavailable"
        ),
        "stage": "index",
    }
    assert context.metadata["termination_reason"] == (
        "driftlock_skill_retrieval_failed"
    )


@pytest.mark.asyncio
async def test_distillation_arms_emit_identical_injected_requests(
    tmp_path: Path,
    harbor_agent: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_network(monkeypatch)
    library = SkillLibrary(tmp_path / "library")
    _admit(library)
    requests = []
    for arm in ("baseline", "localized"):
        embedder = LiteralEmbedder(
            {ACTIVATION: (1.0, 0.0), TASK_INSTRUCTION: (1.0, 0.0)}
        )
        agent, environment = _agent(
            harbor_agent,
            tmp_path,
            library=library,
            embedder=embedder,
            arm=arm,
        )
        await agent.run(TASK_INSTRUCTION, environment, _context())
        requests.append(_CapturingRunner.calls[-1]["goal"])

    assert requests == [EXPECTED_INITIAL_REQUEST, EXPECTED_INITIAL_REQUEST]


def _lhtb_tree(tmp_path: Path) -> Path:
    root = tmp_path / "LHTB"
    task = root / "tasks" / "task-a"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        "[task]\nname = 'long-horizon-terminal-bench/task-a'\n",
        encoding="utf-8",
    )
    return root


def test_runnable_distillation_arms_share_one_skill_configuration_mechanism(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path)
    library = tmp_path / "library"
    library.mkdir()
    configs = {
        arm: build_job_config(
            lhtb_dir=root,
            jobs_dir=tmp_path / "jobs",
            job_name=f"job-{arm}",
            arm=arm,
            tasks=["task-a"],
            skill_library_dir=library,
            skill_embedder_import_path="experiment_embedder:embed",
        )
        for arm in ("skill-baseline", "skill-localized")
    }

    baseline = configs["skill-baseline"]["agents"][0]
    localized = configs["skill-localized"]["agents"][0]
    assert baseline["import_path"] == "driftlock.harbor_agent:LHTBDriftlockAgent"
    assert localized["import_path"] == baseline["import_path"]
    for key in (
        "driftlock_skill_library_dir",
        "driftlock_skill_embedder_import_path",
    ):
        assert baseline["kwargs"][key] == localized["kwargs"][key]
    assert baseline["kwargs"]["driftlock_skill_distillation_arm"] == "baseline"
    assert localized["kwargs"]["driftlock_skill_distillation_arm"] == "localized"
    baseline_mechanism = dict(baseline["kwargs"])
    localized_mechanism = dict(localized["kwargs"])
    del baseline_mechanism["driftlock_skill_distillation_arm"]
    del localized_mechanism["driftlock_skill_distillation_arm"]
    assert baseline_mechanism == localized_mechanism

    reference = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name="job-reference",
        arm="driftlock",
        tasks=["task-a"],
    )["agents"][0]["kwargs"]
    assert not any(key.startswith("driftlock_skill_") for key in reference)


@pytest.mark.parametrize("arm", ["skill-baseline", "skill-localized"])
def test_skill_arms_refuse_incomplete_configuration(tmp_path: Path, arm: str) -> None:
    root = _lhtb_tree(tmp_path)

    with pytest.raises(ValueError, match="skill arms require"):
        build_job_config(
            lhtb_dir=root,
            jobs_dir=tmp_path / "jobs",
            job_name=f"incomplete-{arm}",
            arm=arm,
            tasks=["task-a"],
        )
