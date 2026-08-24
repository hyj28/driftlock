from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

import driftlock.lhtb as lhtb
from driftlock.lhtb import (
    HarborWorkspaceDeltaObserver,
    LHTBRuntimeCompatibilityError,
    LHTBTerminusRuntime,
    WorkspaceDelta,
    WorkspaceSnapshot,
)
from driftlock.models import StepTokenBudgetExhausted


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    cache_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class FakeResponse:
    content: str
    usage: FakeUsage
    model_name: str = "fake-model"
    reasoning_content: str | None = None
    response_id: str | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None
    action: str = "pwd"
    observation: str = "terminal output"
    parser_error: bool = False
    complete: bool = False


class OutputLengthExceededError(Exception):
    def __init__(self, response: FakeResponse) -> None:
        super().__init__("model output reached max_tokens")
        self.response = response
        self.truncated_response = response.content


class FakeLLM:
    def __init__(self, responses: list[FakeResponse | BaseException]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self._llm_kwargs: dict[str, Any] = {}
        self._use_responses_api = False

    async def call(self, prompt: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def get_model_output_limit(self) -> int:
        return 1000


class FakeChat:
    def __init__(self, model: Any, interleaved_thinking: bool = False) -> None:
        self._model = model
        self._messages: list[dict[str, Any]] = []
        self._last_response_id: str | None = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_tokens = 0
        self.total_cost = 0.0
        self.interleaved_thinking = interleaved_thinking
        self.response_chain_resets = 0

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    @property
    def rollout_details(self) -> list[Any]:
        return []

    async def chat(self, prompt: str, **kwargs: Any) -> FakeResponse:
        response = await self._model.call(prompt=prompt, **kwargs)
        self.record_response(prompt, response)
        return response

    def record_response(self, prompt: str, response: FakeResponse) -> None:
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        self.total_cache_tokens += response.usage.cache_tokens
        self.total_cost += response.usage.cost_usd
        self._messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.content},
            ]
        )

    def reset_response_chain(self) -> None:
        self.response_chain_resets += 1
        self._last_response_id = None


@dataclass
class FakeResult:
    content: str | None = None


@dataclass
class FakeObservation:
    results: list[FakeResult]


@dataclass
class FakeMetrics:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None


@dataclass
class FakeToolCall:
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]


@dataclass
class FakeStep:
    step_id: int
    timestamp: str
    source: str
    message: str
    model_name: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[FakeToolCall] | None = None
    observation: FakeObservation | None = None
    metrics: FakeMetrics | None = None


class FakeSession:
    def __init__(self, environment: FakeEnvironment | None = None) -> None:
        self._session_name = "terminus-2"
        self._user = "root"
        self._previous_buffer = "old"
        self.environment = environment
        self.starts = 0
        self.stops = 0
        self.keys: list[list[str]] = []
        self._markers: list[tuple[float, str]] = []
        self._local_asciinema_recording_path: Path | None = None
        self._remote_asciinema_recording_path: PurePosixPath | None = None

    async def get_incremental_output(self) -> str:
        return "Current Terminal Screen:\n$"

    async def is_session_alive(self) -> bool:
        return True

    async def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1

    async def send_keys(self, keys: list[str], **kwargs: Any) -> None:
        self.keys.append(keys)


class FakeAgent:
    def __init__(self, llm: FakeLLM) -> None:
        self._llm = llm
        self._session = FakeSession()
        self._chat: FakeChat | None = None
        self._prompt_template = "TASK:\n{instruction}\n\n{terminal_state}"
        self._llm_call_kwargs: dict[str, Any] = {"temperature": 0.2}
        self._max_episodes = 100
        self._n_episodes = 0
        self._trajectory_steps: list[FakeStep] = []
        self._interleaved_thinking = False
        self._process_reward_tracker = None
        self._enable_summarize = True
        self._pending_completion = False
        self._pending_subagent_refs = None
        self._pending_handoff_prompt = None
        self._termination_reason: str | None = None
        self._run_started_monotonic: float | None = None
        self._original_instruction = ""
        self._context: Any = None
        self._api_request_times: list[float] = []
        self._model_name = "fake-model"
        self._llm_kwargs: dict[str, Any] = {}
        self._last_response_model_name: str | None = None
        self.logs_dir = Path("logs")
        self.mcp_servers: list[Any] = []
        self.dump_count = 0

    def _reset_per_run_state(self) -> None:
        self._trajectory_steps = []
        self._api_request_times = []
        self._n_episodes = 0
        self._pending_completion = False
        self._termination_reason = None
        self._run_started_monotonic = None

    async def _build_skills_section(self, environment: Any) -> str:
        return ""

    def _limit_output_length(self, value: str) -> str:
        return value

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: FakeChat,
        logging_dir: Path | None = None,
        original_instruction: str = "",
        reset_context: bool = True,
    ) -> None:
        del logging_dir, original_instruction, reset_context
        for _episode in range(self._n_episodes, self._max_episodes):
            self._n_episodes += 1
            response = await self._query_llm(
                chat,
                initial_prompt,
                (None, None, None),
            )
            if response.parser_error:
                observation = (
                    "Previous response had parsing errors:\nERROR: invalid response"
                )
                calls = None
            else:
                was_pending = self._pending_completion
                if response.complete:
                    self._pending_completion = True
                    observation = (
                        response.observation
                        if was_pending
                        else "Please confirm completion.\n" + response.observation
                    )
                else:
                    self._pending_completion = False
                    observation = response.observation
                calls = [
                    FakeToolCall(
                        "call", "bash_command", {"keystrokes": response.action}
                    )
                ]
                if response.complete:
                    calls.append(FakeToolCall("done", "mark_task_complete", {}))
            step = FakeStep(
                step_id=len(self._trajectory_steps) + 1,
                timestamp="now",
                source="agent",
                message=response.content,
                model_name=response.model_name,
                tool_calls=calls,
                observation=FakeObservation([FakeResult(observation)]),
                metrics=FakeMetrics(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                ),
            )
            self._trajectory_steps.append(step)
            if response.complete and was_pending:
                self._termination_reason = "confirmed_task_complete"
            else:
                self._termination_reason = "max_turns"

    def _update_context_from_state(self, context: Any) -> None:
        assert self._chat is not None
        context.n_input_tokens = self._chat.total_input_tokens
        context.n_output_tokens = self._chat.total_output_tokens

    def _dump_trajectory(self) -> None:
        self.dump_count += 1


class FakeObserver:
    def __init__(self, canonical: str = "/app") -> None:
        self.canonical = canonical
        self.snapshots = [
            WorkspaceSnapshot({"a": "1"}, "old"),
            WorkspaceSnapshot({"a": "2"}, "new"),
            WorkspaceSnapshot({"a": "2"}, "new"),
            WorkspaceSnapshot({"a": "2"}, "new"),
        ]

    async def canonical_workspace(self) -> str:
        return self.canonical

    async def snapshot(self) -> WorkspaceSnapshot:
        return self.snapshots.pop(0)

    def compare(
        self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> WorkspaceDelta:
        return WorkspaceDelta(("a",) if before != after else (), "step diff")


@dataclass
class RemoteResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class FakeEnvironment:
    def __init__(self, results: list[RemoteResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict[str, Any]] = []

    async def exec(self, command: str, **kwargs: Any) -> RemoteResult:
        self.calls.append({"command": command, **kwargs})
        if command.startswith("realpath -e --"):
            return RemoteResult("/app\n")
        return self.results.pop(0) if self.results else RemoteResult()


@pytest.fixture
def fake_harbor_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    symbols = {
        ("harbor.llms.chat", "Chat"): FakeChat,
        ("harbor.models.trajectories", "Step"): FakeStep,
        ("harbor.models.trajectories", "Observation"): FakeObservation,
        ("harbor.models.trajectories", "ObservationResult"): FakeResult,
        ("harbor.models.trajectories", "Metrics"): FakeMetrics,
    }
    monkeypatch.setattr(
        lhtb, "_harbor_symbol", lambda module, name: symbols[module, name]
    )


def _runtime(
    responses: list[FakeResponse | BaseException],
    *,
    observer: FakeObserver | None = None,
    rate_limit_retries: int = 0,
) -> tuple[LHTBTerminusRuntime, FakeAgent, FakeLLM, Any]:
    llm = FakeLLM(responses)
    agent = FakeAgent(llm)
    context = SimpleNamespace(n_input_tokens=None, n_output_tokens=None)
    environment = FakeEnvironment()
    runtime = LHTBTerminusRuntime(
        agent,
        environment,
        context,
        remote_workspace="/app",
        observer=observer or FakeObserver(),
        require_pinned_harbor=False,
        rate_limit_retries=rate_limit_retries,
        rate_limit_backoff_sec=0.0,
    )
    return runtime, agent, llm, context


async def test_runtime_yields_one_response_and_restores_semantic_chat(
    fake_harbor_symbols: None,
) -> None:
    responses = [
        FakeResponse("first", FakeUsage(20, 4), observation="one"),
        FakeResponse("second", FakeUsage(30, 5), observation="two"),
    ]
    runtime, agent, llm, context = _runtime(responses, observer=FakeObserver())

    prompt = await runtime.prepare_start(
        "fix it", plan="inspect then test", rollback_feedback=None
    )
    assert "must return to the login shell" in prompt
    first = await runtime.start(prompt=prompt, tokens_remaining=10_000)
    second = await runtime.resume(
        first.conversation,
        prompt=first.conversation.next_prompt,
        tokens_remaining=10_000,
    )

    assert runtime.provider_call_count == 2
    assert first.tokens == 24
    assert first.changed_paths == ("a",)
    assert first.diff == "step diff"
    assert second.tokens == 35
    assert second.conversation.episode == 2
    assert len(second.conversation.messages) == 4
    assert 0 < llm.calls[0]["max_tokens"] < 10_000
    assert 0 < llm.calls[1]["max_tokens"] < llm.calls[0]["max_tokens"]
    assert llm.calls[0]["num_retries"] == 0
    assert llm.calls[0]["max_retries"] == 0
    assert agent._max_episodes == 100
    assert agent._n_episodes == 2
    assert context.n_input_tokens == 50
    assert context.n_output_tokens == 9


async def test_runtime_reserves_input_tokens_and_uses_responses_ceiling(
    fake_harbor_symbols: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, agent, llm, _ = _runtime([FakeResponse("ok", FakeUsage(30, 10))])
    llm._use_responses_api = True
    agent._llm_call_kwargs["max_completion_tokens"] = 50
    monkeypatch.setattr(lhtb, "_conservative_input_token_bound", lambda *args: 30)
    prompt = await runtime.prepare_start("task", plan="", rollback_feedback=None)

    await runtime.start(prompt=prompt, tokens_remaining=100)

    assert llm.calls[0]["max_output_tokens"] == 50
    assert "max_tokens" not in llm.calls[0]
    assert "max_completion_tokens" not in llm.calls[0]


async def test_runtime_output_ceiling_preserves_provider_routing_byte_for_byte(
    fake_harbor_symbols: None,
) -> None:
    runtime, agent, llm, _ = _runtime([FakeResponse("ok", FakeUsage(30, 10))])
    agent._llm_call_kwargs.update(
        {
            "max_tokens": 50,
            "extra_body": {
                "provider": {
                    "only": ["baidu/fp8"],
                    "allow_fallbacks": False,
                }
            },
        }
    )
    before = json.dumps(agent._llm_call_kwargs, sort_keys=True, separators=(",", ":"))
    prompt = await runtime.prepare_start("task", plan="", rollback_feedback=None)

    await runtime.start(prompt=prompt, tokens_remaining=10_000)

    assert llm.calls[0]["extra_body"] == {
        "provider": {"only": ["baidu/fp8"], "allow_fallbacks": False}
    }
    after = json.dumps(agent._llm_call_kwargs, sort_keys=True, separators=(",", ":"))
    assert after == before


async def test_runtime_refuses_call_when_input_exhausts_budget(
    fake_harbor_symbols: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _, llm, _ = _runtime([FakeResponse("unused", FakeUsage(1, 1))])
    monkeypatch.setattr(lhtb, "_conservative_input_token_bound", lambda *args: 101)
    prompt = await runtime.prepare_start("task", plan="", rollback_feedback=None)

    with pytest.raises(StepTokenBudgetExhausted, match="cannot cover"):
        await runtime.start(prompt=prompt, tokens_remaining=100)

    assert llm.calls == []


async def test_runtime_rejects_canonical_observer_mismatch(
    fake_harbor_symbols: None,
) -> None:
    runtime, _, _, _ = _runtime([], observer=FakeObserver("/other"))

    with pytest.raises(LHTBRuntimeCompatibilityError, match="different canonical"):
        await runtime.prepare_start("task", plan="", rollback_feedback=None)


async def test_runtime_keeps_physical_accounting_when_rollback_restarts(
    fake_harbor_symbols: None,
) -> None:
    runtime, agent, _, _ = _runtime(
        [
            FakeResponse("wrong", FakeUsage(10, 2)),
            FakeResponse("retry", FakeUsage(11, 3)),
        ]
    )
    prompt = await runtime.prepare_start("task", plan="plan", rollback_feedback=None)
    await runtime.start(prompt=prompt, tokens_remaining=None)
    physical_steps = len(agent._trajectory_steps)

    retry_prompt = await runtime.prepare_start(
        "task", plan="plan", rollback_feedback="wrong branch"
    )
    retry = await runtime.start(prompt=retry_prompt, tokens_remaining=None)

    assert retry.conversation.episode == 1
    assert len(retry.conversation.messages) == 2
    assert "wrong branch" in retry_prompt
    assert runtime.provider_call_count == 2
    assert agent._chat is not None
    assert agent._chat.total_input_tokens == 21
    assert agent._chat.total_output_tokens == 5
    assert len(agent._trajectory_steps) == physical_steps + 2
    assert agent._n_episodes == 2
    assert len(runtime.environment.calls) == 3
    assert "pane_pid" in runtime.environment.calls[1]["command"]


async def test_runtime_records_parser_error_as_billed_boundary(
    fake_harbor_symbols: None,
) -> None:
    response = FakeResponse("bad json", FakeUsage(12, 8), parser_error=True)
    runtime, _, _, _ = _runtime([response])
    prompt = await runtime.prepare_start("task", plan="", rollback_feedback=None)

    boundary = await runtime.start(prompt=prompt, tokens_remaining=10_000)

    assert boundary.tokens == 20
    assert boundary.error is not None
    assert "parsing errors" in boundary.error
    assert boundary.conversation.next_prompt.startswith("Previous response")


async def test_runtime_preserves_harbor_two_response_completion_handshake(
    fake_harbor_symbols: None,
) -> None:
    runtime, _, _, _ = _runtime(
        [
            FakeResponse("done?", FakeUsage(10, 2), complete=True),
            FakeResponse("confirmed", FakeUsage(11, 2), complete=True),
        ]
    )
    prompt = await runtime.prepare_start("task", plan="", rollback_feedback=None)

    first = await runtime.start(prompt=prompt, tokens_remaining=10_000)
    second = await runtime.resume(
        first.conversation,
        prompt=first.conversation.next_prompt,
        tokens_remaining=10_000,
    )

    assert first.conversation.pending_completion
    assert not first.completed
    assert second.conversation.pending_completion
    assert second.completed


async def test_runtime_records_truncation_usage_without_retry(
    fake_harbor_symbols: None,
) -> None:
    response = FakeResponse("truncated", FakeUsage(40, 10))
    runtime, agent, llm, _ = _runtime([OutputLengthExceededError(response)])
    prompt = await runtime.prepare_start("task", plan="", rollback_feedback=None)

    boundary = await runtime.start(prompt=prompt, tokens_remaining=10_000)

    assert runtime.provider_call_count == 1
    assert len(llm.calls) == 1
    assert boundary.tokens == 50
    assert boundary.error == "model output reached max_tokens"
    assert "shorter response" in boundary.conversation.next_prompt
    assert boundary.conversation.messages[-1]["content"] == "truncated"
    assert agent._chat is not None
    assert agent._chat.total_input_tokens == 40
    assert agent._chat.total_output_tokens == 10


async def test_runtime_refuses_unpatched_truncation(
    fake_harbor_symbols: None,
) -> None:
    response = FakeResponse("truncated", FakeUsage(1, 1))
    error = OutputLengthExceededError(response)
    error.response = None
    runtime, _, _, _ = _runtime([error])
    prompt = await runtime.prepare_start("task", plan="", rollback_feedback=None)

    with pytest.raises(LHTBRuntimeCompatibilityError, match="retain"):
        await runtime.start(prompt=prompt, tokens_remaining=10_000)


async def test_workspace_observer_reports_content_and_git_view_changes() -> None:
    environment = FakeEnvironment(
        [
            RemoteResult(
                "d\0./src\0metadata:\0f\0./src/a.py\0metadata:" + "a" * 64 + "\0"
            ),
            RemoteResult("-old\n"),
            RemoteResult(
                "d\0./src\0metadata:\0f\0./src/a.py\0metadata:"
                + "b" * 64
                + "\0l\0./latest\0metadata:7372632f612e7079\0"
            ),
            RemoteResult("+new\n"),
        ]
    )
    observer = HarborWorkspaceDeltaObserver(
        environment, remote_workspace="/app", user="root"
    )

    before = await observer.snapshot()
    after = await observer.snapshot()
    delta = observer.compare(before, after)

    assert delta.changed_paths == ("latest", "src/a.py")
    assert "workspace-before" in delta.diff
    manifest_call = next(
        call for call in environment.calls if "python3" in call["command"]
    )
    assert manifest_call["cwd"] == "/app"


def test_workspace_observer_rejects_root_or_relative_workspace() -> None:
    with pytest.raises(ValueError, match="absolute non-root"):
        HarborWorkspaceDeltaObserver(FakeEnvironment(), remote_workspace="/")
    with pytest.raises(ValueError, match="absolute non-root"):
        HarborWorkspaceDeltaObserver(FakeEnvironment(), remote_workspace="app")


def test_workspace_manifest_supports_newlines_and_rejects_bad_digests() -> None:
    parsed = lhtb._parse_sha256_manifest(
        "d\0./empty\nname\0metadata:\0l\0./link\0metadata:7461726765740a6e616d65\0"
    )

    assert parsed == {
        "empty\nname": "d:metadata:",
        "link": "l:metadata:7461726765740a6e616d65",
    }
    with pytest.raises(RuntimeError, match="SHA-256"):
        lhtb._parse_sha256_manifest("f\0./file\0not-a-digest\0")


@pytest.mark.skipif(not hasattr(os, "listxattr"), reason="requires POSIX xattrs")
def test_workspace_manifest_detects_metadata_and_special_files(tmp_path: Path) -> None:
    file_path = tmp_path / "mode-only.txt"
    file_path.write_text("same content")
    fifo_path = tmp_path / "events.fifo"
    os.mkfifo(fifo_path)

    before = subprocess.run(
        lhtb.HarborWorkspaceDeltaObserver._MANIFEST_COMMAND,
        cwd=tmp_path,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    file_path.chmod(0o700)
    after = subprocess.run(
        lhtb.HarborWorkspaceDeltaObserver._MANIFEST_COMMAND,
        cwd=tmp_path,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    before_manifest = lhtb._parse_sha256_manifest(before)
    after_manifest = lhtb._parse_sha256_manifest(after)
    assert before_manifest["mode-only.txt"] != after_manifest["mode-only.txt"]
    assert after_manifest["events.fifo"].startswith("p:")


def test_single_attempt_configuration_rejects_retry_and_fallback_keys() -> None:
    agent = SimpleNamespace(
        _llm_kwargs={"fallbacks": ["other"]},
        _llm_call_kwargs={"temperature": 0.2},
    )
    llm = SimpleNamespace(_llm_kwargs={"max_retries": 2})

    with pytest.raises(LHTBRuntimeCompatibilityError, match="fallbacks, max_retries"):
        lhtb._validate_single_attempt_configuration(agent, llm)


def test_packaged_harbor_patch_is_available() -> None:
    patch = lhtb.lhtb_harbor_patch_path()

    assert patch.is_file()
    text = patch.read_text()
    assert "DRIFTLOCK_HARBOR_PATCH_VERSION = 11" in text
    # The packaged patch writes the marker the installed module then verifies, so
    # bumping one without the other would ship a tree preflight always rejects.
    assert (
        f"DRIFTLOCK_HARBOR_PATCH_VERSION = {lhtb.DRIFTLOCK_HARBOR_PATCH_VERSION}"
        in text
    )
    assert "_driftlock_finalize_after_agent_run" in text
    assert "cat >>" in text
    assert "block=True" in text
    assert "__driftlock_status=$?" not in text
    # A batch that cannot reach a shell boundary goes back to the model as
    # feedback; only the unreachable last line of defence still raises.
    assert "_driftlock_boundary_feedback" in text


def test_recording_rotation_stays_based_on_original_path() -> None:
    session = FakeSession()
    session._local_asciinema_recording_path = Path("run.cast")
    session._remote_asciinema_recording_path = PurePosixPath("/tmp/run.cast")

    lhtb._rotate_session_recording(session, 1)
    lhtb._rotate_session_recording(session, 2)

    assert session._local_asciinema_recording_path == Path("run.rollback-2.cast")
    assert session._remote_asciinema_recording_path == PurePosixPath(
        "/tmp/run.rollback-2.cast"
    )


async def test_before_restore_replaces_shell_and_verifies_cwd(
    fake_harbor_symbols: None,
) -> None:
    environment = FakeEnvironment(
        [
            RemoteResult(),
            RemoteResult("/app\n"),
            RemoteResult("/app\n"),
        ]
    )
    runtime, agent, _, _ = _runtime([])
    runtime.environment = environment
    runtime._process_baseline = ("2:100", "3:200")
    runtime._canonical_workspace = "/app"
    agent._session.environment = environment
    agent._session._local_asciinema_recording_path = Path("run.cast")
    agent._session._remote_asciinema_recording_path = PurePosixPath("/tmp/run.cast")

    await runtime.before_workspace_restore("/app")

    cleanup = next(
        call["command"] for call in environment.calls if "kill -STOP" in call["command"]
    )
    assert "kill -KILL" in cleanup
    assert "2:100 3:200" in cleanup
    assert "tmux kill-session" in cleanup
    assert "exit 42" in cleanup
    assert agent._session.stops == 1
    assert agent._session.starts == 1
    assert agent._session.keys == [["cd -- /app", "Enter"]]
    assert agent._session._previous_buffer is None
    assert agent._session._local_asciinema_recording_path == Path("run.rollback-1.cast")
    assert agent._session._remote_asciinema_recording_path == PurePosixPath(
        "/tmp/run.rollback-1.cast"
    )


def test_runtime_rejects_process_reward_tracker() -> None:
    agent = FakeAgent(FakeLLM([]))
    agent._process_reward_tracker = object()

    with pytest.raises(LHTBRuntimeCompatibilityError, match="must be disabled"):
        LHTBTerminusRuntime(
            agent,
            object(),
            object(),
            remote_workspace="/app",
            observer=FakeObserver(),
            require_pinned_harbor=False,
        )


def test_runtime_rejects_raw_trajectory_mode() -> None:
    agent = FakeAgent(FakeLLM([]))
    agent._save_raw_content_in_trajectory = True

    with pytest.raises(LHTBRuntimeCompatibilityError, match="raw_content"):
        LHTBTerminusRuntime(
            agent,
            FakeEnvironment(),
            object(),
            remote_workspace="/app",
            observer=FakeObserver(),
            require_pinned_harbor=False,
        )


def test_step_tokens_rejects_missing_provider_usage() -> None:
    step = FakeStep(
        step_id=1,
        timestamp="now",
        source="agent",
        message="response",
        metrics=FakeMetrics(prompt_tokens=None, completion_tokens=2),
    )

    with pytest.raises(LHTBRuntimeCompatibilityError, match="must report"):
        lhtb._step_tokens(step)


def test_generated_process_scripts_are_valid_shell_and_quote_inputs() -> None:
    baseline = lhtb._process_baseline_command("session name; touch /tmp/nope")
    cleanup = lhtb._kill_tmux_tree_command(
        "session name; touch /tmp/nope",
        "/workspace with spaces",
        process_baseline=("2:100", "30:400"),
    )

    for script in (baseline, cleanup):
        result = subprocess.run(
            ["sh", "-n"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    assert "'session name; touch /tmp/nope'" in cleanup
    assert "'/workspace with spaces'" in cleanup


class FakeRateLimitError(Exception):
    """Shaped like litellm.RateLimitError: an HTTP 429 with a status attribute."""

    def __init__(self, message: str = "rate limited") -> None:
        super().__init__(message)
        self.status_code = 429


class FakeServerError(Exception):
    def __init__(self) -> None:
        super().__init__("bad gateway")
        self.status_code = 502


def test_rate_limit_rejection_matches_429_only() -> None:
    assert lhtb.is_rate_limit_rejection(FakeRateLimitError()) is True
    assert lhtb.is_rate_limit_rejection(FakeServerError()) is False
    assert lhtb.is_rate_limit_rejection(TimeoutError("slow")) is False
    assert lhtb.is_rate_limit_rejection(ValueError("nope")) is False


def test_rate_limit_rejection_recognises_the_class_name_without_a_status() -> None:
    # litellm raises RateLimitError; some transports leave status_code unset.
    named = type("RateLimitError", (Exception,), {})()

    assert lhtb.is_rate_limit_rejection(named) is True
    assert lhtb.is_rate_limit_rejection(type("Other", (Exception,), {})()) is False


async def test_counting_llm_retries_a_rate_limited_call_on_the_same_provider() -> None:
    response = FakeResponse("done", FakeUsage(10, 2))
    llm = FakeLLM([FakeRateLimitError(), FakeRateLimitError(), response])
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    counting = lhtb._CountingLLM(
        llm,
        bypass_retry_wrapper=False,
        rate_limit_retries=3,
        rate_limit_backoff_sec=1.0,
        rate_limit_backoff_cap_sec=2.0,
        sleep=sleep,
    )

    assert await counting.call("go", extra_body={"provider": {"only": ["baidu/fp8"]}})
    # Three physical attempts, two of them refused, so one generation was billed.
    assert counting.call_count == 3
    assert counting.rate_limited_call_count == 2
    assert counting.call_count - counting.rate_limited_call_count == 1
    assert len(llm.calls) == 3
    assert slept == [1.0, 2.0]
    # The routing is never rewritten: a retry must not become a provider switch.
    assert {call["extra_body"]["provider"]["only"][0] for call in llm.calls} == {
        "baidu/fp8"
    }


async def test_counting_llm_gives_up_after_the_configured_retries() -> None:
    llm = FakeLLM([FakeRateLimitError() for _ in range(4)])

    async def sleep(seconds: float) -> None:
        return None

    counting = lhtb._CountingLLM(
        llm,
        bypass_retry_wrapper=False,
        rate_limit_retries=2,
        rate_limit_backoff_sec=1.0,
        sleep=sleep,
    )

    with pytest.raises(FakeRateLimitError):
        await counting.call("go")

    assert len(llm.calls) == 3
    assert counting.rate_limited_call_count == 2


async def test_counting_llm_does_not_retry_a_non_rate_limit_failure() -> None:
    llm = FakeLLM([FakeServerError(), FakeResponse("unreachable", FakeUsage(10, 2))])

    counting = lhtb._CountingLLM(
        llm, bypass_retry_wrapper=False, rate_limit_retries=5, rate_limit_backoff_sec=0
    )

    with pytest.raises(FakeServerError):
        await counting.call("go")

    assert len(llm.calls) == 1
    assert counting.rate_limited_call_count == 0


async def test_counting_llm_makes_one_attempt_when_retries_are_disabled() -> None:
    llm = FakeLLM([FakeRateLimitError(), FakeResponse("unreachable", FakeUsage(10, 2))])

    counting = lhtb._CountingLLM(llm, bypass_retry_wrapper=False)

    with pytest.raises(FakeRateLimitError):
        await counting.call("go")

    assert len(llm.calls) == 1
    assert counting.rate_limited_call_count == 0


def test_counting_llm_rejects_a_negative_retry_policy() -> None:
    llm = FakeLLM([])

    with pytest.raises(ValueError, match="rate_limit_retries cannot be negative"):
        lhtb._CountingLLM(llm, bypass_retry_wrapper=False, rate_limit_retries=-1)
    with pytest.raises(ValueError, match="backoff cannot be negative"):
        lhtb._CountingLLM(llm, bypass_retry_wrapper=False, rate_limit_backoff_sec=-1.0)


async def test_a_rate_limited_retry_still_counts_as_one_boundary_call(
    fake_harbor_symbols: None,
) -> None:
    # The boundary check counts *billable* calls. Two 429s and one generation is
    # one step, not three: this is what killed 12 trials on 2026-08-24, when the
    # retry worked and the invariant threw the result away.
    runtime, _agent, llm, _context = _runtime(
        [
            FakeRateLimitError(),
            FakeRateLimitError(),
            FakeResponse("done", FakeUsage(20, 4)),
        ],
        rate_limit_retries=3,
    )

    prompt = await runtime.prepare_start("fix it", plan="p", rollback_feedback=None)
    boundary = await runtime.start(prompt=prompt, tokens_remaining=10_000)

    assert boundary.tokens == 24
    assert len(llm.calls) == 3
    assert runtime.provider_call_count == 3
    assert runtime.rate_limited_call_count == 2


async def test_an_exhausted_rate_limit_retry_propagates(
    fake_harbor_symbols: None,
) -> None:
    # When the provider stays down past the retry budget the step must fail
    # loudly with the provider's own error, not be silently absorbed.
    runtime, _agent, llm, _context = _runtime(
        [FakeRateLimitError() for _ in range(5)], rate_limit_retries=2
    )
    prompt = await runtime.prepare_start("fix it", plan="p", rollback_feedback=None)

    with pytest.raises(FakeRateLimitError):
        await runtime.start(prompt=prompt, tokens_remaining=10_000)

    assert len(llm.calls) == 3
    assert runtime.rate_limited_call_count == 2
