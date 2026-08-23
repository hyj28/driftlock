from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from driftlock.heuristics import HeuristicConfig
from driftlock.lhtb import WorkspaceDelta, WorkspaceSnapshot
from driftlock.lhtb_analysis import _validate_arm_identity
from driftlock.lhtb_experiment import DEFAULT_MODEL, build_job_config
from driftlock.models import JudgeVerdict, RunResult, RunStatus, Verdict
from driftlock.native_lhtb import (
    BilledProviderFailure,
    BilledProviderResponse,
    ContextUsageRecorder,
    LHTBNativeAgentRuntime,
    NativeProcessQuiescer,
    PhysicalProviderBoundaryError,
    ProviderUsage,
    SingleAttemptJSONProvider,
    UnknownBilledProviderUsage,
    append_verifier_feedback,
    apply_native_accounting,
    billed_provider_exception,
    billed_provider_response,
    exact_provider_usage,
    native_checkpoint_store_root,
    set_native_result_metadata,
    set_native_token_limit_metadata,
)
from driftlock.remote import RemoteCheckpointError
from driftlock.runner import RunnerConfig


@dataclass(frozen=True, slots=True)
class LocalResult:
    return_code: int
    stdout: str
    stderr: str


class RecordingRemoteEnvironment:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.uploads: list[tuple[Path, str]] = []
        self.downloads: list[tuple[str, Path]] = []

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        cwd: str | None = None,
    ) -> LocalResult:
        self.calls.append(
            {
                "command": command,
                "timeout_sec": timeout_sec,
                "user": user,
                "cwd": cwd,
            }
        )
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_sec
        )
        return LocalResult(process.returncode or 0, stdout.decode(), stderr.decode())

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        self.uploads.append((source, target_path))
        shutil.copy2(source, target_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        self.downloads.append((source_path, target))
        shutil.copy2(source_path, target)


class RestoreApplyFailureEnvironment(RecordingRemoteEnvironment):
    def __init__(self, workspace: Path) -> None:
        super().__init__()
        self.workspace = workspace
        self.failed_once = False

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        cwd: str | None = None,
    ) -> LocalResult:
        if "cp -a" in command and not self.failed_once:
            self.calls.append(
                {
                    "command": command,
                    "timeout_sec": timeout_sec,
                    "user": user,
                    "cwd": cwd,
                }
            )
            self.failed_once = True
            (self.workspace / "rejected.txt").write_text(
                "partial restore", encoding="utf-8"
            )
            return LocalResult(1, "", "injected restore failure")
        return await super().exec(
            command,
            timeout_sec=timeout_sec,
            user=user,
            cwd=cwd,
        )


class CleanupFailureEnvironment(RecordingRemoteEnvironment):
    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        cwd: str | None = None,
    ) -> LocalResult:
        if "rejected processes survived cleanup" in command:
            self.calls.append(
                {
                    "command": command,
                    "timeout_sec": timeout_sec,
                    "user": user,
                    "cwd": cwd,
                }
            )
            return LocalResult(42, "", "injected process cleanup failure")
        return await super().exec(
            command,
            timeout_sec=timeout_sec,
            user=user,
            cwd=cwd,
        )


class LocalLiteralObserver:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    async def canonical_workspace(self) -> str:
        return str(self.workspace)

    async def snapshot(self) -> WorkspaceSnapshot:
        files = {
            path.relative_to(self.workspace).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(self.workspace.rglob("*"))
            if path.is_file()
        }
        return WorkspaceSnapshot(files=files)

    def compare(
        self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> WorkspaceDelta:
        paths = sorted(set(before.files) | set(after.files))
        changed = tuple(
            path for path in paths if before.files.get(path) != after.files.get(path)
        )
        return WorkspaceDelta(changed, "literal observer diff" if changed else "")


class ScriptedPhysicalCall:
    def __init__(
        self,
        responses: list[BilledProviderResponse | BilledProviderFailure],
        *,
        increments: list[int] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.increments = list(increments or [1] * len(responses))
        self._physical_call_count = 0
        self.prompts: list[str] = []
        self.output_caps: list[int] = []

    @property
    def physical_call_count(self) -> int:
        return self._physical_call_count

    async def __call__(
        self, prompt: str, *, max_output_tokens: int
    ) -> BilledProviderResponse:
        self.prompts.append(prompt)
        self.output_caps.append(max_output_tokens)
        self._physical_call_count += self.increments.pop(0)
        response = self.responses.pop(0)
        if isinstance(response, BilledProviderFailure):
            raise response
        return response


class DriftOnceJudge:
    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, context: Any) -> JudgeVerdict:
        self.calls += 1
        assert context.tokens_remaining == 999_993
        return JudgeVerdict(
            Verdict.DRIFTED,
            "avoid the rejected no-op branch",
            tokens=7,
        )


def _response(
    tool: str,
    arguments: dict[str, Any],
    *,
    input_tokens: int,
    output_tokens: int,
    cache_tokens: int = 0,
    cost_usd: float = 0.0,
    text: str = "",
) -> BilledProviderResponse:
    return BilledProviderResponse(
        json.dumps(
            {
                "text": text,
                "tool_calls": [
                    {
                        "name": tool,
                        "arguments": arguments,
                        "call_id": f"{tool}-1",
                    }
                ],
            }
        ),
        ProviderUsage(
            input_tokens=input_tokens,
            cache_tokens=cache_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ),
    )


def _raw_provider_response(
    *,
    prompt_tokens: object = 11,
    cache_tokens: object = 3,
    completion_tokens: object = 5,
    cost_usd: object = 0.25,
    content: object = "payload",
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            cache_tokens=cache_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        ),
    )


def test_exact_provider_usage_maps_only_exact_finite_provider_fields() -> None:
    assert exact_provider_usage(_raw_provider_response()) == ProviderUsage(
        input_tokens=11,
        cache_tokens=3,
        output_tokens=5,
        cost_usd=0.25,
    )
    assert exact_provider_usage(_raw_provider_response(prompt_tokens=True)) is None
    assert exact_provider_usage(_raw_provider_response(cache_tokens=-1)) is None
    assert exact_provider_usage(_raw_provider_response(completion_tokens=2.5)) is None
    assert exact_provider_usage(_raw_provider_response(cost_usd=float("nan"))) is None
    assert exact_provider_usage(_raw_provider_response(cost_usd=True)) is None


def test_provider_response_and_exception_triage_is_harbor_free() -> None:
    response = _raw_provider_response()
    assert billed_provider_response(response) == BilledProviderResponse(
        "payload", ProviderUsage(11, 3, 5, 0.25)
    )

    truncated = RuntimeError("output limit")
    truncated.response = response
    truncated.truncated_response = '{"text":"partial"}'
    assert billed_provider_exception(truncated) == BilledProviderResponse(
        '{"text":"partial"}',
        ProviderUsage(11, 3, 5, 0.25),
        truncated=True,
    )

    failed = RuntimeError("timeout after response")
    failed.response = response
    with pytest.raises(BilledProviderFailure) as known:
        billed_provider_exception(failed)
    assert known.value.usage == ProviderUsage(11, 3, 5, 0.25)

    with pytest.raises(BilledProviderFailure) as unknown:
        billed_provider_exception(TimeoutError("connection timed out"))
    assert unknown.value.usage is None


def test_native_checkpoint_store_root_is_outside_agent_logs(tmp_path: Path) -> None:
    logs = tmp_path / "trial" / "agent"

    root = native_checkpoint_store_root(logs)

    assert root == tmp_path / "trial" / ".driftlock-native-checkpoints"
    assert logs not in root.parents


def _runtime(
    tmp_path: Path,
    call: ScriptedPhysicalCall,
    *,
    budget: int = 1_000_000,
    heuristic_config: HeuristicConfig | None = None,
    fine_judge: Any = None,
    retain_checkpoints: bool = False,
    environment: RecordingRemoteEnvironment | None = None,
) -> tuple[
    Path,
    RecordingRemoteEnvironment,
    SingleAttemptJSONProvider,
    LHTBNativeAgentRuntime,
]:
    workspace = tmp_path / "remote-workspace"
    remote_tmp = tmp_path / "remote-tmp"
    workspace.mkdir()
    remote_tmp.mkdir()
    resolved_environment = environment or RecordingRemoteEnvironment()
    provider = SingleAttemptJSONProvider(call)
    runtime = LHTBNativeAgentRuntime(
        resolved_environment,
        LocalLiteralObserver(workspace),
        provider,
        remote_workspace=str(workspace),
        store_dir=tmp_path / "host-only-checkpoints",
        remote_tmp_dir=str(remote_tmp),
        user="agent-user",
        runner_config=RunnerConfig(
            max_steps=20,
            max_rollbacks=2,
            checkpoint_interval=5,
            max_tokens=budget,
        ),
        heuristic_config=heuristic_config,
        fine_judge=fine_judge,
        retain_checkpoints=retain_checkpoints,
        agent_max_output_tokens=100,
        agent_min_output_tokens=4,
    )
    return workspace, resolved_environment, provider, runtime


async def test_native_runtime_completes_with_remote_tools_and_exact_usage(
    tmp_path: Path,
) -> None:
    call = ScriptedPhysicalCall(
        [
            _response(
                "read_file",
                {"path": "broken.py"},
                input_tokens=10,
                output_tokens=1,
                cache_tokens=1,
                cost_usd=0.1,
            ),
            _response(
                "write_file",
                {"path": "broken.py", "content": "print('fixed')\n"},
                input_tokens=20,
                output_tokens=2,
                cache_tokens=2,
                cost_usd=0.2,
            ),
            _response(
                "run_shell",
                {"command": "python3 broken.py"},
                input_tokens=30,
                output_tokens=3,
                cache_tokens=3,
                cost_usd=0.3,
            ),
            _response(
                "complete",
                {"summary": "repaired and verified"},
                input_tokens=5,
                output_tokens=1,
                cost_usd=0.05,
            ),
        ]
    )
    workspace, environment, provider, runtime = _runtime(tmp_path, call)
    (workspace / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    result = await runtime.run(goal="repair broken.py")

    assert result.status is RunStatus.COMPLETED
    assert [step.outcome.tokens for step in result.steps] == [11, 22, 33, 6]
    assert result.agent_tokens_used == 72
    assert result.tokens_used == 72
    assert provider.usage.input_tokens == 65
    assert provider.usage.cache_tokens == 6
    assert provider.usage.output_tokens == 7
    assert provider.usage.cost_usd == pytest.approx(0.65)
    assert call.physical_call_count == 4
    assert len(call.prompts) == 4
    assert (workspace / "broken.py").read_text(encoding="utf-8") == ("print('fixed')\n")
    assert result.steps[1].outcome.changed_paths == ("broken.py",)
    assert result.steps[1].outcome.diff == "literal observer diff"
    shell_calls = [
        item for item in environment.calls if "python3 broken.py" in item["command"]
    ]
    assert len(shell_calls) == 1
    assert shell_calls[0]["command"].startswith(f"cd -- {workspace}")
    assert all(item["user"] == "agent-user" for item in environment.calls)
    context = SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        cost_usd=None,
        metadata={},
    )
    ContextUsageRecorder(context).apply(
        context,
        agent_usage=provider.usage,
        agent_request_times_msec=(10.0, 20.0, 30.0, 40.0),
        provider_request_count=4,
    )
    assert context.n_input_tokens == 65
    assert context.n_cache_tokens == 6
    assert context.n_output_tokens == 7
    assert context.cost_usd == pytest.approx(0.65)
    assert context.metadata["driftlock_native_usage"]["provider_request_count"] == 4


def test_native_accounting_reconcile_guards_provider_and_judge_totals() -> None:
    context = SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        cost_usd=None,
        metadata={},
    )
    provider = SimpleNamespace(
        usage=ProviderUsage(8, 2, 3, 0.4),
        exact_usage_request_count=1,
        unknown_billed_request_count=0,
        provider_call_count=1,
    )

    with pytest.raises(RuntimeError, match="provider usage does not reconcile"):
        apply_native_accounting(
            context,
            recorder=ContextUsageRecorder(context),
            runtime=SimpleNamespace(
                agent_tokens_consumed=10,
                judge_tokens_consumed=0,
            ),
            provider=provider,
            agent_request_times_msec=(15.0,),
            reconcile=True,
        )

    judge = SimpleNamespace(
        n_input_tokens=4,
        n_cache_tokens=1,
        n_output_tokens=2,
        cost_usd=0.1,
        request_times_msec=[6.0],
    )
    with pytest.raises(RuntimeError, match="judge usage does not reconcile"):
        apply_native_accounting(
            context,
            recorder=ContextUsageRecorder(context),
            runtime=SimpleNamespace(
                agent_tokens_consumed=11,
                judge_tokens_consumed=5,
            ),
            provider=provider,
            agent_request_times_msec=(15.0,),
            judge=judge,
            reconcile=True,
        )


def test_native_metadata_writers_publish_literal_result_and_budget_fields() -> None:
    context = SimpleNamespace(metadata={"preserved": "yes"})
    result = RunResult(
        status=RunStatus.COMPLETED,
        state={"done": True},
        steps=(),
        rollbacks=(),
        checkpoints=(),
        tokens_used=17,
        agent_tokens_used=12,
        judge_tokens_used=5,
    )
    runtime = SimpleNamespace(tokens_consumed=29)

    set_native_result_metadata(
        context,
        result=result,
        runtime=runtime,
        trial_token_budget=100,
    )

    assert context.metadata == {
        "preserved": "yes",
        "driftlock": {
            "status": "completed",
            "steps": 0,
            "rollbacks": 0,
            "tokens_used": 17,
            "agent_tokens_used": 12,
            "judge_tokens_used": 5,
            "signal_counts": {},
            "trial_tokens_used": 29,
            "trial_token_budget": 100,
        },
        "termination_reason": "confirmed_task_complete",
    }

    set_native_token_limit_metadata(
        context,
        runtime=SimpleNamespace(tokens_consumed=88),
        trial_token_budget=100,
    )
    assert context.metadata["termination_reason"] == "driftlock_token_limit"
    assert context.metadata["driftlock"] == {
        "status": "token_limit",
        "tokens_used": 88,
        "trial_token_budget": 100,
    }


async def test_native_runtime_rolls_back_workspace_and_ephemeral_feedback(
    tmp_path: Path,
) -> None:
    marker = "avoid the rejected no-op branch"
    call = ScriptedPhysicalCall(
        [
            _response(
                "run_shell",
                {"command": "true"},
                input_tokens=5,
                output_tokens=2,
            ),
            _response(
                "complete",
                {"summary": "changed course"},
                input_tokens=6,
                output_tokens=3,
            ),
        ]
    )
    judge = DriftOnceJudge()
    _, environment, provider, runtime = _runtime(
        tmp_path,
        call,
        heuristic_config=HeuristicConfig(
            no_change_steps=1,
            loop_window=10,
            loop_repetitions=10,
            error_window=10,
            command_failure_window=10,
            reward_stall_steps=10,
        ),
        fine_judge=judge,
        retain_checkpoints=True,
    )

    result = await runtime.run(goal="finish safely")

    assert result.status is RunStatus.COMPLETED
    assert len(result.rollbacks) == 1
    assert result.agent_tokens_used == 16
    assert result.judge_tokens_used == 7
    assert result.tokens_used == 23
    assert provider.usage.total_tokens == 16
    assert marker in call.prompts[1]
    restore_commands = [
        item["command"]
        for item in environment.calls
        if "recovery-staging" in item["command"]
    ]
    assert restore_commands
    cleanup_index = next(
        index
        for index, item in enumerate(environment.calls)
        if "rejected processes survived cleanup" in item["command"]
    )
    restore_index = next(
        index
        for index, item in enumerate(environment.calls)
        if "cp -a" in item["command"] and "recovery-staging" not in item["command"]
    )
    assert cleanup_index < restore_index
    manifests = list((tmp_path / "host-only-checkpoints").glob("**/manifest.json"))
    terminal_manifests = [
        path
        for path in manifests
        if json.loads(path.read_text(encoding="utf-8"))["label"] == "terminal"
    ]
    assert len(terminal_manifests) == 1
    terminal_state = terminal_manifests[0].with_name("state.json")
    assert marker not in terminal_state.read_text(encoding="utf-8")
    context = SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        cost_usd=None,
        metadata={},
    )
    ContextUsageRecorder(context).apply(
        context,
        agent_usage=provider.usage,
        judge_usage=ProviderUsage(5, 1, 2, 0.01),
        agent_request_times_msec=(10.0, 20.0),
        judge_request_times_msec=(5.0,),
        provider_request_count=2,
    )
    assert context.n_input_tokens == 16
    assert context.n_cache_tokens == 1
    assert context.n_output_tokens == 7
    assert context.n_input_tokens + context.n_output_tokens == result.tokens_used
    assert context.cost_usd == pytest.approx(0.01)
    assert context.metadata["driftlock_native_usage"]["judge_request_count"] == 1


async def test_native_runtime_stops_before_call_when_budget_cannot_cover_prefill(
    tmp_path: Path,
) -> None:
    call = ScriptedPhysicalCall(
        [_response("complete", {"summary": "unused"}, input_tokens=1, output_tokens=1)]
    )
    _, _, provider, runtime = _runtime(tmp_path, call, budget=1)

    result = await runtime.run(goal="impossible under this budget")

    assert result.status is RunStatus.TOKEN_LIMIT
    assert result.steps == ()
    assert result.tokens_used == 0
    assert provider.usage == ProviderUsage()
    assert call.physical_call_count == 0
    assert call.prompts == []


async def test_native_provider_refuses_hidden_second_physical_call(
    tmp_path: Path,
) -> None:
    call = ScriptedPhysicalCall(
        [_response("complete", {"summary": "unsafe"}, input_tokens=4, output_tokens=1)],
        increments=[2],
    )
    _, _, _, runtime = _runtime(tmp_path, call)

    with pytest.raises(
        PhysicalProviderBoundaryError,
        match="exactly one physical provider request; observed 2",
    ):
        await runtime.run(goal="detect retries")

    assert call.physical_call_count == 2


async def test_native_runtime_bills_known_usage_on_provider_failure_and_continues(
    tmp_path: Path,
) -> None:
    call = ScriptedPhysicalCall(
        [
            _response(
                "run_shell", {"command": "true"}, input_tokens=3, output_tokens=1
            ),
            BilledProviderFailure(
                "upstream disconnected",
                usage=ProviderUsage(input_tokens=7, output_tokens=2, cost_usd=0.09),
            ),
            _response(
                "complete", {"summary": "recovered"}, input_tokens=2, output_tokens=1
            ),
        ]
    )
    _, _, provider, runtime = _runtime(tmp_path, call)

    result = await runtime.run(goal="survive one provider failure")

    assert result.status is RunStatus.COMPLETED
    assert [step.outcome.tokens for step in result.steps] == [4, 9, 3]
    assert "upstream disconnected" in (result.steps[1].outcome.error or "")
    assert result.agent_tokens_used == 16
    assert provider.usage == ProviderUsage(12, 0, 4, 0.09)
    assert call.physical_call_count == 3


async def test_unknown_billed_provider_failure_is_explicit_and_not_counted_as_exact(
    tmp_path: Path,
) -> None:
    call = ScriptedPhysicalCall(
        [BilledProviderFailure("connection timed out", usage=None)]
    )
    _, _, provider, runtime = _runtime(tmp_path, call)

    with pytest.raises(UnknownBilledProviderUsage, match="exact usage is unavailable"):
        await runtime.run(goal="account for the failed request")

    assert provider.usage == ProviderUsage()
    assert provider.exact_usage_request_count == 0
    assert provider.unknown_billed_request_count == 1
    assert provider.provider_call_count == 1
    context = SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        cost_usd=None,
        metadata={},
    )
    apply_native_accounting(
        context,
        recorder=ContextUsageRecorder(context),
        runtime=runtime,
        provider=provider,
        agent_request_times_msec=(123.0,),
        reconcile=False,
    )
    assert context.n_input_tokens == 0
    assert context.n_output_tokens == 0
    assert context.metadata["n_episodes"] == 0
    assert context.metadata["driftlock_native_usage"] == {
        "input_tokens": 0,
        "cache_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "provider_request_count": 0,
        "physical_provider_request_count": 1,
        "unknown_billed_request_count": 1,
        "usage_complete": False,
        "judge_request_count": 0,
    }


async def test_failed_restore_keeps_native_phase_recovery_archive(
    tmp_path: Path,
) -> None:
    marker = "avoid the rejected no-op branch"
    environment = RestoreApplyFailureEnvironment(tmp_path / "remote-workspace")
    call = ScriptedPhysicalCall(
        [
            _response(
                "run_shell", {"command": "true"}, input_tokens=5, output_tokens=2
            ),
            _response(
                "complete", {"summary": "unused"}, input_tokens=6, output_tokens=3
            ),
        ]
    )
    _, _, _, runtime = _runtime(
        tmp_path,
        call,
        heuristic_config=HeuristicConfig(
            no_change_steps=1,
            loop_window=10,
            loop_repetitions=10,
            error_window=10,
            command_failure_window=10,
            reward_stall_steps=10,
        ),
        fine_judge=DriftOnceJudge(),
        environment=environment,
    )

    with pytest.raises(RemoteCheckpointError, match="recovery archive retained"):
        await runtime.run(goal=marker)

    recovery = list(
        (tmp_path / "host-only-checkpoints" / "phase-0" / "recovery").glob(
            "restore-*.tar.gz"
        )
    )
    assert len(recovery) == 1
    assert recovery[0].stat().st_size > 0


async def test_process_cleanup_failure_aborts_before_workspace_restore(
    tmp_path: Path,
) -> None:
    environment = CleanupFailureEnvironment()
    call = ScriptedPhysicalCall(
        [
            _response(
                "run_shell", {"command": "true"}, input_tokens=5, output_tokens=2
            ),
            _response(
                "complete", {"summary": "unused"}, input_tokens=6, output_tokens=3
            ),
        ]
    )
    workspace, _, _, runtime = _runtime(
        tmp_path,
        call,
        heuristic_config=HeuristicConfig(
            no_change_steps=1,
            loop_window=10,
            loop_repetitions=10,
            error_window=10,
            command_failure_window=10,
            reward_stall_steps=10,
        ),
        fine_judge=DriftOnceJudge(),
        environment=environment,
    )
    (workspace / "sentinel.txt").write_text("original", encoding="utf-8")
    sentinel_inode = (workspace / "sentinel.txt").stat().st_ino

    with pytest.raises(RuntimeError, match="quiesce rejected native-agent processes"):
        await runtime.run(goal="rollback without surviving processes")

    assert (workspace / "sentinel.txt").stat().st_ino == sentinel_inode
    assert (workspace / "sentinel.txt").read_text(encoding="utf-8") == "original"
    assert not any("cp -a" in item["command"] for item in environment.calls)


async def test_native_process_quiescer_rejects_malformed_baseline() -> None:
    class MalformedBaselineEnvironment:
        async def exec(
            self,
            command: str,
            *,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ) -> LocalResult:
            del command, timeout_sec, user
            return LocalResult(0, "not-a-process\n", "")

    quiescer = NativeProcessQuiescer(MalformedBaselineEnvironment(), user="agent-user")

    with pytest.raises(RuntimeError, match="process baseline is malformed"):
        await quiescer.prepare()


async def test_native_process_quiescer_freezes_and_kills_nonbaseline_identities() -> (
    None
):
    class ScriptedEnvironment:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def exec(
            self,
            command: str,
            *,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ) -> LocalResult:
            assert timeout_sec == 60
            assert user == "agent-user"
            self.commands.append(command)
            stdout = "2:100\n3:200\n" if len(self.commands) == 1 else ""
            return LocalResult(0, stdout, "")

    environment = ScriptedEnvironment()
    quiescer = NativeProcessQuiescer(environment, user="agent-user")

    await quiescer.prepare()
    await quiescer.before_restore("/app")

    assert len(environment.commands) == 2
    cleanup = environment.commands[1]
    assert "baseline='2:100 3:200'" in cleanup
    assert 'kill -STOP "$pid"' in cleanup
    assert 'kill -KILL "$pid"' in cleanup
    assert "rejected processes survived cleanup" in cleanup


def test_append_verifier_feedback_preserves_history_and_step_count() -> None:
    state = {
        "driftlock_tool_agent": {
            "schema_version": 1,
            "messages": [{"role": "assistant", "content": "initial summary"}],
            "steps": 4,
        }
    }

    updated = append_verifier_feedback(state, "The checksum still fails.")

    assert updated == {
        "driftlock_tool_agent": {
            "schema_version": 1,
            "messages": [
                {"role": "assistant", "content": "initial summary"},
                {
                    "role": "user",
                    "content": (
                        "The task verifier rejected the previous completion. Continue "
                        "from the current workspace and address this feedback:\n"
                        "The checksum still fails."
                    ),
                },
            ],
            "steps": 4,
        }
    }
    assert state["driftlock_tool_agent"]["messages"] == [
        {"role": "assistant", "content": "initial summary"}
    ]


def _lhtb_tree(tmp_path: Path) -> Path:
    root = tmp_path / "LHTB"
    task = root / "tasks" / "task-a"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        "[task]\nname = 'long-horizon-terminal-bench/task-a'\n",
        encoding="utf-8",
    )
    return root


def _identity_payload(agent: dict[str, Any], *, info_name: str) -> dict[str, Any]:
    resolved_agent = {
        "name": None,
        "override_setup_timeout_sec": None,
        "max_timeout_sec": None,
        **agent,
    }
    return {
        "agent_info": {"name": info_name, "version": "0.1.0"},
        "config": {
            "agent": resolved_agent,
            "environment": {
                "type": "docker",
                "import_path": None,
                "force_build": True,
                "delete": True,
                "override_cpus": None,
                "override_memory_mb": None,
                "override_storage_mb": None,
                "override_gpus": None,
                "suppress_override_warnings": False,
                "mounts": None,
                "env": {},
                "kwargs": {},
            },
            "verifier": {
                "override_timeout_sec": None,
                "max_timeout_sec": None,
                "env": {},
                "disable": False,
            },
            "timeout_multiplier": 1.0,
            "agent_timeout_multiplier": None,
            "verifier_timeout_multiplier": None,
            "agent_setup_timeout_multiplier": None,
            "environment_build_timeout_multiplier": None,
            "artifacts": [],
        },
    }


@pytest.mark.parametrize(
    ("arm", "has_judge"),
    [
        ("native-driftlock-heuristic", False),
        ("native-driftlock", True),
    ],
)
def test_native_job_generation_and_analysis_identity(
    tmp_path: Path, arm: str, has_judge: bool
) -> None:
    root = _lhtb_tree(tmp_path)
    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name=arm,
        arm=arm,
        tasks=["task-a"],
        max_total_tokens=123_456,
    )
    agent = config["agents"][0]
    assert agent["import_path"] == (
        "driftlock.harbor_native_agent:LHTBNativeDriftlockAgent"
    )
    assert ("driftlock_judge_model" in agent["kwargs"]) is has_judge
    payload = _identity_payload(agent, info_name="driftlock-native-tool-agent")

    budget, signature = _validate_arm_identity(
        payload, arm, DEFAULT_MODEL, tmp_path / "native-result.json"
    )

    assert budget == 123_456
    assert len(signature) == 64
    payload["agent_info"]["name"] = "driftlock-terminus-2"
    with pytest.raises(ValueError, match="wrong agent config"):
        _validate_arm_identity(
            payload, arm, DEFAULT_MODEL, tmp_path / "mismatched-result.json"
        )


def test_existing_terminus_arm_identity_still_accepts_generated_config(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path)
    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name="driftlock",
        arm="driftlock",
        tasks=["task-a"],
        max_total_tokens=777,
    )
    payload = _identity_payload(config["agents"][0], info_name="driftlock-terminus-2")

    budget, signature = _validate_arm_identity(
        payload, "driftlock", DEFAULT_MODEL, tmp_path / "terminus-result.json"
    )

    assert budget == 777
    assert len(signature) == 64


def test_terminus_config_is_rejected_when_labeled_as_native_arm(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path)
    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name="driftlock",
        arm="driftlock",
        tasks=["task-a"],
    )
    payload = _identity_payload(config["agents"][0], info_name="driftlock-terminus-2")

    with pytest.raises(ValueError, match="wrong agent config"):
        _validate_arm_identity(
            payload,
            "native-driftlock",
            DEFAULT_MODEL,
            tmp_path / "mislabelled-native-result.json",
        )


def test_native_arm_identity_rejects_unsupported_checkpoint_retention(
    tmp_path: Path,
) -> None:
    root = _lhtb_tree(tmp_path)
    config = build_job_config(
        lhtb_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name="native-driftlock",
        arm="native-driftlock",
        tasks=["task-a"],
    )
    config["agents"][0]["kwargs"]["driftlock_retain_checkpoints"] = True
    payload = _identity_payload(
        config["agents"][0], info_name="driftlock-native-tool-agent"
    )

    with pytest.raises(ValueError, match="invalid checkpoint retention"):
        _validate_arm_identity(
            payload,
            "native-driftlock",
            DEFAULT_MODEL,
            tmp_path / "retained-native-result.json",
        )
