from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from driftlock.agent import (
    MAX_EDIT_MATCH_CHARACTERS,
    MAX_EDIT_REPLACEMENT_CHARACTERS,
    AgentCompletion,
    AgentCompletionRequest,
    FileEditStatus,
    ToolCall,
    ToolCallingAgent,
)
from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.local import LocalEnvironment, LocalWorkspaceDeltaObserver
from driftlock.models import DriftContext, JudgeVerdict, RunStatus, StepContext, Verdict
from driftlock.runner import DriftlockRunner, RunnerConfig


class _ScriptedProvider:
    def __init__(self, responses: Sequence[AgentCompletion]) -> None:
        self.responses = list(responses)
        self.requests: list[AgentCompletionRequest] = []

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
        self.requests.append(request)
        return self.responses.pop(0)


class _ConcurrentEnvironment:
    """Run the real protocol while changing the target after staging begins."""

    def __init__(self, workspace: Path, target: Path, content: bytes) -> None:
        self.environment = LocalEnvironment(workspace)
        self.workspace = workspace
        self.target = target
        self.content = content
        self.writer_ran = False

    async def exec(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ):
        if ".driftlock-edit-" not in command:
            return await self.environment.exec(
                command, timeout_sec=timeout_sec, user=user
            )

        async def write_when_staged() -> None:
            while not tuple(self.workspace.glob(".driftlock-edit-*")):
                await asyncio.sleep(0)
            self.target.write_bytes(self.content)
            self.writer_ran = True

        writer = asyncio.create_task(write_when_staged())
        try:
            return await self.environment.exec(
                command, timeout_sec=timeout_sec, user=user
            )
        finally:
            await asyncio.wait_for(writer, timeout=2)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self.environment.upload_file(source_path, target_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.environment.download_file(source_path, target_path)


def _context(state: dict[str, object], *, sequence: int = 1) -> StepContext:
    return StepContext(
        goal="make one bounded edit",
        plan="edit, verify",
        state=state,
        sequence=sequence,
        logical_step=sequence,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=None,
    )


def _edit_call(
    old_text: object = "OLD",
    new_text: object = "NEW",
    *,
    path: object = "target.txt",
    call_id: str = "edit-1",
) -> ToolCall:
    return ToolCall(
        "edit_file",
        {"path": path, "old_text": old_text, "new_text": new_text},
        call_id,
    )


async def _run_edit(
    workspace: Path,
    call: ToolCall,
    *,
    max_tool_output_chars: int = 16_000,
    environment: object | None = None,
):
    provider = _ScriptedProvider([AgentCompletion(tool_calls=(call,), tokens=3)])
    agent = ToolCallingAgent(
        environment or LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        edit_file=True,
        max_tool_output_chars=max_tool_output_chars,
    )
    return await agent(_context(agent.initial_state()))


def _result(outcome) -> dict[str, object]:
    prefix = "edit_file:\n"
    assert outcome.tool_observations[0].startswith(prefix)
    return json.loads(outcome.tool_observations[0][len(prefix) :])


def test_file_edit_status_values_are_complete_and_distinct() -> None:
    assert [(item.name, item.value) for item in FileEditStatus] == [
        ("APPLIED", "applied"),
        ("REFUSED", "refused"),
        ("COULD_NOT_DETERMINE", "could_not_determine"),
    ]


@pytest.mark.parametrize(
    ("content", "status", "reason", "count", "expected"),
    [
        (b"zero\n", "refused", "match_not_found", 0, b"zero\n"),
        (
            b"before OLD after\n",
            "applied",
            "unique_match_replaced",
            1,
            b"before NEW after\n",
        ),
        (
            b"OLD one OLD two OLD\n",
            "refused",
            "match_not_unique",
            3,
            b"OLD one OLD two OLD\n",
        ),
    ],
)
async def test_zero_one_many_match_matrix(
    tmp_path: Path,
    content: bytes,
    status: str,
    reason: str,
    count: int,
    expected: bytes,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(content)

    outcome = await _run_edit(workspace, _edit_call())

    assert target.read_bytes() == expected
    assert _result(outcome)["status"] == status
    assert _result(outcome)["reason"] == reason
    assert _result(outcome)["match_count"] == count
    assert (outcome.error is None) is (status == "applied")


async def test_unique_edit_preserves_every_other_byte_and_reports_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    before_lines = [f"line {number:03d}: untouched\n" for number in range(320)]
    before_lines[173] = "line 173: OLD payload\n"
    target.write_text("".join(before_lines), encoding="utf-8")
    expected_lines = list(before_lines)
    expected_lines[173] = "line 173: NEW payload\n"

    outcome = await _run_edit(workspace, _edit_call())

    assert target.read_text(encoding="utf-8") == "".join(expected_lines)
    result = _result(outcome)
    assert result["status"] == "applied"
    assert result["reason"] == "unique_match_replaced"
    assert result["file_bytes_before"] == 6402
    assert result["file_bytes_after"] == 6402
    assert result["before_sha256"] == (
        "66c5935909d0f23408afb126bfdf12e021e0fc8d864f974a7aced4620833f8eb"
    )
    assert result["after_sha256"] == (
        "0966781f49f7edb444b81054e2b26d3165eae0ed5de304537f055968564c74c6"
    )
    assert outcome.changed_paths == ("target.txt",)


async def test_identical_replacement_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"before OLD after\n")

    outcome = await _run_edit(workspace, _edit_call(new_text="OLD"))

    assert target.read_bytes() == b"before OLD after\n"
    assert _result(outcome)["status"] == "refused"
    assert _result(outcome)["reason"] == "replacement_is_identical"
    assert _result(outcome)["match_count"] == 1


async def test_concurrent_writer_is_not_overwritten_or_mixed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD" + b"x" * 15_000)
    concurrent = b"concurrent writer owns this content\n"
    environment = _ConcurrentEnvironment(workspace, target, concurrent)

    outcome = await _run_edit(
        workspace,
        _edit_call(),
        environment=environment,
    )

    assert environment.writer_ran is True
    assert target.read_bytes() == concurrent
    assert _result(outcome)["status"] == "could_not_determine"
    assert _result(outcome)["reason"] in {
        "file_changed_before_atomic_replace",
        "file_changed_after_atomic_replace",
    }


async def test_file_beyond_read_limit_is_not_edited(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    content = b"a" * 128 + b"OLD"
    target.write_bytes(content)

    outcome = await _run_edit(workspace, _edit_call(), max_tool_output_chars=128)

    assert target.read_bytes() == content
    assert _result(outcome)["status"] == "could_not_determine"
    assert _result(outcome)["reason"] == "file_byte_limit_exceeded"
    assert outcome.tool_audits[0]["limits"]["file_bytes"] == 128


async def test_result_file_size_cap_is_not_crossed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    content = b"OLD" + b"a" * 125
    target.write_bytes(content)

    outcome = await _run_edit(
        workspace,
        _edit_call(new_text="NEWER"),
        max_tool_output_chars=128,
    )

    assert target.read_bytes() == content
    assert _result(outcome)["status"] == "refused"
    assert _result(outcome)["reason"] == "result_file_byte_limit_exceeded"


async def test_non_utf8_file_is_not_edited(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    content = b"prefix\xffOLD suffix"
    target.write_bytes(content)

    outcome = await _run_edit(workspace, _edit_call())

    assert target.read_bytes() == content
    assert _result(outcome)["status"] == "could_not_determine"
    assert _result(outcome)["reason"] == "file_is_not_strict_utf8"


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (
            _edit_call(old_text="x" * (MAX_EDIT_MATCH_CHARACTERS + 1)),
            "match_character_limit_exceeded",
        ),
        (
            _edit_call(new_text="x" * (MAX_EDIT_REPLACEMENT_CHARACTERS + 1)),
            "replacement_character_limit_exceeded",
        ),
    ],
)
async def test_text_caps_are_recorded(
    tmp_path: Path, call: ToolCall, reason: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")

    outcome = await _run_edit(workspace, call)

    assert target.read_bytes() == b"OLD"
    assert _result(outcome)["status"] == "refused"
    assert _result(outcome)["reason"] == reason
    assert outcome.tool_audits[0]["result"]["reason"] == reason


@pytest.mark.parametrize(
    ("arguments", "literal_error"),
    [
        (
            {"path": "target.txt", "old_text": "OLD"},
            "missing required argument(s): new_text",
        ),
        (
            {"path": "target.txt", "old_text": 4, "new_text": "NEW"},
            "old_text must be a non-empty string",
        ),
        (
            {"path": "target.txt", "old_text": "", "new_text": "NEW"},
            "old_text must be a non-empty string",
        ),
    ],
)
async def test_malformed_edit_is_a_recorded_tool_error(
    tmp_path: Path, arguments: dict[str, object], literal_error: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD")

    outcome = await _run_edit(workspace, ToolCall("edit_file", arguments, "malformed"))

    assert target.read_bytes() == b"OLD"
    assert literal_error in (outcome.error or "")
    assert outcome.tool_observations == (
        f"edit_file:\nERROR: malformed arguments for edit_file: {literal_error}",
    )


async def test_runner_rollback_restores_content_from_before_edit(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_bytes(b"OLD\n")
    provider = _ScriptedProvider(
        [
            AgentCompletion(tool_calls=(_edit_call(call_id="one"),), tokens=2),
            AgentCompletion(tool_calls=(_edit_call(call_id="two"),), tokens=2),
            AgentCompletion(
                tool_calls=(
                    ToolCall("complete", {"summary": "rolled back safely"}, "done"),
                ),
                tokens=2,
            ),
        ]
    )
    agent = ToolCallingAgent(
        LocalEnvironment(workspace),
        LocalWorkspaceDeltaObserver(workspace),
        provider,
        edit_file=True,
    )

    class _DriftJudge:
        async def judge(self, _context: DriftContext) -> JudgeVerdict:
            return JudgeVerdict(Verdict.DRIFTED, "repeat edit should roll back")

    result = await DriftlockRunner(
        DirectoryCheckpointStore(workspace, tmp_path / "snapshots"),
        HeuristicJudge(
            HeuristicConfig(
                no_change_steps=10,
                loop_window=2,
                loop_repetitions=2,
                error_window=10,
                command_failure_window=10,
                reward_stall_steps=10,
            )
        ),
        fine_judge=_DriftJudge(),
        config=RunnerConfig(max_steps=3, max_rollbacks=1, checkpoint_interval=10),
    ).run(
        goal="edit and exercise rollback",
        step=agent,
        initial_state=agent.initial_state(),
    )

    assert result.status is RunStatus.COMPLETED
    assert result.steps[0].outcome.changed_paths == ("target.txt",)
    assert len(result.rollbacks) == 1
    assert target.read_bytes() == b"OLD\n"
