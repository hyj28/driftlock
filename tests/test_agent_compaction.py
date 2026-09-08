from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from driftlock.agent import (
    MIN_MAX_HISTORY_CHARACTERS,
    AgentCompletion,
    AgentCompletionRequest,
    AgentConversationCodec,
    AgentStateError,
    ConversationCompactionStatus,
    ToolCall,
    ToolCallingAgent,
    compact_conversation_history,
    conversation_history_characters,
)
from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.lhtb import WorkspaceDelta, WorkspaceSnapshot
from driftlock.models import RunStatus, StepContext
from driftlock.runner import DriftlockRunner, RunnerConfig


class ScriptedProvider:
    def __init__(self, responses: Sequence[AgentCompletion]) -> None:
        self.responses = list(responses)
        self.requests: list[AgentCompletionRequest] = []

    async def __call__(self, request: AgentCompletionRequest) -> AgentCompletion:
        self.requests.append(request)
        return self.responses.pop(0)


@dataclass(frozen=True, slots=True)
class EmptyObserver:
    async def canonical_workspace(self) -> str:
        return "/remote/workspace"

    async def snapshot(self) -> WorkspaceSnapshot:
        return WorkspaceSnapshot(files={})

    def compare(
        self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> WorkspaceDelta:
        return WorkspaceDelta()


class UnusedEnvironment:
    async def exec(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("no environment tool should run")


def _context(
    state: Mapping[str, Any], *, sequence: int = 1, logical_step: int = 1
) -> StepContext:
    return StepContext(
        goal="repair the parser",
        plan="inspect, patch, verify",
        state=state,
        sequence=sequence,
        logical_step=logical_step,
        attempt=1,
        rollback_feedback=None,
        tokens_remaining=None,
    )


def _paired_history(
    count: int, *, content_characters: int = 300
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for number in range(count):
        call_id = f"read-{number}"
        history.extend(
            (
                {
                    "role": "assistant",
                    "content": f"inspect file {number}",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "name": "read_file",
                            "arguments": {"path": f"file-{number}.txt"},
                        }
                    ],
                    "truncated": False,
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "read_file",
                    "content": chr(97 + number % 26) * content_characters,
                    "is_error": False,
                },
            )
        )
    return history


def _assert_no_orphaned_tools(messages: Sequence[Mapping[str, Any]]) -> None:
    declared: Counter[str] = Counter()
    answered: Counter[str] = Counter()
    for message in messages:
        for call in message.get("tool_calls", []):
            declared[call["id"]] += 1
        if message.get("role") == "tool":
            answered[message["tool_call_id"]] += 1
    assert declared == answered


def _complete(summary: str = "done") -> AgentCompletion:
    return AgentCompletion(
        tool_calls=(ToolCall("complete", {"summary": summary}, "complete-1"),),
        tokens=7,
    )


def _agent(provider: ScriptedProvider, *, bound: int = 900) -> ToolCallingAgent:
    return ToolCallingAgent(
        UnusedEnvironment(),
        EmptyObserver(),
        provider,
        max_history_characters=bound,
    )


def test_compaction_under_bound_returns_an_equal_defensive_copy() -> None:
    history = [
        {"role": "assistant", "content": "inspected parser.py", "tool_calls": []},
        {"role": "user", "content": "continue with the failing branch"},
    ]
    before = json.dumps(history, separators=(",", ":"), ensure_ascii=False)

    result = compact_conversation_history(history, max_characters=900, step=3)

    assert result.status is ConversationCompactionStatus.UNCHANGED
    assert result.messages == history
    assert result.messages is not history
    assert result.messages[0] is not history[0]
    assert (
        json.dumps(result.messages, separators=(",", ":"), ensure_ascii=False) == before
    )
    assert result.audit is None


def test_compaction_keeps_tool_calls_and_results_as_indivisible_units() -> None:
    history = _paired_history(5)

    result = compact_conversation_history(history, max_characters=900, step=6)

    assert result.status is ConversationCompactionStatus.COMPACTED
    assert len(result.messages) == 3
    assert [message["role"] for message in result.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert result.messages[1]["tool_calls"][0]["id"] == "read-4"
    assert result.messages[2]["tool_call_id"] == "read-4"
    _assert_no_orphaned_tools(result.messages)
    assert conversation_history_characters(result.messages) < 900
    assert result.audit is not None
    assert result.audit.step == 6
    assert result.audit.affected_message_count == 8
    assert result.audit.retained_message_count == 2
    assert result.audit.summary.startswith("Conversation context was compacted locally")


async def test_tool_agent_common_path_is_byte_identical_and_provider_free(
    tmp_path: Path,
) -> None:
    history = [
        {"role": "assistant", "content": "established invariant", "tool_calls": []},
        {"role": "user", "content": "keep going"},
    ]
    provider = ScriptedProvider([_complete()])
    agent = _agent(provider)
    state = AgentConversationCodec().encode(history, steps=2)

    outcome = await agent(_context(state, sequence=3, logical_step=3))

    assert len(provider.requests) == 1
    sent_history = provider.requests[0].messages[2:]
    assert json.dumps(sent_history, separators=(",", ":")) == json.dumps(
        history, separators=(",", ":")
    )
    assert outcome.context_compactions == ()
    assert outcome.tokens == 7
    assert tmp_path.exists()


async def test_tool_agent_compacts_before_request_and_records_the_step_audit() -> None:
    history = _paired_history(8)
    codec = AgentConversationCodec()
    provider = ScriptedProvider([_complete("continued after compaction")])
    agent = _agent(provider)

    outcome = await agent(
        _context(codec.encode(history, steps=8), sequence=9, logical_step=9)
    )

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.messages[0]["role"] == "system"
    assert "Goal:\nrepair the parser" in request.messages[1]["content"]
    assert "Plan:\ninspect, patch, verify" in request.messages[1]["content"]
    sent_history = request.messages[2:]
    assert conversation_history_characters(sent_history) < 900
    assert "driftlock_context_compactions" not in json.dumps(request.messages)
    _assert_no_orphaned_tools(sent_history)
    assert len(outcome.context_compactions) == 1
    assert outcome.context_compactions[0]["status"] == "compacted"
    assert outcome.context_compactions[0]["step"] == 9
    assert outcome.context_compactions[0]["before_characters"] > 900
    assert outcome.context_compactions[0]["after_characters"] < 900
    assert outcome.context_compactions[0]["affected_message_count"] == 14


async def test_compacted_codec_state_round_trips_and_resumes_equivalently() -> None:
    codec = AgentConversationCodec()
    compacted = compact_conversation_history(
        _paired_history(8), max_characters=900, step=9
    )
    encoded = codec.encode(compacted.messages, steps=8)
    serialized = json.loads(json.dumps(encoded))

    messages, steps = codec.decode(serialized)

    assert messages == compacted.messages
    assert steps == 8
    first_provider = ScriptedProvider([_complete("first")])
    second_provider = ScriptedProvider([_complete("second")])
    first = await _agent(first_provider)(_context(encoded, sequence=9, logical_step=9))
    second = await _agent(second_provider)(
        _context(serialized, sequence=9, logical_step=9)
    )
    assert first_provider.requests[0] == second_provider.requests[0]
    assert AgentConversationCodec().decode(first.state)[1] == 9
    assert AgentConversationCodec().decode(second.state)[1] == 9


async def test_checkpoints_before_and_after_compaction_restore_correct_histories(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DirectoryCheckpointStore(workspace, tmp_path / "checkpoints")
    codec = AgentConversationCodec()
    full_history = _paired_history(8)
    before_state = codec.encode(full_history, steps=8)
    before_checkpoint = store.create(before_state, step=8, label="before")
    first_provider = ScriptedProvider([_complete("first compacted step")])
    compacted_outcome = await _agent(first_provider)(
        _context(before_state, sequence=9, logical_step=9)
    )
    after_checkpoint = store.create(compacted_outcome.state, step=9, label="after")

    restored_after = store.restore(after_checkpoint)
    after_provider = ScriptedProvider([_complete("resumed after checkpoint")])
    after_outcome = await _agent(after_provider)(
        _context(restored_after, sequence=10, logical_step=10)
    )
    assert len(after_provider.requests) == 1
    assert after_outcome.completed is True
    _assert_no_orphaned_tools(after_provider.requests[0].messages[2:])

    restored_before = store.restore(before_checkpoint)
    restored_messages, restored_steps = codec.decode(restored_before)
    assert restored_messages == full_history
    assert restored_steps == 8
    before_provider = ScriptedProvider([_complete("compacted again")])
    retried = await _agent(before_provider)(
        _context(restored_before, sequence=10, logical_step=9)
    )
    assert len(retried.context_compactions) == 1
    assert retried.context_compactions[0]["step"] == 9
    assert retried.context_compactions[0]["affected_message_count"] == 14


async def test_runner_rollback_restores_precompaction_state_and_compacts_again(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DirectoryCheckpointStore(workspace, tmp_path / "checkpoints")
    provider = ScriptedProvider(
        [
            AgentCompletion(text="continue one", tokens=1),
            AgentCompletion(text="continue two", tokens=1),
            AgentCompletion(text="different retry", tokens=1),
        ]
    )
    agent = _agent(provider)
    initial_state = AgentConversationCodec().encode(_paired_history(8), steps=8)
    judge = HeuristicJudge(
        HeuristicConfig(
            no_change_steps=2,
            loop_window=5,
            loop_repetitions=5,
            error_window=5,
            reward_stall_steps=5,
            corroborating_signals=frozenset(),
        )
    )

    result = await DriftlockRunner(
        store,
        judge,
        config=RunnerConfig(
            max_steps=3,
            max_rollbacks=1,
            checkpoint_interval=1,
        ),
    ).run(
        goal="repair the parser",
        plan="inspect, patch, verify",
        step=agent,
        initial_state=initial_state,
    )

    assert result.status is RunStatus.STEP_LIMIT
    assert len(result.rollbacks) == 1
    assert [record.logical_step for record in result.steps] == [1, 2, 1]
    assert result.steps[0].outcome.context_compactions[0]["step"] == 9
    assert result.steps[2].outcome.context_compactions[0]["step"] == 9
    assert len(provider.requests) == 3
    assert conversation_history_characters(provider.requests[2].messages[2:-1]) < 900
    _assert_no_orphaned_tools(provider.requests[2].messages[2:-1])


def test_recompacting_oversized_compacted_history_terminates_nonempty() -> None:
    first = compact_conversation_history(
        _paired_history(20, content_characters=600),
        max_characters=1_800,
        step=21,
    )
    oversized = [
        *first.messages,
        {"role": "user", "content": "new detail " + "z" * 2_000},
    ]

    second = compact_conversation_history(
        oversized,
        max_characters=700,
        step=22,
    )

    assert second.status is ConversationCompactionStatus.SUMMARY_ONLY
    assert second.messages
    assert conversation_history_characters(second.messages) < 700
    assert second.audit is not None
    ledger = second.messages[0]["driftlock_context_compactions"]
    assert ledger["total_event_count"] == 2
    assert [audit["step"] for audit in ledger["recent_events"]] == [21, 22]
    assert [audit["status"] for audit in ledger["recent_events"]] == [
        "compacted",
        "summary_only",
    ]


def test_recent_retention_degrades_predictably_without_a_unit_count_cliff() -> None:
    retained_by_unit_count = [
        compact_conversation_history(
            _paired_history(unit_count),
            max_characters=900,
            step=unit_count,
        ).audit.retained_message_count
        for unit_count in range(9, 14)
    ]
    retained_by_bound = [
        compact_conversation_history(
            _paired_history(20),
            max_characters=bound,
            step=20,
        )
        for bound in (1_800, 1_500, 1_200, 1_000, 900, 700, 512)
    ]

    assert retained_by_unit_count == [2, 2, 2, 2, 2]
    assert [result.audit.retained_message_count for result in retained_by_bound] == [
        6,
        4,
        4,
        2,
        2,
        2,
        0,
    ]
    assert [result.status.value for result in retained_by_bound] == [
        "compacted",
        "compacted",
        "compacted",
        "compacted",
        "compacted",
        "compacted",
        "summary_only",
    ]


def test_durable_compaction_audit_is_a_bounded_ledger() -> None:
    messages: Sequence[Mapping[str, Any]] = _paired_history(5)
    for step in range(1, 201):
        messages = [
            *messages,
            {"role": "user", "content": "z" * 1_000},
        ]
        messages = compact_conversation_history(
            messages,
            max_characters=900,
            step=step,
        ).messages

    ledger = messages[0]["driftlock_context_compactions"]
    raw_state = json.dumps(messages, separators=(",", ":"), ensure_ascii=False)
    unchanged = compact_conversation_history(
        messages,
        max_characters=900,
        step=201,
    )

    assert ledger["schema_version"] == 2
    assert ledger["retention_policy"] == {
        "recent_event_limit": 16,
        "older_events": "aggregate_counts_and_extrema",
        "full_summary": "current_summary_message_only",
    }
    assert ledger["total_event_count"] == 200
    assert ledger["aggregated_event_count"] == 184
    assert len(ledger["recent_events"]) == 16
    assert ledger["aggregated"]["first_recorded_step"] == 1
    assert ledger["aggregated"]["last_recorded_step"] == 184
    assert ledger["aggregated"]["total_affected_message_count"] == 377
    assert len(raw_state) < 4_000
    assert unchanged.status is ConversationCompactionStatus.UNCHANGED
    assert unchanged.messages == messages


def test_under_bound_legacy_unbounded_audits_migrate_to_the_bounded_ledger() -> None:
    legacy_audits = [
        {
            "schema_version": 1,
            "status": "compacted",
            "step": step,
            "before_characters": 1_000,
            "after_characters": 100,
            "affected_message_count": 2,
            "retained_message_count": 2,
            "summary": "s" * 300,
        }
        for step in range(1, 201)
    ]
    history = [
        {
            "role": "user",
            "content": "Conversation context was compacted locally.",
            "driftlock_context_compactions": legacy_audits,
        }
    ]
    assert len(json.dumps(history, separators=(",", ":"))) > 80_000

    result = compact_conversation_history(
        history,
        max_characters=900,
        step=201,
    )

    ledger = result.messages[0]["driftlock_context_compactions"]
    assert result.status is ConversationCompactionStatus.UNCHANGED
    assert ledger["total_event_count"] == 200
    assert ledger["aggregated_event_count"] == 184
    assert len(ledger["recent_events"]) == 16
    assert len(json.dumps(result.messages, separators=(",", ":"))) < 4_000


@pytest.mark.parametrize("padding", ["", "x" * 1_000])
def test_orphaned_tool_result_is_rejected_consistently_at_every_size(
    padding: str,
) -> None:
    history = [
        {
            "role": "tool",
            "tool_call_id": "missing",
            "name": "read_file",
            "content": padding,
        }
    ]

    with pytest.raises(AgentStateError, match="orphaned tool result"):
        compact_conversation_history(history, max_characters=900, step=1)


@pytest.mark.parametrize("padding", ["", "x" * 1_000])
async def test_agent_records_malformed_pairing_without_calling_provider(
    padding: str,
) -> None:
    history = [
        {
            "role": "tool",
            "tool_call_id": "missing",
            "name": "read_file",
            "content": padding,
        }
    ]
    provider = ScriptedProvider([])
    agent = _agent(provider)

    outcome = await agent(_context(AgentConversationCodec().encode(history, steps=3)))

    assert provider.requests == []
    assert outcome.action == "Reject malformed conversation state"
    assert outcome.error == (
        "Malformed conversation state: tool-agent message 0 is an orphaned tool result"
    )
    assert outcome.tokens == 0
    assert AgentConversationCodec().decode(outcome.state)[1] == 4


@pytest.mark.parametrize(
    "content",
    [b"bytes", {"set-value"}],
)
def test_non_json_content_has_one_public_agent_state_error(content: object) -> None:
    history = [{"role": "user", "content": content}]

    with pytest.raises(
        AgentStateError, match="tool-agent messages must be JSON-compatible objects"
    ):
        conversation_history_characters(history)
    with pytest.raises(
        AgentStateError, match="tool-agent messages must be JSON-compatible objects"
    ):
        compact_conversation_history(history, max_characters=900, step=1)


def test_circular_content_has_one_public_agent_state_error() -> None:
    message: dict[str, Any] = {"role": "user"}
    message["content"] = message
    history = [message]

    with pytest.raises(
        AgentStateError, match="tool-agent messages must be JSON-compatible objects"
    ):
        conversation_history_characters(history)
    with pytest.raises(
        AgentStateError, match="tool-agent messages must be JSON-compatible objects"
    ):
        compact_conversation_history(history, max_characters=900, step=1)


def test_minimum_history_bound_is_public_and_documented() -> None:
    assert MIN_MAX_HISTORY_CHARACTERS == 512
    assert "MIN_MAX_HISTORY_CHARACTERS" in compact_conversation_history.__doc__
    assert "MIN_MAX_HISTORY_CHARACTERS" in ToolCallingAgent.__doc__

    with pytest.raises(
        ValueError, match="max_characters must be an integer of at least 512"
    ):
        compact_conversation_history([], max_characters=511, step=0)


async def test_far_over_bound_keeps_non_history_prompt_turns_for_next_request() -> None:
    provider = ScriptedProvider([_complete()])
    agent = _agent(provider, bound=700)
    history = _paired_history(20, content_characters=300)
    context = _context(AgentConversationCodec().encode(history, steps=20))

    outcome = await agent(context)
    request_messages = provider.requests[0].messages

    assert request_messages[0]["role"] == "system"
    assert request_messages[1] == {
        "role": "user",
        "content": ("Goal:\nrepair the parser\n\nPlan:\ninspect, patch, verify"),
    }
    assert conversation_history_characters(request_messages[2:]) < 700
    assert outcome.completed is True
