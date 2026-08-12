"""No-network smoke test against the patched, pinned Harbor implementation.

Run explicitly after installing the LHTB Harbor checkout; this file is outside the
default driftlock test path because Harbor is intentionally not a package dependency.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.models.agent.context import AgentContext

from driftlock import LHTBTerminusRuntime, WorkspaceDelta, WorkspaceSnapshot


class Session:
    _session_name = "terminus-2"
    _user = "root"

    async def get_incremental_output(self) -> str:
        return "Current Terminal Screen:\nroot@container:/app#"

    async def is_session_alive(self) -> bool:
        return True

    async def send_keys(self, keys: Any, **kwargs: Any) -> None:
        return None


class Observer:
    async def snapshot(self) -> WorkspaceSnapshot:
        return WorkspaceSnapshot({})

    def compare(
        self, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> WorkspaceDelta:
        return WorkspaceDelta()


class Environment:
    async def exec(self, command: str, **kwargs: Any) -> Any:
        return SimpleNamespace(stdout="2:100\n", stderr="", return_code=0)


@pytest.mark.asyncio
async def test_real_pinned_terminus_loop_yields_after_one_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Terminus2(
        logs_dir=tmp_path,
        model_name="fake-provider/fake-model",
        enable_summarize=False,
        record_terminal_session=False,
    )
    agent._session = Session()
    agent._dump_trajectory = lambda: None
    calls = 0

    async def fake_completion(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        assert kwargs["max_tokens"] == 100
        assert kwargs["num_retries"] == 0

        class Completion(dict):
            pass

        response = Completion(
            {
                "model": "fake-model",
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"analysis":"inspect","plan":"list files",'
                                '"commands":[],"task_complete":false}'
                            ),
                            "reasoning_content": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        response.usage = SimpleNamespace(
            prompt_tokens=23,
            completion_tokens=9,
        )
        return response

    monkeypatch.setattr(litellm, "acompletion", fake_completion)
    context = AgentContext()
    runtime = LHTBTerminusRuntime(
        agent,
        Environment(),
        context,
        remote_workspace="/app",
        observer=Observer(),
    )

    prompt = await runtime.prepare_start(
        "inspect the workspace", plan="list files", rollback_feedback=None
    )
    boundary = await runtime.start(prompt=prompt, tokens_remaining=100)

    assert calls == 1
    assert runtime.provider_call_count == 1
    assert boundary.tokens == 32
    assert boundary.conversation.episode == 1
    assert boundary.conversation.messages[-2]["content"] == prompt
    assert len(agent._trajectory_steps) == 2
