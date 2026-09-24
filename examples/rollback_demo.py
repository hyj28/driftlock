"""Demonstrate driftlock's rollback control flow with a scripted stand-in.

No model is called. The step function deliberately plants a short action loop so
the coarse heuristic has drift to catch; this demonstrates runner control flow,
not model behaviour.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from driftlock.checkpoints import DirectoryCheckpointStore
from driftlock.heuristics import HeuristicConfig, HeuristicJudge
from driftlock.models import (
    Checkpoint,
    DriftSignal,
    RunStatus,
    StepContext,
    StepOutcome,
    StepRecord,
)
from driftlock.runner import DriftlockRunner, RunnerConfig


class TimelineCheckpointStore(DirectoryCheckpointStore):
    """Print stable display names when public checkpoint operations occur."""

    def __init__(
        self,
        workspace: Path,
        store_dir: Path,
        *,
        ignore: tuple[str, ...] = (),
    ) -> None:
        super().__init__(workspace, store_dir, ignore=ignore)
        self.checkpoint_names: dict[str, str] = {}

    def create(
        self,
        state: Mapping[str, Any],
        *,
        step: int,
        parent_id: str | None = None,
        label: str | None = None,
    ) -> Checkpoint:
        checkpoint = super().create(
            state,
            step=step,
            parent_id=parent_id,
            label=label,
        )
        name = f"checkpoint-{len(self.checkpoint_names)}"
        self.checkpoint_names[checkpoint.checkpoint_id] = name
        print(
            f"checkpoint: {name} ({label}, logical step {step}, "
            f"phase={state.get('phase')})"
        )
        return checkpoint

    def restore(self, checkpoint: Checkpoint) -> dict[str, Any]:
        state = super().restore(checkpoint)
        name = self.checkpoint_names[checkpoint.checkpoint_id]
        print(
            f"rollback: restored {name} "
            f"({checkpoint.label}, logical step {checkpoint.step}, "
            f"phase={state.get('phase')})"
        )
        return state


class TimelineHeuristicJudge(HeuristicJudge):
    """Print the public coarse signals that initiate a review."""

    def evaluate(self, steps: list[StepRecord]) -> tuple[DriftSignal, ...]:
        signals = super().evaluate(steps)
        if signals and self.initiates_review(signals):
            descriptions = ", ".join(
                f"{signal.kind} (lookback {signal.lookback} steps)"
                for signal in signals
            )
            print(f"coarse trigger: {descriptions}")
        return signals


async def run_demo() -> None:
    print("driftlock rollback demo")
    print("scripted stand-in: no model is called; the stall is planted deliberately")
    print(
        "no fine judge: heuristics-only mode rolls every coarse trigger back unreviewed"
    )

    with TemporaryDirectory(prefix="driftlock-rollback-demo-") as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        workspace.mkdir()
        progress = workspace / "progress.txt"
        detour = workspace / "detour.txt"

        async def scripted_step(context: StepContext) -> StepOutcome:
            if context.attempt == 1 and context.logical_step <= 7:
                action = f"write healthy progress {context.logical_step}"
                progress.write_text(f"healthy {context.logical_step}", encoding="utf-8")
                state = {"phase": f"healthy-{context.logical_step}"}
                changed_paths = ("progress.txt",)
            elif context.attempt == 1:
                action = "repeat stalled detour"
                detour.write_text(
                    f"unproductive retry {context.logical_step - 2}\n",
                    encoding="utf-8",
                )
                state = {"phase": "stalled"}
                changed_paths = ("detour.txt",)
            else:
                observed_progress = (
                    progress.read_text(encoding="utf-8")
                    if progress.exists()
                    else "<missing>"
                )
                observed_detour = detour.exists()
                print(
                    f"retry observes: progress.txt={observed_progress!r}, "
                    f"detour.txt exists={observed_detour}"
                )
                restored = (
                    context.rollback_feedback is not None
                    and context.state == {"phase": "healthy-4"}
                    and observed_progress == "healthy 4"
                    and not observed_detour
                )
                if not restored:
                    raise RuntimeError(
                        "demo invariant failed: checkpoint restoration was incomplete"
                    )
                action = "finish from restored draft"
                progress.write_text(
                    "outline\nhealthy draft\nfinished after rollback\n",
                    encoding="utf-8",
                )
                state = {"phase": "finished"}
                changed_paths = ("progress.txt",)

            print(
                f"step: sequence={context.sequence} logical={context.logical_step} "
                f"attempt={context.attempt} action={action}"
            )
            return StepOutcome(
                action=action,
                state=state,
                changed_paths=changed_paths,
                completed=context.attempt > 1,
            )

        store = TimelineCheckpointStore(workspace, root / "checkpoint-store")
        judge = TimelineHeuristicJudge(HeuristicConfig())
        result = await DriftlockRunner(
            store,
            judge,
            config=RunnerConfig(
                max_steps=20,
                max_rollbacks=1,
                checkpoint_interval=4,
                checkpoint_on_exit=True,
            ),
        ).run(
            goal="finish the scripted workspace task",
            plan="make progress, stage a stall, then recover",
            step=scripted_step,
            initial_state={"phase": "new"},
        )

        if not result.rollbacks:
            raise RuntimeError("demo invariant failed: the run did not roll back")
        if result.status is not RunStatus.COMPLETED:
            raise RuntimeError(
                f"demo invariant failed: unexpected RunStatus {result.status.value}"
            )

        print(f"RunStatus: {result.status.value}")
        print(f"rollbacks: {len(result.rollbacks)}")


if __name__ == "__main__":
    asyncio.run(run_demo())
