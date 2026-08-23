"""Zero-token trajectory drift heuristics."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from driftlock.models import DriftSignal, StepRecord

SIGNAL_KINDS = frozenset(
    {
        "no_file_change",
        "action_loop",
        "error_spike",
        "sustained_command_failure",
        "reward_stall",
    }
)
"""Every signal kind :meth:`HeuristicJudge.evaluate` can emit."""


@dataclass(frozen=True, slots=True)
class HeuristicConfig:
    """Thresholds for the coarse drift detector."""

    no_change_steps: int = 4
    loop_window: int = 6
    loop_repetitions: int = 3
    error_window: int = 5
    error_rate: float = 0.6
    command_failure_window: int = 8
    command_failure_rate: float = 1.0
    reward_stall_steps: int = 5
    reward_epsilon: float = 1e-6
    corroborating_signals: frozenset[str] = field(
        default_factory=lambda: frozenset({"no_file_change"})
    )

    def __post_init__(self) -> None:
        integer_fields = (
            self.no_change_steps,
            self.loop_window,
            self.loop_repetitions,
            self.error_window,
            self.command_failure_window,
            self.reward_stall_steps,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("heuristic window sizes must be positive")
        if self.loop_repetitions > self.loop_window:
            raise ValueError("loop_repetitions cannot exceed loop_window")
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")
        if not 0.0 <= self.command_failure_rate <= 1.0:
            raise ValueError("command_failure_rate must be between 0 and 1")
        if self.reward_epsilon < 0:
            raise ValueError("reward_epsilon cannot be negative")
        unknown = frozenset(self.corroborating_signals) - SIGNAL_KINDS
        if unknown:
            raise ValueError(
                f"unknown corroborating signal kinds: {', '.join(sorted(unknown))}"
            )
        if frozenset(self.corroborating_signals) == SIGNAL_KINDS:
            raise ValueError(
                "at least one signal kind must be able to initiate a fine review"
            )
        object.__setattr__(
            self, "corroborating_signals", frozenset(self.corroborating_signals)
        )


class HeuristicJudge:
    """Detect suspicious stalls, loops, error spikes, and reward plateaus."""

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()

    @property
    def history_window(self) -> int:
        """Number of observations needed before every detector is meaningful."""

        return max(
            self.config.no_change_steps,
            self.config.loop_window,
            self.config.error_window,
            self.config.command_failure_window,
            self.config.reward_stall_steps,
        )

    def initiates_review(self, signals: tuple[DriftSignal, ...]) -> bool:
        """Whether this signal set is strong enough to spend a fine-judge call.

        A signal kind listed in :attr:`HeuristicConfig.corroborating_signals` is
        evidence, not cause: it is passed to the fine judge when something else
        fires, but on its own it never opens a review. ``no_file_change`` is
        corroborating by default because an agent that is reading rather than
        writing is exploring, not drifting -- the 2026-08-23 diagnostic run raised
        it alone 109 times and the fine judge rejected all 109.
        """

        return any(
            signal.kind not in self.config.corroborating_signals for signal in signals
        )

    def evaluate(self, steps: list[StepRecord]) -> tuple[DriftSignal, ...]:
        if not steps:
            return ()
        signals: list[DriftSignal] = []
        config = self.config

        no_change = steps[-config.no_change_steps :]
        if len(no_change) == config.no_change_steps and all(
            step.outcome.workspace_delta_observed and not step.outcome.changed_paths
            for step in no_change
        ):
            signals.append(
                DriftSignal(
                    "no_file_change",
                    f"no files changed in the last {config.no_change_steps} steps",
                    lookback=config.no_change_steps,
                )
            )

        loop_steps = steps[-config.loop_window :]
        fingerprints = [_action_fingerprint(step.outcome.action) for step in loop_steps]
        repeated = Counter(fingerprints).most_common(1)
        if (
            len(loop_steps) == config.loop_window
            and repeated
            and repeated[0][0]
            and repeated[0][1] >= config.loop_repetitions
        ):
            signals.append(
                DriftSignal(
                    "action_loop",
                    f"the same action appeared {repeated[0][1]} times in the last "
                    f"{config.loop_window} steps",
                    lookback=config.loop_window,
                )
            )

        error_steps = steps[-config.error_window :]
        if len(error_steps) == config.error_window:
            failures = sum(step.outcome.error is not None for step in error_steps)
            rate = failures / config.error_window
            if rate >= config.error_rate:
                signals.append(
                    DriftSignal(
                        "error_spike",
                        f"error rate is {rate:.0%} over the last "
                        f"{config.error_window} steps",
                        lookback=config.error_window,
                    )
                )

        command_steps = steps[-config.command_failure_window :]
        if len(command_steps) == config.command_failure_window:
            all_failed = sum(
                step.outcome.commands_run > 0
                and step.outcome.commands_failed == step.outcome.commands_run
                for step in command_steps
            )
            rate = all_failed / config.command_failure_window
            if rate >= config.command_failure_rate:
                signals.append(
                    DriftSignal(
                        "sustained_command_failure",
                        f"all commands failed in {all_failed} of the last "
                        f"{config.command_failure_window} steps",
                        lookback=config.command_failure_window,
                    )
                )

        reward_steps = [
            step.outcome.reward
            for step in steps[-config.reward_stall_steps :]
            if step.outcome.reward is not None
        ]
        if len(reward_steps) == config.reward_stall_steps and (
            max(reward_steps) - min(reward_steps) <= config.reward_epsilon
        ):
            signals.append(
                DriftSignal(
                    "reward_stall",
                    "reward did not improve in the last "
                    f"{config.reward_stall_steps} steps",
                    lookback=config.reward_stall_steps,
                )
            )
        return tuple(signals)


def _action_fingerprint(action: str) -> str:
    normalized = re.sub(r"\s+", " ", action.strip().lower())
    return normalized[:500]
