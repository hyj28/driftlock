"""Process-table observations shared by native and Terminus runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProcessTableStatus(StrEnum):
    """Whether a process table was observed, absent, or unreadable."""

    OBSERVED = "observed"
    ABSENT = "absent"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class ProcessIdentitySnapshot:
    """One process-table observation and its stable PID/start-time identities."""

    status: ProcessTableStatus
    identities: tuple[str, ...] = ()
    kernel: str | None = None


# Versioning prevents an empty or truncated legacy stream from looking observed.
PROCESS_TABLE_SNAPSHOT_PREFIX = "driftlock-process-table-v1:"
# A separate tagged line keeps kernel metadata out of the identity namespace.
PROCESS_TABLE_KERNEL_PREFIX = "driftlock-process-kernel-v1:"
# Darwin has no procfs process table; unknown kernels remain fail-closed.
PROCESS_TABLE_UNAVAILABLE_KERNELS = frozenset({"Darwin"})


def valid_process_identity(value: str) -> bool:
    """Return whether *value* is a Linux PID/start-time identity."""

    pid, separator, started = value.partition(":")
    return (
        separator == ":"
        and pid.isascii()
        and pid.isdigit()
        and int(pid) > 0
        and started.isascii()
        and started.isdigit()
        and int(started) >= 0
    )


def parse_process_identity_snapshot(stdout: str | None) -> ProcessIdentitySnapshot:
    """Parse a versioned snapshot without confusing unavailable with empty."""

    lines = (stdout or "").splitlines()
    if not lines:
        return ProcessIdentitySnapshot(ProcessTableStatus.UNREADABLE)

    first = lines[0]
    if first.startswith(PROCESS_TABLE_SNAPSHOT_PREFIX):
        raw_status = first.removeprefix(PROCESS_TABLE_SNAPSHOT_PREFIX)
        try:
            status = ProcessTableStatus(raw_status)
        except ValueError as error:
            raise ValueError("process snapshot has an unknown status") from error
        remaining = lines[1:]
        kernel = None
        if remaining and remaining[0].startswith(PROCESS_TABLE_KERNEL_PREFIX):
            kernel = remaining[0].removeprefix(PROCESS_TABLE_KERNEL_PREFIX)
            if (
                not kernel
                or len(kernel) > 64
                or not kernel.isascii()
                or any(
                    not (character.isalnum() or character in "._-")
                    for character in kernel
                )
            ):
                raise ValueError("process snapshot has invalid kernel metadata")
            remaining = remaining[1:]
        identities = tuple(remaining)
    else:
        # Older test doubles and remote agents emitted identities without a header.
        # Non-empty legacy output is still observable; empty output is handled above.
        status = ProcessTableStatus.OBSERVED
        identities = tuple(lines)
        kernel = None

    if any(not valid_process_identity(value) for value in identities) or len(
        identities
    ) != len(set(identities)):
        raise ValueError("process snapshot contains malformed identities")
    if status is not ProcessTableStatus.OBSERVED and identities:
        raise ValueError("an unavailable process snapshot cannot contain identities")
    return ProcessIdentitySnapshot(status, identities, kernel)
