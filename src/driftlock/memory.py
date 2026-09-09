"""Bounded, auditable persistence for unvalidated agent memories.

Memory deliberately has no admission gate.  A current entry is therefore a hint
about where to inspect, never evidence that outranks a fresh workspace observation.
Corrections and revocations append immutable events so the bad claim remains
attributable, while retrieval exposes only the latest active event.  Rejected
operations live in the caller's bounded audit result rather than an on-disk ledger:
otherwise attempts made after the store-size cap would recreate unbounded durable
state or require silently dropping older audit evidence.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

# Two hundred fifty-six facts cover recurring repository conventions without
# letting an agent turn memory into an unbounded shadow knowledge base.
DEFAULT_MAX_MEMORY_ENTRIES = 256

# Two thousand characters fit a specific operational fact plus its caveats while
# preventing transcripts, source files, and command logs from becoming memories.
DEFAULT_MAX_MEMORY_CONTENT_CHARACTERS = 2_000

# Two MiB is the ordinary budget available to new claims; remediation may use only
# the separately bounded reserve below, keeping the absolute durable size finite.
DEFAULT_MAX_MEMORY_STORE_BYTES = 2 * 1024 * 1024

# Sixty-four KiB is withheld from new records so retracting or correcting claims
# remains possible when the ordinary store budget is full. The sum of the normal
# budget and this remediation reserve is the absolute durable-size ceiling.
DEFAULT_MAX_MEMORY_REMEDIATION_BYTES = 64 * 1024

# Five hundred characters permit a useful correction or revocation explanation
# without allowing audit reasons to become an unbounded secondary content field.
DEFAULT_MAX_MEMORY_REASON_CHARACTERS = 500

# One hundred twenty-eight characters fit stable task and run identifiers while
# bounding provenance independently of memory content and total serialized size.
DEFAULT_MAX_MEMORY_PROVENANCE_ID_CHARACTERS = 128

# Runner counters are small in practice; this explicit ceiling also bounds their
# decimal JSON representation inside revocation-reserve calculations.
DEFAULT_MAX_MEMORY_PROVENANCE_COUNTER = 999_999_999

# A task cannot legitimately revise one claim this many times. The independent
# ceiling lets checkpoint and disk decoders reject adversarial event arrays before
# materializing them.
DEFAULT_MAX_MEMORY_EVENTS_PER_ENTRY = 1_024

# Five hundred characters retain actionable tool failures while explicit length
# and digest metadata represents larger errors without silently truncating them.
DEFAULT_MAX_MEMORY_AUDIT_ERROR_CHARACTERS = 500

# Thirty-two recent recovery examples make crash/corruption handling inspectable;
# aggregate counts retain the total without an unbounded in-memory event list.
DEFAULT_MAX_MEMORY_RECOVERY_EXAMPLES = 32

# Version one is explicit because strict disk decoding must reject shapes whose
# semantics this implementation cannot prove rather than guessing at migrations.
MEMORY_SCHEMA_VERSION = 1

# Generated identifiers never exceed this shape; the expression also prevents
# filenames from escaping the entries directory when records are read by id.
_MEMORY_ID = re.compile(r"memory-[0-9]{6}")

# These signatures intentionally prefer false positives over persisting a secret.
# Memories are hints, so rejecting one is cheaper than making a credential durable.
_CREDENTIAL_PATTERNS = (
    re.compile(r"-----BEGIN [^-\r\n]*PRIVATE KEY(?: BLOCK)?-----", re.IGNORECASE),
    re.compile(
        r"(?:^|[^A-Za-z0-9])(?:[A-Za-z0-9]+[_-])*"
        r"(?:api[_-]?key|access[_-]?key|client[_-]?secret|secret|token|"
        r"password|passwd|credential)s?\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"),
)


class MemoryEntryStatus(StrEnum):
    """Whether a logical memory may enter a retrieval snapshot."""

    ACTIVE = "active"
    REVOKED = "revoked"


class MemoryOperation(StrEnum):
    """Mutations accepted by the memory store and agent tool."""

    RECORD = "record"
    CORRECT = "correct"
    REVOKE = "revoke"


class MemoryMutationStatus(StrEnum):
    """Whether a requested mutation changed durable state."""

    APPLIED = "applied"
    REJECTED = "rejected"


class MemoryRecoveryKind(StrEnum):
    """Recoverable anomalies observed while scanning the durable store."""

    STALE_TEMPORARY_FILE = "stale_temporary_file"
    MALFORMED_ENTRY = "malformed_entry"


class MemoryStoreError(ValueError):
    """Base class for a refused memory operation or unreadable store."""


class MemoryStoreFormatError(MemoryStoreError):
    """Durable memory data does not have the exact supported shape."""


@dataclass(frozen=True, slots=True)
class MemoryStoreConfig:
    """Independent bounds for logical entries, content, and durable bytes."""

    max_entries: int = DEFAULT_MAX_MEMORY_ENTRIES
    max_content_characters: int = DEFAULT_MAX_MEMORY_CONTENT_CHARACTERS
    max_store_bytes: int = DEFAULT_MAX_MEMORY_STORE_BYTES
    max_remediation_bytes: int = DEFAULT_MAX_MEMORY_REMEDIATION_BYTES
    max_reason_characters: int = DEFAULT_MAX_MEMORY_REASON_CHARACTERS
    max_provenance_id_characters: int = DEFAULT_MAX_MEMORY_PROVENANCE_ID_CHARACTERS
    max_events_per_entry: int = DEFAULT_MAX_MEMORY_EVENTS_PER_ENTRY

    def __post_init__(self) -> None:
        for name in (
            "max_entries",
            "max_content_characters",
            "max_store_bytes",
            "max_remediation_bytes",
            "max_reason_characters",
            "max_provenance_id_characters",
            "max_events_per_entry",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise MemoryStoreError(f"{name} must be a positive integer")
        required = _required_revocation_reserve(self)
        if self.max_remediation_bytes < required:
            raise MemoryStoreError(
                "max_remediation_bytes must reserve at least "
                f"{required} bytes for one worst-case terminal revocation"
            )


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    """Host-supplied identity of the task run and step making a claim."""

    task_id: str
    run_id: str
    sequence: int
    logical_step: int
    attempt: int

    def __post_init__(self) -> None:
        for name in ("task_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise MemoryStoreError(
                    f"memory provenance {name} must be non-empty text"
                )
            if "\n" in value or "\r" in value or "\x00" in value:
                raise MemoryStoreError(f"memory provenance {name} must be one line")
        for name in ("sequence", "logical_step", "attempt"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise MemoryStoreError(
                    f"memory provenance {name} must be a positive integer"
                )
            if value > DEFAULT_MAX_MEMORY_PROVENANCE_COUNTER:
                raise MemoryStoreError(
                    f"memory provenance {name} exceeds the "
                    f"{DEFAULT_MAX_MEMORY_PROVENANCE_COUNTER} limit"
                )

    def validate_for(self, config: MemoryStoreConfig) -> None:
        for name in ("task_id", "run_id"):
            value = getattr(self, name)
            if len(value) > config.max_provenance_id_characters:
                raise MemoryStoreError(
                    f"memory provenance {name} exceeds the "
                    f"{config.max_provenance_id_characters}-character limit"
                )
            if _looks_like_credential(value):
                raise MemoryStoreError(
                    f"memory provenance {name} looks like a credential"
                )
            if not _is_valid_utf8(value):
                raise MemoryStoreError(
                    f"memory provenance {name} must be valid UTF-8 text"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "logical_step": self.logical_step,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, value: object) -> MemoryProvenance:
        expected = {"task_id", "run_id", "sequence", "logical_step", "attempt"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise MemoryStoreFormatError("memory provenance fields are malformed")
        try:
            return cls(
                task_id=value.get("task_id"),
                run_id=value.get("run_id"),
                sequence=value.get("sequence"),
                logical_step=value.get("logical_step"),
                attempt=value.get("attempt"),
            )
        except MemoryStoreError as error:
            raise MemoryStoreFormatError(
                f"memory provenance is invalid: {error}"
            ) from error


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """One immutable successful write, correction, or revocation."""

    revision: int
    operation: MemoryOperation
    status_after: MemoryEntryStatus
    provenance: MemoryProvenance
    content: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision <= 0
        ):
            raise MemoryStoreFormatError(
                "memory event revision must be a positive integer"
            )
        if not isinstance(self.operation, MemoryOperation):
            raise MemoryStoreFormatError("memory event operation has wrong type")
        if not isinstance(self.status_after, MemoryEntryStatus):
            raise MemoryStoreFormatError("memory event status has wrong type")
        if not isinstance(self.provenance, MemoryProvenance):
            raise MemoryStoreFormatError("memory event provenance has wrong type")
        if self.content is not None and not isinstance(self.content, str):
            raise MemoryStoreFormatError("memory event content has wrong type")
        if self.reason is not None and not isinstance(self.reason, str):
            raise MemoryStoreFormatError("memory event reason has wrong type")

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "operation": self.operation.value,
            "status_after": self.status_after.value,
            "provenance": self.provenance.to_dict(),
            "content": self.content,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: object, *, index: int) -> MemoryEvent:
        expected = {
            "revision",
            "operation",
            "status_after",
            "provenance",
            "content",
            "reason",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise MemoryStoreFormatError(f"memory event {index} fields are malformed")
        try:
            operation = MemoryOperation(value.get("operation"))
            status = MemoryEntryStatus(value.get("status_after"))
        except (TypeError, ValueError) as error:
            raise MemoryStoreFormatError(
                f"memory event {index} has an invalid enum value"
            ) from error
        revision = value.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0:
            raise MemoryStoreFormatError(
                f"memory event {index} revision must be a positive integer"
            )
        content = value.get("content")
        reason = value.get("reason")
        if content is not None and not isinstance(content, str):
            raise MemoryStoreFormatError(f"memory event {index} content has wrong type")
        if reason is not None and not isinstance(reason, str):
            raise MemoryStoreFormatError(f"memory event {index} reason has wrong type")
        return cls(
            revision=revision,
            operation=operation,
            status_after=status,
            provenance=MemoryProvenance.from_dict(value.get("provenance")),
            content=content,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A logical memory and its complete, bounded successful-event history."""

    memory_id: str
    events: tuple[MemoryEvent, ...]

    def __post_init__(self) -> None:
        _validate_memory_id(self.memory_id)
        if not isinstance(self.events, tuple) or not self.events:
            raise MemoryStoreFormatError("memory events must be a non-empty tuple")
        if any(not isinstance(event, MemoryEvent) for event in self.events):
            raise MemoryStoreFormatError("memory events contain an invalid value")
        for index, event in enumerate(self.events, 1):
            if event.revision != index:
                raise MemoryStoreFormatError(
                    "memory event revisions are not contiguous"
                )
            if index == 1:
                if (
                    event.operation is not MemoryOperation.RECORD
                    or event.status_after is not MemoryEntryStatus.ACTIVE
                    or event.content is None
                    or event.reason is not None
                ):
                    raise MemoryStoreFormatError(
                        "first memory event must record content"
                    )
            elif event.operation is MemoryOperation.CORRECT:
                if (
                    event.status_after is not MemoryEntryStatus.ACTIVE
                    or event.content is None
                    or not event.reason
                ):
                    raise MemoryStoreFormatError("memory correction event is malformed")
            elif event.operation is MemoryOperation.REVOKE:
                if (
                    event.status_after is not MemoryEntryStatus.REVOKED
                    or event.content is not None
                    or not event.reason
                ):
                    raise MemoryStoreFormatError("memory revocation event is malformed")
            else:
                raise MemoryStoreFormatError(
                    "record may only be the first memory event"
                )
            if (
                index < len(self.events)
                and event.status_after is MemoryEntryStatus.REVOKED
            ):
                raise MemoryStoreFormatError("revoked memory has later events")

    @property
    def status(self) -> MemoryEntryStatus:
        return self.events[-1].status_after

    @property
    def revision(self) -> int:
        return self.events[-1].revision

    @property
    def current_content(self) -> str | None:
        if self.status is MemoryEntryStatus.REVOKED:
            return None
        return self.events[-1].content

    @property
    def current_provenance(self) -> MemoryProvenance:
        return self.events[-1].provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "memory_id": self.memory_id,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, value: object) -> MemoryEntry:
        expected = {"schema_version", "memory_id", "events"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise MemoryStoreFormatError("memory entry fields are malformed")
        if value.get("schema_version") != MEMORY_SCHEMA_VERSION:
            raise MemoryStoreFormatError("unsupported memory schema version")
        raw_events = value.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise MemoryStoreFormatError("memory entry events must be a non-empty list")
        try:
            return cls(
                memory_id=value.get("memory_id"),
                events=tuple(
                    MemoryEvent.from_dict(event, index=index)
                    for index, event in enumerate(raw_events)
                ),
            )
        except MemoryStoreFormatError:
            raise
        except (TypeError, ValueError) as error:
            raise MemoryStoreFormatError(f"memory entry is invalid: {error}") from error


@dataclass(frozen=True, slots=True)
class MemoryMutationResult:
    """A bounded audit result for one applied or rejected operation."""

    status: MemoryMutationStatus
    operation: MemoryOperation | None
    memory_id: str | None
    limits: Mapping[str, Mapping[str, Any]]
    changes: tuple[Mapping[str, Any], ...] = ()
    provenance: MemoryProvenance | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, MemoryMutationStatus):
            raise TypeError("status must be a MemoryMutationStatus")
        if self.operation is not None and not isinstance(
            self.operation, MemoryOperation
        ):
            raise TypeError("operation must be a MemoryOperation or None")
        if self.memory_id is not None:
            _validate_memory_id(self.memory_id)
        if not isinstance(self.limits, Mapping):
            raise TypeError("limits must be a mapping")
        if not isinstance(self.changes, tuple) or any(
            not isinstance(change, Mapping) for change in self.changes
        ):
            raise TypeError("changes must be a tuple of mappings")
        if self.provenance is not None and not isinstance(
            self.provenance, MemoryProvenance
        ):
            raise TypeError("provenance must be a MemoryProvenance or None")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if self.status is MemoryMutationStatus.APPLIED:
            if (
                self.operation is None
                or self.memory_id is None
                or self.provenance is None
                or self.error is not None
            ):
                raise ValueError("applied memory result is incomplete")
        elif self.error is None:
            raise ValueError("rejected memory result needs an error")

    def to_report(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "mode": "agent-memory-mutation",
            "status": self.status.value,
            "operation": self.operation.value if self.operation is not None else None,
            "memory_id": self.memory_id,
            "changes": [dict(change) for change in self.changes],
            "limits": {name: dict(value) for name, value in self.limits.items()},
        }
        if self.provenance is not None:
            result["provenance"] = self.provenance.to_dict()
        if self.error is not None:
            result["error"] = self.error
        return result

    def to_observation(self) -> str:
        return json.dumps(
            self.to_report(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )


@dataclass(frozen=True, slots=True)
class MemoryRecoveryEvent:
    """One bounded example of a store anomaly handled without global failure."""

    kind: MemoryRecoveryKind
    origin: str
    detail: str

    def to_report(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "origin": self.origin,
            "detail": self.detail,
        }


class MemoryStore:
    """Directory-backed memories with immutable successful mutation history."""

    def __init__(
        self,
        root: Path | str,
        *,
        config: MemoryStoreConfig | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.entries = self.root / "entries"
        self.config = config or MemoryStoreConfig()
        if not isinstance(self.config, MemoryStoreConfig):
            raise TypeError("config must be a MemoryStoreConfig")
        try:
            self.entries.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MemoryStoreError(f"could not create memory store: {error}") from error
        self._lock_path = self.root / ".mutation.lock"
        self._recovery_lock = threading.Lock()
        self._recovery_counts: Counter[str] = Counter()
        self._recovery_examples: list[MemoryRecoveryEvent] = []
        self._last_scan_failures: list[MemoryRecoveryEvent] = []

    @property
    def recovery_report(self) -> dict[str, Any]:
        with self._recovery_lock:
            return {
                "total_event_count": sum(self._recovery_counts.values()),
                "kind_counts": dict(sorted(self._recovery_counts.items())),
                "recent_examples": [
                    event.to_report() for event in self._recovery_examples
                ],
                "retained_example_limit": DEFAULT_MAX_MEMORY_RECOVERY_EXAMPLES,
            }

    @property
    def last_scan_failures(self) -> tuple[MemoryRecoveryEvent, ...]:
        with self._recovery_lock:
            return tuple(self._last_scan_failures)

    def memory_ids(self) -> tuple[str, ...]:
        """List every logical entry in deterministic identifier order."""

        with self._recovery_lock:
            self._last_scan_failures = []
        identifiers: list[str] = []
        try:
            children = tuple(self.entries.iterdir())
        except OSError as error:
            raise MemoryStoreFormatError(
                f"could not list memory entries: {error}"
            ) from error
        for path in children:
            if path.name.startswith(".tmp-"):
                self._record_recovery(
                    MemoryRecoveryEvent(
                        MemoryRecoveryKind.STALE_TEMPORARY_FILE,
                        path.name,
                        "ignored by readers; a locked mutation may remove it",
                    ),
                    scan_failure=False,
                )
                continue
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                self._record_recovery(
                    MemoryRecoveryEvent(
                        MemoryRecoveryKind.MALFORMED_ENTRY,
                        path.name,
                        "unexpected item in memory entries directory",
                    ),
                    scan_failure=True,
                )
                continue
            memory_id = path.stem
            try:
                _validate_memory_id(memory_id)
            except MemoryStoreFormatError as error:
                self._record_recovery(
                    MemoryRecoveryEvent(
                        MemoryRecoveryKind.MALFORMED_ENTRY,
                        path.name,
                        str(error),
                    ),
                    scan_failure=True,
                )
                continue
            identifiers.append(memory_id)
        if len(identifiers) != len(set(identifiers)):
            raise MemoryStoreFormatError("memory entry identifiers are not unique")
        return tuple(sorted(identifiers))

    def read(self, memory_id: str) -> MemoryEntry:
        """Read and strictly validate one durable memory entry."""

        _validate_memory_id(memory_id)
        path = self.entries / f"{memory_id}.json"
        try:
            size = path.stat().st_size
            if size > self._hard_store_byte_limit:
                raise MemoryStoreFormatError(
                    f"memory entry exceeds the absolute store byte limit: {path}"
                )
            data = json.loads(path.read_bytes().decode("utf-8"))
        except FileNotFoundError:
            raise MemoryStoreError(f"memory {memory_id!r} does not exist") from None
        except MemoryStoreFormatError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MemoryStoreFormatError(
                f"memory entry is not valid UTF-8 JSON: {path}"
            ) from error
        try:
            entry = MemoryEntry.from_dict(data)
        except MemoryStoreFormatError as error:
            raise MemoryStoreFormatError(
                f"malformed memory entry {path}: {error}"
            ) from error
        if entry.memory_id != memory_id:
            raise MemoryStoreFormatError(
                f"memory id does not match its filename: {path}"
            )
        self._validate_loaded_entry(entry, path)
        return entry

    def all_entries(self) -> tuple[MemoryEntry, ...]:
        memory_ids = self.memory_ids()
        scan_failures = list(self.last_scan_failures)
        entries: list[MemoryEntry] = []
        failures: list[MemoryStoreFormatError] = []
        for memory_id in memory_ids:
            try:
                entries.append(self.read(memory_id))
            except MemoryStoreFormatError as error:
                failures.append(error)
                self._record_recovery(
                    MemoryRecoveryEvent(
                        MemoryRecoveryKind.MALFORMED_ENTRY,
                        memory_id,
                        _bounded_audit_error(str(error)),
                    ),
                    scan_failure=True,
                )
        if not entries:
            if failures:
                raise failures[0]
            if scan_failures:
                failure = scan_failures[0]
                raise MemoryStoreFormatError(
                    f"malformed memory store item {failure.origin}: {failure.detail}"
                )
        return tuple(entries)

    def current_entries(self) -> tuple[MemoryEntry, ...]:
        """Return only current active claims; superseded text stays audit-only."""

        return tuple(
            entry
            for entry in self.all_entries()
            if entry.status is MemoryEntryStatus.ACTIVE
        )

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a bounded exact snapshot suitable for agent checkpoint state."""

        with self._mutation_lock():
            entries = self.all_entries()
            return {
                "schema_version": MEMORY_SCHEMA_VERSION,
                "entries": [entry.to_dict() for entry in entries],
            }

    def restore_checkpoint_state(
        self,
        value: object,
        *,
        task_id: str,
        run_id: str,
    ) -> None:
        """Restore this run's effects while preserving unrelated concurrent writes.

        The runner retains the rejected tool calls in its step audit.  The durable
        memory view itself returns to the checkpoint snapshot, just as the workspace
        does.  A conflicting write to the same entry by another run is refused
        rather than silently overwritten.
        """

        snapshot = _decode_checkpoint_state(value, self.config)
        snapshot_by_id = {entry.memory_id: entry for entry in snapshot}
        with self._mutation_lock():
            current = self.all_entries()
            current_by_id = {entry.memory_id: entry for entry in current}
            replacements: list[MemoryEntry] = []
            removals: list[str] = []
            for memory_id, prior in snapshot_by_id.items():
                present = current_by_id.get(memory_id)
                if present is None:
                    raise MemoryStoreError(
                        f"cannot restore memory checkpoint: {memory_id!r} disappeared"
                    )
                if present == prior:
                    continue
                prefix = present.events[: len(prior.events)]
                suffix = present.events[len(prior.events) :]
                if prefix != prior.events or not suffix:
                    raise MemoryStoreError(
                        "cannot restore memory checkpoint after a conflicting "
                        f"rewrite of {memory_id!r}"
                    )
                if any(
                    event.provenance.task_id != task_id
                    or event.provenance.run_id != run_id
                    for event in suffix
                ):
                    raise MemoryStoreError(
                        "cannot restore memory checkpoint across another run's "
                        f"write to {memory_id!r}"
                    )
                replacements.append(prior)
            for memory_id, present in current_by_id.items():
                if memory_id in snapshot_by_id:
                    continue
                ownership = tuple(
                    event.provenance.task_id == task_id
                    and event.provenance.run_id == run_id
                    for event in present.events
                )
                if not any(ownership):
                    continue
                first_owned = ownership.index(True)
                if not all(ownership[first_owned:]):
                    raise MemoryStoreError(
                        "cannot restore memory checkpoint across another run's "
                        f"write to {memory_id!r}"
                    )
                retained = present.events[:first_owned]
                if retained:
                    replacements.append(MemoryEntry(memory_id, retained))
                else:
                    removals.append(memory_id)

            for entry in replacements:
                self._write_entry(entry, _serialize_entry(entry))
            for memory_id in removals:
                try:
                    (self.entries / f"{memory_id}.json").unlink()
                except OSError as error:
                    raise MemoryStoreError(
                        f"could not restore memory checkpoint for {memory_id!r}: "
                        f"{error}"
                    ) from error

    def record(
        self, content: object, provenance: MemoryProvenance
    ) -> MemoryMutationResult:
        with self._mutation_lock():
            return self._record_locked(content, provenance)

    def _record_locked(
        self, content: object, provenance: MemoryProvenance
    ) -> MemoryMutationResult:
        memory_ids = self.memory_ids()
        self.all_entries()
        entry_count = len(memory_ids)
        before_bytes = self._store_bytes()
        normalized, error = self._validate_content(content)
        if error is not None:
            return self._reject(
                MemoryOperation.RECORD,
                None,
                provenance,
                error,
                entry_count,
                before_bytes,
                content_length=_safe_length(content),
                hit="entry_length" if "character limit" in error else None,
            )
        provenance_error = self._provenance_error(provenance)
        if provenance_error is not None:
            return self._reject(
                MemoryOperation.RECORD,
                None,
                None,
                provenance_error,
                entry_count,
                before_bytes,
                content_length=len(normalized),
            )
        if entry_count >= self.config.max_entries:
            return self._reject(
                MemoryOperation.RECORD,
                None,
                provenance,
                f"memory entry count reached max_entries ({self.config.max_entries})",
                entry_count,
                before_bytes,
                content_length=len(normalized),
                hit="entry_count",
            )
        memory_id = self._next_id(memory_ids)
        entry = MemoryEntry(
            memory_id,
            (
                MemoryEvent(
                    1,
                    MemoryOperation.RECORD,
                    MemoryEntryStatus.ACTIVE,
                    provenance,
                    content=normalized,
                ),
            ),
        )
        return self._persist(
            MemoryOperation.RECORD,
            entry,
            previous_size=0,
            entry_count_before=entry_count,
            store_bytes_before=before_bytes,
            content_length=len(normalized),
            provenance=provenance,
            changes=(
                {
                    "kind": "recorded",
                    "revision": 1,
                    "content_sha256": _text_sha256(normalized),
                    "content_character_count": len(normalized),
                },
            ),
        )

    def correct(
        self,
        memory_id: object,
        content: object,
        reason: object,
        provenance: MemoryProvenance,
    ) -> MemoryMutationResult:
        with self._mutation_lock():
            return self._revise(
                MemoryOperation.CORRECT, memory_id, content, reason, provenance
            )

    def revoke(
        self,
        memory_id: object,
        reason: object,
        provenance: MemoryProvenance,
    ) -> MemoryMutationResult:
        with self._mutation_lock():
            return self._revise(
                MemoryOperation.REVOKE, memory_id, None, reason, provenance
            )

    def record_rejected_attempt(
        self,
        *,
        operation: MemoryOperation | None,
        memory_id: object,
        provenance: MemoryProvenance | None,
        error: str,
    ) -> MemoryMutationResult:
        """Audit malformed agent input without retaining its potentially secret text."""

        if not isinstance(error, str):
            error = f"memory rejection error has wrong type: {type(error).__name__}"
        try:
            entry_count = len(self.memory_ids())
            store_bytes = self._store_bytes()
        except MemoryStoreError:
            entry_count = 0
            store_bytes = 0
        safe_id = (
            memory_id
            if isinstance(memory_id, str) and _MEMORY_ID.fullmatch(memory_id)
            else None
        )
        return self._reject(
            operation,
            safe_id,
            provenance,
            error,
            entry_count,
            store_bytes,
        )

    def _revise(
        self,
        operation: MemoryOperation,
        memory_id: object,
        content: object,
        reason: object,
        provenance: MemoryProvenance,
    ) -> MemoryMutationResult:
        memory_ids = self.memory_ids()
        entries = self.all_entries()
        entry_count = len(memory_ids)
        before_bytes = self._store_bytes()
        if not isinstance(memory_id, str) or _MEMORY_ID.fullmatch(memory_id) is None:
            return self._reject(
                operation,
                None,
                provenance if isinstance(provenance, MemoryProvenance) else None,
                "memory_id has invalid format",
                entry_count,
                before_bytes,
            )
        provenance_error = self._provenance_error(provenance)
        if provenance_error is not None:
            return self._reject(
                operation,
                memory_id,
                None,
                provenance_error,
                entry_count,
                before_bytes,
            )
        normalized_reason, reason_error = self._validate_reason(reason)
        if reason_error is not None:
            return self._reject(
                operation,
                memory_id,
                provenance,
                reason_error,
                entry_count,
                before_bytes,
            )
        normalized_content: str | None = None
        if operation is MemoryOperation.CORRECT:
            normalized_content, content_error = self._validate_content(content)
            if content_error is not None:
                return self._reject(
                    operation,
                    memory_id,
                    provenance,
                    content_error,
                    entry_count,
                    before_bytes,
                    content_length=_safe_length(content),
                    hit="entry_length" if "character limit" in content_error else None,
                )
        by_id = {entry.memory_id: entry for entry in entries}
        prior = by_id.get(memory_id)
        if prior is None:
            return self._reject(
                operation,
                memory_id,
                provenance,
                f"memory {memory_id!r} does not exist",
                entry_count,
                before_bytes,
                content_length=_safe_length(normalized_content),
            )
        if prior.status is MemoryEntryStatus.REVOKED:
            return self._reject(
                operation,
                memory_id,
                provenance,
                "revoked memory is terminal",
                entry_count,
                before_bytes,
                content_length=_safe_length(normalized_content),
            )
        if (
            operation is MemoryOperation.CORRECT
            and normalized_content == prior.current_content
        ):
            return self._reject(
                operation,
                memory_id,
                provenance,
                "memory correction content is identical to the current claim",
                entry_count,
                before_bytes,
                content_length=_safe_length(normalized_content),
            )
        revision = prior.revision + 1
        if revision > self.config.max_events_per_entry:
            return self._reject(
                operation,
                memory_id,
                provenance,
                "memory entry reached the configured event-count bound "
                f"({self.config.max_events_per_entry})",
                entry_count,
                before_bytes,
                content_length=_safe_length(normalized_content),
            )
        status_after = (
            MemoryEntryStatus.ACTIVE
            if operation is MemoryOperation.CORRECT
            else MemoryEntryStatus.REVOKED
        )
        event = MemoryEvent(
            revision,
            operation,
            status_after,
            provenance,
            content=normalized_content,
            reason=normalized_reason,
        )
        entry = MemoryEntry(memory_id, (*prior.events, event))
        prior_content = prior.current_content
        assert prior_content is not None
        changes: tuple[Mapping[str, Any], ...]
        if operation is MemoryOperation.CORRECT:
            assert normalized_content is not None
            changes = (
                {
                    "kind": "corrected",
                    "revision": revision,
                    "superseded_revision": prior.revision,
                    "before_content_sha256": _text_sha256(prior_content),
                    "after_content_sha256": _text_sha256(normalized_content),
                    "content_character_count": len(normalized_content),
                    "reason_sha256": _text_sha256(normalized_reason),
                    "reason_character_count": len(normalized_reason),
                },
            )
        else:
            changes = (
                {
                    "kind": "revoked",
                    "revision": revision,
                    "revoked_revision": prior.revision,
                    "content_sha256": _text_sha256(prior_content),
                    "reason_sha256": _text_sha256(normalized_reason),
                    "reason_character_count": len(normalized_reason),
                },
            )
        return self._persist(
            operation,
            entry,
            previous_size=len(_serialize_entry(prior)),
            entry_count_before=entry_count,
            store_bytes_before=before_bytes,
            content_length=_safe_length(normalized_content),
            provenance=provenance,
            changes=changes,
        )

    def _persist(
        self,
        operation: MemoryOperation,
        entry: MemoryEntry,
        *,
        previous_size: int,
        entry_count_before: int,
        store_bytes_before: int,
        content_length: int,
        provenance: MemoryProvenance,
        changes: tuple[Mapping[str, Any], ...],
    ) -> MemoryMutationResult:
        document = _serialize_entry(entry)
        after_bytes = store_bytes_before - previous_size + len(document)
        entry_count_after = entry_count_before + (1 if previous_size == 0 else 0)
        if operation is MemoryOperation.RECORD:
            store_limit = self.config.max_store_bytes
        elif operation is MemoryOperation.CORRECT:
            store_limit = self._hard_store_byte_limit - _required_revocation_reserve(
                self.config
            )
        else:
            store_limit = self._hard_store_byte_limit
        if after_bytes > store_limit:
            if operation is MemoryOperation.RECORD:
                limit_name = "max_store_bytes"
            elif operation is MemoryOperation.CORRECT:
                limit_name = "the correction limit with revocation headroom"
            else:
                limit_name = "the absolute store byte limit"
            return self._reject(
                operation,
                None if operation is MemoryOperation.RECORD else entry.memory_id,
                provenance,
                f"memory store would exceed {limit_name} ({store_limit})",
                entry_count_before,
                store_bytes_before,
                content_length=content_length,
                hit="store_size",
            )
        self._write_entry(entry, document)
        result = MemoryMutationResult(
            MemoryMutationStatus.APPLIED,
            operation,
            entry.memory_id,
            self._limits(
                entry_count_before,
                entry_count_after,
                content_length,
                store_bytes_before,
                after_bytes,
            ),
            changes=changes,
            provenance=provenance,
        )
        return result

    def _write_entry(self, entry: MemoryEntry, document: bytes) -> None:
        final = self.entries / f"{entry.memory_id}.json"
        temporary = self.entries / f".tmp-{entry.memory_id}.json"
        try:
            with temporary.open("xb") as handle:
                handle.write(document)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, final)
        except OSError as error:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise MemoryStoreError(
                f"could not persist memory {entry.memory_id!r}: {error}"
            ) from error

    def _reject(
        self,
        operation: MemoryOperation | None,
        memory_id: str | None,
        provenance: MemoryProvenance | None,
        error: str,
        entry_count: int,
        store_bytes: int,
        *,
        content_length: int = 0,
        hit: str | None = None,
    ) -> MemoryMutationResult:
        safe_provenance = (
            provenance
            if isinstance(provenance, MemoryProvenance)
            and self._provenance_error(provenance) is None
            else None
        )
        result = MemoryMutationResult(
            MemoryMutationStatus.REJECTED,
            operation,
            memory_id,
            self._limits(
                entry_count,
                entry_count,
                content_length,
                store_bytes,
                store_bytes,
                hit=hit,
            ),
            provenance=safe_provenance,
            error=_bounded_audit_error(error),
        )
        return result

    def _limits(
        self,
        count_before: int,
        count_after: int,
        content_length: int,
        bytes_before: int,
        bytes_after: int,
        *,
        hit: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        return {
            "entry_count": {
                "limit": self.config.max_entries,
                "before": count_before,
                "after": count_after,
                "hit": hit == "entry_count",
            },
            "entry_length": {
                "limit": self.config.max_content_characters,
                "attempted": content_length,
                "hit": hit == "entry_length",
            },
            "store_size": {
                "limit": self.config.max_store_bytes,
                "before": bytes_before,
                "after": bytes_after,
                "hit": hit == "store_size",
            },
            "remediation_reserve": {
                "limit": self.config.max_remediation_bytes,
                "absolute_store_limit": self._hard_store_byte_limit,
                "used_before": max(0, bytes_before - self.config.max_store_bytes),
                "used_after": max(0, bytes_after - self.config.max_store_bytes),
            },
        }

    def _validate_loaded_entry(self, entry: MemoryEntry, path: Path) -> None:
        if len(entry.events) > self.config.max_events_per_entry:
            raise MemoryStoreFormatError(
                f"memory entry exceeds the event-count bound: {path}"
            )
        for event in entry.events:
            try:
                event.provenance.validate_for(self.config)
            except MemoryStoreError as error:
                raise MemoryStoreFormatError(
                    f"memory entry has invalid provenance: {path}: {error}"
                ) from error
            if event.content is not None:
                _loaded_text_guard(
                    event.content,
                    self.config.max_content_characters,
                    "content",
                    path,
                )
            if event.reason is not None:
                _loaded_text_guard(
                    event.reason,
                    self.config.max_reason_characters,
                    "reason",
                    path,
                )

    def _validate_content(self, value: object) -> tuple[str, str | None]:
        if not isinstance(value, str) or not value.strip():
            return "", "memory content must be non-empty text"
        normalized = value.strip()
        if len(normalized) > self.config.max_content_characters:
            return normalized, (
                "memory content exceeds the "
                f"{self.config.max_content_characters}-character limit"
            )
        if "\x00" in normalized:
            return normalized, "memory content contains a null character"
        if not _is_valid_utf8(normalized):
            return normalized, "memory content must be valid UTF-8 text"
        if _looks_like_credential(normalized):
            return normalized, "memory content looks like a credential"
        return normalized, None

    def _validate_reason(self, value: object) -> tuple[str, str | None]:
        if not isinstance(value, str) or not value.strip():
            return "", "memory correction or revocation reason must be non-empty text"
        normalized = value.strip()
        if len(normalized) > self.config.max_reason_characters:
            return normalized, (
                "memory reason exceeds the "
                f"{self.config.max_reason_characters}-character limit"
            )
        if "\x00" in normalized or not _is_valid_utf8(normalized):
            return normalized, "memory reason must be valid UTF-8 text"
        if _looks_like_credential(normalized):
            return normalized, "memory reason looks like a credential"
        return normalized, None

    def _provenance_error(self, value: object) -> str | None:
        if not isinstance(value, MemoryProvenance):
            return "provenance must be a MemoryProvenance"
        try:
            value.validate_for(self.config)
        except MemoryStoreError as error:
            return str(error)
        return None

    def _store_bytes(self) -> int:
        total = 0
        for memory_id in self.memory_ids():
            path = self.entries / f"{memory_id}.json"
            try:
                total += path.stat().st_size
            except OSError as error:
                raise MemoryStoreFormatError(
                    f"could not stat memory entry: {path}"
                ) from error
        return total

    @property
    def _hard_store_byte_limit(self) -> int:
        return self.config.max_store_bytes + self.config.max_remediation_bytes

    @contextmanager
    def _mutation_lock(self) -> Iterator[None]:
        try:
            handle = self._lock_path.open("a+b")
        except OSError as error:
            raise MemoryStoreError(
                f"could not open memory mutation lock: {error}"
            ) from error
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError as error:
                raise MemoryStoreError(
                    f"could not acquire memory mutation lock: {error}"
                ) from error
            self._clean_stale_temporaries()
            yield
        finally:
            with suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _clean_stale_temporaries(self) -> None:
        """Remove abandoned write files only while no other writer can own them."""

        try:
            temporary_paths = tuple(
                path for path in self.entries.iterdir() if path.name.startswith(".tmp-")
            )
        except OSError as error:
            raise MemoryStoreFormatError(
                f"could not inspect temporary memory entries: {error}"
            ) from error
        for path in temporary_paths:
            try:
                path.unlink()
                detail = "removed while holding the mutation lock"
            except OSError:
                detail = "ignored because locked cleanup failed"
            self._record_recovery(
                MemoryRecoveryEvent(
                    MemoryRecoveryKind.STALE_TEMPORARY_FILE,
                    path.name,
                    detail,
                ),
                scan_failure=False,
            )

    def _record_recovery(
        self, event: MemoryRecoveryEvent, *, scan_failure: bool
    ) -> None:
        with self._recovery_lock:
            self._recovery_counts[event.kind.value] += 1
            if len(self._recovery_examples) < DEFAULT_MAX_MEMORY_RECOVERY_EXAMPLES:
                self._recovery_examples.append(event)
            if (
                scan_failure
                and len(self._last_scan_failures) < DEFAULT_MAX_MEMORY_RECOVERY_EXAMPLES
            ):
                self._last_scan_failures.append(event)

    @staticmethod
    def _next_id(memory_ids: tuple[str, ...]) -> str:
        number = (
            max(
                (int(memory_id.removeprefix("memory-")) for memory_id in memory_ids),
                default=0,
            )
            + 1
        )
        if number > 999_999:
            raise MemoryStoreError("memory identifier space is exhausted")
        return f"memory-{number:06d}"


def _validate_memory_id(memory_id: object) -> None:
    if not isinstance(memory_id, str) or _MEMORY_ID.fullmatch(memory_id) is None:
        raise MemoryStoreFormatError("memory id has invalid format")


def _decode_checkpoint_state(
    value: object, config: MemoryStoreConfig
) -> tuple[MemoryEntry, ...]:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "entries"}:
        raise MemoryStoreFormatError("memory checkpoint state fields are malformed")
    if value.get("schema_version") != MEMORY_SCHEMA_VERSION:
        raise MemoryStoreFormatError("unsupported memory checkpoint schema version")
    raw_entries = value.get("entries")
    if not isinstance(raw_entries, list):
        raise MemoryStoreFormatError("memory checkpoint entries must be a list")
    if len(raw_entries) > config.max_entries:
        raise MemoryStoreFormatError(
            "memory checkpoint exceeds the configured entry-count bound"
        )
    entries_list: list[MemoryEntry] = []
    serialized_bytes = 0
    for raw_entry in raw_entries:
        raw_events = raw_entry.get("events") if isinstance(raw_entry, Mapping) else None
        if (
            isinstance(raw_events, list)
            and len(raw_events) > config.max_events_per_entry
        ):
            raise MemoryStoreFormatError(
                "memory checkpoint entry exceeds the configured event-count bound"
            )
        entry = MemoryEntry.from_dict(raw_entry)
        if len(entry.events) > config.max_events_per_entry:
            raise MemoryStoreFormatError(
                "memory checkpoint entry exceeds the configured event-count bound"
            )
        for event in entry.events:
            event.provenance.validate_for(config)
            if event.content is not None:
                _checkpoint_text_guard(
                    event.content,
                    config.max_content_characters,
                    "content",
                )
            if event.reason is not None:
                _checkpoint_text_guard(
                    event.reason,
                    config.max_reason_characters,
                    "reason",
                )
        serialized_bytes += len(_serialize_entry(entry))
        if serialized_bytes > config.max_store_bytes + config.max_remediation_bytes:
            raise MemoryStoreFormatError(
                "memory checkpoint exceeds the absolute store byte limit"
            )
        entries_list.append(entry)
    entries = tuple(entries_list)
    identifiers = tuple(entry.memory_id for entry in entries)
    if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(
        set(identifiers)
    ):
        raise MemoryStoreFormatError(
            "memory checkpoint entries must have unique sorted identifiers"
        )
    return entries


def _required_revocation_reserve(config: MemoryStoreConfig) -> int:
    # JSON escaping can expand one character to six ASCII bytes (``\\u0001``).
    # The fixed allowance covers field names, counters, punctuation, and indentation.
    return 2_048 + 6 * (
        config.max_reason_characters + 2 * config.max_provenance_id_characters
    )


def _checkpoint_text_guard(value: str, limit: int, name: str) -> None:
    if (
        not value.strip()
        or len(value) > limit
        or "\x00" in value
        or not _is_valid_utf8(value)
    ):
        raise MemoryStoreFormatError(
            f"memory checkpoint {name} violates its configured bound"
        )
    if _looks_like_credential(value):
        raise MemoryStoreFormatError(
            f"memory checkpoint {name} looks like a credential"
        )


def _serialize_entry(entry: MemoryEntry) -> bytes:
    return (
        json.dumps(entry.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _safe_length(value: object) -> int:
    return len(value) if isinstance(value, str) else 0


def _bounded_audit_error(value: str) -> str:
    if len(value) <= DEFAULT_MAX_MEMORY_AUDIT_ERROR_CHARACTERS:
        return value
    return (
        "memory error detail omitted for audit bound; "
        f"character_count={len(value)}; sha256={_text_sha256(value)}"
    )


def _looks_like_credential(value: str) -> bool:
    return any(pattern.search(value) is not None for pattern in _CREDENTIAL_PATTERNS)


def _is_valid_utf8(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _loaded_text_guard(value: str, limit: int, name: str, path: Path) -> None:
    if (
        not value.strip()
        or len(value) > limit
        or "\x00" in value
        or not _is_valid_utf8(value)
    ):
        raise MemoryStoreFormatError(f"memory {name} violates its bound: {path}")
    if _looks_like_credential(value):
        raise MemoryStoreFormatError(f"memory {name} looks like a credential: {path}")
