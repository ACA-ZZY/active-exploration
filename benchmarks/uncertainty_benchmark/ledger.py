"""Idempotent true-model call ledger for resumable benchmark blocks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROLES = {
    "initial_train",
    "initial_holdout",
    "acquisition_train",
    "acquisition_audit",
    "final_audit",
    "reference",
    "calibration",
    "design",
    "retry",
}

@dataclass(frozen=True)
class CallEvent:
    event_id: str
    role: str
    method: str
    round_index: int
    logical_calls: int
    unique_inputs: int
    physical_calls: int
    artifact: str

class CallLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._events: dict[str, CallEvent] = {}
        if self.path.is_file():
            body = json.loads(self.path.read_text(encoding="utf-8"))
            if body.get("schema") != "uncertainty-benchmark-ledger-v1":
                raise RuntimeError(f"unsupported ledger schema in {self.path}")
            for row in body.get("events", []):
                event = CallEvent(**row)
                self._events[event.event_id] = event

    @property
    def events(self) -> tuple[CallEvent, ...]:
        return tuple(self._events[key] for key in sorted(self._events))

    def record(self, event: CallEvent) -> None:
        if event.role not in ROLES:
            raise ValueError(f"unregistered call role {event.role!r}")
        if min(event.logical_calls, event.unique_inputs, event.physical_calls) < 0:
            raise ValueError("call counts must be non-negative")
        previous = self._events.get(event.event_id)
        if previous is not None:
            if previous != event:
                raise RuntimeError(f"ledger event drift on resume: {event.event_id}")
            return
        self._events[event.event_id] = event
        self._flush()

    def _flush(self) -> None:
        from .checkpoint import atomic_write_json

        atomic_write_json(self.path, self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        by_role: dict[str, dict[str, int]] = {}
        for event in self._events.values():
            row = by_role.setdefault(
                event.role,
                {"logical_calls": 0, "unique_inputs": 0, "physical_calls": 0},
            )
            row["logical_calls"] += event.logical_calls
            row["unique_inputs"] += event.unique_inputs
            row["physical_calls"] += event.physical_calls
        return {
            "schema": "uncertainty-benchmark-ledger-v1",
            "events": [asdict(event) for event in self.events],
            "totals_by_role": by_role,
            "totals": {
                key: sum(row[key] for row in by_role.values())
                for key in ("logical_calls", "unique_inputs", "physical_calls")
            },
        }

