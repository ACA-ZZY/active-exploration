"""True-response call ledger, kept by role"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

DEPLOYMENT_ROLES = ("initial_dataset", "acquisition_audit")
INSTRUMENT_ROLES = ("evaluation_audit", "reference_chain")

@dataclass
class Ledger:
    calls: Counter = field(default_factory=Counter)
    unique_inputs: Counter = field(default_factory=Counter)

    def record(self, role: str, n: int, unique: int | None = None) -> None:
        if role not in DEPLOYMENT_ROLES + INSTRUMENT_ROLES:
            raise ValueError(f"unregistered ledger role: {role}")
        self.calls[role] += int(n)
        self.unique_inputs[role] += int(n if unique is None else unique)

    @property
    def deployment(self) -> int:
        return sum(self.calls[r] for r in DEPLOYMENT_ROLES)

    @property
    def instrumentation(self) -> int:
        return sum(self.calls[r] for r in INSTRUMENT_ROLES)

    def as_row(self) -> dict:
        row = {f"calls_{k}": int(v) for k, v in self.calls.items()}
        row.update({f"unique_{k}": int(v) for k, v in self.unique_inputs.items()})
        row["deployment_total"] = self.deployment
        row["instrumentation_total"] = self.instrumentation
        return row

    def merge(self, other: "Ledger") -> "Ledger":
        merged = Ledger()
        merged.calls = self.calls + other.calls
        merged.unique_inputs = self.unique_inputs + other.unique_inputs
        return merged
