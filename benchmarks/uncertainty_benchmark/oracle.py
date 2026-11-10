"""Transactional boundary between acquisition and the physical response."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checkpoint import BlockStore, safe_artifact_name
from .ledger import CallEvent, CallLedger

@dataclass(frozen=True)
class QueryResult:
    x: np.ndarray
    y: np.ndarray
    artifact: Path
    reused: bool
    origin_physical_calls: int

class CachedOracle:
    """Evaluate a frozen proposal once and persist labels before returning"""

    def __init__(self, problem, store: BlockStore, ledger: CallLedger):
        self.problem = problem
        self.store = store
        self.ledger = ledger

    def query(
        self,
        event_id: str,
        x: np.ndarray,
        *,
        role: str,
        method: str = "shared",
        round_index: int = 0,
    ) -> QueryResult:
        values = np.ascontiguousarray(np.atleast_2d(np.asarray(x, dtype=np.float64)))
        if values.shape[1] != int(self.problem.dimension):
            raise ValueError(
                f"oracle expected {self.problem.dimension} columns, got {values.shape[1]}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("oracle inputs must be finite")

        relative = Path("queries") / f"{safe_artifact_name(event_id)}.npz"
        cached = self.store.load_npz(relative)
        reused = cached is not None
        if cached is None:
            response = np.asarray(self.problem.evaluate(values), dtype=np.float64).reshape(-1)
            if response.shape != (values.shape[0],) or not np.all(np.isfinite(response)):
                raise RuntimeError("physical response returned invalid labels")
            self.store.save_npz_once(
                relative,
                x=values,
                y=response,
                origin_physical_calls=np.asarray(values.shape[0], dtype=np.int64),
            )
            cached = self.store.load_npz(relative)
            if cached is None:  # pragma: no cover - filesystem failure
                raise RuntimeError(f"query artefact disappeared after write: {relative}")

        stored_x = np.ascontiguousarray(np.asarray(cached["x"], dtype=np.float64))
        stored_y = np.asarray(cached["y"], dtype=np.float64).reshape(-1)
        if not np.array_equal(stored_x, values):
            raise RuntimeError(f"query id reused for different inputs: {event_id}")
        if stored_y.shape != (values.shape[0],) or not np.all(np.isfinite(stored_y)):
            raise RuntimeError(f"corrupt oracle label artefact: {relative}")
        origin_calls = int(np.asarray(cached["origin_physical_calls"]).item())
        if origin_calls != values.shape[0]:
            raise RuntimeError(f"invalid physical-call count in {relative}")

        unique = int(np.unique(values, axis=0).shape[0])
        self.ledger.record(
            CallEvent(
                event_id=event_id,
                role=role,
                method=method,
                round_index=int(round_index),
                logical_calls=int(values.shape[0]),
                unique_inputs=unique,
                physical_calls=origin_calls,
                artifact=str(relative).replace("\\", "/"),
            )
        )
        stored_x.setflags(write=False)
        stored_y.setflags(write=False)
        return QueryResult(stored_x, stored_y, self.store.path(relative), reused, origin_calls)

