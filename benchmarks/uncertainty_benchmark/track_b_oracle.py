"""Fail-closed true-model transaction boundary for Track-B B1"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checkpoint import BlockStore, safe_artifact_name
from .ledger import CallEvent, CallLedger

class UnknownInflightQuery(RuntimeError):
    """An intent exists without a durable response; manual reconciliation required."""

@dataclass(frozen=True)
class TrackBQueryResult:
    x: np.ndarray
    y: np.ndarray
    artifact: Path
    reused: bool
    origin_physical_calls: int

class TrackBTransactionalOracle:
    def __init__(self, problem, store: BlockStore, ledger: CallLedger):
        self.problem = problem
        self.store = store
        self.ledger = ledger

    @staticmethod
    def _intent_payload(
        event_id: str,
        values: np.ndarray,
        *,
        role: str,
        method: str,
        round_index: int,
    ) -> dict[str, object]:
        return {
            "x": values,
            "event_id": np.asarray(event_id),
            "role": np.asarray(role),
            "method": np.asarray(method),
            "round_index": np.asarray(int(round_index), dtype=np.int64),
        }

    @staticmethod
    def _scalar_text(value: np.ndarray) -> str:
        return str(np.asarray(value).item())

    def _validate_intent(
        self,
        cached: dict[str, np.ndarray],
        event_id: str,
        values: np.ndarray,
        *,
        role: str,
        method: str,
        round_index: int,
    ) -> None:
        stored = np.ascontiguousarray(np.asarray(cached.get("x"), dtype=np.float64))
        if (
            stored.shape != values.shape
            or not np.array_equal(stored, values)
            or self._scalar_text(cached.get("event_id")) != event_id
            or self._scalar_text(cached.get("role")) != role
            or self._scalar_text(cached.get("method")) != method
            or int(np.asarray(cached.get("round_index")).item()) != int(round_index)
        ):
            raise RuntimeError(f"Track-B oracle intent drift: {event_id}")

    def query(
        self,
        event_id: str,
        x: np.ndarray,
        *,
        role: str,
        method: str,
        round_index: int = 0,
    ) -> TrackBQueryResult:
        values = np.ascontiguousarray(np.atleast_2d(np.asarray(x, dtype=np.float64)))
        if values.shape[1] != int(self.problem.dimension):
            raise ValueError(
                f"oracle expected {self.problem.dimension} columns, got {values.shape[1]}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("oracle inputs must be finite")

        safe = safe_artifact_name(event_id)
        intent_relative = Path("oracle_intents") / f"{safe}.npz"
        query_relative = Path("queries") / f"{safe}.npz"
        response_cached = self.store.load_npz(query_relative)
        intent_cached = self.store.load_npz(intent_relative)
        if response_cached is not None and intent_cached is None:
            raise RuntimeError(f"Track-B response has no prior intent: {event_id}")

        created_intent = False
        if intent_cached is None:
            _, created_intent = self.store.save_npz_once(
                intent_relative,
                **self._intent_payload(
                    event_id,
                    values,
                    role=role,
                    method=method,
                    round_index=round_index,
                ),
            )
            intent_cached = self.store.load_npz(intent_relative)
            if intent_cached is None:  # pragma: no cover - filesystem failure
                raise RuntimeError(f"Track-B oracle intent disappeared: {event_id}")
        self._validate_intent(
            intent_cached,
            event_id,
            values,
            role=role,
            method=method,
            round_index=round_index,
        )

        reused = response_cached is not None
        if response_cached is None:
            if not created_intent:
                self.store.write_json(
                    Path("oracle_status") / f"{safe}.json",
                    {
                        "schema": "track-b-oracle-unknown-inflight-v1",
                        "status": "unknown_inflight",
                        "event_id": event_id,
                        "role": role,
                        "method": method,
                        "round_index": int(round_index),
                        "intent": str(intent_relative).replace("\\", "/"),
                        "automatic_re_evaluation_forbidden": True,
                    },
                )
                raise UnknownInflightQuery(
                    f"Track-B intent exists without a durable response for {event_id}; "
                    "HF completion is unknown and automatic re-evaluation is forbidden"
                )
            try:
                response = np.asarray(
                    self.problem.evaluate(values), dtype=np.float64
                ).reshape(-1)
            except Exception:
                self.store.write_json(
                    Path("oracle_status") / f"{safe}.json",
                    {
                        "schema": "track-b-oracle-unknown-inflight-v1",
                        "status": "unknown_inflight",
                        "event_id": event_id,
                        "role": role,
                        "method": method,
                        "round_index": int(round_index),
                        "intent": str(intent_relative).replace("\\", "/"),
                        "automatic_re_evaluation_forbidden": True,
                        "reason": "physical evaluation raised before durable response commit",
                    },
                )
                raise
            if response.shape != (values.shape[0],) or not np.all(np.isfinite(response)):
                raise RuntimeError("physical response returned invalid labels")
            self.store.save_npz_once(
                query_relative,
                x=values,
                y=response,
                origin_physical_calls=np.asarray(values.shape[0], dtype=np.int64),
            )
            response_cached = self.store.load_npz(query_relative)
            if response_cached is None:  # pragma: no cover - filesystem failure
                raise RuntimeError(f"Track-B response disappeared after write: {event_id}")

        stored_x = np.ascontiguousarray(np.asarray(response_cached.get("x"), dtype=np.float64))
        stored_y = np.asarray(response_cached.get("y"), dtype=np.float64).reshape(-1)
        if (
            stored_x.shape != values.shape
            or not np.array_equal(stored_x, values)
            or stored_y.shape != (values.shape[0],)
            or not np.all(np.isfinite(stored_y))
        ):
            raise RuntimeError(f"corrupt Track-B oracle response: {event_id}")
        origin_calls = int(np.asarray(response_cached.get("origin_physical_calls")).item())
        if origin_calls != values.shape[0]:
            raise RuntimeError(f"invalid Track-B physical-call count: {event_id}")

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
                artifact=str(query_relative).replace("\\", "/"),
            )
        )
        stored_x.setflags(write=False)
        stored_y.setflags(write=False)
        return TrackBQueryResult(
            x=stored_x,
            y=stored_y,
            artifact=self.store.path(query_relative),
            reused=reused,
            origin_physical_calls=origin_calls,
        )
