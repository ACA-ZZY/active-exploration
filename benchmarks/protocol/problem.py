"""Problem interface shared by every case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import numpy as np

Direction = Literal["upper", "lower"]

class ResponseFunction(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray: ...

@dataclass(frozen=True)
class Problem:
    """One case definition."""

    name: str
    dimension: int
    adverse: Direction
    recipe: str
    response: ResponseFunction
    reference_probability: float | None = None
    units: str = ""
    feature_map: Callable[[np.ndarray], np.ndarray] | None = None
    sequence_map: Callable[[np.ndarray], np.ndarray] | None = None
    subset_sampler: str = "pcn"
    feature_tag: str = ""

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(self.response(np.atleast_2d(x)), dtype=float).reshape(-1)
        if values.shape[0] != np.atleast_2d(x).shape[0]:
            raise ValueError(f"{self.name}: response must return one value per row")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{self.name}: response returned a non-finite value")
        return values

def adverse_order(values: np.ndarray, adverse: Direction) -> np.ndarray:
    """Indices from most adverse to least adverse, ties broken stably."""
    if adverse == "upper":
        return np.argsort(-np.asarray(values, dtype=float), kind="stable")
    return np.argsort(np.asarray(values, dtype=float), kind="stable")

def in_conditional_domain(
    values: np.ndarray, threshold: float, adverse: Direction
) -> np.ndarray:
    """Membership of the conditional level defined by ``threshold``."""
    values = np.asarray(values, dtype=float)
    return values >= threshold if adverse == "upper" else values <= threshold

def exceedance_fraction(
    values: np.ndarray, threshold: float, adverse: Direction
) -> float:
    return float(np.mean(in_conditional_domain(values, threshold, adverse)))
