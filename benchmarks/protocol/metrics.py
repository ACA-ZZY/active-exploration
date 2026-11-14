"""Per-level surrogate diagnostics and chain-level trajectory metrics"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr

from .problem import Direction, in_conditional_domain

@dataclass(frozen=True)
class LevelMetrics:
    level: int
    nominal_probability: float
    n_audited: int
    precision: float
    raw_mae: float
    mean_corrected_mae: float
    spearman: float
    bias: float
    proxy_cutoff: float
    reference_cutoff: float
    cutoff_gap: float
    cutoff_ratio: float
    enrichment: float
    resolved: bool

    def as_row(self) -> dict[str, Any]:
        return asdict(self)

def level_metrics(
    *,
    level: int,
    nominal_probability: float,
    proxy_values: np.ndarray,
    true_values: np.ndarray,
    proxy_cutoff: float,
    reference_cutoff: float,
    adverse: Direction,
) -> LevelMetrics:
    """Score one audited level"""

    proxy_values = np.asarray(proxy_values, dtype=float).reshape(-1)
    true_values = np.asarray(true_values, dtype=float).reshape(-1)
    if proxy_values.shape != true_values.shape:
        raise ValueError("audited proxy and true values must align")

    hits = in_conditional_domain(true_values, reference_cutoff, adverse)
    precision = float(np.mean(hits))

    error = proxy_values - true_values
    bias = float(np.mean(error))
    raw_mae = float(np.mean(np.abs(error)))
    mean_corrected_mae = float(np.mean(np.abs(error - bias)))

    if np.ptp(proxy_values) == 0 or np.ptp(true_values) == 0:
        rho = float("nan")
    else:
        rho = float(spearmanr(proxy_values, true_values).statistic)

    return LevelMetrics(
        level=level,
        nominal_probability=nominal_probability,
        n_audited=int(proxy_values.size),
        precision=precision,
        raw_mae=raw_mae,
        mean_corrected_mae=mean_corrected_mae,
        spearman=rho,
        bias=bias,
        proxy_cutoff=float(proxy_cutoff),
        reference_cutoff=float(reference_cutoff),
        cutoff_gap=float(proxy_cutoff - reference_cutoff),
        cutoff_ratio=float(proxy_cutoff / reference_cutoff)
        if reference_cutoff != 0
        else float("nan"),
        enrichment=float(precision / nominal_probability)
        if nominal_probability > 0
        else float("nan"),
        resolved=bool(precision > 0),
    )

def dynamic_range_retention(
    proxy_cutoffs: Sequence[float], reference_cutoffs: Sequence[float]
) -> float:
    """Share of the reference threshold span that the surrogate chain covers"""

    proxy = np.asarray(proxy_cutoffs, dtype=float)
    reference = np.asarray(reference_cutoffs, dtype=float)
    reference_span = reference[-1] - reference[0]
    if reference_span == 0:
        return float("nan")
    return float((proxy[-1] - proxy[0]) / reference_span)

def unresolved_levels(metrics: Sequence[LevelMetrics]) -> list[int]:
    """Levels the surrogate chain never actually reached."""
    return [m.level for m in metrics if not m.resolved]
