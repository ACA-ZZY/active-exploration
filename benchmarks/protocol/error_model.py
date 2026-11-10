"""Conditional error model with an out-of-sample family choice"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

CANDIDATES: dict[str, Any] = {
    "normal": stats.norm,
    "student_t": stats.t,
    "ged": stats.gennorm,
    "skew_normal": stats.skewnorm,
    "logistic": stats.logistic,
}

@dataclass(frozen=True)
class FittedErrorModel:
    family: str
    parameters: tuple[float, ...]
    validation_loglik: float
    validation_ks: float
    validation_ks_pvalue: float
    n_fit: int
    n_validation: int

    @property
    def distribution(self):
        return CANDIDATES[self.family](*self.parameters)

    def mean_correction(self) -> float:
        return float(self.distribution.mean())

    def exceedance(self, margin: np.ndarray | float) -> np.ndarray:
        """P(e > margin) under the fitted error law."""
        return np.asarray(self.distribution.sf(margin), dtype=float)

@dataclass
class Selection:
    chosen: FittedErrorModel
    ranking: list[FittedErrorModel] = field(default_factory=list)

    def report(self) -> list[dict[str, Any]]:
        return [
            {
                "family": m.family,
                "validation_loglik": m.validation_loglik,
                "validation_ks": m.validation_ks,
                "validation_ks_pvalue": m.validation_ks_pvalue,
                "chosen": m.family == self.chosen.family,
            }
            for m in self.ranking
        ]

def select_error_model(
    fit_residuals: np.ndarray,
    validation_residuals: np.ndarray,
    families: dict[str, Any] | None = None,
) -> Selection:
    """Fit every candidate on ``fit_residuals`` and rank on ``validation_residuals``"""

    fit_residuals = np.asarray(fit_residuals, dtype=float).reshape(-1)
    validation_residuals = np.asarray(validation_residuals, dtype=float).reshape(-1)
    if fit_residuals.size == 0 or validation_residuals.size == 0:
        raise ValueError("both residual sets must be non-empty")

    families = CANDIDATES if families is None else families
    fitted: list[FittedErrorModel] = []

    for name, family in families.items():
        try:
            parameters = family.fit(fit_residuals)
            frozen = family(*parameters)
            loglik = float(np.sum(frozen.logpdf(validation_residuals)))
            if not np.isfinite(loglik):
                continue
            ks = stats.kstest(validation_residuals, frozen.cdf)
        except Exception:
            continue
        fitted.append(
            FittedErrorModel(
                family=name,
                parameters=tuple(float(p) for p in parameters),
                validation_loglik=loglik,
                validation_ks=float(ks.statistic),
                validation_ks_pvalue=float(ks.pvalue),
                n_fit=int(fit_residuals.size),
                n_validation=int(validation_residuals.size),
            )
        )

    if not fitted:
        raise RuntimeError("no candidate error model could be fitted")

    fitted.sort(key=lambda m: m.validation_loglik, reverse=True)
    return Selection(chosen=fitted[0], ranking=fitted)
