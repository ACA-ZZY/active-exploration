"""Probability estimates on the surrogate chain"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SAMPLER, SamplerConfig
from .error_model import FittedErrorModel
from .problem import Direction, in_conditional_domain
from .resolution import Resolution

@dataclass(frozen=True)
class ProbabilityEstimate:
    threshold: float
    chain_factor: float
    terminal_fraction_raw: float
    terminal_fraction_corrected: float
    raw: float
    corrected: float
    sub_resolution: bool

    def as_row(self) -> dict:
        return {
            "threshold": self.threshold,
            "chain_factor": self.chain_factor,
            "terminal_fraction_raw": self.terminal_fraction_raw,
            "terminal_fraction_corrected": self.terminal_fraction_corrected,
            "probability_raw": self.raw,
            "probability_corrected": self.corrected,
            "sub_resolution": self.sub_resolution,
        }

def conditional_exceedance(
    proxy_values: np.ndarray,
    threshold: float,
    model: FittedErrorModel,
    adverse: Direction,
) -> float:
    """Mean over the population of ``P(true response beyond threshold)``"""

    margin = np.asarray(proxy_values, dtype=float).reshape(-1) - float(threshold)
    distribution = model.distribution
    probabilities = (
        distribution.cdf(margin) if adverse == "upper" else distribution.sf(margin)
    )
    return float(np.mean(probabilities))

def estimate(
    terminal_values: np.ndarray,
    threshold: float,
    level_index: int,
    model: FittedErrorModel | None,
    adverse: Direction,
    resolution: Resolution,
    cfg: SamplerConfig = SAMPLER,
) -> ProbabilityEstimate:
    """Subset probability at ``threshold`` from the terminal population."""

    chain_factor = cfg.conditional_probability**level_index
    raw_fraction = float(
        np.mean(in_conditional_domain(terminal_values, threshold, adverse))
    )
    corrected_fraction = (
        conditional_exceedance(terminal_values, threshold, model, adverse)
        if model is not None
        else float("nan")
    )
    corrected = chain_factor * corrected_fraction
    return ProbabilityEstimate(
        threshold=float(threshold),
        chain_factor=chain_factor,
        terminal_fraction_raw=raw_fraction,
        terminal_fraction_corrected=corrected_fraction,
        raw=chain_factor * raw_fraction,
        corrected=corrected,
        sub_resolution=bool(np.isfinite(corrected))
        and resolution.is_sub_resolution(corrected),
    )
