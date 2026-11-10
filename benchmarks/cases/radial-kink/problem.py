"""Radial shell kink problem, eight dimensions"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

BETA = 8.7
KAPPA = 1.0
Q0 = 4.3
DIMENSION = 8
NAME = "radial_kink"

CONFIG = {'case': {'slug': 'radial-kink',
          'name': 'radial_kink',
          'dimension': 8,
          'adverse': 'lower',
          'status': 'ready'},
 'seeds': {'root': 20260826,
           'unconditional': 1,
           'proxy_chain': 2,
           'reference_chain': 3,
           'audit': 4,
           'error_model': 5}}

def response(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    q = (np.sum(x * x, axis=1) - DIMENSION) / 4.0
    return BETA - q - KAPPA * np.maximum(q - Q0, 0.0)

def reference_probability() -> float:
    q_star = (BETA + KAPPA * Q0) / (1.0 + KAPPA)
    return float(chi2.sf(DIMENSION + 4.0 * q_star, DIMENSION))

def failure_probability(threshold: float) -> float:
    """Exact ``P[g <= t]``."""
    t = float(threshold)
    q_t = BETA - t if t >= BETA - Q0 else (BETA + KAPPA * Q0 - t) / (1.0 + KAPPA)
    return float(chi2.sf(DIMENSION + 4.0 * q_t, DIMENSION))

def threshold_for_probability(probability: float) -> float:
    """Exact inverse of :func:`failure_probability`."""
    q_p = (float(chi2.isf(float(probability), DIMENSION)) - DIMENSION) / 4.0
    if q_p <= Q0:
        return BETA - q_p
    return BETA + KAPPA * Q0 - (1.0 + KAPPA) * q_p

def build():
    from protocol.problem import Problem

    return Problem(
        name=NAME,
        dimension=DIMENSION,
        adverse="lower",
        response=response,
        reference_probability=reference_probability(),
        recipe="deep",
    )
