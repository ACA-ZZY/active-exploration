"""Tail kink problem, fifty dimensions"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

BETA = 4.8
KAPPA = 1.0
S0 = 3.2
DIMENSION = 50
NAME = "tail_kink"

CONFIG = {'case': {'slug': 'tail-kink',
          'name': 'tail_kink',
          'dimension': 50,
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
    s = x.sum(axis=1) / np.sqrt(DIMENSION)
    return BETA - s - KAPPA * np.maximum(s - S0, 0.0)

def reference_probability() -> float:
    s_star = (BETA + KAPPA * S0) / (1.0 + KAPPA)
    return float(norm.sf(s_star))

def failure_probability(threshold: float) -> float:
    """Exact ``P[g <= t]``."""
    t = float(threshold)
    s_t = BETA - t if t >= BETA - S0 else (BETA + KAPPA * S0 - t) / (1.0 + KAPPA)
    return float(norm.sf(s_t))

def threshold_for_probability(probability: float) -> float:
    """Exact inverse of :func:`failure_probability`."""
    s_p = float(norm.isf(float(probability)))
    if s_p <= S0:
        return BETA - s_p
    return BETA + KAPPA * S0 - (1.0 + KAPPA) * s_p

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
