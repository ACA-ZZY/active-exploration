"""Hysteretic oscillator, forty-eight dimensions, the surrogate experiment lane."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

NAME = "hysteretic_lab"
DIMENSION = 48

TAIL_INCREMENT_RATIOS = (1.10, 1.13, 1.48, 2.51)
YIELD_FRACTION_BY_DEPTH = {"all": 0.0220, "0.1": 0.22, "0.01": 0.74,
                           "0.001": 0.94, "0.0001": 1.00}

CONFIG = {'case': {'slug': 'hysteretic-lab',
          'name': 'hysteretic_lab',
          'dimension': 48,
          'adverse': 'upper',
          'status': 'ready'},
 'seeds': {'root': 20260826,
           'unconditional': 1,
           'proxy_chain': 2,
           'reference_chain': 3,
           'audit': 4,
           'error_model': 5}}

def spec():
    import hysteretic_oscillator as solver

    return solver.DEFAULT_SPEC

def response(x: np.ndarray) -> np.ndarray:
    """Peak displacement for a batch of 48-dimensional excitation coefficients."""
    import hysteretic_oscillator as solver

    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != DIMENSION:
        raise ValueError(f"{NAME}: expected {DIMENSION} inputs per row")
    return solver.response(x)

def diagnostics(x: np.ndarray):
    """Yielding and ductility, for blind-spot auditing only."""
    import hysteretic_oscillator as solver

    return solver.evaluate_batch(np.atleast_2d(np.asarray(x, dtype=float)))

DECIMATION = 4        # 50 Hz -> 12.5 Hz; the band needs 4.8 Hz

def excitation_history(x: np.ndarray) -> np.ndarray:
    """The ground acceleration record itself, decimated four times"""
    import hysteretic_oscillator as solver

    x = np.atleast_2d(np.asarray(x, dtype=float))
    return solver.excitation(x, solver.DEFAULT_SPEC)[:, ::DECIMATION]

def build():
    from protocol.problem import Problem

    return Problem(
        name=NAME,
        dimension=DIMENSION,
        adverse="upper",
        response=response,
        reference_probability=None,   # produced by the reference chain
        recipe="recurrent",
        sequence_map=excitation_history,
        subset_sampler="componentwise_amh",
    )
