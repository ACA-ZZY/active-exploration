"""Published 2-D cantilever-beam reliability benchmark."""

from __future__ import annotations

import numpy as np

NAME = "cantilever_beam_2d"
DIMENSION = 2

_MODULUS_PA = 2.6e4 * 1.0e6        # 2.6e4 MPa in Pa
_SPAN_MM = 6.0e3

CONFIG = {'case': {'slug': 'cantilever_beam_2d',
          'name': 'cantilever_beam_2d',
          'dimension': 2,
          'adverse': 'lower',
          'status': 'ready'},
 'seeds': {'root': 20260826,
           'unconditional': 1,
           'proxy_chain': 2,
           'reference_chain': 3,
           'audit': 4,
           'error_model': 5}}

def inputs(x: np.ndarray) -> np.ndarray:
    """Map latent N(0,1)^2 to (W [Pa], H [mm])."""
    z = np.atleast_2d(np.asarray(x, dtype=float))
    return np.column_stack([1000.0 + 200.0 * z[:, 0], 250.0 + 37.5 * z[:, 1]])

def inputs_log(x: np.ndarray) -> np.ndarray:
    """Log-physical coordinates (log W, log H)."""
    phys = inputs(x)
    out = np.asarray(phys, dtype=float).copy()
    out[:, 0] = np.log(out[:, 0])          # log W
    out[:, 1] = np.log(out[:, 1])          # log H
    return out

def response(x: np.ndarray) -> np.ndarray:
    w, h = inputs(x)[:, 0], inputs(x)[:, 1]
    return (_SPAN_MM / 325.0
            - 12.0 / 8.0 * _SPAN_MM ** 4 / _MODULUS_PA * w / h ** 3)

def build():
    from protocol.problem import Problem

    return Problem(
        name=NAME,
        dimension=DIMENSION,
        adverse="lower",
        response=response,
        recipe="network_16x64x16_refine_lr1e3_e200",
        units="mm",
        feature_map=inputs,
        subset_sampler="joint_rw_amh",
    )
