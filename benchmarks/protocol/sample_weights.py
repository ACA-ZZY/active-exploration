"""Differentiated sampling weights."""

from __future__ import annotations

import numpy as np

def _neighbour_distance(y: np.ndarray) -> np.ndarray:
    """Local spacing of each sample once the group is ranked by response."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size < 2:
        return np.ones_like(y)
    order = np.argsort(y)
    gaps = np.diff(y[order])
    spacing = np.empty(y.size, dtype=float)
    spacing[0] = gaps[0]
    spacing[-1] = gaps[-1]
    if y.size > 2:
        spacing[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
    out = np.empty_like(spacing)
    out[order] = spacing
    return out

def sparsity_weights(y: np.ndarray, groups: np.ndarray | None = None, *,
                     cap: float = 5.0, gamma: float = 1.5) -> np.ndarray:
    """Differentiated sampling weights with the ratio capped"""
    y = np.asarray(y, dtype=float).reshape(-1)
    groups = np.zeros(y.size, dtype=int) if groups is None else np.asarray(groups).reshape(-1)
    weights = np.empty(y.size, dtype=float)
    for g in np.unique(groups):
        mask = groups == g
        spacing = _neighbour_distance(y[mask])
        mean = spacing.mean()
        ratio = spacing / mean if mean > 0 else np.ones_like(spacing)
        weights[mask] = 0.5 + 0.5 * np.clip(ratio, 0.0, cap) ** gamma
    return weights / weights.mean()

def quantile_weights(y: np.ndarray, groups: np.ndarray | None = None, *,
                     adverse: str, scale: float = 4.0,
                     gamma: float = 1.5) -> np.ndarray:
    """Weight by how adverse the response is, not by how isolated it is"""
    if adverse not in ("upper", "lower"):
        raise ValueError(f"adverse must be 'upper' or 'lower', got {adverse!r}")
    y = np.asarray(y, dtype=float).reshape(-1)
    ordering = y if adverse == "upper" else -y
    groups = np.zeros(y.size, dtype=int) if groups is None else np.asarray(groups).reshape(-1)
    weights = np.empty(y.size, dtype=float)
    for g in np.unique(groups):
        mask = groups == g
        n = int(mask.sum())
        rank = np.argsort(np.argsort(ordering[mask])) / max(n - 1, 1)
        weights[mask] = 0.5 + 0.5 * scale * rank ** gamma
    return weights / weights.mean()

SCHEMES = {"sparsity": sparsity_weights, "quantile": quantile_weights}

def build(scheme: str | None, y, groups=None, **kwargs) -> np.ndarray | None:
    """Dispatch, passing each scheme only the knobs it actually has"""
    if not scheme:
        return None
    if scheme not in SCHEMES:
        raise ValueError(f"unknown sampling scheme {scheme!r}; have {sorted(SCHEMES)}")
    fn = SCHEMES[scheme]
    import inspect

    params = inspect.signature(fn).parameters
    if "adverse" in params and kwargs.get("adverse") is None:
        raise ValueError(
            f"the {scheme!r} scheme ranks by how adverse a row is, so it needs "
            "the problem's adverse direction; construct the Surrogate with "
            "adverse=problem.adverse")
    return fn(y, groups, **{k: v for k, v in kwargs.items() if k in params})

def effective_sample_size(weights: np.ndarray | None) -> float:
    """Kish's measure: how many evenly drawn samples this weighting is worth."""
    if weights is None:
        return float("nan")
    w = np.asarray(weights, dtype=float).reshape(-1)
    return float(w.sum() ** 2 / np.sum(w ** 2))
