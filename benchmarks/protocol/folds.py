"""Ten-fold split of the initial dataset"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DATA, DataConfig

@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray
    fit: np.ndarray
    validation: np.ndarray

    def check(self, total: int, cfg: DataConfig) -> None:
        parts = [self.train, self.fit, self.validation]
        joined = np.concatenate(parts)
        if joined.size != total or np.unique(joined).size != total:
            raise ValueError(f"fold {self.index}: split is not a partition")
        if self.fit.size != cfg.fit_size:
            raise ValueError(f"fold {self.index}: fitting set must stay at {cfg.fit_size}")

def make_folds(
    n_samples: int, rng: np.random.Generator, cfg: DataConfig = DATA
) -> list[Fold]:
    """Partition ``n_samples`` indices into ten training/fitting/validation folds"""

    if n_samples < cfg.holdout_size + cfg.n_folds:
        raise ValueError("dataset too small for the frozen split")

    order = rng.permutation(n_samples)
    holdout = cfg.holdout_size
    stride = max(1, n_samples // cfg.n_folds)
    folds: list[Fold] = []

    for index in range(cfg.n_folds):
        lo = (index * stride) % n_samples
        take = np.arange(lo, lo + holdout) % n_samples
        held = order[take]
        mask = np.ones(n_samples, dtype=bool)
        mask[take] = False
        validation = held[: cfg.validation_size]
        fit = held[cfg.validation_size :]
        train = order[mask]
        fold = Fold(index=index, train=train, fit=fit, validation=validation)
        fold.check(n_samples, cfg)
        folds.append(fold)

    return folds
