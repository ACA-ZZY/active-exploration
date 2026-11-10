"""Freeze the selected seeds, audit them, and assemble the level table"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DATA, DataConfig
from .metrics import LevelMetrics, dynamic_range_retention, level_metrics, unresolved_levels
from .problem import Direction, ResponseFunction
from .sampler import SubsetRun

@dataclass
class AuditedLevel:
    level: int
    inputs: np.ndarray
    proxy_values: np.ndarray
    true_values: np.ndarray
    to_train: np.ndarray          # row indices within this level bound for training
    to_fit: np.ndarray            # row indices bound for the error-model fitting set

@dataclass
class ChainAudit:
    levels: list[AuditedLevel] = field(default_factory=list)
    metrics: list[LevelMetrics] = field(default_factory=list)
    dynamic_range_retention: float = float("nan")
    true_evaluations: int = 0
    unique_true_inputs: int = 0

    @property
    def unresolved(self) -> list[int]:
        return unresolved_levels(self.metrics)

    def training_tail(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([lv.inputs[lv.to_train] for lv in self.levels])
        y = np.concatenate([lv.true_values[lv.to_train] for lv in self.levels])
        return x, y

    def fitting_tail(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.concatenate([lv.inputs[lv.to_fit] for lv in self.levels])
        y = np.concatenate([lv.true_values[lv.to_fit] for lv in self.levels])
        p = np.concatenate([lv.proxy_values[lv.to_fit] for lv in self.levels])
        return x, y, p

    def table(self) -> list[dict]:
        return [m.as_row() for m in self.metrics]

def audit_chain(
    proxy_run: SubsetRun,
    reference_run: SubsetRun,
    true_response: ResponseFunction,
    adverse: Direction,
    rng: np.random.Generator,
    cfg: DataConfig = DATA,
) -> ChainAudit:
    """Audit every level of a surrogate chain against the reference thresholds."""

    if len(proxy_run.levels) != len(reference_run.levels):
        raise ValueError("proxy and reference chains must have the same depth")

    audit = ChainAudit()
    for proxy_level, reference_level in zip(proxy_run.levels, reference_run.levels):
        take = cfg.audit_per_level
        inputs = proxy_level.seeds[:take]
        proxy_values = proxy_level.seed_values[:take]
        if inputs.shape[0] != take:
            raise ValueError(
                f"level {proxy_level.index}: expected {take} selected seeds, "
                f"got {inputs.shape[0]}"
            )

        true_values = np.asarray(true_response(inputs), dtype=float).reshape(-1)
        audit.true_evaluations += take

        split = rng.permutation(take)
        audit.levels.append(
            AuditedLevel(
                level=proxy_level.index,
                inputs=inputs.copy(),
                proxy_values=proxy_values.copy(),
                true_values=true_values,
                to_train=split[: cfg.audit_to_train],
                to_fit=split[cfg.audit_to_train : cfg.audit_to_train + cfg.audit_to_fit],
            )
        )
        audit.metrics.append(
            level_metrics(
                level=proxy_level.index,
                nominal_probability=proxy_level.nominal_probability,
                proxy_values=proxy_values,
                true_values=true_values,
                proxy_cutoff=proxy_level.threshold,
                reference_cutoff=reference_level.threshold,
                adverse=adverse,
            )
        )

    audit.dynamic_range_retention = dynamic_range_retention(
        proxy_run.thresholds, reference_run.thresholds
    )
    if audit.levels:
        all_inputs = np.vstack([level.inputs for level in audit.levels])
        audit.unique_true_inputs = int(np.unique(all_inputs, axis=0).shape[0])
    return audit
