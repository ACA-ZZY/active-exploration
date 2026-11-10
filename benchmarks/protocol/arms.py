"""The arms: base learner, unconditional baselines, refined learners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audit import ChainAudit, audit_chain
from .config import DATA, SAMPLER, DataConfig
from .error_model import Selection, select_error_model
from .folds import Fold
from .ledger import Ledger
from .metrics import LevelMetrics
from .probability import ProbabilityEstimate, estimate
from .problem import Direction, Problem
from .resolution import Resolution
from .sampler import SubsetRun, run_subset_simulation
from .surrogate import Surrogate

@dataclass
class ArmResult:
    arm: str
    fold: int
    recipe: str
    fold_lineage: int          # distinct inputs that trained THIS fold's model
    dataset_lineage: int       # distinct labels the arm's construction consumed
    training_calls: list[int]
    proxy_run: SubsetRun
    audit: ChainAudit
    error_model: Selection
    probabilities: list[ProbabilityEstimate]
    ledger: Ledger
    acquisition_audit: ChainAudit | None = None

    @property
    def metrics(self) -> list[LevelMetrics]:
        return self.audit.metrics

    def summary(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "fold": self.fold,
            "recipe": self.recipe,
            "fold_lineage": self.fold_lineage,
            "dataset_lineage": self.dataset_lineage,
            "training_calls": list(self.training_calls),
            "dynamic_range_retention": self.audit.dynamic_range_retention,
            "unresolved_levels": self.audit.unresolved,
            "error_family": self.error_model.chosen.family,
            "error_family_ranking": self.error_model.report(),
            "levels": self.audit.table(),
            "probabilities": [p.as_row() for p in self.probabilities],
            **self.ledger.as_row(),
        }

def _fit_error_model(
    surrogate: Surrogate,
    problem: Problem,
    pool: np.ndarray,
    pool_values: np.ndarray,
    fold: Fold,
    extra_x: np.ndarray | None = None,
    extra_y: np.ndarray | None = None,
) -> Selection:
    """Residuals ``e = D_hat - D_true`` on data outside the training lineage."""
    fit_x = pool[fold.fit]
    fit_y = pool_values[fold.fit]
    if extra_x is not None and extra_x.size:
        fit_x = np.vstack([fit_x, extra_x])
        fit_y = np.concatenate([fit_y, extra_y])
    fit_residuals = surrogate.predict(fit_x) - fit_y

    validation_x = pool[fold.validation]
    validation_residuals = surrogate.predict(validation_x) - pool_values[fold.validation]
    return select_error_model(fit_residuals, validation_residuals)

def _evaluate(
    *,
    arm: str,
    fold: Fold,
    surrogate: Surrogate,
    problem: Problem,
    reference: SubsetRun,
    reference_thresholds: np.ndarray,
    resolution: Resolution,
    selection: Selection,
    rng: np.random.Generator,
    ledger: Ledger,
) -> tuple[SubsetRun, ChainAudit, list[ProbabilityEstimate]]:
    """Run the arm's own chain, audit it, and price the reference thresholds."""
    proxy_run = run_subset_simulation(
        surrogate, dimension=problem.dimension, adverse=problem.adverse, rng=rng,
        kernel=getattr(problem, "subset_sampler", "pcn"),
    )
    audit = audit_chain(proxy_run, reference, problem.evaluate, problem.adverse, rng)
    ledger.record("evaluation_audit", audit.true_evaluations)

    terminal = proxy_run.levels[-1].values
    probabilities = [
        estimate(
            terminal_values=terminal,
            threshold=float(threshold),
            level_index=SAMPLER.n_populations - 1,
            model=selection.chosen,
            adverse=problem.adverse,
            resolution=resolution,
        )
        for threshold in reference_thresholds
    ]
    return proxy_run, audit, probabilities

def build_base_arm(
    *,
    name: str,
    fold: Fold,
    problem: Problem,
    pool: np.ndarray,
    pool_values: np.ndarray,
    reference: SubsetRun,
    resolution: Resolution,
    seeds: dict[str, int],
    cfg: DataConfig = DATA,
) -> tuple[ArmResult, Surrogate]:
    """A, or any ``B_N``: one training call, no acquisition."""
    ledger = Ledger()
    ledger.record("initial_dataset", pool.shape[0])

    surrogate = Surrogate(seed=seeds["model"], recipe=problem.recipe,
                          adverse=problem.adverse,
                          feature_map=problem.feature_map,
                          sequence_map=problem.sequence_map).fit(
        pool[fold.train], pool_values[fold.train]
    )
    selection = _fit_error_model(surrogate, problem, pool, pool_values, fold)

    proxy_run, audit, probabilities = _evaluate(
        arm=name,
        fold=fold,
        surrogate=surrogate,
        problem=problem,
        reference=reference,
        reference_thresholds=reference.thresholds,
        resolution=resolution,
        selection=selection,
        rng=np.random.default_rng(seeds["evaluate"]),
        ledger=ledger,
    )
    result = ArmResult(
        arm=name,
        fold=fold.index,
        recipe=problem.recipe,
        fold_lineage=int(fold.train.size),
        dataset_lineage=int(pool.shape[0]),
        training_calls=list(surrogate.training_calls),
        proxy_run=proxy_run,
        audit=audit,
        error_model=selection,
        probabilities=probabilities,
        ledger=ledger,
    )
    return result, surrogate

def build_refined_arm(
    *,
    fold: Fold,
    problem: Problem,
    pool: np.ndarray,
    pool_values: np.ndarray,
    base: Surrogate,
    reference: SubsetRun,
    resolution: Resolution,
    seeds: dict[str, int],
    cfg: DataConfig = DATA,
    arm_name: str = "C",
    carried_fit_x: np.ndarray | None = None,
    carried_fit_y: np.ndarray | None = None,
    carried_tail_x: np.ndarray | None = None,
    carried_tail_y: np.ndarray | None = None,
    full_retrain: bool = False,
) -> tuple[ArmResult, Surrogate, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One closed-loop refinement from ``base``."""
    ledger = Ledger()
    ledger.record("initial_dataset", pool.shape[0])

    frozen = base.clone_frozen(seed=seeds["model"])
    acquisition_rng = np.random.default_rng(seeds["acquire"])
    acquisition_run = run_subset_simulation(
        frozen, dimension=problem.dimension, adverse=problem.adverse,
        rng=acquisition_rng,
        kernel=getattr(problem, "subset_sampler", "pcn"),
    )
    acquisition = audit_chain(
        acquisition_run, reference, problem.evaluate, problem.adverse, acquisition_rng
    )
    ledger.record("acquisition_audit", acquisition.true_evaluations)

    tail_x, tail_y = acquisition.training_tail()
    fit_x, fit_y, _ = acquisition.fitting_tail()

    if carried_tail_x is not None and carried_tail_x.size:
        all_tail_x = np.vstack([carried_tail_x, tail_x])
        all_tail_y = np.concatenate([carried_tail_y, tail_y])
    else:
        all_tail_x, all_tail_y = tail_x, tail_y

    if full_retrain:
        use_tail_x, use_tail_y = all_tail_x, all_tail_y
    else:
        use_tail_x, use_tail_y = tail_x, tail_y
    n_unconditional = use_tail_x.shape[0]
    if n_unconditional > fold.train.size:
        raise ValueError("not enough training rows to balance the tail labels")
    reuse = np.random.default_rng(seeds["reuse"]).permutation(fold.train.size)
    reuse_idx = fold.train[reuse[:n_unconditional]]
    refine_x = np.vstack([pool[reuse_idx], use_tail_x])
    refine_y = np.concatenate([pool_values[reuse_idx], use_tail_y])
    if not full_retrain and refine_x.shape[0] != cfg.finetune_rows:
        raise ValueError(
            f"refinement call must hold {cfg.finetune_rows} rows, "
            f"got {refine_x.shape[0]}"
        )

    refined = base.clone_frozen(seed=seeds["model"])
    refined.continue_fit(refine_x, refine_y, new_labels=tail_x.shape[0])

    if carried_fit_x is not None and carried_fit_x.size:
        fit_x = np.vstack([carried_fit_x, fit_x])
        fit_y = np.concatenate([carried_fit_y, fit_y])
    selection = _fit_error_model(
        refined, problem, pool, pool_values, fold, extra_x=fit_x, extra_y=fit_y
    )
    proxy_run, audit, probabilities = _evaluate(
        arm=arm_name,
        fold=fold,
        surrogate=refined,
        problem=problem,
        reference=reference,
        reference_thresholds=reference.thresholds,
        resolution=resolution,
        selection=selection,
        rng=np.random.default_rng(seeds["evaluate"]),
        ledger=ledger,
    )
    result = ArmResult(
        arm=arm_name,
        fold=fold.index,
        recipe=problem.recipe,
        fold_lineage=int(base.lineage + tail_x.shape[0]),
        dataset_lineage=int(pool.shape[0] + refined.lineage - base.lineage
                            + (base.lineage - fold.train.size)),
        training_calls=list(refined.training_calls),
        proxy_run=proxy_run,
        audit=audit,
        error_model=selection,
        probabilities=probabilities,
        ledger=ledger,
        acquisition_audit=acquisition,
    )
    return result, refined, fit_x, fit_y, all_tail_x, all_tail_y
