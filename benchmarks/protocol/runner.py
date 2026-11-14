"""Case orchestration"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import resolution as resolution_module
from .arms import ArmResult, build_base_arm, build_refined_arm
from .config import BASELINE, DATA, SAMPLER
from .folds import make_folds
from .ledger import Ledger
from .problem import Problem
from .sampler import SubsetRun, run_subset_simulation

STREAM = {
    "unconditional": 1,
    "reference_chain": 2,
    "folds": 3,
    "model": 4,
    "acquire": 5,
    "evaluate": 6,
    "reuse": 7,
}

@dataclass
class CaseResult:
    problem: str
    reference: SubsetRun
    arms: list[ArmResult]
    ledger: Ledger
    resolution: Any

    def fold_table(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for arm in self.arms:
            for level in arm.audit.metrics:
                row = level.as_row()
                row.update(arm=arm.arm, fold=arm.fold,
                           fold_lineage=arm.fold_lineage,
                           dataset_lineage=arm.dataset_lineage)
                rows.append(row)
        return rows

    def aggregate(self) -> list[dict[str, Any]]:
        """Median and fold quantiles per arm and level."""
        keys = (
            "precision",
            "raw_mae",
            "mean_corrected_mae",
            "spearman",
            "proxy_cutoff",
            "cutoff_gap",
        )
        buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for arm in self.arms:
            for level in arm.audit.metrics:
                buckets.setdefault((arm.arm, level.level), []).append(level.as_row())

        summary: list[dict[str, Any]] = []
        for (arm_name, level_index), rows in sorted(buckets.items()):
            entry: dict[str, Any] = {
                "arm": arm_name,
                "level": level_index,
                "nominal_probability": rows[0]["nominal_probability"],
                "n_folds": len(rows),
                "reference_cutoff": rows[0]["reference_cutoff"],
            }
            for key in keys:
                values = np.array([r[key] for r in rows], dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    entry[f"{key}_median"] = float("nan")
                    entry[f"{key}_fold_q25"] = float("nan")
                    entry[f"{key}_fold_q75"] = float("nan")
                    continue
                entry[f"{key}_median"] = float(np.median(finite))
                entry[f"{key}_fold_q25"] = float(np.quantile(finite, 0.25))
                entry[f"{key}_fold_q75"] = float(np.quantile(finite, 0.75))
            summary.append(entry)
        return summary

def _spawn(root: int, stream: str, *tags: int) -> int:
    seed = np.random.SeedSequence([root, STREAM[stream], *tags])
    return int(seed.generate_state(1, dtype=np.uint32)[0])

def run_case(
    problem: Problem,
    *,
    root_seed: int,
    baselines: Iterable[int] = (),
    n_folds: int | None = None,
    reference_run: SubsetRun | None = None,
    stages: int = 1,
) -> CaseResult:
    """Run the reference chain, then A, the requested ``B_N`` and C, fold by fold."""

    folds_wanted = DATA.n_folds if n_folds is None else int(n_folds)
    anchor = DATA.initial_size
    baselines = [int(n) for n in baselines if int(n) != anchor]
    cap = BASELINE.cap_for(anchor)
    for n in baselines:
        if n > cap:
            raise ValueError(f"B_{n} exceeds the cap of {cap} (20x anchor {anchor})")

    ledger = Ledger()
    resolution = resolution_module.declare()

    if reference_run is None:
        reference_rng = np.random.default_rng(_spawn(root_seed, "reference_chain"))
        reference_run = run_subset_simulation(
            problem.evaluate,
            dimension=problem.dimension,
            adverse=problem.adverse,
            rng=reference_rng,
            kernel=getattr(problem, "subset_sampler", "pcn"),
        )
    ledger.record("reference_chain", reference_run.total_evaluations)

    largest = max([DATA.initial_size, *baselines])
    pool_rng = np.random.default_rng(_spawn(root_seed, "unconditional"))
    pool = pool_rng.standard_normal((largest, problem.dimension))
    pool_values = problem.evaluate(pool)
    ledger.record("initial_dataset", largest)

    arms: list[ArmResult] = []
    fold_rng = np.random.default_rng(_spawn(root_seed, "folds"))
    anchor_folds = make_folds(DATA.initial_size, fold_rng)

    for fold in anchor_folds[:folds_wanted]:
        base_result, base_model = build_base_arm(
            name="A",
            fold=fold,
            problem=problem,
            pool=pool[: DATA.initial_size],
            pool_values=pool_values[: DATA.initial_size],
            reference=reference_run,
            resolution=resolution,
            seeds={
                "model": _spawn(root_seed, "model", 0, fold.index),
                "evaluate": _spawn(root_seed, "evaluate", 0, fold.index),
            },
        )
        arms.append(base_result)

        (refined_result, refined_model, carry_x, carry_y,
         carry_tail_x, carry_tail_y) = build_refined_arm(
            fold=fold,
            problem=problem,
            pool=pool[: DATA.initial_size],
            pool_values=pool_values[: DATA.initial_size],
            base=base_model,
            reference=reference_run,
            resolution=resolution,
            seeds={
                "model": _spawn(root_seed, "model", 1, fold.index),
                "acquire": _spawn(root_seed, "acquire", 1, fold.index),
                "evaluate": _spawn(root_seed, "evaluate", 1, fold.index),
                "reuse": _spawn(root_seed, "reuse", 1, fold.index),
            },
        )
        arms.append(refined_result)

        loop_model = refined_model
        for k in range(2, int(stages) + 1):
            (loop_result, loop_model, carry_x, carry_y,
             carry_tail_x, carry_tail_y) = build_refined_arm(
                fold=fold,
                problem=problem,
                pool=pool[: DATA.initial_size],
                pool_values=pool_values[: DATA.initial_size],
                base=loop_model,
                reference=reference_run,
                resolution=resolution,
                seeds={
                    "model": _spawn(root_seed, "model", 88 + k, fold.index),
                    "acquire": _spawn(root_seed, "acquire", 88 + k, fold.index),
                    "evaluate": _spawn(root_seed, "evaluate", 88 + k, fold.index),
                    "reuse": _spawn(root_seed, "reuse", 88 + k, fold.index),
                },
                arm_name=f"C{k}",
                carried_fit_x=carry_x,
                carried_fit_y=carry_y,
                carried_tail_x=carry_tail_x,
                carried_tail_y=carry_tail_y,
                full_retrain=True,
            )
            arms.append(loop_result)

        for order, n in enumerate(baselines, start=2):
            baseline_folds = make_folds(n, np.random.default_rng(_spawn(root_seed, "folds", n)))
            result, _ = build_base_arm(
                name=f"B_{n}",
                fold=baseline_folds[fold.index],
                problem=problem,
                pool=pool[:n],
                pool_values=pool_values[:n],
                reference=reference_run,
                resolution=resolution,
                seeds={
                    "model": _spawn(root_seed, "model", order, fold.index),
                    "evaluate": _spawn(root_seed, "evaluate", order, fold.index),
                },
            )
            arms.append(result)

    for arm in arms:
        for role in ("acquisition_audit", "evaluation_audit"):
            if arm.ledger.calls[role]:
                ledger.record(role, arm.ledger.calls[role])

    return CaseResult(
        problem=problem.name,
        reference=reference_run,
        arms=arms,
        ledger=ledger,
        resolution=resolution,
    )

def write_results(result: CaseResult, out_dir: Path, root_seed: int) -> None:
    """Write the level table, the aggregate and a reproducibility manifest."""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = result.fold_table()
    with (out_dir / "level_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = result.aggregate()
    with (out_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    manifest = {
        "problem": result.problem,
        "surrogate_recipe": result.arms[0].recipe if result.arms else None,
        "root_seed": root_seed,
        "reference_thresholds": result.reference.thresholds.tolist(),
        "nominal_probabilities": result.reference.nominal_probabilities.tolist(),
        "reference_dynamic_range": result.reference.dynamic_range(),
        "resolution": {
            "n_fit": result.resolution.n_fit,
            "empirical": result.resolution.empirical,
            "upper_bound": result.resolution.upper_bound,
            "overall": result.resolution.overall,
        },
        "ledger": {
            **result.ledger.as_row(),
            "single_deployment": DATA.initial_size
            + DATA.audit_per_level * SAMPLER.n_populations,
            "cross_validation_extra": result.ledger.calls["acquisition_audit"]
            - DATA.audit_per_level * SAMPLER.n_populations,
            "note": (
                "single_deployment is the headline cost of one practical run: "
                "the initial dataset plus one acquisition loop. The extra "
                "acquisition loops of cross-validation and every evaluation "
                "audit and reference call are listed apart and must not be "
                "folded into it."
            ),
        },
        "arms": [arm.summary() for arm in result.arms],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "protocol": {
            "population": SAMPLER.population,
            "conditional_probability": SAMPLER.conditional_probability,
            "n_populations": SAMPLER.n_populations,
            "target_acceptance": SAMPLER.target_acceptance,
            "seed_retained_as_first_state": True,
            "mh_steps_per_chain": SAMPLER.n_steps,
            "fold_split": [DATA.train_size, DATA.fit_size, DATA.validation_size],
            "audit_per_level": DATA.audit_per_level,
            "finetune_rows": DATA.finetune_rows,
        },
        "fold_spread_note": (
            "Quantiles are across folds of one shared dataset. Folds are not "
            "independent replicates: do not read them as confidence intervals "
            "and do not run paired tests across them."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
