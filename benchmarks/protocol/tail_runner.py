"""Case orchestration under the tail-holdout organisation."""

from __future__ import annotations

import csv, json, platform, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import resolution as resolution_module
from .audit import audit_chain
from .config import SAMPLER, TAIL_HOLDOUT, SamplerConfig, TailHoldoutConfig
from .error_model import select_error_model
from .ledger import Ledger
from .metrics import dynamic_range_retention, level_metrics
from .probability import estimate
from .problem import Problem
from .sampler import SubsetRun, run_subset_simulation
from .surrogate import Surrogate
from .tail_holdout import build_arms

STREAM = {"pool": 1, "reference": 2, "model": 4, "acquire": 5,
          "evaluate": 6, "reuse": 7, "holdout": 8, "audit": 9}

def _spawn(root: int, stream: str, *tags: int) -> int:
    return int(np.random.SeedSequence([root, STREAM[stream], *tags])
               .generate_state(1, dtype=np.uint32)[0])

@dataclass
class ArmScore:
    arm: str
    replicate: int
    recipe: str
    error_family: str
    n_fit: int
    metrics: list
    acceptance_rates: list[float | None]
    dynamic_range: float
    probabilities: list
    training_calls: list
    dataset_lineage: int

@dataclass
class TailCaseResult:
    problem: str
    reference: SubsetRun
    scores: list[ArmScore] = field(default_factory=list)
    ledger: Ledger = field(default_factory=Ledger)
    acquisition_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def rows(self) -> list[dict[str, Any]]:
        out = []
        for s in self.scores:
            for m in s.metrics:
                r = m.as_row()
                rates = getattr(s, "acceptance_rates", ())
                observed = rates[m.level] if m.level < len(rates) else None
                r["acceptance_rate"] = (
                    float(observed) if observed is not None else float("nan")
                )
                r.update(arm=s.arm, replicate=s.replicate, recipe=s.recipe,
                         error_family=s.error_family, n_fit=s.n_fit,
                         dataset_lineage=s.dataset_lineage)
                out.append(r)
        return out

    def aggregate(self) -> list[dict[str, Any]]:
        keys = ("precision", "raw_mae", "mean_corrected_mae", "spearman",
                "acceptance_rate",
                "proxy_cutoff", "cutoff_gap")
        buckets: dict[tuple[str, int], list] = {}
        for s in self.scores:
            for m in s.metrics:
                row = m.as_row()
                rates = getattr(s, "acceptance_rates", ())
                observed = rates[m.level] if m.level < len(rates) else None
                row["acceptance_rate"] = (
                    float(observed) if observed is not None else float("nan")
                )
                buckets.setdefault((s.arm, m.level), []).append(row)
        out = []
        for (arm, level), rs in sorted(buckets.items()):
            e = {"arm": arm, "level": level, "n_replicates": len(rs),
                 "nominal_probability": rs[0]["nominal_probability"],
                 "reference_cutoff": rs[0]["reference_cutoff"]}
            for k in keys:
                v = np.array([r[k] for r in rs], dtype=float)
                v = v[np.isfinite(v)]
                e[f"{k}_median"] = float(np.median(v)) if v.size else float("nan")
                e[f"{k}_q25"] = float(np.quantile(v, .25)) if v.size else float("nan")
                e[f"{k}_q75"] = float(np.quantile(v, .75)) if v.size else float("nan")
            out.append(e)
        return out

def _audit_cfg(sampler):
    """The evaluation audit, sized to what a level actually has"""
    import dataclasses

    from .config import DATA

    return dataclasses.replace(DATA, audit_per_level=sampler.n_seed)

def _evaluate_chain(problem, model, reference, rng, sampler=SAMPLER,
                    audit_cfg=None):
    """One proxy chain and its audit, shared by every scoring of that model."""
    run = run_subset_simulation(model, dimension=problem.dimension,
                                adverse=problem.adverse, rng=rng, cfg=sampler,
                                kernel=getattr(problem, "subset_sampler", "pcn"))
    audit = audit_chain(run, reference, problem.evaluate, problem.adverse, rng,
                        **({} if audit_cfg is None else {"cfg": audit_cfg}))
    return run, audit

def _score(arm, rep, problem, run, audit, reference, res, sel, n_fit,
           training_calls, lineage, recipe):
    terminal = run.levels[-1].values
    probs = [estimate(terminal_values=terminal, threshold=float(t),
                      level_index=len(run.levels) - 1, model=sel.chosen,
                      adverse=problem.adverse, resolution=res).as_row()
             for t in reference.thresholds]
    return ArmScore(arm=arm, replicate=rep, recipe=recipe,
                    error_family=sel.chosen.family, n_fit=n_fit,
                    metrics=audit.metrics,
                    acceptance_rates=[level.acceptance_rate for level in run.levels],
                    dynamic_range=audit.dynamic_range_retention,
                    probabilities=probs, training_calls=training_calls,
                    dataset_lineage=lineage)

def run_tail_case(problem: Problem, *, root_seed: int,
                  baselines: Iterable[int] = (),
                  n_replicates: int | None = None,
                  replicate_start: int = 0,
                  stages: int = 1,
                  checkpoints: bool = True,
                  sampler: "SamplerConfig" = SAMPLER,
                  evaluation_sampler: "SamplerConfig | None" = None,
                  reference_sampler: "SamplerConfig | None" = None,
                  expected_reference_thresholds: Iterable[float] | None = None,
                  accumulate: bool | None = None,
                  cfg: TailHoldoutConfig = TAIL_HOLDOUT) -> TailCaseResult:
    reps = cfg.n_replicates if n_replicates is None else int(n_replicates)
    if reps < 1 or int(replicate_start) < 0:
        raise ValueError("replicate count must be positive and replicate_start non-negative")
    baselines = [int(n) for n in baselines if int(n) != cfg.train_size]
    cap = cfg.train_size * 20
    for n in baselines:
        if n > cap:
            raise ValueError(f"B_{n} exceeds the cap of {cap}")

    ledger = Ledger()
    res_main = resolution_module.declare(cfg.fit_size)
    res_diag = resolution_module.declare(cfg.fit_split[1])

    evaluation_cfg = sampler if evaluation_sampler is None else evaluation_sampler
    reference_cfg = evaluation_cfg if reference_sampler is None else reference_sampler
    samplers = {
        "acquisition": sampler,
        "evaluation": evaluation_cfg,
        "reference": reference_cfg,
    }
    if len({value.n_populations for value in samplers.values()}) != 1:
        raise ValueError("acquisition, evaluation and reference must use the same number of levels")
    if len({value.conditional_probability for value in samplers.values()}) != 1:
        raise ValueError("acquisition, evaluation and reference must use the same p0")
    reference = run_subset_simulation(problem.evaluate, dimension=problem.dimension,
                                      cfg=reference_cfg,
                                      adverse=problem.adverse,
                                      rng=np.random.default_rng(_spawn(root_seed, "reference")),
                                      kernel=getattr(problem, "subset_sampler", "pcn"))
    if expected_reference_thresholds is not None:
        expected = np.asarray(list(expected_reference_thresholds), dtype=float)
        if expected.shape != reference.thresholds.shape or not np.allclose(
            reference.thresholds, expected, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                "recomputed reference thresholds do not match the frozen manifest: "
                f"got {reference.thresholds.tolist()}, expected {expected.tolist()}"
            )
    ledger.record("reference_chain", reference.total_evaluations)

    result = TailCaseResult(problem=problem.name, reference=reference, ledger=ledger)

    for rep in range(int(replicate_start), int(replicate_start) + reps):
        built = build_arms(problem, reference,
                           np.random.default_rng(_spawn(root_seed, "pool", rep)),
                           {"model": _spawn(root_seed, "model", rep),
                            "acquire": _spawn(root_seed, "acquire", rep),
                            "reuse": _spawn(root_seed, "reuse", rep),
                            "holdout": _spawn(root_seed, "holdout", rep)},
                           cfg=cfg,
                           stages=stages, checkpoints=checkpoints, sampler=sampler,
                           accumulate=accumulate)
        ledger.record("initial_dataset", cfg.initial_size)
        harvests = [(1, built["harvest"])]
        for k in range(2, int(stages) + 1):
            if f"harvest{k}" in built:
                harvests.append((k, built[f"harvest{k}"]))
        for loop, harvested in harvests:
            audited_x = np.vstack([harvested.train_x, harvested.holdout_x])
            unique_audited = int(np.unique(audited_x, axis=0).shape[0])
            per_level_train_unique = [
                int(np.unique(
                    harvested.train_x[i * cfg.seeds_per_level:
                                      (i + 1) * cfg.seeds_per_level], axis=0
                ).shape[0])
                for i in range(sampler.n_populations)
            ]
            per_level_holdout_unique = [
                int(np.unique(
                    harvested.holdout_x[i * cfg.extra_per_level:
                                        (i + 1) * cfg.extra_per_level], axis=0
                ).shape[0])
                for i in range(sampler.n_populations)
            ]
            result.acquisition_diagnostics.append({
                "replicate": rep,
                "loop": loop,
                "rows": int(harvested.true_calls),
                "unique_inputs": unique_audited,
                "per_level_train_rows": cfg.seeds_per_level,
                "per_level_train_unique": per_level_train_unique,
                "per_level_holdout_rows": cfg.extra_per_level,
                "per_level_holdout_unique": per_level_holdout_unique,
                "per_level_precision": [
                    float(value) for value in harvested.per_level_precision
                ],
            })
            ledger.record("acquisition_audit", harvested.true_calls,
                          unique=unique_audited)

        def err(model, fx, fy, vx, vy):
            return select_error_model(model.predict(fx) - fy, model.predict(vx) - vy)

        tf, uf = cfg.fit_split
        tv, uv = cfg.validation_split
        unc_fx = built["fit_x"][tf:]; unc_fy = built["fit_y"][tf:]
        unc_vx = built["validation_x"][tv:]; unc_vy = built["validation_y"][tv:]

        chains = {}
        chain_specs = [("A", built["base"], 0), ("C", built["refined"], 1)]
        for k in range(2, int(stages) + 1):
            if f"refined{k}" in built:
                chain_specs.append((f"C{k}", built[f"refined{k}"], k))
        for tag, model, slot in chain_specs:
            run, audit = _evaluate_chain(
                problem, model, reference,
                np.random.default_rng(_spawn(root_seed, "evaluate", rep, slot)),
                sampler=evaluation_cfg, audit_cfg=_audit_cfg(evaluation_cfg))
            chains[tag] = (run, audit)
            ledger.record("evaluation_audit", audit.true_evaluations,
                          unique=audit.unique_true_inputs)

        for arm, chain_tag, model, lin, fx, fy, vx, vy, res, nfit in (
            ("A", "A", built["base"], cfg.initial_size,
             unc_fx, unc_fy, unc_vx, unc_vy, res_diag, uf),
            ("C", "C", built["refined"], cfg.initial_size + cfg.tail_train,
             built["fit_x"], built["fit_y"], built["validation_x"],
             built["validation_y"], res_main, cfg.fit_size),
            ("C_uncond", "C", built["refined"], cfg.initial_size + cfg.tail_train,
             unc_fx, unc_fy, unc_vx, unc_vy, res_diag, uf),
        ) + tuple(
            (f"C{k}", f"C{k}", built[f"refined{k}"],
             cfg.initial_size + k * cfg.tail_train,
             built[f"fit{k}_x"], built[f"fit{k}_y"],
             built[f"validation{k}_x"], built[f"validation{k}_y"],
             res_main, cfg.fit_size)
            for k in range(2, int(stages) + 1) if f"refined{k}" in built):
            run, audit = chains[chain_tag]
            sel = err(model, fx, fy, vx, vy)
            result.scores.append(
                _score(arm, rep, problem, run, audit, reference, res, sel, nfit,
                       list(model.training_calls), lin, problem.recipe))

        for order, n in enumerate(baselines, start=10):
            prng = np.random.default_rng(_spawn(root_seed, "pool", rep, n))
            bx = prng.standard_normal((n + cfg.holdout_size, problem.dimension))
            by = problem.evaluate(bx)
            ledger.record("initial_dataset", bx.shape[0])
            bm = Surrogate(seed=_spawn(root_seed, "model", rep, n),
                           adverse=problem.adverse,
                           recipe=problem.recipe,
                           feature_map=problem.feature_map,
                          sequence_map=problem.sequence_map).fit(bx[:n], by[:n])
            hf, hv = bx[n:n + uf], bx[n + uf:n + uf + uv]
            fit_prediction = bm.predict(hf)
            validation_prediction = bm.predict(hv)
            if not (np.all(np.isfinite(fit_prediction)) and
                    np.all(np.isfinite(validation_prediction))):
                raise RuntimeError(
                    f"B_{n} replicate {rep}: recurrent surrogate produced "
                    f"non-finite holdout predictions "
                    f"(fit {np.count_nonzero(~np.isfinite(fit_prediction))}/"
                    f"{fit_prediction.size}, validation "
                    f"{np.count_nonzero(~np.isfinite(validation_prediction))}/"
                    f"{validation_prediction.size}); error calibration is invalid"
                )
            sel = select_error_model(
                fit_prediction - by[n:n + uf],
                validation_prediction - by[n + uf:n + uf + uv],
            )
            run, audit = _evaluate_chain(
                problem, bm, reference,
                np.random.default_rng(_spawn(root_seed, "evaluate", rep, order)),
                sampler=evaluation_cfg, audit_cfg=_audit_cfg(evaluation_cfg))
            ledger.record("evaluation_audit", audit.true_evaluations,
                          unique=audit.unique_true_inputs)
            result.scores.append(
                _score(f"B_{n}", rep, problem, run, audit, reference, res_diag,
                       sel, uf, list(bm.training_calls), n, problem.recipe))

    return result

def _closed_loops(result: TailCaseResult) -> int:
    """How many closed loops the run actually completed"""
    import re

    return len({s.arm for s in result.scores if re.fullmatch(r"C\d*", s.arm)})

def _torch_environment() -> dict:
    """Torch version and device used by the recipes, when relevant."""
    import sys as _sys

    torch = _sys.modules.get("torch")
    if torch is None:
        return {}
    from .surrogate import resolve_device

    env = {"torch": torch.__version__, "device": resolve_device()}
    if env["device"].startswith("cuda") and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env

def write_tail_results(result: TailCaseResult, out_dir: Path, root_seed: int,
                       cfg: TailHoldoutConfig = TAIL_HOLDOUT,
                       sampler: SamplerConfig = SAMPLER,
                       evaluation_sampler: SamplerConfig | None = None,
                       reference_sampler: SamplerConfig | None = None,
                       provenance: dict | None = None,
                       accumulate: bool | None = None) -> None:
    """``accumulate`` mirrors ``run_tail_case``'s argument of the same name"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = result.rows()
    with (out_dir / "level_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    agg = result.aggregate()
    with (out_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
    import dataclasses

    def sampler_record(value: SamplerConfig) -> dict:
        record = dataclasses.asdict(value)
        record.update(n_seed=value.n_seed, chain_length=value.chain_length,
                      n_steps=value.n_steps)
        return record

    evaluation_cfg = sampler if evaluation_sampler is None else evaluation_sampler
    reference_cfg = evaluation_cfg if reference_sampler is None else reference_sampler
    manifest = {
        "problem": result.problem, "organisation": cfg.name, "root_seed": root_seed,
        "provenance": {} if provenance is None else provenance,
        "surrogate_recipe": result.scores[0].recipe if result.scores else None,
        "reference_thresholds": result.reference.thresholds.tolist(),
        "acquisition_diagnostics": result.acquisition_diagnostics,
        "acquisition_sampler": sampler_record(sampler),
        "evaluation_sampler": sampler_record(evaluation_cfg),
        "reference_sampler": sampler_record(reference_cfg),
        "data_organisation": {
            "initial_size": cfg.initial_size, "train_size": cfg.train_size,
            "unconditional_holdout": cfg.unconditional_holdout,
            "audit_per_level": cfg.audit_per_level,
            "seeds_per_level": cfg.seeds_per_level,
            "extra_per_level": cfg.extra_per_level,
            "acquisition_calls_per_loop": cfg.acquisition_calls,
            "evaluation_audit_per_level": evaluation_cfg.n_seed,
            "evaluation_audit_per_chain": (
                evaluation_cfg.n_seed * evaluation_cfg.n_populations
            ),
            "tail_train": cfg.tail_train, "tail_holdout": cfg.tail_holdout,
            "refinement_rows": cfg.refinement_rows,
            "fit_split_tail_unconditional": list(cfg.fit_split),
            "validation_split_tail_unconditional": list(cfg.validation_split),
            "deployment_calls": result.ledger.deployment,
            "closed_loops": _closed_loops(result),
            "weight_group": getattr(cfg, "weight_group", "call"),
            "n_replicates": (max((s.replicate for s in result.scores),
                                 default=-1) + 1),
            "n_replicates_configured": cfg.n_replicates,
            "accumulate_harvests": (cfg.accumulate_harvests if accumulate is None
                                    else bool(accumulate)),
        },
        "resolution": {
            "main_n_fit": cfg.fit_size,
            "main_overall": resolution_module.declare(cfg.fit_size).overall,
            "diagnostic_n_fit": cfg.fit_split[1],
            "diagnostic_overall": resolution_module.declare(cfg.fit_split[1]).overall,
            "note": (f"C uses the tail-informed model fitted on {cfg.fit_size} rows; "
                     f"A, B_N and C_uncond use unconditional-only models fitted on "
                     f"{cfg.fit_split[1]}, so their resolution is coarser and the two "
                     "are not compared head-to-head."),
        },
        "ledger": result.ledger.as_row(),
        "replicate_note": ("Independent replicates, not folds: each draws its own "
                           "dataset and acquisition stream. Not a confidence interval; "
                           "no paired test across replicates."),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform(),
                        **_torch_environment()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
