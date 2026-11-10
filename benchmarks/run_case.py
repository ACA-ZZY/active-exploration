#!/usr/bin/env python3
"""Run one case under the frozen protocol"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def _bind_requested_cpu_core() -> dict | None:
    """Bind the real interpreter, not the short-lived Windows venv launcher."""
    raw = os.environ.get("PROTOCOL_CPU_CORE")
    if raw is None:
        return None
    try:
        core = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid PROTOCOL_CPU_CORE={raw!r}") from exc
    try:
        import psutil

        process = psutil.Process()
        process.cpu_affinity([core])
        applied = [int(value) for value in process.cpu_affinity()]
    except Exception as exc:
        raise RuntimeError(f"could not bind worker process to CPU core {core}: {exc}") from exc
    if applied != [core]:
        raise RuntimeError(
            f"requested CPU core {core}, worker process reports affinity {applied}"
        )
    return {"pid": int(os.getpid()), "requested_core": core, "applied_cores": applied}

PROCESS_AFFINITY = _bind_requested_cpu_core()

from protocol.config import TAIL_HOLDOUT
from protocol.runner import run_case, write_results          # noqa: E402
from protocol.tail_runner import run_tail_case, write_tail_results  # noqa: E402

def load_case(slug: str):
    case_dir = ROOT / "cases" / slug
    if not case_dir.is_dir():
        raise SystemExit(f"unknown case: {slug}")
    spec = importlib.util.spec_from_file_location(f"case_{slug}", case_dir / "problem.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return case_dir, getattr(module, "CONFIG", {}), module.build()

_MISSING = object()

def _nested_value(data: dict, path: tuple[str, ...]):
    """Return one declared expected-config value, or ``_MISSING``"""

    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current

def _same_expected_value(expected, actual) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) \
            and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual),
                            rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(
            _same_expected_value(left, right)
            for left, right in zip(expected, actual)
        )
    return expected == actual

def _uses_log_inputs(problem) -> bool:
    tag = str(getattr(problem, "feature_tag", "")).lower()
    feature = getattr(problem, "feature_map", None)
    name = str(getattr(feature, "__name__", "")).lower()
    return (
        tag == "log_inputs" or tag.startswith("log_") or tag.endswith("_log")
        or name == "inputs_log" or name.startswith("log_") or name.endswith("_log")
    )

def _validate_expected_config(
    expected: dict,
    *,
    case_slug: str,
    problem,
    root_seed: int,
    cfg,
    sampler,
    stages: int,
    baselines: list[int],
    requested_replicates: int | None,
    replicate_start: int,
    accumulate: bool,
) -> None:
    """Validate every recognised declaration before any simulation or fit"""

    if not isinstance(expected, dict):
        raise ValueError("expected-config root must be a JSON object")

    from protocol.surrogate import RECIPES

    errors: list[str] = []
    recipe_key = str(problem.recipe)
    recipe = RECIPES.get(recipe_key)
    if recipe is None:
        errors.append(f"recipe: runtime recipe {recipe_key!r} is not registered")

    base_rate = None
    refine_rate = None
    base_epochs = None
    refine_epochs = None
    recipe_family = None
    hidden_layer_sizes = None
    no_log_output = None
    if recipe is not None:
        recipe_family = str(getattr(recipe, "name", recipe_key))
        hidden_layer_sizes = getattr(recipe, "hidden_layer_sizes", None)
        if hidden_layer_sizes is not None:
            hidden_layer_sizes = [int(width) for width in hidden_layer_sizes]
        base_rate = getattr(recipe, "learning_rate_init", None)
        if base_rate is None:
            base_rate = getattr(recipe, "learning_rate", None)
        refine_rate = getattr(recipe, "refine_learning_rate", None)
        if refine_rate is None:
            refine_rate = base_rate
        base_epochs = getattr(recipe, "max_iter", None)
        refine_epochs = getattr(recipe, "refine_iter", None)
        no_log_output = not bool(
            getattr(recipe, "log_shift_target", False)
            or getattr(recipe, "log_target", False)
        )

    resolved_baselines = [
        int(size) for size in baselines if int(size) != int(cfg.train_size)
    ]
    fit_rows_by_round = [
        int(cfg.tail_train) + (
            loop * int(cfg.tail_train) if accumulate else int(cfg.tail_train)
        )
        for loop in range(1, int(stages) + 1)
    ]

    checks = [
        (("case",), case_slug),
        (("problem",), problem.name),
        (("root_seed",), int(root_seed)),
        (("recipe",), recipe_key),
        (("recipe_override",), recipe_key),
        (("no_log_inputs",), not _uses_log_inputs(problem)),
        (("no_log_output",), no_log_output),
        (("base", "train_size"), int(cfg.train_size)),
        (("base", "recipe_family"), recipe_family),
        (("base", "hidden_layer_sizes"), hidden_layer_sizes),
        (("base", "learning_rate"), base_rate),
        (("base", "frozen_learning_rate"), base_rate),
        (("base", "epochs"), base_epochs),
        (("surrogate", "recipe"), recipe_key),
        (("surrogate", "hidden_layer_sizes"), hidden_layer_sizes),
        (("surrogate", "base_learning_rate"), base_rate),
        (("surrogate", "refinement_learning_rate"), refine_rate),
        (("refinement", "learning_rate"), refine_rate),
        (("refinement", "epochs_per_round"), refine_epochs),
        (("refinement", "unconditional_replay_rows"), int(cfg.tail_train)),
        (("refinement", "tail_rows_per_round"), int(cfg.tail_train)),
        (("refinement", "accumulate_tail_rows"), bool(accumulate)),
        (("refinement", "fit_rows_by_round"), fit_rows_by_round),
        (("data_organisation", "train_size"), int(cfg.train_size)),
        (("data_organisation", "tail_train_per_loop"), int(cfg.tail_train)),
        (("data_organisation", "unconditional_replay_per_loop"),
         int(cfg.tail_train)),
        (("data_organisation", "accumulate_harvests"), bool(accumulate)),
        (("sampler", "kernel"),
         str(getattr(problem, "subset_sampler", "pcn"))),
        (("sampler", "population"), int(sampler.population)),
        (("sampler", "population_per_level"), int(sampler.population)),
        (("sampler", "conditional_probability"),
         float(sampler.conditional_probability)),
        (("sampler", "n_seed"), int(sampler.n_seed)),
        (("sampler", "n_populations"), int(sampler.n_populations)),
        (("sampler", "chain_length"), int(sampler.chain_length)),
        (("sampler", "target_acceptance"), float(sampler.target_acceptance)),
        (("arms", "A"), int(cfg.train_size)),
        (("arms", "B_N"), resolved_baselines),
        (("arms", "C_rounds"), int(stages)),
        (("arms", "C_stages"), int(stages)),
    ]
    for path, actual in checks:
        declared = _nested_value(expected, path)
        if declared is _MISSING:
            continue
        if not _same_expected_value(declared, actual):
            dotted = ".".join(path)
            errors.append(f"{dotted}: expected {declared!r}, resolved {actual!r}")

    for repeat_key in ("n_replicates", "replicates"):
        declared = _nested_value(expected, (repeat_key,))
        if declared is _MISSING:
            continue
        try:
            campaign_replicates = int(declared)
        except (TypeError, ValueError):
            errors.append(f"{repeat_key}: expected a positive integer, got {declared!r}")
            continue
        if campaign_replicates < 1:
            errors.append(f"{repeat_key}: expected a positive integer, got {declared!r}")
            continue
        if requested_replicates is None:
            configured = int(cfg.n_replicates)
            if configured != campaign_replicates:
                errors.append(
                    f"{repeat_key}: expected {campaign_replicates}, "
                    f"configured {configured}"
                )
        else:
            start = int(replicate_start)
            count = int(requested_replicates)
            if count < 1 or start < 0 or start + count > campaign_replicates:
                errors.append(
                    f"{repeat_key}: expected campaign size {campaign_replicates}, "
                    f"requested replicate slice [{start}, {start + count})"
                )

    if errors:
        detail = "\n  - ".join(errors)
        raise ValueError(f"expected-config mismatch:\n  - {detail}")

def _read_expected_config(path: str | Path) -> tuple[dict, dict]:
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse expected-config {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"expected-config {resolved} must contain a JSON object")
    record = {
        "path": str(resolved),
    }
    return data, record

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--baselines", type=int, nargs="*", default=[],
                        help=f"B_N sizes; capped at 20x the anchor")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--stages", type=int, default=1,
                        help="number of closed loops: 1 = C only, 2 = add C2. "
                             "A second loop costs a second round of acquisition "
                             "audit calls, which the ledger records")
    parser.add_argument("--organisation", choices=("tail_holdout", "legacy"),
                        default="tail_holdout",
                        help="tail_holdout is the frozen organisation; "
                             "legacy reproduces the 2026-08-24 one")
    parser.add_argument("--replicates", type=int, default=None,
                        help="independent replicates under tail_holdout")
    parser.add_argument("--replicate-start", type=int, default=0,
                        help="global replicate id of the first requested repeat; "
                             "used to shard independent repeats across processes")
    parser.add_argument("--recipe", default=None,
                        help="override the case's declared surrogate recipe; "
                             "for recipe comparisons, not for production runs")
    parser.add_argument("--no-feature-map", action="store_true",
                        help="drop the case's physics input transform; for "
                             "recipe comparisons, not for production runs")
    parser.add_argument("--log-inputs", action="store_true",
                        help="use the case's log-physical feature map "
                             "(log of the positive inputs) instead of the raw "
                             "physical inputs; for representation comparisons")
    parser.add_argument("--no-accumulate", action="store_true",
                        help="fit each closed loop on its own harvest only, "
                             "the behaviour before 2026-08-26")
    parser.add_argument("--scale", type=int, default=1,
                        help="multiply every count in the data organisation and "
                             "the sampler by this factor; the reference chain "
                             "scales with it, so a scaled run is not comparable "
                             "with an unscaled one")
    parser.add_argument("--subset-population", type=int, default=None,
                        help="change only C's acquisition population per level. "
                             "A/C/B_N evaluation stays on the pre-existing sampler; "
                             "p0 stays at 0.10 and harvested seeds follow p0, "
                             "and the original 50 extra tail holdout rows per "
                             "level are retained. Unlike --scale, the base "
                             "training/holdout and error-model splits do not move")
    parser.add_argument("--reference-population", type=int, default=None,
                        help="set the true-model reference population independently. "
                             "Use 1000 to compare a population-2000 treatment "
                             "against the existing frozen reference ruler")
    parser.add_argument("--reference-manifest", default=None,
                        help="manifest that freezes the expected reference "
                             "thresholds; the recomputed true-model reference "
                             "must match it before any surrogate is trained")
    parser.add_argument("--train-size", type=int, default=None,
                        help="depart from the frozen 5,000 training rows. "
                             "Only this count moves: the sampler, the audit, "
                             "the refinement call and the reference chain are "
                             "untouched, so runs at different sizes measure "
                             "against the same reference and ARE comparable - "
                             "unlike --scale, which moves everything. Recorded "
                             "in the manifest's data_organisation")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="recompute every closed loop instead of reusing "
                             "stored ones; the stored loops are byte-identical "
                             "rebuilds, so this only costs time")
    parser.add_argument("--ema", type=float, default=None,
                        help="closed-loop navigator damping: the next "
                             "acquisition runs the convex combination "
                             "(1-ema)*previous_navigator + ema*fresh_fine_tune "
                             "over the trainable tensors; None keeps the frozen "
                             "0 (no damping)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and print the tail-holdout budgets, then "
                             "exit before true-model calls or surrogate training")
    parser.add_argument("--experiment-id", default=None,
                        help="experiment identifier recorded in the manifest")
    parser.add_argument("--expected-config", default=None,
                        help="experiment configuration checked against the requested settings")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.replicate_start < 0:
        parser.error("--replicate-start must be non-negative")
    if args.organisation != "tail_holdout" and args.replicate_start != 0:
        parser.error("--replicate-start is available only under tail_holdout")

    if args.subset_population is not None and args.scale != 1:
        parser.error("--subset-population and --scale describe different experiments; "
                     "use only one")
    if args.organisation != "tail_holdout" and (
        args.subset_population is not None or args.reference_population is not None
    ):
        parser.error("population overrides are available only under tail_holdout")
    if args.organisation != "tail_holdout" and args.dry_run:
        parser.error("--dry-run budget inspection is available only under tail_holdout")
    population_override = (
        args.subset_population is not None or args.reference_population is not None
    )
    if population_override and args.out is None and not args.dry_run:
        parser.error("population overrides require an explicit --out so the frozen "
                     "results directory cannot be overwritten")

    case_dir, config, problem = load_case(args.case)
    from dataclasses import replace as _replace
    if args.no_feature_map:
        problem = _replace(problem, feature_map=None)
    if args.recipe:
        problem = _replace(problem, recipe=args.recipe)
    if args.log_inputs:
        case_module = sys.modules.get(f"case_{args.case}")
        if case_module is None or not hasattr(case_module, "inputs_log"):
            parser.error(f"--log-inputs: case {args.case} has no inputs_log map")
        problem = _replace(problem, feature_map=case_module.inputs_log,
                           feature_tag="log_inputs")
        print(f"feature map  log-inputs (departure from the case default)")
    if config["case"]["status"] == "blocked":
        raise SystemExit(
            f"{args.case} is blocked: see the case README. Clear the gate before running."
        )

    root_seed = args.seed if args.seed is not None else int(config["seeds"]["root"])
    out_dir = Path(args.out) if args.out else case_dir / "results"

    print(f"case         {problem.name}  ({problem.dimension}-D, adverse={problem.adverse})")
    print(f"subset MCMC  {getattr(problem, 'subset_sampler', 'pcn')}")
    print(f"organisation {args.organisation}")
    print(f"recipe       {problem.recipe}"
          f"{'  (overridden)' if args.recipe else ''}")
    print(f"seed         {root_seed}")
    print(f"baselines    {args.baselines or 'none'}")

    if args.organisation == "tail_holdout":
        from protocol.config import (sampler_with_population, scaled,
                                     with_subset_population)

        sampler_cfg, cfg = scaled(args.scale)
        if args.train_size is None and args.case != "hysteretic-lab":
            cfg = _replace(cfg, train_size=max(1, round(1000 * args.scale)))
        evaluation_cfg = sampler_cfg
        reference_cfg = evaluation_cfg
        if args.subset_population is not None:
            sampler_cfg, cfg = with_subset_population(
                args.subset_population, sampler=sampler_cfg, holdout=cfg
            )
            print(f"subset pop   {sampler_cfg.population} per level; "
                  f"p0={sampler_cfg.conditional_probability:.0%}, "
                  f"{sampler_cfg.n_seed} seeds + {cfg.extra_per_level} holdout")
            print(f"evaluate pop {evaluation_cfg.population} per level "
                  "(A/C/B_N common frozen evaluator)")
        if args.reference_population is not None:
            reference_cfg = sampler_with_population(
                args.reference_population, sampler=evaluation_cfg
            )
            print(f"reference pop {reference_cfg.population} per level "
                  "(independent fixed ruler)")
        elif args.subset_population is not None:
            print(f"reference pop {reference_cfg.population} per level "
                  "(default frozen ruler)")
        reference_source = None
        expected_reference_thresholds = None
        if args.reference_manifest is not None:
            reference_path = Path(args.reference_manifest).resolve()
            reference_bytes = reference_path.read_bytes()
            reference_data = json.loads(reference_bytes.decode("utf-8"))
            expected_reference_thresholds = reference_data["reference_thresholds"]
            reference_source = {
                "path": str(reference_path),
                "thresholds": expected_reference_thresholds,
            }
            print(f"reference src {reference_path}")
        if args.train_size is not None:
            cfg = _replace(cfg, train_size=int(args.train_size))
            if cfg.train_size == TAIL_HOLDOUT.train_size:
                print(f"train size   {cfg.train_size}  (frozen base retained)")
            else:
                print(f"train size   {cfg.train_size}  (departure from the frozen "
                      f"{TAIL_HOLDOUT.train_size})")
            print(f"refinement   {cfg.refinement_rows} rows = "
                  f"{cfg.refinement_rows / cfg.train_size:.0%} of the training set")
        if args.ema is not None:
            if not 0.0 <= args.ema <= 1.0:
                parser.error("--ema must be within [0, 1]")
            cfg = _replace(cfg, navigator_ema=float(args.ema))
            print(f"navigator ema {cfg.navigator_ema}  (departure from the frozen 0)")
        if args.scale != 1:
            print(f"scale        x{args.scale}  (population {sampler_cfg.population}, "
                  f"train {cfg.train_size}, audit/level {cfg.audit_per_level})")
        expected_config_record = None
        if args.expected_config is not None:
            try:
                expected_config, expected_config_record = _read_expected_config(
                    args.expected_config
                )
                _validate_expected_config(
                    expected_config,
                    case_slug=args.case,
                    problem=problem,
                    root_seed=root_seed,
                    cfg=cfg,
                    sampler=sampler_cfg,
                    stages=args.stages,
                    baselines=args.baselines,
                    requested_replicates=args.replicates,
                    replicate_start=args.replicate_start,
                    accumulate=not args.no_accumulate,
                )
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
            print(f"expected cfg validated  {expected_config_record['path']}")
        if args.dry_run:
            print(f"tail labels  {cfg.tail_train} per loop = "
                  f"{cfg.tail_train / cfg.train_size:.0%} of the training set")
            print(f"replay rows  {cfg.tail_train} per loop")
            print(f"audit calls  {cfg.acquisition_calls} per loop")
            print(f"A/C deploy   {cfg.deployment_calls(args.stages)} per replicate")
            baseline_deployment = sum(
                int(n) + cfg.holdout_size
                for n in args.baselines if int(n) != cfg.train_size
            )
            if baseline_deployment:
                print(f"ABC deploy   {cfg.deployment_calls(args.stages) + baseline_deployment} "
                      "per replicate, including B_N datasets")
            print("dry run      no true-model calls, training, checkpoints or results")
            return 0
        started_at = datetime.now(timezone.utc)
        started_clock = perf_counter()
        provenance = {
            "experiment_id": args.experiment_id,
            "runner": str(Path(__file__).resolve()),
            "argv": list(sys.argv),
            "started_at_utc": started_at.isoformat(),
            "reference_source": reference_source,
            "expected_config": expected_config_record,
            "cpu_affinity": PROCESS_AFFINITY,
        }
        result = run_tail_case(problem, root_seed=root_seed, stages=args.stages,
                               sampler=sampler_cfg,
                               evaluation_sampler=evaluation_cfg,
                               reference_sampler=reference_cfg,
                               expected_reference_thresholds=expected_reference_thresholds,
                               cfg=cfg,
                               checkpoints=not args.no_checkpoints,
                               accumulate=False if args.no_accumulate else None,
                               baselines=args.baselines,
                               n_replicates=args.replicates,
                               replicate_start=args.replicate_start)
        provenance.update(
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=perf_counter() - started_clock,
            exit_code=0,
        )
        write_tail_results(result, out_dir, root_seed, cfg=cfg,
                           sampler=sampler_cfg,
                           evaluation_sampler=evaluation_cfg,
                           reference_sampler=reference_cfg,
                           provenance=provenance,
                           accumulate=False if args.no_accumulate else None)
        print(f"\nreference thresholds  {result.reference.thresholds.round(4).tolist()}")
        print(f"deployment per run    {cfg.deployment_calls(args.stages)} "
              f"({cfg.initial_size} initial + {args.stages} x "
              f"{cfg.acquisition_calls} audited)")
        print(f"refinement rows       {cfg.refinement_rows} "
              f"({cfg.tail_train} tail + {cfg.tail_train} unconditional)")
        replicate_ids = sorted({s.replicate for s in result.scores})
        n_rep = len(replicate_ids)
        print(f"holdout               {cfg.holdout_size} "
              f"({cfg.tail_holdout} tail + {cfg.unconditional_holdout} unconditional)"
              f" -> fit {cfg.fit_size}, validate {cfg.validation_size}")
        print(f"replicates            {n_rep}  (independent, not folds)")
        print(f"replicate ids         {replicate_ids}")
        print(f"deployment calls      {result.ledger.deployment}")
        print(f"instrumentation calls {result.ledger.instrumentation}")
    else:
        result = run_case(problem, root_seed=root_seed, baselines=args.baselines,
                          n_folds=args.folds, stages=args.stages)
        write_results(result, out_dir, root_seed)
        print(f"\nreference thresholds  {result.reference.thresholds.round(4).tolist()}")
        print(f"resolution            {result.resolution.overall:.3e}")
        print(f"deployment calls      {result.ledger.deployment}")
        print(f"instrumentation calls {result.ledger.instrumentation}")
    print(f"\nwritten to {out_dir}")
    return 0

