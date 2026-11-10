from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from random_field_analysis.reliability.conditional_ged import bucket_by_slope, fit_conditional_ged_runtime, load_error_yhat_fold, split_fit_test

def _write_runtime(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    details = model.fit_details
    lines = [
        "Mode: conditional_yhat",
        "ParamForm: spline",
        f"Beta: {model.global_beta:.17g}",
        f"GlobalBeta: {model.global_beta:.17g}",
        f"Loc: {model.global_loc:.17g}",
        f"GlobalLoc: {model.global_loc:.17g}",
        f"GlobalScale: {model.global_scale:.17g}",
        f"Scale: {model.global_scale:.17g}",
        f"TargetBins: {details['n_bins_target']}",
        f"Bins: {len(model.edges) - 1}",
        f"SigmaInflation: {model.sigma_inflation:.17g}",
        f"BetaMin: {model.beta_min:.17g}",
        f"BetaMax: {model.beta_max:.17g}",
        f"SCALE_FLOOR: {model.scale_floor:.17g}",
        f"SCALE_CAP: {model.scale_cap:.17g}",
        f"SIGMA_MONOTONE: {details['sigma_monotone']}",
        f"SIGMA_MONOTONE_INCREASING: {details['sigma_monotone_increasing']}",
        "ConditionVar: y_hat",
    ]
    for key, values in [
        ("BinEdges", model.edges), ("AnchorX", details["anchors_x"]),
        ("BinBetas", model.betas), ("BinLocs", model.locs),
        ("BinScales", model.scales), ("BinCounts", details["counts"]),
    ]:
        lines.append(key + ": " + ",".join(f"{value:.17g}" for value in values))
    for name, tck in [("SplineBeta", model.beta_tck), ("SplineLoc", model.loc_tck), ("SplineLogScale", model.log_scale_tck)]:
        for suffix in ("t", "c"):
            lines.append(name + suffix.upper() + ": " + ",".join(f"{value:.17g}" for value in tck[suffix]))
        lines.append(f"{name}K: {tck['k']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/conditional_ged.yaml")
    parser.add_argument("--fold-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    with (ROOT / args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    fold_dir = ROOT / (args.fold_dir or cfg["fold_dir"])
    out_dir = ROOT / (args.output_dir or cfg["output_dir"])
    err, yhat = load_error_yhat_fold(fold_dir)
    test_count = int(cfg.get("test_count", 500))
    split = split_fit_test(err, yhat, test_count)
    fit_err, fit_yhat = split["fit"]
    num_slopes = int(cfg.get("num_slopes", 10))
    pair_path = fold_dir / "fs_eval_pairs.csv"
    pairs = pd.read_csv(pair_path) if pair_path.exists() else None
    if pairs is not None and "slope" in pairs.columns:
        slope_ids = sorted(pairs["slope"].unique())
        if len(slope_ids) != num_slopes:
            raise ValueError("Paired slope count differs from configuration")
        fit_slopes = pairs["slope"].to_numpy()[test_count:]
        buckets = {sid: (fit_err[fit_slopes == label], fit_yhat[fit_slopes == label]) for sid, label in enumerate(slope_ids)}
    else:
        buckets = bucket_by_slope(fit_err, fit_yhat, test_count, num_slopes)
    for slope, (err_s, yhat_s) in buckets.items():
        if len(err_s) < int(cfg.get("min_samples_per_slope", 30)):
            raise ValueError(f"Not enough fitting rows for slope {slope}")
        runtime = fit_conditional_ged_runtime(err_s, yhat_s, scale_inflation=float(cfg.get("scale_multiplier", 1.12)))
        if len(runtime.edges) - 1 != int(cfg.get("bins", 4)):
            raise ValueError(f"Slope {slope} has insufficient distinct predictions for four bins")
        _write_runtime(runtime, out_dir / f"slope_{slope}" / "ged_params.txt")
    print(out_dir)

if __name__ == "__main__":
    main()
