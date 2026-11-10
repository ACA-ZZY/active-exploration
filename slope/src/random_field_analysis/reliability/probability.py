from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gennorm

from .conditional_ged import load_conditional_ged

def model_path(base: Path, fold: str, slope: int) -> Path:
    paths = [base / fold / "Verify" / f"slope_{slope}" / "ged_params.txt", base / fold / f"slope_{slope}" / "ged_params.txt"]
    if base.name == fold or (base.name == "Verify" and base.parent.name == fold):
        paths.extend([base / "Verify" / f"slope_{slope}" / "ged_params.txt", base / f"slope_{slope}" / "ged_params.txt"])
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing GED parameters for {fold}, slope {slope}")

def load_candidates(path: Path, cfg: dict) -> pd.DataFrame:
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    columns = []
    for column in frame.columns:
        match = re.fullmatch(r"subset_(\d+)_sim_(\d+)", str(column))
        if match:
            columns.append((int(match.group(1)), int(match.group(2)), column))
    if not columns:
        return frame
    columns.sort()
    num_slopes = int(cfg.get("num_slopes", 10))
    if len(frame) != num_slopes:
        raise ValueError("Wide subset table must contain one row per slope")
    first_level = int(cfg.get("first_level", 1))
    rows = []
    for slope in range(num_slopes):
        for subset, sample, column in columns:
            rows.append({"slope": slope, "level": subset + first_level, "sample": sample, "y_hat": frame.iloc[slope][column]})
    return pd.DataFrame(rows)

def evaluate_candidates(frame: pd.DataFrame, cfg: dict, ged_dir: Path):
    slope_col = str(cfg.get("slope_column", "slope"))
    level_col = str(cfg.get("level_column", "level"))
    yhat_col = str(cfg.get("yhat_column", "y_hat"))
    weight_col = cfg.get("weight_column")
    probability_col = cfg.get("region_probability_column")
    fold_col = cfg.get("fold_column")
    required = {slope_col, level_col, yhat_col}
    required.update(column for column in (weight_col, probability_col, fold_col) if column)
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Candidate data must contain {sorted(required)}")
    out = frame.copy().reset_index(drop=True)
    for column in (slope_col, level_col):
        values = pd.to_numeric(out[column], errors="raise").to_numpy(dtype=float)
        if not np.all(np.isfinite(values) & (values == np.floor(values))):
            raise ValueError(f"{column} must contain finite integers")
        out[column] = values.astype(int)
    predictions = pd.to_numeric(out[yhat_col], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Predictions must be finite FOS values")
    out[yhat_col] = predictions
    if fold_col:
        out["_fold"] = out[fold_col].astype(str)
    else:
        out["_fold"] = str(cfg.get("fold", "fold_1"))
    keep_ratio = float(cfg.get("keep_ratio", 0.1))
    threshold = float(cfg.get("failure_threshold", 1.0))
    first_level = int(cfg.get("first_level", 1))
    max_levels = int(cfg.get("max_probability_levels", 4))
    error_sign = int(cfg.get("error_sign", 1))
    if not 0 < keep_ratio < 1 or first_level < 0 or max_levels < 1 or error_sign not in {-1, 1}:
        raise ValueError("Invalid conditional-sampling settings")
    out["p_fail_conditional_ged"] = np.nan
    out["fos_corrected_center"] = np.nan
    level_rows = []
    slope_rows = []
    for (fold, slope), slope_frame in out.groupby(["_fold", slope_col], sort=True):
        model = load_conditional_ged(model_path(ged_dir, str(fold), int(slope)))
        local_levels = []
        for level, batch in slope_frame.groupby(level_col, sort=True):
            y = batch[yhat_col].to_numpy(dtype=float)
            _, location, _ = model.conditional_params(y)
            center = y - error_sign * location
            point_probabilities = model.p_fail(y, threshold=threshold, error_sign=error_sign)
            out.loc[batch.index, "p_fail_conditional_ged"] = point_probabilities
            out.loc[batch.index, "fos_corrected_center"] = center
            if weight_col:
                weights = pd.to_numeric(batch[weight_col], errors="raise").to_numpy(dtype=float)
                if not np.all(np.isfinite(weights) & (weights >= 0)) or not weights.sum() > 0:
                    raise ValueError("Within-level weights must be finite, nonnegative and have positive sum")
                conditional_mean = float(np.average(point_probabilities, weights=weights))
            else:
                conditional_mean = float(np.mean(point_probabilities))
            if probability_col:
                region_probabilities = pd.to_numeric(batch[probability_col], errors="raise").to_numpy(dtype=float)
                if not np.all(np.isfinite(region_probabilities)) or np.any(region_probabilities != region_probabilities[0]):
                    raise ValueError("Region probability must be constant within each slope and level")
                region_probability = float(region_probabilities[0])
                if not 0 < region_probability <= 1:
                    raise ValueError("Region probability must be in (0, 1]")
            else:
                region_probability = keep_ratio ** int(level)
            rank = max(1, int(np.ceil(keep_ratio * len(center))))
            corrected_threshold = float(np.sort(center)[rank - 1])
            row = {
                "fold": str(fold), "slope": int(slope), "level": int(level), "sample_count": len(y),
                "region_probability": region_probability, "corrected_threshold": corrected_threshold,
                "conditional_failure_probability": conditional_mean,
                "failure_probability": region_probability * conditional_mean,
            }
            level_rows.append(row)
            local_levels.append(row)
        required_levels = list(range(first_level, first_level + max_levels))
        available = {row["level"]: row for row in local_levels}
        if any(level not in available for level in required_levels):
            raise ValueError(f"Missing probability levels for {fold}, slope {slope}")
        eligible = [available[level] for level in required_levels]
        selected = next((row for row in eligible if row["corrected_threshold"] <= threshold), eligible[-1])
        slope_rows.append({
            "fold": str(fold), "slope": int(slope), "selected_level": selected["level"],
            "threshold_reached": selected["corrected_threshold"] <= threshold,
            "region_probability": selected["region_probability"],
            "conditional_failure_probability": selected["conditional_failure_probability"],
            "failure_probability": selected["failure_probability"],
        })
    return out.drop(columns="_fold"), pd.DataFrame(level_rows), pd.DataFrame(slope_rows)

def response_intervals(frame: pd.DataFrame, cfg: dict, ged_dir: Path, confidence: float = 0.95) -> pd.DataFrame:
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be in (0, 1)")
    slope_col = str(cfg.get("slope_column", "slope"))
    level_col = str(cfg.get("level_column", "level"))
    yhat_col = str(cfg.get("yhat_column", "y_hat"))
    fold_col = cfg.get("fold_column")
    error_sign = int(cfg.get("error_sign", 1))
    source = frame.copy()
    source["_fold"] = source[fold_col].astype(str) if fold_col else str(cfg.get("fold", "fold_1"))
    rows = []
    for (fold, slope, level), batch in source.groupby(["_fold", slope_col, level_col], sort=True):
        model = load_conditional_ged(model_path(ged_dir, str(fold), int(slope)))
        boundary = float(np.max(batch[yhat_col].to_numpy(dtype=float)))
        beta, location, scale = model.conditional_params(np.asarray(boundary))
        limits = boundary - error_sign * gennorm.ppf([(1 - confidence) / 2, (1 + confidence) / 2], beta, loc=location, scale=scale)
        rows.append({"fold": str(fold), "slope": int(slope), "level": int(level), "confidence": confidence,
                     "predicted_boundary": boundary, "corrected_center": float(boundary - error_sign * location),
                     "lower": float(np.min(limits)), "upper": float(np.max(limits))})
    return pd.DataFrame(rows)
