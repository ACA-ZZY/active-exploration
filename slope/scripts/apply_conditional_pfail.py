from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from random_field_analysis.reliability.probability import evaluate_candidates, load_candidates, response_intervals

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/surrogate_sus.yaml")
    parser.add_argument("--candidate-file")
    parser.add_argument("--conditional-ged-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    with (ROOT / args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    candidate_file = ROOT / (args.candidate_file or cfg["candidate_file"])
    out_csv = ROOT / (args.output or cfg["output_csv"])
    ged_dir = ROOT / (args.conditional_ged_dir or cfg["conditional_ged_dir"])
    candidates = load_candidates(candidate_file, cfg)
    records, levels, slopes = evaluate_candidates(candidates, cfg, ged_dir)
    intervals = response_intervals(candidates, cfg, ged_dir, float(cfg.get("confidence", 0.95)))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(out_csv, index=False)
    levels.to_csv(out_csv.with_name(out_csv.stem + "_by_level.csv"), index=False)
    slopes.to_csv(out_csv.with_name(out_csv.stem + "_by_slope.csv"), index=False)
    intervals.to_csv(out_csv.with_name(out_csv.stem + "_intervals.csv"), index=False)
    print(slopes.to_string(index=False))

if __name__ == "__main__":
    main()
