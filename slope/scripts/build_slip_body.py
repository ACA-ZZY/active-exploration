from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from random_field_analysis.slip.stablegrow import body_mask_to_edge_label, stablegrow_body_mask

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slip_body.yaml")
    args = parser.parse_args()

    root = ROOT
    with (root / args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    values = np.load(root / cfg["input_npy"])
    domain = np.load(root / cfg["domain_mask_npy"]).astype(bool)
    classify, labels, body, valid, centers = stablegrow_body_mask(
        values,
        domain,
        air_threshold=float(cfg.get("air_threshold", 0.0)),
        vertical_axis=int(cfg.get("vertical_axis", 1)),
        min_stable_region=int(cfg.get("min_stable_region", 400)),
    )
    edge = body_mask_to_edge_label(body, valid)

    out = root / cfg["output_npy"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), classify=classify, labels=labels, body=body, valid=valid, edge=edge, centers=centers)

if __name__ == "__main__":
    main()
