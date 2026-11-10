from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def _path(root: Path, value):
    if value in (None, ""):
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proxy_train.yaml")
    args = parser.parse_args()

    root = ROOT
    with (root / args.config).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    from random_field_analysis.training.kfold import TrainConfig, run_kfold_training

    config = TrainConfig(
        data_root=_path(root, raw["data_root"]),
        fos_file=_path(root, raw["fos_file"]),
        output_root=_path(root, raw["output_root"]),
        kfold=int(raw.get("kfold", 10)),
        num_slopes=int(raw.get("num_slopes", 10)),
        epochs=int(raw.get("epochs", 50)),
        branch_finetune_epochs=int(raw.get("branch_finetune_epochs", 3)),
        batch_size=int(raw.get("batch_size", 1)),
        learning_rate=float(raw.get("learning_rate", 1e-4)),
        fs_scale_min=float(raw.get("fs_scale_min", 0.5)),
        fs_scale_max=float(raw.get("fs_scale_max", 3.5)),
        validation_holdout=int(raw.get("validation_holdout", 1500)),
        device=str(raw.get("device", "cuda:0")),
        seed=int(raw.get("seed", 42)),
        pretrained_checkpoint=_path(root, raw.get("pretrained_checkpoint")),
    )
    run_kfold_training(config)

if __name__ == "__main__":
    main()
