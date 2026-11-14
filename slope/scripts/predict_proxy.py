from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def _normalize_input(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32, copy=True)
    parts = path.stem.split(",")
    if len(parts) >= 5:
        max1, min1, max2, min2 = map(float, parts[1:5])
        if max1 <= min1 or max2 <= min2:
            raise ValueError("Input normalization limits must increase")
        mask = arr == -1
        arr[0] = (arr[0] - min1) / (max1 - min1)
        arr[1] = (arr[1] - min2) / (max2 - min2)
        arr[mask] = -1
    return arr

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/surrogate/5BUCN.pt")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/predictions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--legacy-pickle", action="store_true")
    parser.add_argument("--fos-min", type=float)
    parser.add_argument("--fos-max", type=float)
    args = parser.parse_args()

    root = ROOT
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from random_field_analysis.models.checkpoints import load_checkpoint, restore_fos

    model = load_checkpoint(ckpt, device=args.device, trust_legacy_pickle=args.legacy_pickle)
    if (args.fos_min is None) != (args.fos_max is None):
        parser.error("--fos-min and --fos-max must be supplied together")
    fos_range = None if args.fos_min is None else (args.fos_min, args.fos_max)
    model.eval()
    x = torch.from_numpy(_normalize_input(input_path)).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        fos, disp = model(x)
        fos = restore_fos(fos, model, fos_range)
    np.savez_compressed(out_dir / f"{input_path.stem}_prediction.npz", fos=fos.cpu().numpy(), displacement=disp.cpu().numpy())

if __name__ == "__main__":
    main()
