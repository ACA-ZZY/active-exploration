from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from random_field_analysis.models.checkpoints import load_checkpoint, save_state_dict_checkpoint

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fos-min", type=float, required=True)
    parser.add_argument("--fos-max", type=float, required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    model = load_checkpoint(input_path, device=args.device, trust_legacy_pickle=True)
    save_state_dict_checkpoint(model, output_path, fos_range=(args.fos_min, args.fos_max))
    print(output_path)

if __name__ == "__main__":
    main()
