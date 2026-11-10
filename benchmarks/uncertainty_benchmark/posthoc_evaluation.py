from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .common_student import restore_student
from .contracts import available_cases, load_case_contract
from .strategies import IMPLEMENTED_METHODS

DATA = Path(__file__).resolve().parents[2] / "data"

def evaluate_tail(student, case, training_x=None):
    with np.load(DATA / "common_data" / case / "tail/samples.npz") as z:
        x, y, bands = z["x_latent"], z["y_hf"], z["partition"]
    if training_x is not None:
        train = {row.tobytes() for row in np.ascontiguousarray(training_x, dtype=np.float64)}
        if any(row.tobytes() in train for row in np.ascontiguousarray(x, dtype=np.float64)):
            raise ValueError("tail evaluation inputs overlap the training inputs")
    scale = json.loads((DATA / "references" / case / "nominal_scale.json").read_text())["iqr"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid response IQR")
    prediction = np.asarray(student.predict(x)).reshape(-1)
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid predictions")
    errors = []
    for band in range(1, 6):
        selected = bands == band
        if selected.sum() != 1000:
            raise ValueError("each tail band requires 1000 evaluation inputs")
        errors.append(float(np.mean(np.abs(prediction[selected] - y[selected])) / scale))
    return dict(e1=errors[0], e2=errors[1], e3=errors[2], e4=errors[3], e5=errors[4],
                E_tail=float(np.mean(errors)), shallow=float(np.mean(errors[:2])),
                middle=float(np.mean(errors[2:4])), deepest=errors[4])

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=available_cases())
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    resolved = load_case_contract(args.case)
    request = json.loads((args.block_dir / "request.json").read_text())["request"]
    if request["case"] != args.case:
        raise ValueError("block belongs to another case")
    with np.load(args.block_dir / "shared/base_training.npz") as z:
        initial_x = z["x"]
    rows = []
    for method in IMPLEMENTED_METHODS:
        folder = args.block_dir / "methods" / method.lower().replace("-", "_")
        for round_index in range(1, resolved.contract.rounds + 1):
            current = folder / f"round_{round_index:02d}"
            if not (current / "complete.json").is_file():
                continue
            with (current / "student.pkl").open("rb") as f:
                student = restore_student(pickle.load(f), resolved.problem, device=args.device)
            with np.load(current / "training_data.npz") as z:
                train_x = np.vstack([initial_x, z["tail_x"]])
            rows.append(dict(method=method, round=round_index,
                             **evaluate_tail(student, args.case, train_x)))
    output = args.block_dir / "tail_evaluation.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
