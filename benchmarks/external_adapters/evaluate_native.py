from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from uncertainty_benchmark.contracts import available_cases
from uncertainty_benchmark.posthoc_evaluation import DATA, evaluate_tail

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=available_cases())
    parser.add_argument("--method", choices=["AK-SS", "BSS", "MF-AL"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--matlab", default="matlab")
    args = parser.parse_args(argv)
    if args.method == "MF-AL":
        from .mf_al import restore_model
        terminal = args.run_dir / "models/native_terminal/model.pkl"
        if not terminal.exists():
            terminal = args.run_dir / "models/native_budget/model.pkl"
        model = restore_model(terminal, args.case)
    elif args.method == "AK-SS":
        with (args.run_dir / "native_gp.pkl").open("rb") as f:
            gp = pickle.load(f)

        class Model:
            def predict(self, x):
                return np.asarray(gp.getMetaModel()(x)).reshape(-1)

        model = Model()
    else:
        from scipy.io import savemat
        from .bss_cases.runner import BSS_ROOT, ADAPTER_DIR, _quote_matlab

        with np.load(DATA / "common_data" / args.case / "tail/samples.npz") as z:
            inputs = z["x_latent"]
        evaluation_input = args.run_dir / "tail_evaluation_inputs.mat"
        output = args.run_dir / "tail_predictions.json"
        savemat(evaluation_input, {"x": inputs})
        expression = (
            f"addpath('{_quote_matlab(ADAPTER_DIR)}'); "
            f"evaluate_bss_tail('{_quote_matlab(BSS_ROOT)}',"
            f"'{_quote_matlab(args.run_dir / 'result.mat')}',"
            f"'{_quote_matlab(evaluation_input)}','{_quote_matlab(output)}');"
        )
        subprocess.run([args.matlab, "-batch", expression], check=True)
        predictions = np.asarray(json.loads(output.read_text()), dtype=float)

        class Model:
            def predict(self, x):
                if not np.array_equal(x, inputs):
                    raise ValueError("evaluation input order differs")
                return predictions

        model = Model()
    result = evaluate_tail(model, args.case)
    (args.run_dir / "tail_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
