# Random Field Analysis

This package contains the three-dimensional ResUNet, training and inference routines, sliding-body reconstruction, and conditional generalized error distribution (GED) calculations used for the random-field slope application.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

## Training and prediction

```bash
python scripts/train_proxy.py --config configs/proxy_train.yaml
```

Training reads voxel inputs and displacement labels from the `Input` and `Label` directories under `data_root`. The FOS labels are read from `fos_file`. Each checkpoint records the FOS normalization limits. Validation writes paired predictions, true values, errors and slope identities to `fs_eval_pairs.csv` in original FOS units.

```bash
python scripts/predict_proxy.py \
  --checkpoint outputs/kfold/fold_1/fold_1UCN_state_dict.pt \
  --input <input-npy> \
  --output outputs/predictions
```

The saved FOS prediction is restored to its original units using checkpoint metadata. Legacy full-model files require `--legacy-pickle`. The fold-1 model uses FOS normalization limits of 0.5 and 3.5.

```bash
python scripts/predict_proxy.py \
  --checkpoint weights/kfold/refinement/fold_1/fold_1UCN.pt \
  --input <input-npy> \
  --legacy-pickle --fos-min 0.5 --fos-max 3.5
```

The legacy `nerual` classes are included for loading the released full-model files. A legacy checkpoint can be converted without training.

```bash
python scripts/convert_legacy_checkpoint.py \
  --input weights/kfold/refinement/fold_1/fold_1UCN.pt \
  --output outputs/fold_1_state_dict.pt \
  --fos-min 0.5 --fos-max 3.5
```

The separate `weights/surrogate/5BUCN.pt` file has no recorded FOS normalization metadata. Its original limits must be supplied from its training configuration before interpreting its output in FOS units.

## Conditional GED fitting

```bash
python scripts/fit_conditional_ged.py --config configs/conditional_ged.yaml
```

The fitting routine uses the original four-bin anchor estimates, shrinkage, linear smoothing splines and scale multiplier of 1.12. It fits each slope separately on rows after the first 500, with the preceding rows retained for evaluation. Input can be paired `fs_eval_pairs.csv`, `fs_pred.txt` with `fs_error.xlsx`, or `fsV.txt` with `fs_error.xlsx`. Errors are predicted FOS minus true FOS. Explicit slope identities in the paired file are used when available. Input files may use the interleaved slope order.

The output directory is `outputs/conditional_ged/fold_1/Verify/slope_*/ged_params.txt`. Parameters retain the spline knots, coefficients, degrees, bin anchors, counts and scale multiplier used by the inference routine. `--fold-dir` and `--output-dir` select another fold and its destination.

## Conditional failure probabilities

Candidate CSV files contain `slope`, `level` and `y_hat`. Slopes are indexed from 0 to 9. Level 1 is the first conditional subset with original probability 0.1, followed by levels 2 to 5. A wide workbook with columns `subset_0_sim_0` through `subset_4_sim_999` and one row per slope is also accepted.

```bash
python scripts/apply_conditional_pfail.py --config configs/surrogate_sus.yaml
```

This command reads the newly fitted parameters from `outputs/conditional_ged`. To use the released fold-1 parameters, run:

```bash
python scripts/apply_conditional_pfail.py \
  --config configs/surrogate_sus.yaml \
  --conditional-ged-dir weights/conditional_ged \
  --candidate-file <candidate-csv-or-xlsx>
```

For each slope, the routine evaluates the conditional failure probability at every candidate. It selects the first of levels 1 to 4 whose corrected lower-decile threshold reaches FOS 1. If none reaches it, it uses level 4. The selected level mean is multiplied by its original probability, `0.1 ** level`. Separate output files contain candidate probabilities, level summaries, slope estimates and response intervals. The response intervals use the maximum predicted FOS at each of the five recorded levels.

Optional `weight_column` values weight samples within a level. `region_probability_column` supplies the original probability of that level and is constant within its slope and level. A `fold_column` permits several folds in one candidate file, with parameters loaded separately for each fold.

## Sliding-body reconstruction

```bash
python scripts/build_slip_body.py --config configs/slip_body.yaml
```

The procedure processes displacement fields using displacement clustering and basal stable-region connectivity.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```
