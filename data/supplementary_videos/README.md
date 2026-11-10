# Supplementary video source data

The five NPZ files contain the recorded inputs, true responses, and predictions used by Supplementary Videos 1–5. Each file is an unchanged copy of the existing video source array. No response calculation, model training, prediction, subsampling, or deduplication was performed during packaging.

| Video | File | Initial training inputs | Acquired inputs | Fixed evaluation inputs | Input dimension |
|---|---|---:|---:|---:|---:|
| 1 | `two-sided-kink.npz` | 1,000 | 2,000 | 5,000 | 2 |
| 2 | `radial-kink.npz` | 1,000 | 2,000 | 5,000 | 8 |
| 3 | `tail-kink.npz` | 1,000 | 2,000 | 5,000 | 50 |
| 4 | `hysteretic-lab.npz` | 5,000 | 2,000 | 5,000 | 48 |
| 5 | `cantilever_beam_2d.npz` | 1,000 | 2,000 | 5,000 | 2 |

## Array fields

The prefixes `initial`, `query`, and `fixed` identify the three input groups in the table. For each group, `_x` contains the full latent input coordinates, `_y` the recorded true responses, `_coordinate` the coordinate used in the response plot, and `_prediction` the predictions from the five recorded models. Prediction columns follow `stage = ['A', 'R1', 'R2', 'R3', 'R4']`, with `A` denoting the initial model. The arrays retain their original floating-point precision.

`query_round` gives acquisition rounds 1–4, with 500 rows per round. Within each round, rows 0–99, 100–199, 200–299, 300–399, and 400–499 originate from the initial candidate layer and the four successive conditional layers. The five levels therefore each contribute 100 queries per round. `query_input_id` gives the recorded input identity.

`fixed_sample_id`, `fixed_input_id`, `fixed_weight`, and `fixed_partition` identify and weight the fixed evaluation states. The fixed states were used to record model predictions and were not added to the training set in the video. Repeated chain states and their weights are retained. `dataset_id` identifies this fixed dataset; `model_state_id` identifies the five model states in the same order as `stage`.

The three kink responses are dimensionless, oscillator responses are in metres, and beam responses are in millimetres. `manifest.json` provides the array shapes and types, file hashes, response-window thresholds, input projections, and display settings for each video. For the two-sided kink, the response panel uses the absolute value of the stored coordinate. Smooth motion between recorded states is a display interpolation rather than an additional model evaluation.

The 44,000 sample rows, true responses, all five model predictions, input identities, projections, acquisition rounds, source levels, response-window memberships, fixed-state identities, and fixed weights were checked against the video tables removed from the supplementary Source Data workbook. Differences within 5 × 10⁻¹⁴ relative and absolute tolerances arise from Excel decimal serialization. The NPZ copies themselves are byte-identical to the original files.

```python
import numpy as np

with np.load('two-sided-kink.npz', allow_pickle=False) as data:
    inputs = data['query_x']
    truth = data['query_y']
    predictions = data['query_prediction']
    stages = data['stage']
```
