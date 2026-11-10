from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy.stats import gennorm

from random_field_analysis.reliability.conditional_ged import fit_conditional_ged_runtime, load_conditional_ged
from random_field_analysis.reliability.probability import evaluate_candidates, model_path

ROOT = Path(__file__).resolve().parents[1]

def test_refitted_spline_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    predictions = np.linspace(0.7, 2.0, 150)
    errors = 0.01 * predictions + rng.normal(0, 0.025, len(predictions))
    model = fit_conditional_ged_runtime(errors, predictions)
    spec = importlib.util.spec_from_file_location("fit_script", ROOT / "scripts/fit_conditional_ged.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "ged_params.txt"
    module._write_runtime(model, path)
    restored = load_conditional_ged(path)
    assert model.param_form == "spline"
    assert len(model.edges) - 1 == 4
    assert restored.sigma_inflation == 1.12
    assert "PitX" not in path.read_text()
    for expected, actual in zip(model.conditional_params(predictions), restored.conditional_params(predictions)):
        np.testing.assert_array_equal(actual, expected)

def test_failure_probability_keeps_slopes_and_levels_separate(tmp_path):
    for slope in (0, 1):
        target = tmp_path / "fold_1" / "Verify" / f"slope_{slope}" / "ged_params.txt"
        target.parent.mkdir(parents=True)
        target.write_text("Mode: single\nBeta: 2\nLoc: 0\nScale: 0.1\nSigmaInflation: 1.12\n")
    frame = pd.DataFrame({
        "slope": [0, 0, 0, 0, 1, 1, 1, 1],
        "level": [1, 1, 2, 2, 1, 1, 2, 2],
        "y_hat": [.85, .9, .7, .8, 1.2, 1.3, .95, 1.1],
    })
    cfg = {"keep_ratio": .1, "first_level": 1, "max_probability_levels": 2}
    records, levels, slopes = evaluate_candidates(frame, cfg, tmp_path)
    first = slopes.loc[slopes.slope == 0].iloc[0]
    second = slopes.loc[slopes.slope == 1].iloc[0]
    assert first.selected_level == 1
    assert second.selected_level == 2
    expected_first = .1 * np.mean(1 - gennorm.cdf(np.array([.85, .9]) - 1, 2, scale=.112))
    expected_second = .01 * np.mean(1 - gennorm.cdf(np.array([.95, 1.1]) - 1, 2, scale=.112))
    assert first.failure_probability == pytest.approx(expected_first)
    assert second.failure_probability == pytest.approx(expected_second)
    assert len(records) == 8 and len(levels) == 4

def test_missing_fold_cannot_reuse_another_fold(tmp_path):
    target = tmp_path / "fold_1" / "Verify" / "slope_0" / "ged_params.txt"
    target.parent.mkdir(parents=True)
    target.write_text("Mode: single\nBeta: 2\nLoc: 0\nScale: 0.1\n")
    with pytest.raises(FileNotFoundError):
        model_path(tmp_path, "fold_2", 0)

@pytest.mark.parametrize("relative", [
    "weights/surrogate/5BUCN.pt",
    "weights/kfold/base/fold_1/fold_1UCN.pt",
    "weights/kfold/refinement/fold_1/fold_1UCN.pt",
])
def test_released_legacy_checkpoints_forward(relative):
    torch = pytest.importorskip("torch")
    from random_field_analysis.models.checkpoints import load_checkpoint, restore_fos
    torch.set_num_threads(1)
    model = load_checkpoint(ROOT / relative, trust_legacy_pickle=True).eval()
    with torch.inference_mode():
        fos, displacement = model(torch.zeros(1, 2, 16, 16, 16))
    assert fos.shape == (1,)
    assert displacement.shape == (1, 3, 16, 16, 16)
    assert torch.isfinite(fos).all()
    with pytest.raises(ValueError):
        restore_fos(fos, model)

def test_checkpoint_retains_fos_units(tmp_path):
    torch = pytest.importorskip("torch")
    from random_field_analysis.models.checkpoints import build_model, load_checkpoint, restore_fos, save_state_dict_checkpoint
    torch.set_num_threads(1)
    model = build_model()
    path = tmp_path / "model.pt"
    save_state_dict_checkpoint(model, path, fos_range=(.5, 3.5))
    loaded = load_checkpoint(path)
    values = torch.tensor([0., .5, 1.])
    torch.testing.assert_close(restore_fos(values, loaded), torch.tensor([.5, 2., 3.5]))
    with pytest.raises(ValueError):
        restore_fos(values, loaded, (0., 1.))
