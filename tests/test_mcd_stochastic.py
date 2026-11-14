"""MC-Dropout regression checks: no fitting and no HF response evaluation."""
from pathlib import Path
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch
from torch import nn

if os.environ.get("MCD_SURROGATE_PATH"):
    spec = importlib.util.spec_from_file_location(
        "_mcd_under_test", os.environ["MCD_SURROGATE_PATH"])
    code = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = code
    spec.loader.exec_module(code)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from protocol import surrogate as code

torch.set_num_threads(1)

class IdentityScale:
    def transform(self, x):
        return np.asarray(x)

class AffineScale:
    def inverse_transform(self, x):
        return np.asarray(x) * 7.0 + 100.0

def selector(kind="deep"):
    if kind == "network":
        recipe = code.NetworkRecipe(hidden_layer_sizes=(12, 12), dropout=0.2)
    elif kind == "recurrent":
        recipe = code.RecurrentRecipe(hidden=4, layers=2, steps=6, dropout=0.2)
    else:
        recipe = code.DeepRecipe(width=12, depth=3, dropout=0.2)
    s = code.Surrogate(seed=571, recipe=recipe, device="cpu")
    s._scaler, s._y_scaler = IdentityScale(), AffineScale()
    if kind == "network":
        s._model = s._shallow_module(3)
    elif kind == "recurrent":
        _, recurrence, _ = code._torch_modules()
        torch.manual_seed(571)
        s._model = [recurrence(3, 6, 4, 2, True, dropout=0.2)]
        s._driving = lambda x: (np.asarray(x, dtype=np.float32), True)
        s._scale_drive = lambda x: np.asarray(x, dtype=np.float32)
    else:
        s._model = s._deep_module(3)
    return s

@pytest.mark.parametrize("dimension", [8, 50])
def test_formal_plain_deep_recipe_really_contains_dropout(dimension):
    s = code.Surrogate(seed=571, recipe=code.DeepRecipe(dropout=0.1), device="cpu")
    m = s._deep_module(dimension)
    assert [n.p for n in m.modules() if isinstance(n, nn.Dropout)] == [0.1] * 3
    assert m(torch.zeros(2, dimension)).shape == (2, 1)

def test_zero_dropout_keeps_original_plain_architecture_and_initial_weights():
    s = code.Surrogate(seed=571, recipe=code.DeepRecipe(width=12, depth=3), device="cpu")
    got = s._deep_module(3)
    torch.manual_seed(571)
    expected = nn.Sequential(nn.Linear(3, 12), nn.GELU(), nn.Linear(12, 12),
                             nn.GELU(), nn.Linear(12, 12), nn.GELU(), nn.Linear(12, 1))
    assert list(got.state_dict()) == list(expected.state_dict())
    for key in got.state_dict():
        assert torch.equal(got.state_dict()[key], expected.state_dict()[key])

@pytest.mark.parametrize("kind", ["deep", "network", "recurrent"])
def test_thirty_actual_passes_variation_physical_units_and_reproducibility(kind):
    s = selector(kind)
    m = s._model[0] if isinstance(s._model, list) else s._model
    x = np.random.default_rng(93).normal(size=(16, 3))
    raw = []
    hook = m.register_forward_hook(lambda module, args, output:
                                  raw.append(output.detach().cpu().numpy().copy()))
    mean, variance = s.predict_stochastic(x, passes=30, seed=123)
    hook.remove()
    assert len(raw) == 30
    assert any(not np.array_equal(raw[0], r) for r in raw[1:])
    expected = np.asarray(raw, dtype=np.float64).reshape(30, -1) * 7.0 + 100.0
    np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-14, atol=1e-13)
    np.testing.assert_allclose(variance, expected.var(axis=0, ddof=1), rtol=1e-12)
    assert mean.dtype == variance.dtype == np.float64
    assert np.all(variance > 0.0)
    again = s.predict_stochastic(x, passes=30, seed=123)
    assert np.array_equal(mean, again[0])
    assert np.array_equal(variance, again[1])
    assert all(not layer.training for layer in m.modules())
    if kind == "recurrent":
        assert not m.gru.training
    else:
        assert np.array_equal(s.predict(x), s.predict(x))

@pytest.mark.parametrize("kind", ["deep", "recurrent"])
def test_legacy_checkpoint_with_recipe_only_dropout_is_rejected(kind):
    s = selector(kind)
    legacy = nn.Sequential(nn.Linear(3, 12), nn.GELU(), nn.Linear(12, 1))
    s._model = [legacy] if kind == "recurrent" else legacy
    with pytest.raises(RuntimeError, match="legacy checkpoint.*retrained"):
        s.predict_stochastic(np.ones((4, 3)), seed=42)

def test_identical_float32_draws_have_exactly_zero_variance():
    draws = np.tile(np.array([0.1, 1e10, 1e30], dtype=np.float32), (30, 1))
    mean, variance = code.Surrogate._stochastic_moments(draws)
    assert np.array_equal(mean, draws[0].astype(np.float64))
    assert np.array_equal(variance, np.zeros(3))

def test_batch_normalization_is_never_updated_by_stochastic_forward():
    s = selector()
    s._model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(0.2), nn.Linear(3, 1))
    bn = s._model[0]
    before = [bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone()]
    s.predict_stochastic(np.ones((16, 3)), seed=42)
    after = [bn.running_mean, bn.running_var, bn.num_batches_tracked]
    assert all(torch.equal(a, b) for a, b in zip(before, after))

@pytest.mark.parametrize("kind", ["deep", "recurrent"])
def test_dropout_is_disabled_after_a_forward_exception(kind):
    s = selector(kind)
    m = s._model[0] if isinstance(s._model, list) else s._model
    def fail(module, args):
        raise RuntimeError("deliberate forward error")
    hook = m.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="deliberate forward error"):
        s.predict_stochastic(np.ones((4, 3)), seed=42)
    hook.remove()
    assert all(not layer.training for layer in m.modules())

def test_invalid_passes_and_nonfinite_outputs_are_not_silently_ranked():
    s = selector()
    with pytest.raises(ValueError, match="at least two"):
        s.predict_stochastic(np.ones((4, 3)), passes=1)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        code.Surrogate._stochastic_moments([np.array([np.nan])] * 30)

def test_log_decode_follows_inverse_scaling_for_each_draw():
    from dataclasses import replace
    s = selector("recurrent")
    s.recipe = replace(s.recipe, log_target=True)
    class SmallScale:
        def inverse_transform(self, raw):
            return raw * 0.2 + 1.0
    s._y_scaler = SmallScale()
    raw = []
    hook = s._model[0].register_forward_hook(lambda module, args, output:
                  raw.append(output.detach().cpu().numpy().copy()))
    mean, variance = s.predict_stochastic(np.ones((8, 3)), seed=31)
    hook.remove()
    physical = np.exp(np.asarray(raw, dtype=np.float64).reshape(30, -1) * 0.2 + 1.0)
    np.testing.assert_allclose(mean, physical.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(variance, physical.var(axis=0, ddof=1), rtol=1e-12)

def test_acquisition_selects_the_actual_thirty_pass_variance_topk():
    from types import SimpleNamespace
    if os.environ.get("MCD_BENCHMARKS"):
        sys.path.insert(0, os.environ["MCD_BENCHMARKS"])
    from uncertainty_benchmark.seed_blocks import SeedBlock
    from uncertainty_benchmark.strategies import MCDropoutVarT30Strategy

    s = selector()
    x = np.random.default_rng(411).normal(size=(100, 3))
    seeds = SeedBlock(2026090701, 0)
    _, variance = s.predict_stochastic(
        x, seed=seeds.seed("mc_dropout_passes", round_index=1))
    contract = SimpleNamespace(dimension=3, train_per_round=5,
                               audit_per_round=3, acquisition_per_round=8)
    proposed = MCDropoutVarT30Strategy().propose(
        SimpleNamespace(members=[s]), contract, seeds, 1, set(),
        pool_size=len(x), candidate_chunks=[(0, x)])
    assert set(proposed.metadata["selected_source_indices"]) == set(np.argsort(variance)[-8:])
    assert proposed.train_x.shape == (5, 3)
    assert proposed.audit_x.shape == (3, 3)
    assert len(np.unique(np.vstack([proposed.train_x, proposed.audit_x]), axis=0)) == 8
