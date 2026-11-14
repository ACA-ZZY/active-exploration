from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from uncertainty_benchmark.contracts import available_cases, load_case_contract
from uncertainty_benchmark.seed_blocks import SeedBlock
from uncertainty_benchmark.strategies import DirectionCorrectQBC, IMPLEMENTED_METHODS

@pytest.mark.parametrize("case", available_cases())
def test_released_case_contract(case):
    resolved = load_case_contract(case)
    assert resolved.reference_thresholds.shape == (5,)
    assert resolved.contract.train_size == (5000 if case == "hysteretic-lab" else 1000)
    assert resolved.problem.subset_sampler == resolved.contract.sampler_kernel

@pytest.mark.parametrize("adverse,expected", [("lower", [0.1, 0.2]), ("upper", [0.8, 0.9])])
def test_qbc_selects_the_correct_tail(adverse, expected):
    from types import SimpleNamespace

    class Committee:
        members = [None] * 5

        def mean_variance(self, x):
            return x[:, 0], np.full(len(x), 0.01)

    contract = SimpleNamespace(dimension=1, adverse=adverse, train_per_round=1,
                               audit_per_round=1, acquisition_per_round=2)
    x = np.array([[0.1], [0.2], [0.8], [0.9]])
    p = DirectionCorrectQBC().propose(Committee(), contract, SeedBlock(2026090701, 0),
                                     1, set(), candidate_chunks=[(0, x)], pool_size=4)
    assert sorted(np.r_[p.train_x[:, 0], p.audit_x[:, 0]]) == expected

def test_mf_probability_and_sampling():
    from external_adapters.mf_al import membership_probability, componentwise_step
    p = membership_probability(np.array([-1., 0., 1.]), np.array([0., 1., 0.]), 0.)
    assert np.array_equal(p, [0., .5, 1.])
    rng = np.random.default_rng(829)
    x = rng.normal(size=(20000, 3))
    y = componentwise_step(x, rng)
    assert abs(float(y.mean())) < 0.03
    assert np.max(np.abs(y.var(axis=0) - 1.)) < 0.05

@pytest.mark.parametrize("case", available_cases())
def test_all_seven_methods_and_resume(case, monkeypatch, tmp_path):
    import torch
    from protocol.surrogate import RECIPES
    import uncertainty_benchmark.contracts as contracts
    import uncertainty_benchmark.run_block as runner
    from external_adapters.lcmd_tp_adapter import BMDAL_ROOT

    if not BMDAL_ROOT.is_dir():
        pytest.skip("install the pinned BMDAL dependency to run integration checks")
    torch.set_num_threads(1)
    original = contracts._SPECS[case]
    tiny = replace(original, train_size=30, initial_holdout=5, replay_size=10,
                   rounds=2, train_per_round=5, audit_per_round=5, subset_population=40)
    monkeypatch.setitem(contracts._SPECS, case, tiny)
    recipe = RECIPES[original.base_recipe]
    changes = dict(max_iter=2, refine_iter=2)
    if recipe.name == "network":
        changes["hidden_layer_sizes"] = (8, 8)
    elif recipe.name == "deep":
        changes.update(width=8, depth=2)
    else:
        changes.update(hidden=4, layers=1)
    monkeypatch.setitem(RECIPES, original.base_recipe, replace(recipe, **changes))
    monkeypatch.setitem(contracts._SPECS, case, replace(tiny, refine_epochs=2))
    monkeypatch.setattr(runner, "QBC_POOL_SIZE", 80)
    monkeypatch.setattr(runner, "QBC_CHUNK_SIZE", 32)

    out = tmp_path / case
    result = runner.run_seed_block(case, block=0, root_seed=2026090701,
                                   output=out, methods=IMPLEMENTED_METHODS, device="cpu")
    assert set(result["methods"]) == set(IMPLEMENTED_METHODS)
    assert result["ledger"]["totals"]["physical_calls"] == 35 + 7 * 2 * 10
    assert result["audit_labels_used_for_training"] == 0
    for method in IMPLEMENTED_METHODS:
        p = out / runner._round_dir(method, 2) / "training_data.npz"
        with np.load(p) as z:
            assert len(z["tail_x"]) == 10
            assert len(z["replay_x"]) == 10
    before = (out / "ledger.json").read_bytes()
    again = runner.run_seed_block(case, block=0, root_seed=2026090701,
                                  output=out, methods=IMPLEMENTED_METHODS, device="cpu")
    assert again["ledger"] == result["ledger"]
    assert (out / "ledger.json").read_bytes() == before

def test_native_budget_and_tail_kink_entry():
    from external_adapters import otak_runner
    from external_adapters.bss_cases import runner
    assert otak_runner.HF_CAP == runner.HF_BUDGET == 50000
    assert otak_runner.CASES["tail-kink"] == 50
    assert "tail-kink" in runner.CASES

def test_bss_tail_kink_formula():
    from external_adapters.otak_runner import load_response
    response = load_response("tail-kink")
    x = np.vstack([np.zeros(50), np.full(50, 1.)])
    s = x.sum(axis=1) / np.sqrt(50.)
    assert np.array_equal(response(x), 4.8 - s - np.maximum(s - 3.2, 0.))

def test_tail_metric_matches_the_population_band_definition():
    from uncertainty_benchmark.posthoc_evaluation import DATA, evaluate_tail
    import json

    class TrueResponse:
        def predict(self, x):
            return load_case_contract("two-sided-kink").problem.evaluate(x)

    exact = evaluate_tail(TrueResponse(), "two-sided-kink")
    assert max(exact.values()) < 1e-12
    scale = json.loads((DATA / "references/two-sided-kink/nominal_scale.json").read_text())["iqr"]

    class ShiftedResponse:
        def predict(self, x):
            return TrueResponse().predict(x) + scale

    shifted = evaluate_tail(ShiftedResponse(), "two-sided-kink")
    assert all(abs(v - 1.) < 1e-12 for v in shifted.values())
