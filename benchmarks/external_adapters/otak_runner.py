"""Named ot-AK AK-SS comparator, with isolated RNG and from-zero HF accounting"""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import traceback
from types import SimpleNamespace
import numpy as np
from uncertainty_benchmark.checkpoint import BlockStore
from uncertainty_benchmark.ledger import CallLedger
from uncertainty_benchmark.oracle import CachedOracle
from uncertainty_benchmark.seed_blocks import SeedBlock
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party' / 'otak'
COMMIT = '15fa3a663fe4fa9ad1a6b5655b20f922cbf1c281'
METHOD = 'AK-SS-otAK-M32'
CASES = {'two-sided-kink': 2, 'radial-kink': 8, 'tail-kink': 50, 'cantilever_beam_2d': 2}
HF_CAP = 50000
GRID = (1000, 2000, 3000, 4000, 5000)
PACKAGES = {'openturns': '1.20.post3', 'numpy': '1.26.4', 'scipy': '1.11.4'}

def verify_runtime_source():
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions != PACKAGES:
        raise RuntimeError(f'isolated runtime drift: {versions}')
    observed = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if observed != COMMIT:
        raise RuntimeError('ot-AK commit drift')
    subprocess.run(['git', '-C', str(UPSTREAM), 'diff', '--exit-code', '--quiet'], check=True)
    return versions

def instrument_compute(native_class, ot):
    """Fail closed unless the exact two reviewed instrumentation sites exist."""
    source = textwrap.dedent(inspect.getsource(native_class.compute_proba))
    reset = 'ot.RandomGenerator.SetSeed(1)'
    observer = '        res = SS_kr.getResult()'
    if source.count(reset) != 1 or source.count(observer) != 1:
        raise RuntimeError('upstream instrumentation sites changed')
    adapted = source.replace(reset, 'pass')
    initial = 'DoE_inputs = exp_LHS.generate()'
    if source.count(initial) != 1:
        raise RuntimeError('upstream initial design site changed')
    adapted = adapted.replace(initial, 'DoE_inputs = ot.Sample(self._initial_x)')
    adapted = adapted.replace(observer, observer + '\n        self._record_native(SS_kr)')
    adapted = adapted.replace('my_krig = algokriging.getResult()', 'my_krig = algokriging.getResult(); self._last_kriging = my_krig; self._last_design = (DoE_inputs, DoE_responses)')
    namespace = {'ot': ot, 'np': np}
    exec(compile(adapted, '<verified-otak-compute-instrumentation>', 'exec'), namespace)
    return (namespace['compute_proba'], source, adapted)

def load_response(slug):
    path = ROOT / 'cases' / slug / 'problem.py'
    spec = importlib.util.spec_from_file_location('otak_physics_' + slug.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.DIMENSION != CASES[slug]:
        raise RuntimeError('physics dimension drift')
    return module.response

def run_case(slug, block, output, *, root_seed=2026090701, cpu_core=4, timeout_seconds=86400):
    import psutil
    process = psutil.Process()
    if hasattr(process, 'cpu_affinity'):
        process.cpu_affinity([cpu_core])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    versions = verify_runtime_source()
    sys.path.insert(0, str(UPSTREAM))
    import openturns as ot
    import otak
    ot.TBB.SetThreadsNumber(1)
    streams = SeedBlock(root_seed, block)
    raw_seed = streams.seed('native_pipeline', method=METHOD)
    dimension = CASES[slug]
    n0 = 1000
    total_cap = n0 + HF_CAP
    request = dict(schema='otak-project-request-v1', method=METHOD, case=slug, block=block, root_seed=root_seed, named_uint64_seed=raw_seed, openturns_seed=raw_seed % 2 ** 32, dimension=dimension, hf_cap=HF_CAP, total_hf_cap=total_cap, initial_design=n0, initial_design_distribution='shared original-distribution inputs', input_distribution='independent standard Gaussian', event='g <= 0', nominal_initial_pool=1000, subset_initial_outer_sampling=1000, growing_pool='native additive nominal MC growth until CoV or HF cap', target_probability=0.1, proposal_range=1.0, coefficient_variation_target=0.05, u_stop=2.0, basis='constant', kernel='Matern nu=1.5, anisotropic', upstream_commit=COMMIT, actual_hf_from_zero=True, free_initial_data=False, reference_probability_visible=False, reference_labels_visible=False, diagnostic_budget_grid=list(GRID), diagnostic_rng_feedback=False, native_confidence_interval='N/A; native CoV retained, not a GP-error CI', cpu_core=cpu_core, tbb_threads=1, runtime=versions)
    store = BlockStore(Path(output), request)
    with store.lock():
        done = store.read_json('complete.json')
        if done:
            return store.read_json('summary.json')
        if store.path('started.json').exists():
            raise RuntimeError('partial native run preserved; do not automatically repeat HF')
        ledger = CallLedger(store.path('ledger.json'))
        response = load_response(slug)
        oracle = CachedOracle(SimpleNamespace(dimension=dimension, evaluate=response), store, ledger)
        counter = {'rows': 0, 'events': 0}
        (adapted, original, changed) = instrument_compute(otak.AK_SSAlgorithm, ot)
        store.write_json('instrumentation.json', dict(removed_internal_fixed_seed=True, added_native_readout_observer=True, acquisition_or_gp_math_changed=False, source=original, adapted_source=changed))
        store.write_json('started.json', dict(status='running', monotonic_start=time.monotonic()))

        def evaluate_many(values):
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError('native time limit reached')
            x = np.asarray(values, dtype=np.float64).reshape(-1, dimension)
            if counter['rows'] + len(x) > total_cap:
                raise RuntimeError('additional-query ceiling exceeded')
            event = f"native-query-{counter['events']:05d}"
            store.save_npz_once(Path('oracle_intents') / f'{event}.npz', x=x)
            role = 'initial_train' if counter['rows'] < n0 else 'acquisition_train'
            result = oracle.query(event, x, role=role, method=METHOD)
            counter['rows'] += len(x)
            counter['events'] += 1
            store.write_json('oracle_status.json', dict(returned_rows=counter['rows'], completed_events=counter['events'], last_completed_event=event, status='response_persisted'))
            return result.y[:, None].tolist()

        def evaluate_one(x):
            return evaluate_many([x])[0]
        (native_readouts, diagnostics) = ([], {})
        started = time.monotonic()
        diagnostic_seconds = 0.0

        class ObservedAKSS(otak.AK_SSAlgorithm):
            compute_proba = adapted

            def _record_native(self, sampler):
                result = sampler.getResult()
                native_readouts.append(dict(hf_cost=counter['rows'], probability=float(result.getProbabilityEstimate()), coefficient_of_variation=float(result.getCoefficientOfVariation()), nominal_pool_size=int(self.n_SS), elapsed_seconds=time.monotonic() - started))
                store.write_json('native_readouts.json', native_readouts)

            def compute_U(self, kriging, evaluated):
                nonlocal diagnostic_seconds
                values = super().compute_U(kriging, evaluated)
                if not np.isfinite(values).all():
                    raise RuntimeError('nonfinite native U criterion; no project-result retuning')
                cost = counter['rows']
                if cost in GRID and cost not in diagnostics:
                    prior_state = ot.RandomGenerator.GetState()
                    before = time.monotonic()
                    try:
                        seed = streams.seed('independent_budget_readout', method=METHOD, member=cost)
                        ot.RandomGenerator.SetSeed(seed % 2 ** 32)
                        event = ot.ThresholdEvent(ot.CompositeRandomVector(kriging.getMetaModel(), ot.RandomVector(self.distrib)), ot.LessOrEqual(), 0.0)
                        sampler = ot.SubsetSampling(event, 1.0, 0.1)
                        sampler.setMaximumOuterSampling(self.n_SS)
                        sampler.run()
                        result = sampler.getResult()
                        diagnostics[cost] = dict(hf_cost=cost, probability=float(result.getProbabilityEstimate()), coefficient_of_variation=float(result.getCoefficientOfVariation()), nominal_pool_size=int(self.n_SS), additional_hf=0, role='independent_readout_no_rng_or_label_feedback')
                        store.write_json('budget_readouts.json', diagnostics)
                    finally:
                        ot.RandomGenerator.SetState(prior_state)
                        diagnostic_seconds += time.monotonic() - before
                return values
        try:
            ot.RandomGenerator.SetSeed(raw_seed % 2 ** 32)
            distribution = ot.Normal(dimension)
            truth = ot.PythonFunction(dimension, 1, evaluate_one, evaluate_many)
            event = ot.ThresholdEvent(ot.CompositeRandomVector(truth, ot.RandomVector(distribution)), ot.LessOrEqual(), 0.0)
            model = ObservedAKSS(event, 1000, n0, HF_CAP, ot.ConstantBasisFactory(dimension).build(), ot.MaternModel([1.] * dimension, 1.5), proposal_range=1.0, target_proba=0.1, cv_target=0.05, criterion=2.0, verbose=False)
            model._initial_x = np.random.default_rng(streams.seed("initial_inputs")).standard_normal((1250, dimension))[:n0]
            model.compute_proba()
            if not native_readouts:
                raise RuntimeError('native run has no probability readout')
            if counter['rows'] != ledger.as_dict()['totals']['physical_calls']:
                raise RuntimeError('true-row accounting mismatch')
            store.save_pickle('native_gp.pkl', model._last_kriging)
            store.save_npz(Path('native_gp_training.npz'), x=np.asarray(model._last_design[0]), y=np.asarray(model._last_design[1]).reshape(-1))
            final = native_readouts[-1]
            if not 0 <= final['probability'] <= 1:
                raise RuntimeError('native probability out of range')
            budget_rows = []
            for budget in GRID:
                if budget >= counter['rows']:
                    row = dict(final, budget=budget, readout='native_final_carry_forward')
                elif budget in diagnostics:
                    row = dict(diagnostics[budget], budget=budget, readout='budget_checkpoint')
                else:
                    row = dict(budget=budget, status='unavailable', reason='no valid checkpoint')
                budget_rows.append(row)
            summary = dict(schema='otak-project-run-v1', status='complete', method=METHOD, case=slug, block=block, hf_cap=HF_CAP, hf_physical_rows=counter['rows'], native_reported_hf=model.getSimBudget(), probability=final['probability'], coefficient_of_variation=final['coefficient_of_variation'], stopping_reason='hf_cap' if counter['rows'] >= total_cap else 'native_stopping_rule', budget_curve=budget_rows, native_readouts=native_readouts, elapsed_seconds=time.monotonic() - started, diagnostic_seconds=diagnostic_seconds, initial_holdout_rows=0, audit_labels_used=0, caveats=['This is the ot-AK implementation, not a verified byte-for-byte Huang reproduction.', 'Native CoV measures simulation variation; it is not a complete metamodel-error interval.', 'No valid estimate is replaced by a reference probability.'])
            store.write_json('summary.json', summary)
            store.write_json('complete.json', dict(status='complete'))
            return summary
        except Exception:
            store.write_json('failure.json', dict(status='failed', case=slug, block=block, returned_hf_rows=counter['rows'], error=traceback.format_exc(), elapsed_seconds=time.monotonic() - started, pending_intent_caveat='Unmatched persisted intents may contain an unknown in-flight call; never count them as zero or auto-repeat.'))
            raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--block', required=True, type=int)
    parser.add_argument('--root-seed', type=int, default=2026090701)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cpu-core', type=int, default=4)
    parser.add_argument('--timeout-seconds', type=float, default=86400)
    args = parser.parse_args()
    result = run_case(args.case, args.block, args.out, root_seed=args.root_seed, cpu_core=args.cpu_core, timeout_seconds=args.timeout_seconds)
    print(json.dumps({key: result[key] for key in ('status', 'case', 'block', 'probability', 'hf_physical_rows')}), flush=True)
if __name__ == '__main__':
    main()
