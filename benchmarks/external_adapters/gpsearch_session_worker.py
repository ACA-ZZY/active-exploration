"""Checkpointed native gpsearch session actions for formal US-LW collection."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
import uuid
from pathlib import Path
import gpsearch_worker as stable
SCHEMA = 'gpsearch.session-action.v1'
EXPECTED_COMMIT = stable.EXPECTED_COMMIT
METHOD = 'US-LW-GP-Seq'
PREDICTION_CHUNK_SIZE = 4096
AUDIT_POOL_SIZE = 1000000
AUDIT_CHUNK_SIZE = 4096
AUDIT_TOP_K = 250
FORMAL_GP = {'kernel': 'gpsearch.core.kernels.RBF', 'ard': True, 'normalize_y': True, 'fix_noise': True, 'noise_variance': 0.001, 'noise_variance_units': 'standardized-output variance', 'num_restarts': 10, 'optimizer': 'bfgs', 'max_iters': 1000}
FORMAL_ACQUISITION = {'criterion': 'gpsearch.core.acquisitions.US_LW', 'method': 'l-bfgs-b', 'num_restarts': 10, 'init_method': None, 'max_iters': 1000, 'ftol': 1e-09, 'gtol': 1e-07, 'gmm_components': 2, 'gmm_covariance_type': 'full'}

class SessionInputError(ValueError):
    """Raised when an action could cross the frozen information boundary."""

def _canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def _atomic_json(path, value):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / '.{}.{}.tmp'.format(target.name, uuid.uuid4().hex)
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()

def _atomic_npz(path, np, **arrays):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / '.{}.{}.tmp'.format(target.name, uuid.uuid4().hex)
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()

def _require_fields(body, required, optional=()):
    if not isinstance(body, dict):
        raise SessionInputError('request must be a JSON object')
    missing = sorted(set(required) - set(body))
    extra = sorted(set(body) - set(required) - set(optional))
    if missing:
        raise SessionInputError('request is missing fields: {}'.format(', '.join(missing)))
    if extra:
        raise SessionInputError('request has unsupported fields: {}'.format(', '.join(extra)))

def _load_request(path):
    request_path = Path(path).resolve()
    with request_path.open('r', encoding='utf-8') as stream:
        body = json.load(stream)
    common = {'schema_version', 'action', 'request_id', 'expected_third_party_commit', 'dimension', 'domain'}
    if body.get('schema_version') != SCHEMA:
        raise SessionInputError('unsupported request schema')
    if body.get('expected_third_party_commit') != EXPECTED_COMMIT:
        raise SessionInputError('third-party commit declaration differs from pinned checkout')
    action = body.get('action')
    action_fields = {'initialize': common | {'seed', 'expected_initial_count', 'initial_data_npz', 'checkpoint_out'}, 'propose': common | {'seed', 'checkpoint_in', 'expected_model'}, 'observe': common | {'seed', 'checkpoint_in', 'checkpoint_out', 'expected_model', 'observation_npz', 'expected_proposal_x'}, 'audit': common | {'seed', 'candidate_seed', 'checkpoint_in', 'expected_model', 'prior_audit_npz', 'selection_npz', 'pool_size', 'chunk_size', 'top_k'}}
    if action not in action_fields:
        raise SessionInputError('action must be initialize, propose, observe or audit')
    _require_fields(body, action_fields[action])
    dimension = int(body['dimension'])
    if dimension <= 0:
        raise SessionInputError('dimension must be positive')
    domain = body['domain']
    if domain != [[-6.0, 6.0] for _ in range(dimension)]:
        raise SessionInputError('formal latent domain is frozen to [-6,6]^d')
    for field in ('seed', 'candidate_seed'):
        if field in body:
            value = int(body[field])
            if value < 0 or value > 2 ** 32 - 1:
                raise SessionInputError('{} must be uint32'.format(field))
            body[field] = value
    body['dimension'] = dimension
    return (request_path, body)

def _runtime(api):
    return {'python_executable': sys.executable, 'python': platform.python_version(), 'numpy': api['np'].__version__, 'scipy': api['scipy'].__version__, 'GPy': api['GPy'].__version__, 'KDEpy': api['KDEpy'].__version__, 'scikit_learn': api['sklearn'].__version__, 'prediction_memory_adapter': {'kind': 'chunked_predict_only', 'chunk_size': PREDICTION_CHUNK_SIZE, 'density_points_unchanged': True, 'density_point_order_unchanged': True, 'authors_kde_gmm_and_acquisition_unchanged': True}, 'checkpoint_adapter': {'issue': 'GPy 1.9.9 compressed JSON load drops fixed-parameter constraint', 'rule': 'verify serialized nugget remains exactly 1e-3, then restore only the fixed constraint before any optimization', 'nugget_value_changed': False}}

def _base_response(request_path, request, api):
    return {'schema_version': SCHEMA, 'request_id': request['request_id'], 'request_path': str(request_path), 'action': request['action'], 'method': METHOD, 'third_party': {'repository': 'https://github.com/ablancha/gpsearch', 'commit': EXPECTED_COMMIT, 'api': ['gpsearch.core.acquisitions.US_LW', 'gpsearch.core.likelihood.Likelihood', 'gpsearch.core.minimizers.funmin', 'gpsearch.core.kernels.RBF']}, 'runtime': _runtime(api), 'formal_gp': dict(FORMAL_GP), 'formal_acquisition': dict(FORMAL_ACQUISITION)}

def _matrix(value, name, np, columns=None):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise SessionInputError('{} must be a matrix'.format(name))
    if columns is not None and array.shape[1] != columns:
        raise SessionInputError('{} has the wrong dimension'.format(name))
    if not np.all(np.isfinite(array)):
        raise SessionInputError('{} contains non-finite values'.format(name))
    return np.ascontiguousarray(array)

def _load_oracle_artifact(path, np, expected_rows, dimension):
    source = Path(path).resolve()
    if not source.is_file():
        raise SessionInputError('oracle artifact is absent: {}'.format(source))
    expected_fields = {'x', 'y', 'origin_physical_calls'}
    with np.load(str(source), allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise SessionInputError('oracle artifact fields must be exactly {}; got {}'.format(sorted(expected_fields), sorted(archive.files)))
        x = _matrix(archive['x'], 'x', np, columns=dimension)
        y_original = np.asarray(archive['y'], dtype=np.float64)
        y = y_original.reshape(-1, 1)
        origin_calls = int(np.asarray(archive['origin_physical_calls']).item())
    if x.shape[0] != expected_rows or y.shape[0] != expected_rows:
        raise SessionInputError('oracle artifact row count differs from the request')
    if origin_calls != expected_rows:
        raise SessionInputError('oracle artifact physical-call count is inconsistent')
    if not np.all(np.isfinite(y)):
        raise SessionInputError('oracle artifact y contains non-finite values')
    return {'path': str(source), 'x': x, 'y': y, 'origin_physical_calls': origin_calls}

def _model_xy(model, np):
    x = np.ascontiguousarray(np.asarray(model.X, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(model.Y, dtype=np.float64).reshape(-1))
    return (x, y)

def _model_state(model, np):
    (x, y) = _model_xy(model, np)
    return {'rows': int(x.shape[0]), 'dimension': int(x.shape[1]), 'parameters': stable._model_parameters(model, np), 'normalizer_enabled': bool(model.normalizer is not None), 'noise_fixed': bool(model.Gaussian_noise.variance.is_fixed)}

def _validate_model(model, expected, dimension, np):
    state = _model_state(model, np)
    required = {'rows', 'dimension', 'training_data'}
    if set(expected) != required:
        raise SessionInputError('expected_model fields are incomplete or unsupported')
    for key in ('rows', 'dimension'):
        if state[key] != expected[key]:
            raise SessionInputError('checkpoint model drift for {}: expected {!r}, got {!r}'.format(key, expected[key], state[key]))
    with np.load(expected['training_data'], allow_pickle=False) as arrays:
        x, y = _model_xy(model, np)
        if not np.array_equal(x, arrays['x']) or not np.array_equal(y, arrays['y']):
            raise SessionInputError('checkpoint training data differ from the requested data')
    if state['dimension'] != dimension:
        raise SessionInputError('checkpoint dimension differs from request')
    if not state['normalizer_enabled'] or not state['noise_fixed']:
        raise SessionInputError('checkpoint violates normalize_y/fixed-noise contract')
    noise = state['parameters']['noise_variance']
    if not math.isclose(noise, FORMAL_GP['noise_variance'], rel_tol=0.0, abs_tol=1e-12):
        raise SessionInputError('checkpoint nugget differs from fixed 1e-3 variance')
    return state

def _load_checkpoint(path, expected, dimension, api):
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise SessionInputError('checkpoint is absent: {}'.format(checkpoint))
    try:
        model = api['GPy'].core.Model.load_model(str(checkpoint))
    except Exception as exc:
        raise SessionInputError('GPy could not load checkpoint: {}'.format(exc))
    restored_constraint = not bool(model.Gaussian_noise.variance.is_fixed)
    if restored_constraint:
        observed_noise = stable._scalar(model.Gaussian_noise.variance.values, api['np'])
        if not math.isclose(observed_noise, FORMAL_GP['noise_variance'], rel_tol=0.0, abs_tol=1e-12):
            raise SessionInputError('checkpoint lost its fixed constraint and its nugget value also drifted')
        model.Gaussian_noise.variance.fix(FORMAL_GP['noise_variance'])
    state = _validate_model(model, expected, dimension, api['np'])
    state['checkpoint_load_adapter'] = {'gpy_1_9_9_fixed_noise_constraint_restored': restored_constraint, 'noise_value_changed': False}
    return (checkpoint, model, state)

def _save_checkpoint(model, path, api):
    record = stable._save_checkpoint(model, path)
    record['model'] = _model_state(model, api['np'])
    x, y = _model_xy(model, api['np'])
    training_path = Path(str(path) + '.training.npz').resolve()
    _atomic_npz(training_path, api['np'], x=x, y=y)
    record['training_data'] = str(training_path)
    return record

def _build_inputs(dimension, domain, api):
    np = api['np']
    return api['GaussianInputs'](domain, np.zeros(dimension, dtype=np.float64), np.eye(dimension, dtype=np.float64))

class _ChunkedPredictProxy(object):
    """Delegate a GPy model while splitting only its ``predict`` allocation."""

    def __init__(self, model, np, chunk_size=PREDICTION_CHUNK_SIZE):
        self._model = model
        self._np = np
        self._chunk_size = int(chunk_size)

    def predict(self, x, *args, **kwargs):
        values = self._np.atleast_2d(self._np.asarray(x, dtype=self._np.float64))
        pieces = []
        for start in range(0, len(values), self._chunk_size):
            result = self._model.predict(values[start:start + self._chunk_size], *args, **kwargs)
            if not pieces:
                pieces = [[] for _ in range(len(result))]
            for (slot, array) in zip(pieces, result):
                slot.append(self._np.asarray(array))
        return tuple((self._np.concatenate(slot, axis=0) for slot in pieces))

    def __getattr__(self, name):
        return getattr(self._model, name)

def _likelihood(model, inputs, seed, api, chunked=True):
    wrapped = _ChunkedPredictProxy(model, api['np']) if chunked else model
    return api['Likelihood'](wrapped, inputs, kwargs_gmm={'n_components': FORMAL_ACQUISITION['gmm_components'], 'covariance_type': FORMAL_ACQUISITION['gmm_covariance_type'], 'random_state': int(seed)})

def _likelihood_state(likelihood, dimension, np):
    knots = np.asarray(likelihood.fy_interp.get_knots(), dtype=np.float64)
    coeffs = np.asarray(likelihood.fy_interp.get_coeffs(), dtype=np.float64)
    gmm = likelihood.gmm
    return {'weight_type': likelihood.weight_type, 'fit_gmm': bool(likelihood.fit_gmm), 'internal_density_samples': 100000 if dimension <= 2 else 1000000, 'internal_gmm_resample_rows': 20000, 'prediction_chunk_size': PREDICTION_CHUNK_SIZE, 'kde_spline': {'knot_count': int(knots.size), 'coefficient_count': int(coeffs.size), 'knot_min': float(np.min(knots)), 'knot_max': float(np.max(knots)), 'coefficient_min': float(np.min(coeffs)), 'coefficient_max': float(np.max(coeffs))}, 'gmm': {'n_components': int(gmm.n_components), 'covariance_type': str(gmm.covariance_type), 'weights': np.asarray(gmm.weights_, dtype=float).tolist(), 'means': np.asarray(gmm.means_, dtype=float).tolist(), 'covariances': np.asarray(gmm.covariances_, dtype=float).tolist()}}

def _point_diagnostics(model, likelihood, inputs, x, np):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    (mean, variance) = model.predict_noiseless(x)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    variance = np.asarray(variance, dtype=np.float64).reshape(-1)
    if model.normalizer:
        mean_standardized = np.asarray(model.normalizer.normalize(mean), dtype=float)
        variance_standardized = variance / float(model.normalizer.std) ** 2
    else:
        mean_standardized = mean
        variance_standardized = variance
    output_density = np.asarray(likelihood.fy_interp(mean_standardized), dtype=np.float64).reshape(-1)
    nominal_density = np.asarray(inputs.pdf(x), dtype=np.float64).reshape(-1)
    weight_gmm = np.asarray(likelihood.evaluate(x), dtype=np.float64).reshape(-1)
    tiny = np.finfo(np.float64).tiny
    return {'predicted_mean': mean.tolist(), 'predicted_variance': variance.tolist(), 'predicted_mean_standardized': mean_standardized.tolist(), 'predicted_variance_standardized': variance_standardized.tolist(), 'nominal_input_density': nominal_density.tolist(), 'output_density_at_predicted_mean': output_density.tolist(), 'log_output_density_at_predicted_mean': np.log(np.maximum(output_density, tiny)).tolist(), 'raw_importance_ratio': (nominal_density / np.maximum(output_density, tiny)).tolist(), 'gmm_likelihood_weight': weight_gmm.tolist()}

def _optimise_gp(model, seed, api):
    np = api['np']
    (stdout, stderr) = (io.StringIO(), io.StringIO())
    started = time.perf_counter()
    with stable._seeded_author_rng(np, int(seed)):
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            model.optimize_restarts(num_restarts=FORMAL_GP['num_restarts'], optimizer=FORMAL_GP['optimizer'], max_iters=FORMAL_GP['max_iters'], verbose=False)
    return {'elapsed_seconds': time.perf_counter() - started, 'captured_stdout': stdout.getvalue(), 'captured_stderr': stderr.getvalue()}

def _initialize(request_path, request, api):
    np = api['np']
    data = _load_oracle_artifact(request['initial_data_npz'], np, int(request['expected_initial_count']), request['dimension'])
    model = api['GPy'].models.GPRegression(X=data['x'], Y=data['y'], kernel=api['RBF'](input_dim=request['dimension'], ARD=True), normalizer=True, noise_var=FORMAL_GP['noise_variance'])
    model.Gaussian_noise.variance.fix(FORMAL_GP['noise_variance'])
    before = _model_state(model, np)
    optimisation = _optimise_gp(model, request['seed'], api)
    checkpoint = _save_checkpoint(model, request['checkpoint_out'], api)
    response = _base_response(request_path, request, api)
    response.update({'status': 'PASS', 'input': {'kind': 'shared_initial_train_only', 'path': data['path'], 'rows': len(data['x']), 'holdout_test_reference_feedback': False}, 'model_before': before, 'model_after': checkpoint['model'], 'gp_optimization': optimisation, 'checkpoint': checkpoint})
    return response

def _observe(request_path, request, api):
    np = api['np']
    (checkpoint_in, model, before) = _load_checkpoint(request['checkpoint_in'], request['expected_model'], request['dimension'], api)
    observation = _load_oracle_artifact(request['observation_npz'], np, 1, request['dimension'])
    if not np.array_equal(observation['x'], np.asarray(request['expected_proposal_x'], dtype=np.float64)):
        raise SessionInputError('oracle observation does not match the selected input')
    (old_x, old_y) = _model_xy(model, np)
    if np.any(np.all(old_x == observation['x'], axis=1)):
        raise SessionInputError('native optimiser proposed an already-observed input')
    updated_x = np.vstack((old_x, observation['x']))
    updated_y = np.concatenate((old_y, observation['y'].reshape(-1)))[:, None]
    model.set_XY(updated_x, updated_y)
    optimisation = _optimise_gp(model, request['seed'], api)
    checkpoint = _save_checkpoint(model, request['checkpoint_out'], api)
    response = _base_response(request_path, request, api)
    response.update({'status': 'PASS', 'checkpoint_in': {'path': str(checkpoint_in)}, 'observation': {'path': observation['path'], 'physical_calls': 1, 'feedback_role': 'confirmed_acquisition_train_only'}, 'model_before': before, 'model_after': checkpoint['model'], 'gp_optimization': optimisation, 'checkpoint': checkpoint})
    return response

def _propose(request_path, request, api):
    np = api['np']
    (checkpoint, model, state) = _load_checkpoint(request['checkpoint_in'], request['expected_model'], request['dimension'], api)
    inputs = _build_inputs(request['dimension'], request['domain'], api)
    (stdout, stderr) = (io.StringIO(), io.StringIO())
    started_density = time.perf_counter()
    with stable._seeded_author_rng(np, request['seed']):
        likelihood = _likelihood(model, inputs, request['seed'], api, chunked=True)
        density_elapsed = time.perf_counter() - started_density
        criterion = api['US_LW'](model, inputs, likelihood=likelihood)
        started_opt = time.perf_counter()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            proposed_x = api['funmin'](criterion.evaluate, criterion.jacobian, inputs, opt_method=FORMAL_ACQUISITION['method'], kwargs_op={'options': {'disp': False, 'maxiter': FORMAL_ACQUISITION['max_iters'], 'ftol': FORMAL_ACQUISITION['ftol'], 'gtol': FORMAL_ACQUISITION['gtol']}}, num_restarts=FORMAL_ACQUISITION['num_restarts'], parallel_restarts=False, init_method=FORMAL_ACQUISITION['init_method'])
        opt_elapsed = time.perf_counter() - started_opt
    proposed_x = np.asarray(proposed_x, dtype=np.float64).reshape(1, -1)
    objective = stable._scalar(criterion.evaluate(proposed_x), np)
    gradient = np.asarray(criterion.jacobian(proposed_x), dtype=float).reshape(-1)
    domain = np.asarray(request['domain'], dtype=float)
    finite_difference = stable._bounded_finite_difference(criterion, proposed_x, domain, 0.0001, np)
    relative_error = stable._relative_error(gradient, finite_difference, np)
    absolute_error = float(np.max(np.abs(gradient - finite_difference)))
    projected = stable._projected_gradient(gradient, proposed_x, domain, np)
    gradient_pass = bool(np.all(np.isfinite(gradient)) and np.all(np.isfinite(finite_difference)) and (relative_error <= 0.01 or absolute_error <= 1e-06))
    stationarity_pass = bool(np.all(np.isfinite(projected)) and np.max(np.abs(projected)) <= 0.001)
    (model_x, _) = _model_xy(model, np)
    duplicate = bool(np.any(np.all(model_x == proposed_x, axis=1)))
    in_domain = bool(np.all((proposed_x[0] >= domain[:, 0]) & (proposed_x[0] <= domain[:, 1])))
    finite = bool(np.all(np.isfinite(proposed_x)) and np.isfinite(objective))
    passed = gradient_pass and stationarity_pass and (not duplicate) and in_domain and finite
    response = _base_response(request_path, request, api)
    response.update({'status': 'PASS' if passed else 'FAIL', 'checkpoint': {'path': str(checkpoint)}, 'model': state, 'likelihood': _likelihood_state(likelihood, request['dimension'], np), 'proposal': {'x': proposed_x[0].tolist(), 'objective_score_minimized': objective, 'utility_score': -objective, 'objective_gradient': gradient.tolist(), 'bounded_finite_difference_gradient_eps_1e-4': finite_difference.tolist(), 'analytic_vs_finite_difference_relative_error': relative_error, 'analytic_vs_finite_difference_absolute_error': absolute_error, 'gradient_validation_pass': gradient_pass, 'projected_objective_gradient': projected.tolist(), 'projected_objective_gradient_linf': float(np.max(np.abs(projected))), 'box_stationarity_pass': stationarity_pass, 'duplicates_existing_train': duplicate, 'inside_frozen_domain': in_domain, 'finite': finite, 'status': 'pending_oracle' if passed else 'rejected_before_oracle', 'diagnostics': _point_diagnostics(model, likelihood, inputs, proposed_x, np)}, 'density_elapsed_seconds': density_elapsed, 'acquisition_optimization': {'elapsed_seconds': opt_elapsed, 'captured_stdout': stdout.getvalue(), 'captured_stderr': stderr.getvalue()}, 'information_boundary': 'checkpoint contains shared initial train plus preceding confirmed US-LW training labels only; no audit/test/reference labels'})
    return response

def _load_prior_audit(path, dimension, np):
    if path is None:
        return (np.empty((0, dimension), dtype=np.float64), None)
    source = Path(path).resolve()
    if not source.is_file():
        raise SessionInputError('prior_audit_npz is absent')
    with np.load(str(source), allow_pickle=False) as archive:
        if set(archive.files) != {'x'}:
            raise SessionInputError('prior_audit_npz may contain only x (never y)')
        x = _matrix(archive['x'], 'prior audit x', np, columns=dimension)
    return (x, {'path': str(source), 'rows': len(x)})

def _row_keys(values, np):
    values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if values.ndim != 2:
        raise ValueError('row-key input must be a matrix')
    dtype = np.dtype((np.void, values.dtype.itemsize * values.shape[1]))
    return values.view(dtype).reshape(-1)

def _merge_top(top_x, top_scores, top_ordinals, x, scores, ordinals, k, np):
    if len(top_x):
        combined_x = np.vstack((top_x, x))
        combined_scores = np.concatenate((top_scores, scores))
        combined_ordinals = np.concatenate((top_ordinals, ordinals))
    else:
        (combined_x, combined_scores, combined_ordinals) = (x, scores, ordinals)
    order = np.lexsort((combined_ordinals, combined_scores))
    keep = []
    seen = set()
    for index in order:
        key = combined_x[int(index)].tobytes()
        if key in seen:
            continue
        seen.add(key)
        keep.append(int(index))
        if len(keep) == k:
            break
    keep = np.asarray(keep, dtype=np.int64)
    return (combined_x[keep], combined_scores[keep], combined_ordinals[keep])

def _audit(request_path, request, api):
    np = api['np']
    if int(request['pool_size']) != AUDIT_POOL_SIZE:
        raise SessionInputError('formal audit pool is frozen to 1,000,000')
    if int(request['chunk_size']) != AUDIT_CHUNK_SIZE:
        raise SessionInputError('formal audit chunk size is frozen to 4096')
    if int(request['top_k']) != AUDIT_TOP_K:
        raise SessionInputError('formal audit selection is frozen to 250')
    (checkpoint, model, state) = _load_checkpoint(request['checkpoint_in'], request['expected_model'], request['dimension'], api)
    (prior_audit, prior_record) = _load_prior_audit(request['prior_audit_npz'], request['dimension'], np)
    (model_x, _) = _model_xy(model, np)
    excluded = np.vstack((model_x, prior_audit))
    excluded_keys = _row_keys(excluded, np)
    inputs = _build_inputs(request['dimension'], request['domain'], api)
    started_density = time.perf_counter()
    with stable._seeded_author_rng(np, request['seed']):
        likelihood = _likelihood(model, inputs, request['seed'], api, chunked=True)
    density_elapsed = time.perf_counter() - started_density
    criterion = api['US_LW'](model, inputs, likelihood=likelihood)
    candidate_rng = np.random.RandomState(request['candidate_seed'])
    all_scores = np.empty(AUDIT_POOL_SIZE, dtype=np.float64)
    top_x = np.empty((0, request['dimension']), dtype=np.float64)
    top_scores = np.empty(0, dtype=np.float64)
    top_ordinals = np.empty(0, dtype=np.int64)
    accepted = 0
    raw_draws = 0
    excluded_matches = 0
    nonfinite = 0
    started_scoring = time.perf_counter()
    domain = np.asarray(request['domain'], dtype=np.float64)
    while accepted < AUDIT_POOL_SIZE:
        requested = min(AUDIT_CHUNK_SIZE, AUDIT_POOL_SIZE - accepted)
        draw = candidate_rng.normal(size=(requested, request['dimension']))
        raw_draws += len(draw)
        inside = np.all((draw >= domain[:, 0]) & (draw <= domain[:, 1]), axis=1)
        points = np.ascontiguousarray(draw[inside], dtype=np.float64)
        if len(points) > AUDIT_POOL_SIZE - accepted:
            points = points[:AUDIT_POOL_SIZE - accepted]
        if not len(points):
            continue
        scores = np.asarray(criterion.evaluate(points), dtype=np.float64).reshape(-1)
        all_scores[accepted:accepted + len(points)] = scores
        finite_mask = np.isfinite(scores)
        nonfinite += int(np.sum(~finite_mask))
        eligible = finite_mask.copy()
        if len(excluded_keys):
            matches = np.isin(_row_keys(points, np), excluded_keys)
            excluded_matches += int(np.sum(matches))
            eligible &= ~matches
        if np.any(eligible):
            ordinals = accepted + np.nonzero(eligible)[0]
            (top_x, top_scores, top_ordinals) = _merge_top(top_x, top_scores, top_ordinals, points[eligible], scores[eligible], ordinals.astype(np.int64), AUDIT_TOP_K, np)
        accepted += len(points)
    scoring_elapsed = time.perf_counter() - started_scoring
    if nonfinite:
        raise RuntimeError('US-LW produced {} non-finite audit scores'.format(nonfinite))
    if len(top_x) != AUDIT_TOP_K:
        raise RuntimeError('audit stream did not yield 250 unique eligible points')
    order = np.lexsort((top_ordinals, top_scores))
    (top_x, top_scores, top_ordinals) = (top_x[order], top_scores[order], top_ordinals[order])
    diagnostics = _point_diagnostics(model, likelihood, inputs, top_x, np)
    selection_path = Path(request['selection_npz']).resolve()
    _atomic_npz(selection_path, np, x=top_x, objective_score=top_scores, utility_score=-top_scores, candidate_ordinal=top_ordinals, predicted_mean=np.asarray(diagnostics['predicted_mean'], dtype=np.float64), predicted_variance=np.asarray(diagnostics['predicted_variance'], dtype=np.float64), output_density_at_predicted_mean=np.asarray(diagnostics['output_density_at_predicted_mean'], dtype=np.float64), log_output_density_at_predicted_mean=np.asarray(diagnostics['log_output_density_at_predicted_mean'], dtype=np.float64), gmm_likelihood_weight=np.asarray(diagnostics['gmm_likelihood_weight'], dtype=np.float64))
    retained_one_dimensional = math.erf(6.0 / math.sqrt(2.0))
    retained_mass = retained_one_dimensional ** request['dimension']
    quantile_probabilities = np.asarray([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0], dtype=float)
    response = _base_response(request_path, request, api)
    response.update({'status': 'PASS', 'checkpoint': {'path': str(checkpoint)}, 'model': state, 'likelihood': _likelihood_state(likelihood, request['dimension'], np), 'audit_stream': {'distribution': 'N(0,I) rejected outside frozen [-6,6]^d domain', 'candidate_seed': request['candidate_seed'], 'accepted_rows': AUDIT_POOL_SIZE, 'raw_draws': raw_draws, 'outside_domain_rejections': raw_draws - AUDIT_POOL_SIZE, 'theoretical_retained_mass': retained_mass, 'theoretical_truncation_mass_loss': 1.0 - retained_mass, 'chunk_size': AUDIT_CHUNK_SIZE, 'score_quantile_probabilities': quantile_probabilities.tolist(), 'objective_score_quantiles': np.quantile(all_scores, quantile_probabilities).tolist(), 'nonfinite_scores': nonfinite, 'excluded_existing_train_and_prior_audit_matches': excluded_matches, 'full_candidate_matrix_persisted': False, 'full_score_vector_persisted': False, 'kind': 'post-round method-internal diagnostic pool; not the native continuous training optimiser and not GP feedback'}, 'prior_audit': prior_record, 'selection': {'path': str(selection_path), 'rows': AUDIT_TOP_K, 'best_objective': float(top_scores[0]), 'worst_selected_objective': float(top_scores[-1]), 'labels_present': False}, 'density_elapsed_seconds': density_elapsed, 'audit_scoring_elapsed_seconds': scoring_elapsed, 'information_boundary': 'audit candidates and scores use the frozen GP only; selected audit labels are evaluated by the outer collector and never returned here'})
    return response

def run_action(request_path):
    api = stable.load_official_api()
    (request_path, request) = _load_request(request_path)
    handlers = {'initialize': _initialize, 'propose': _propose, 'observe': _observe, 'audit': _audit}
    return handlers[request['action']](request_path, request, api)

def inspect_runtime():
    api = stable.load_official_api()
    source = Path(stable.GPSEARCH_ROOT) / 'gpsearch' / 'core' / 'likelihood.py'
    text = source.read_text(encoding='utf-8')
    required_literals = ['n_samples = int(1e5)', 'n_samples = int(1e6)', 'size=20000']
    missing = [value for value in required_literals if value not in text]
    return {'status': 'PASS' if not missing else 'FAIL', 'schema_version': SCHEMA, 'third_party_commit': stable.verify_checkout(), 'runtime': _runtime(api), 'formal_gp': FORMAL_GP, 'formal_acquisition': FORMAL_ACQUISITION, 'author_likelihood_source': str(source), 'author_density_literals_verified': not missing, 'missing_literals': missing, 'checkpoint_api': {'save_model': hasattr(api['GPy'].core.Model, 'save_model'), 'load_model': hasattr(api['GPy'].core.Model, 'load_model')}, 'oracle_api_present': False}

def _max_abs_relative(reference, candidate, np):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    absolute = float(np.max(np.abs(reference - candidate)))
    scale = max(1e-15, float(np.max(np.abs(reference))))
    return (absolute, absolute / scale)

def self_test_chunking(seed=20260907):
    """Zero-HF equivalence fixture for the prediction-only memory adapter."""
    api = stable.load_official_api()
    np = api['np']
    x_train = np.linspace(-2.5, 2.5, 17)[:, None]
    y_train = (0.2 * x_train + np.sin(1.7 * x_train)).reshape(-1, 1)
    model = api['GPy'].models.GPRegression(X=x_train, Y=y_train, kernel=api['RBF'](input_dim=1, ARD=True, variance=1.3, lengthscale=np.asarray([0.75])), normalizer=True, noise_var=FORMAL_GP['noise_variance'])
    model.Gaussian_noise.variance.fix(FORMAL_GP['noise_variance'])
    probe = np.linspace(-5.5, 5.5, 257)[:, None]
    (full_mean, full_variance) = model.predict(probe)
    (chunk_mean, chunk_variance) = _ChunkedPredictProxy(model, np, 17).predict(probe)
    (mean_abs, mean_rel) = _max_abs_relative(full_mean, chunk_mean, np)
    (var_abs, var_rel) = _max_abs_relative(full_variance, chunk_variance, np)
    inputs = api['GaussianInputs']([[-6.0, 6.0]], np.zeros(1), np.eye(1))
    with stable._seeded_author_rng(np, seed):
        full_likelihood = _likelihood(model, inputs, seed, api, chunked=False)
    with stable._seeded_author_rng(np, seed):
        chunk_likelihood = _likelihood(model, inputs, seed, api, chunked=True)
    full_criterion = api['US_LW'](model, inputs, likelihood=full_likelihood)
    chunk_criterion = api['US_LW'](model, inputs, likelihood=chunk_likelihood)
    criterion_probe = np.asarray([[-3.1], [-0.7], [0.2], [1.8], [4.9]])
    full_value = full_criterion.evaluate(criterion_probe)
    chunk_value = chunk_criterion.evaluate(criterion_probe)
    full_gradient = full_criterion.jacobian(criterion_probe)
    chunk_gradient = chunk_criterion.jacobian(criterion_probe)
    (value_abs, value_rel) = _max_abs_relative(full_value, chunk_value, np)
    (gradient_abs, gradient_rel) = _max_abs_relative(full_gradient, chunk_gradient, np)
    tolerances = {'prediction_mean_max_abs': 1e-12, 'prediction_variance_max_abs': 1e-12, 'uslw_value_max_abs': 1e-10, 'uslw_gradient_max_abs': 1e-09}
    checks = {'prediction_mean': mean_abs <= tolerances['prediction_mean_max_abs'], 'prediction_variance': var_abs <= tolerances['prediction_variance_max_abs'], 'uslw_value': value_abs <= tolerances['uslw_value_max_abs'], 'uslw_gradient': gradient_abs <= tolerances['uslw_gradient_max_abs']}
    with tempfile.TemporaryDirectory(prefix='gpsearch_session_') as temporary:
        checkpoint = Path(temporary) / 'model.zip'
        saved = _save_checkpoint(model, checkpoint, api)
        loaded = api['GPy'].core.Model.load_model(str(checkpoint))
        loaded_state = _model_state(loaded, np)
        old_x, old_y = _model_xy(model, np)
        new_x, new_y = _model_xy(loaded, np)
        checks['checkpoint_roundtrip'] = np.array_equal(old_x, new_x) and np.array_equal(old_y, new_y)
    return {'status': 'PASS' if all(checks.values()) else 'FAIL', 'schema_version': SCHEMA, 'third_party_commit': stable.verify_checkout(), 'runtime': _runtime(api), 'fixture': {'kind': 'synthetic analytic array; zero project/HF calls', 'train_rows': len(x_train), 'prediction_probe_rows': len(probe), 'criterion_probe_rows': len(criterion_probe), 'dimension': 1, 'seed': seed}, 'tolerances': tolerances, 'checks': checks, 'errors': {'prediction_mean': {'max_abs': mean_abs, 'max_relative': mean_rel}, 'prediction_variance': {'max_abs': var_abs, 'max_relative': var_rel}, 'uslw_value': {'max_abs': value_abs, 'max_relative': value_rel}, 'uslw_gradient': {'max_abs': gradient_abs, 'max_relative': gradient_rel}}, 'interpretation': 'Only the GPy prediction allocation is partitioned; density samples, their order, KDE/GMM fitting and native US_LW equations are unchanged. BLAS batching can in principle change last-bit reduction order.', 'hf_calls': 0}

def self_test_session(seed=20260907):
    """Zero-HF transaction fixture for initialize -> propose -> observe."""
    api = stable.load_official_api()
    np = api['np']
    dimension = 1
    domain = [[-6.0, 6.0]]
    x = np.linspace(-2.2, 2.2, 15)[:, None]
    y = (0.15 * x[:, 0] + np.sin(1.9 * x[:, 0])).astype(np.float64)

    def expected(checkpoint):
        model = checkpoint['model']
        return {'rows': model['rows'], 'dimension': model['dimension'], 'training_data': checkpoint['training_data']}
    with tempfile.TemporaryDirectory(prefix='gpsearch_session_actions_') as temporary:
        root = Path(temporary)
        initial_path = root / 'initial.npz'
        _atomic_npz(initial_path, np, x=x, y=y, origin_physical_calls=np.asarray(len(x), dtype=np.int64))
        initialize_request = {'schema_version': SCHEMA, 'action': 'initialize', 'request_id': 'zero-hf-session.initialize', 'expected_third_party_commit': EXPECTED_COMMIT, 'dimension': dimension, 'domain': domain, 'seed': seed, 'expected_initial_count': len(x), 'initial_data_npz': str(initial_path), 'checkpoint_out': str(root / 'initial.zip')}
        initialize_path = root / 'initialize.json'
        _atomic_json(initialize_path, initialize_request)
        (_, initialize_request) = _load_request(initialize_path)
        initialized = _initialize(initialize_path, initialize_request, api)
        propose_request = {'schema_version': SCHEMA, 'action': 'propose', 'request_id': 'zero-hf-session.propose', 'expected_third_party_commit': EXPECTED_COMMIT, 'dimension': dimension, 'domain': domain, 'seed': seed + 1 & 4294967295, 'checkpoint_in': initialized['checkpoint']['path'], 'expected_model': expected(initialized['checkpoint'])}
        propose_path = root / 'propose.json'
        _atomic_json(propose_path, propose_request)
        (_, propose_request) = _load_request(propose_path)
        proposed = _propose(propose_path, propose_request, api)
        point = np.asarray(proposed['proposal']['x'], dtype=np.float64).reshape(1, 1)
        observed_y = np.asarray([0.15 * point[0, 0] + np.sin(1.9 * point[0, 0])], dtype=np.float64)
        observation_path = root / 'observation.npz'
        _atomic_npz(observation_path, np, x=point, y=observed_y, origin_physical_calls=np.asarray(1, dtype=np.int64))
        observe_request = {'schema_version': SCHEMA, 'action': 'observe', 'request_id': 'zero-hf-session.observe', 'expected_third_party_commit': EXPECTED_COMMIT, 'dimension': dimension, 'domain': domain, 'seed': seed + 2 & 4294967295, 'checkpoint_in': initialized['checkpoint']['path'], 'checkpoint_out': str(root / 'observed.zip'), 'expected_model': expected(initialized['checkpoint']), 'observation_npz': str(observation_path), 'expected_proposal_x': point.tolist()}
        observe_path = root / 'observe.json'
        _atomic_json(observe_path, observe_request)
        (_, observe_request) = _load_request(observe_path)
        observed = _observe(observe_path, observe_request, api)
        audit_request = {'schema_version': SCHEMA, 'action': 'audit', 'request_id': 'zero-hf-session.audit', 'expected_third_party_commit': EXPECTED_COMMIT, 'dimension': dimension, 'domain': domain, 'seed': seed + 3 & 4294967295, 'candidate_seed': seed + 4 & 4294967295, 'checkpoint_in': observed['checkpoint']['path'], 'expected_model': expected(observed['checkpoint']), 'prior_audit_npz': None, 'selection_npz': str(root / 'audit_selection.npz'), 'pool_size': AUDIT_POOL_SIZE, 'chunk_size': AUDIT_CHUNK_SIZE, 'top_k': AUDIT_TOP_K}
        audit_path = root / 'audit.json'
        _atomic_json(audit_path, audit_request)
        (_, audit_request) = _load_request(audit_path)
        audited = _audit(audit_path, audit_request, api)
        with np.load(audited['selection']['path'], allow_pickle=False) as archive:
            audit_fields = set(archive.files)
            audit_x = np.asarray(archive['x'], dtype=np.float64)
        checks = {'initialize_pass': initialized['status'] == 'PASS', 'proposal_pass': proposed['status'] == 'PASS', 'proposal_pending_oracle': proposed['proposal']['status'] == 'pending_oracle', 'observe_pass': observed['status'] == 'PASS', 'one_row_appended': observed['model_after']['rows'] == len(x) + 1, 'checkpoint_roundtrip': Path(observed['checkpoint']['path']).is_file(), 'audit_pass': audited['status'] == 'PASS', 'audit_pool_exact': audited['audit_stream']['accepted_rows'] == AUDIT_POOL_SIZE, 'audit_top_250_unique': audit_x.shape == (AUDIT_TOP_K, dimension) and np.unique(audit_x, axis=0).shape[0] == AUDIT_TOP_K, 'audit_selection_has_no_labels': 'y' not in audit_fields, 'no_project_data': True}
        audit_selection_record = dict(audited['selection'])
        audit_selection_record['path'] = '<ephemeral zero-HF fixture>'
        compact = {'initial_model': initialized['model_after'], 'proposal': proposed['proposal'], 'observed_model': observed['model_after'], 'initial_gp_elapsed_seconds': initialized['gp_optimization']['elapsed_seconds'], 'density_elapsed_seconds': proposed['density_elapsed_seconds'], 'proposal_elapsed_seconds': proposed['acquisition_optimization']['elapsed_seconds'], 'post_label_gp_elapsed_seconds': observed['gp_optimization']['elapsed_seconds'], 'audit': {'selection': audit_selection_record, 'stream': audited['audit_stream'], 'density_elapsed_seconds': audited['density_elapsed_seconds'], 'scoring_elapsed_seconds': audited['audit_scoring_elapsed_seconds']}}
    return {'status': 'PASS' if all(checks.values()) else 'FAIL', 'schema_version': SCHEMA, 'third_party_commit': stable.verify_checkout(), 'runtime': _runtime(api), 'fixture': {'kind': 'synthetic analytic function; zero project/HF calls', 'initial_rows': len(x), 'dimension': dimension, 'seed': seed}, 'checks': checks, 'transaction': compact, 'hf_calls': 0}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--request', type=Path)
    group.add_argument('--inspect-runtime', action='store_true')
    group.add_argument('--self-test-chunking', action='store_true')
    group.add_argument('--self-test-session', action='store_true')
    parser.add_argument('--response', type=Path)
    args = parser.parse_args(argv)
    if args.inspect_runtime:
        report = inspect_runtime()
    elif args.self_test_chunking:
        report = self_test_chunking()
    elif args.self_test_session:
        report = self_test_session()
    else:
        report = run_action(args.request)
    if args.response:
        _atomic_json(args.response, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get('status') == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
