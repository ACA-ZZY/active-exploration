"""Isolated worker for the pinned authors' gpsearch implementation"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
EXPECTED_COMMIT = '8c5d77b3f0df803029d26828966274d6185c12e4'
REQUEST_SCHEMA = 'gpsearch.sequential-step.v1'
MORETEST_ROOT = Path(__file__).resolve().parent.parent
GPSEARCH_ROOT = MORETEST_ROOT / 'third_party' / 'gpsearch'

class GPSearchUnavailable(RuntimeError):
    """Raised when the authors' exact checkout/runtime is unavailable."""

def verify_checkout() -> str:
    if not GPSEARCH_ROOT.is_dir():
        raise GPSearchUnavailable(f'gpsearch checkout is absent: {GPSEARCH_ROOT}')
    try:
        proc = subprocess.run(['git', '-C', str(GPSEARCH_ROOT), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GPSearchUnavailable(f'Cannot verify gpsearch checkout: {exc}') from exc
    head = proc.stdout.strip()
    if head != EXPECTED_COMMIT:
        raise GPSearchUnavailable(f'gpsearch provenance mismatch: expected {EXPECTED_COMMIT}, found {head}')
    return head

def load_official_api():
    verify_checkout()
    checkout = str(GPSEARCH_ROOT)
    if checkout not in sys.path:
        sys.path.insert(0, checkout)
    try:
        import GPy
        import KDEpy
        import numpy as np
        import scipy
        import sklearn
        from gpsearch.core.acquisitions import IVR_LW, US_LW
        from gpsearch.core.inputs import GaussianInputs, LogNormalInputs, UniformInputs
        from gpsearch.core.kernels import RBF
        from gpsearch.core.likelihood import Likelihood
        from gpsearch.core.minimizers import funmin
        from gpsearch.core.utils import jacobian_fdiff
    except ImportError as exc:
        raise GPSearchUnavailable('Pinned gpsearch requires its isolated Python 3.8 / GPy 1.9.9 runtime') from exc
    return {'GPy': GPy, 'KDEpy': KDEpy, 'np': np, 'scipy': scipy, 'sklearn': sklearn, 'IVR_LW': IVR_LW, 'Likelihood': Likelihood, 'GaussianInputs': GaussianInputs, 'LogNormalInputs': LogNormalInputs, 'US_LW': US_LW, 'UniformInputs': UniformInputs, 'funmin': funmin, 'jacobian_fdiff': jacobian_fdiff, 'RBF': RBF}

def _require_keys(mapping, required, allowed, label):
    if not isinstance(mapping, dict):
        raise ValueError('{} must be a JSON object'.format(label))
    missing = sorted(set(required) - set(mapping))
    unexpected = sorted(set(mapping) - set(allowed))
    if missing:
        raise ValueError('{} is missing: {}'.format(label, ', '.join(missing)))
    if unexpected:
        raise ValueError('{} contains unsupported fields: {}'.format(label, ', '.join(unexpected)))

def _load_request(request_path):
    request_path = Path(request_path).resolve()
    with request_path.open('r', encoding='utf-8') as stream:
        request = json.load(stream)
    allowed = {'schema_version', 'request_id', 'method', 'data_npz', 'input_distribution', 'seed', 'budget', 'gp', 'optimizer', 'expected_third_party_commit'}
    _require_keys(request, {'schema_version', 'method', 'data_npz', 'input_distribution', 'seed', 'budget', 'gp', 'optimizer'}, allowed, 'request')
    if request['schema_version'] != REQUEST_SCHEMA:
        raise ValueError('unsupported request schema: {}'.format(request['schema_version']))
    expected = request.get('expected_third_party_commit', EXPECTED_COMMIT)
    if expected != EXPECTED_COMMIT:
        raise ValueError('request expected commit {} but this adapter is pinned to {}'.format(expected, EXPECTED_COMMIT))
    method = str(request['method']).upper()
    if method not in {'US_LW', 'IVR_LW'}:
        raise ValueError('method must be US_LW or IVR_LW')
    request['method'] = method
    try:
        seed = int(request['seed'])
    except (TypeError, ValueError) as exc:
        raise ValueError('seed must be an integer') from exc
    if seed < 0 or seed > 2 ** 32 - 1:
        raise ValueError('seed must be in [0, 2**32-1]')
    request['seed'] = seed
    data_path = Path(request['data_npz'])
    if not data_path.is_absolute():
        data_path = request_path.parent / data_path
    request['data_npz'] = str(data_path.resolve())
    return (request_path, request)

def _numeric_matrix(values, name, np, *, columns=None):
    array = np.asarray(values)
    if array.dtype.kind not in 'biufc':
        raise ValueError('{} must be a numeric array'.format(name))
    array = np.asarray(array, dtype=np.float64)
    if name.endswith('_y'):
        array = array.reshape(-1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError('{} must be a two-dimensional sample matrix'.format(name))
    if columns is not None and array.shape[1] != columns:
        raise ValueError('{} has {} columns; expected {}'.format(name, array.shape[1], columns))
    if not np.all(np.isfinite(array)):
        raise ValueError('{} contains non-finite values'.format(name))
    return array

def _load_training_data(data_path, np):
    data_path = Path(data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError('training NPZ is absent: {}'.format(data_path))
    allowed = {'initial_train_x', 'initial_train_y', 'confirmed_train_x', 'confirmed_train_y'}
    required = {'initial_train_x', 'initial_train_y'}
    with np.load(str(data_path), allow_pickle=False) as archive:
        names = set(archive.files)
        missing = sorted(required - names)
        unexpected = sorted(names - allowed)
        if missing:
            raise ValueError('training NPZ is missing: {}'.format(', '.join(missing)))
        if unexpected:
            raise ValueError('training NPZ may contain only initial/confirmed training data; rejected fields: {}'.format(', '.join(unexpected)))
        has_history_x = 'confirmed_train_x' in names
        has_history_y = 'confirmed_train_y' in names
        if has_history_x != has_history_y:
            raise ValueError('confirmed_train_x and confirmed_train_y must be supplied together')
        initial_x = _numeric_matrix(archive['initial_train_x'], 'initial_train_x', np)
        initial_y = _numeric_matrix(archive['initial_train_y'], 'initial_train_y', np)
        if has_history_x:
            history_x = _numeric_matrix(archive['confirmed_train_x'], 'confirmed_train_x', np, columns=initial_x.shape[1])
            history_y = _numeric_matrix(archive['confirmed_train_y'], 'confirmed_train_y', np)
        else:
            history_x = np.empty((0, initial_x.shape[1]), dtype=np.float64)
            history_y = np.empty((0, 1), dtype=np.float64)
    if len(initial_x) != len(initial_y):
        raise ValueError('initial_train_x/y row counts differ')
    if len(history_x) != len(history_y):
        raise ValueError('confirmed_train_x/y row counts differ')
    if len(initial_x) < 2:
        raise ValueError('at least two initial observations are required')
    return {'initial_x': initial_x, 'initial_y': initial_y, 'history_x': history_x, 'history_y': history_y, 'x': np.vstack((initial_x, history_x)), 'y': np.vstack((initial_y, history_y)), 'path': str(data_path)}

def _validate_budget(budget, history_count):
    fields = {'oracle_total', 'training_quota', 'audit_quota', 'confirmed_training_calls', 'confirmed_audit_calls'}
    _require_keys(budget, fields, fields, 'budget')
    parsed = {}
    for key in fields:
        try:
            value = int(budget[key])
        except (TypeError, ValueError) as exc:
            raise ValueError('budget.{} must be an integer'.format(key)) from exc
        if value < 0:
            raise ValueError('budget.{} must be non-negative'.format(key))
        parsed[key] = value
    if parsed['training_quota'] + parsed['audit_quota'] != parsed['oracle_total']:
        raise ValueError('training_quota + audit_quota must equal oracle_total')
    if parsed['confirmed_training_calls'] != history_count:
        raise ValueError('confirmed training history has {} rows but budget declares {}'.format(history_count, parsed['confirmed_training_calls']))
    consumed = parsed['confirmed_training_calls'] + parsed['confirmed_audit_calls']
    if consumed >= parsed['oracle_total']:
        raise ValueError('oracle budget is exhausted')
    if parsed['confirmed_training_calls'] >= parsed['training_quota']:
        raise ValueError('training-feedback quota is exhausted')
    return parsed

def _positive_definite_covariance(value, dimension, np):
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.ndim == 0:
        covariance = np.eye(dimension) * float(covariance)
    elif covariance.ndim == 1:
        if len(covariance) != dimension:
            raise ValueError('distribution covariance diagonal has wrong length')
        covariance = np.diag(covariance)
    if covariance.shape != (dimension, dimension):
        raise ValueError('distribution covariance has wrong shape')
    if not np.all(np.isfinite(covariance)):
        raise ValueError('distribution covariance contains non-finite values')
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError('distribution covariance must be symmetric')
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError('distribution covariance must be positive definite') from exc
    return covariance

def _build_inputs(specification, dimension, api):
    np = api['np']
    allowed_common = {'kind', 'domain', 'mean', 'covariance'}
    _require_keys(specification, {'kind', 'domain'}, allowed_common, 'input_distribution')
    domain = np.asarray(specification['domain'], dtype=np.float64)
    if domain.shape != (dimension, 2):
        raise ValueError('input_distribution.domain must have shape [dimension, 2]')
    if not np.all(np.isfinite(domain)) or not np.all(domain[:, 0] < domain[:, 1]):
        raise ValueError('each input domain bound must be finite and increasing')
    kind = str(specification['kind']).lower()
    if kind == 'uniform':
        if 'mean' in specification or 'covariance' in specification:
            raise ValueError('uniform input_distribution does not accept mean/covariance')
        inputs = api['UniformInputs'](domain.tolist())
    elif kind in {'gaussian', 'lognormal'}:
        if 'mean' not in specification or 'covariance' not in specification:
            raise ValueError('{} distribution requires mean and covariance'.format(kind))
        mean = np.asarray(specification['mean'], dtype=np.float64).reshape(-1)
        if mean.shape != (dimension,) or not np.all(np.isfinite(mean)):
            raise ValueError('distribution mean has wrong shape or non-finite values')
        covariance = _positive_definite_covariance(specification['covariance'], dimension, np)
        cls = api['GaussianInputs'] if kind == 'gaussian' else api['LogNormalInputs']
        inputs = cls(domain.tolist(), mean, covariance)
    else:
        raise ValueError('input_distribution.kind must be uniform, gaussian, or lognormal')
    return (inputs, {'kind': kind, 'domain': domain.tolist(), 'mean': specification.get('mean'), 'covariance': specification.get('covariance')})

def _validated_gp_options(options):
    defaults = {'normalize_y': True, 'fix_noise': True, 'noise_variance': 1e-08, 'num_restarts': 10, 'optimizer': 'bfgs', 'max_iters': 1000}
    if options is None:
        raise ValueError('gp options must be explicit in a formal request')
    _require_keys(options, set(defaults), set(defaults), 'gp')
    merged = dict(defaults)
    merged.update(options)
    if not isinstance(merged['normalize_y'], bool) or not isinstance(merged['fix_noise'], bool):
        raise ValueError('gp.normalize_y and gp.fix_noise must be JSON booleans')
    merged['noise_variance'] = float(merged['noise_variance'])
    merged['num_restarts'] = int(merged['num_restarts'])
    merged['max_iters'] = int(merged['max_iters'])
    merged['optimizer'] = str(merged['optimizer']).lower()
    if not merged['noise_variance'] > 0:
        raise ValueError('gp.noise_variance must be positive')
    if merged['num_restarts'] <= 0 or merged['max_iters'] <= 0:
        raise ValueError('gp.num_restarts and gp.max_iters must be positive')
    if merged['optimizer'] not in {'bfgs', 'lbfgsb', 'scg', 'tnc'}:
        raise ValueError('unsupported GPy optimizer')
    return merged

def _validated_acquisition_options(options):
    defaults = {'method': 'l-bfgs-b', 'num_restarts': None, 'init_method': None, 'max_iters': 1000, 'ftol': 1e-09, 'gtol': 1e-07, 'gmm_components': 2, 'gmm_covariance_type': 'full'}
    if options is None:
        raise ValueError('optimizer options must be explicit in a formal request')
    _require_keys(options, set(defaults), set(defaults), 'optimizer')
    merged = dict(defaults)
    merged.update(options)
    merged['method'] = str(merged['method']).lower()
    if merged['method'] not in {'l-bfgs-b', 'slsqp', 'tnc'}:
        raise ValueError('optimizer.method must be l-bfgs-b, slsqp, or tnc')
    if merged['num_restarts'] is not None:
        merged['num_restarts'] = int(merged['num_restarts'])
        if merged['num_restarts'] < 0:
            raise ValueError('optimizer.num_restarts must be non-negative')
    if merged['init_method'] not in {None, 'sample_fun'}:
        raise ValueError('optimizer.init_method must be null or sample_fun')
    merged['max_iters'] = int(merged['max_iters'])
    merged['ftol'] = float(merged['ftol'])
    merged['gtol'] = float(merged['gtol'])
    merged['gmm_components'] = int(merged['gmm_components'])
    merged['gmm_covariance_type'] = str(merged['gmm_covariance_type'])
    if merged['max_iters'] <= 0 or merged['gmm_components'] <= 0:
        raise ValueError('optimizer max_iters/components must be positive')
    if merged['gmm_covariance_type'] not in {'full', 'tied', 'diag', 'spherical'}:
        raise ValueError('unsupported GMM covariance type')
    return merged

@contextlib.contextmanager
def _seeded_author_rng(np, seed):
    """Pin the otherwise-unseeded Generator in the authors' Likelihood._fit_gmm."""
    original_default_rng = np.random.default_rng

    def seeded_default_rng(*args, **kwargs):
        if not args and (not kwargs):
            return original_default_rng(seed)
        return original_default_rng(*args, **kwargs)
    np.random.seed(seed)
    np.random.default_rng = seeded_default_rng
    try:
        yield
    finally:
        np.random.default_rng = original_default_rng

def _scalar(value, np):
    return float(np.asarray(value, dtype=float).reshape(-1)[0])

def _model_parameters(model, np):
    return {'kernel': 'gpsearch.core.kernels.RBF', 'variance': _scalar(model.kern.variance.values, np), 'lengthscale': np.asarray(model.kern.lengthscale.values, dtype=float).reshape(-1).tolist(), 'noise_variance': _scalar(model.Gaussian_noise.variance.values, np), 'log_likelihood': float(model.log_likelihood()), 'parameter_names': [str(name) for name in model.parameter_names()], 'parameter_array': np.asarray(model.param_array, dtype=float).tolist()}

def _save_checkpoint(model, checkpoint_path):
    target = Path(checkpoint_path).resolve()
    if target.suffix.lower() != '.zip':
        target = target.with_suffix(target.suffix + '.zip' if target.suffix else '.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_base = target.parent / '.{}.{}'.format(target.stem, uuid.uuid4().hex)
    temporary_zip = Path(str(temporary_base) + '.zip')
    try:
        model.save_model(str(temporary_base), compress=True, save_data=True)
        os.replace(str(temporary_zip), str(target))
    finally:
        if temporary_zip.exists():
            temporary_zip.unlink()
    return {'path': str(target), 'format': 'GPy-json-gzip'}

def _atomic_json(path, payload):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / '.{}.{}.tmp'.format(target.name, uuid.uuid4().hex)
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()

def run_sequential_step(request_path, checkpoint_path=None):
    """Return one native US-LW/IVR-LW proposal without evaluating its oracle"""
    api = load_official_api()
    np = api['np']
    (request_path, request) = _load_request(request_path)
    data = _load_training_data(request['data_npz'], np)
    budget = _validate_budget(request['budget'], len(data['history_x']))
    (inputs, distribution) = _build_inputs(request['input_distribution'], data['x'].shape[1], api)
    gp_options = _validated_gp_options(request.get('gp'))
    acquisition_options = _validated_acquisition_options(request.get('optimizer'))
    seed = request['seed']
    model = api['GPy'].models.GPRegression(X=data['x'], Y=data['y'], kernel=api['RBF'](input_dim=data['x'].shape[1], ARD=True), normalizer=gp_options['normalize_y'], noise_var=gp_options['noise_variance'])
    if gp_options['fix_noise']:
        model.Gaussian_noise.variance.fix(gp_options['noise_variance'])
    before = _model_parameters(model, np)
    (gp_stdout, gp_stderr) = (io.StringIO(), io.StringIO())
    started_gp = time.perf_counter()
    with _seeded_author_rng(np, seed):
        with contextlib.redirect_stdout(gp_stdout), contextlib.redirect_stderr(gp_stderr):
            model.optimize_restarts(num_restarts=gp_options['num_restarts'], optimizer=gp_options['optimizer'], max_iters=gp_options['max_iters'], verbose=False)
    elapsed_gp = time.perf_counter() - started_gp
    after = _model_parameters(model, np)
    with _seeded_author_rng(np, seed):
        likelihood = api['Likelihood'](model, inputs, kwargs_gmm={'n_components': acquisition_options['gmm_components'], 'covariance_type': acquisition_options['gmm_covariance_type'], 'random_state': seed})
        criterion_class = api[request['method']]
        criterion = criterion_class(model, inputs, likelihood=likelihood)
        (acquisition_stdout, acquisition_stderr) = (io.StringIO(), io.StringIO())
        started_acquisition = time.perf_counter()
        with contextlib.redirect_stdout(acquisition_stdout), contextlib.redirect_stderr(acquisition_stderr):
            proposed_x = api['funmin'](criterion.evaluate, criterion.jacobian, inputs, opt_method=acquisition_options['method'], kwargs_op={'options': {'disp': False, 'maxiter': acquisition_options['max_iters'], 'ftol': acquisition_options['ftol'], 'gtol': acquisition_options['gtol']}}, num_restarts=acquisition_options['num_restarts'], parallel_restarts=False, init_method=acquisition_options['init_method'])
        elapsed_acquisition = time.perf_counter() - started_acquisition
    proposed_x = np.asarray(proposed_x, dtype=np.float64).reshape(1, -1)
    objective = _scalar(criterion.evaluate(proposed_x), np)
    gradient = np.asarray(criterion.jacobian(proposed_x), dtype=float).reshape(-1)
    domain = np.asarray(distribution['domain'], dtype=float)
    finite_difference = _bounded_finite_difference(criterion, proposed_x, domain, 0.0001, np)
    gradient_relative_error = _relative_error(gradient, finite_difference, np)
    gradient_absolute_error = float(np.max(np.abs(gradient - finite_difference)))
    gradient_validation_pass = bool(np.all(np.isfinite(gradient)) and np.all(np.isfinite(finite_difference)) and (gradient_relative_error <= 0.01 or gradient_absolute_error <= 1e-06))
    projected_gradient = _projected_gradient(gradient, proposed_x, domain, np)
    stationarity_pass = bool(np.all(np.isfinite(projected_gradient)) and float(np.max(np.abs(projected_gradient))) <= 0.001)
    if proposed_x.shape[1] != data['x'].shape[1]:
        raise RuntimeError("authors' optimizer returned a point with the wrong dimension")
    if not np.all(np.isfinite(proposed_x)) or not np.isfinite(objective):
        raise RuntimeError("authors' optimizer returned a non-finite proposal or score")
    if not np.all((proposed_x[0] >= domain[:, 0]) & (proposed_x[0] <= domain[:, 1])):
        raise RuntimeError("authors' optimizer returned a point outside the declared domain")
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = _save_checkpoint(model, checkpoint_path)
    proposal_ordinal = budget['confirmed_training_calls'] + budget['confirmed_audit_calls']
    return {'status': 'PASS' if gradient_validation_pass and stationarity_pass else 'FAIL', 'schema_version': REQUEST_SCHEMA, 'request_id': request.get('request_id'), 'request_path': str(request_path), 'third_party': {'repository': 'https://github.com/ablancha/gpsearch', 'commit': EXPECTED_COMMIT, 'api': ['gpsearch.core.acquisitions.{}'.format(request['method']), 'gpsearch.core.likelihood.Likelihood', 'gpsearch.core.minimizers.funmin', 'gpsearch.core.kernels.RBF']}, 'runtime': {'python_executable': sys.executable, 'python': platform.python_version(), 'numpy': np.__version__, 'scipy': api['scipy'].__version__, 'GPy': api['GPy'].__version__, 'KDEpy': api['KDEpy'].__version__, 'scikit_learn': api['sklearn'].__version__, 'reproducibility_shim': 'seeded np.random.default_rng used internally by Likelihood._fit_gmm; formulas unchanged'}, 'data': {'path': data['path'], 'initial_training_count': len(data['initial_x']), 'confirmed_training_history_count': len(data['history_x']), 'effective_gp_training_count': len(data['x']), 'input_dimension': data['x'].shape[1], 'allowed_npz_fields': ['initial_train_x', 'initial_train_y', 'confirmed_train_x', 'confirmed_train_y'], 'audit_test_reference_feedback': False}, 'budget': budget, 'input_distribution': distribution, 'method': request['method'], 'proposal': {'x': proposed_x[0].tolist(), 'objective_score_minimized': objective, 'utility_score': -objective, 'objective_gradient': gradient.tolist(), 'bounded_finite_difference_gradient_eps_1e-4': finite_difference.tolist(), 'analytic_vs_finite_difference_relative_error': gradient_relative_error, 'analytic_vs_finite_difference_absolute_error': gradient_absolute_error, 'gradient_validation_pass': gradient_validation_pass, 'objective_gradient_linf': float(np.max(np.abs(gradient))), 'projected_objective_gradient': projected_gradient.tolist(), 'projected_objective_gradient_linf': float(np.max(np.abs(projected_gradient))), 'box_stationarity_pass': stationarity_pass, 'stream_ordinal_zero_based': proposal_ordinal, 'oracle_cost_if_confirmed': 1, 'status': 'pending_oracle'}, 'gp_optimization': {'options': gp_options, 'elapsed_seconds': elapsed_gp, 'parameters_before': before, 'parameters_after': after, 'captured_stdout': gp_stdout.getvalue(), 'captured_stderr': gp_stderr.getvalue()}, 'acquisition_optimization': {'options': acquisition_options, 'elapsed_seconds': elapsed_acquisition, 'captured_stdout': acquisition_stdout.getvalue(), 'captured_stderr': acquisition_stderr.getvalue(), 'sequential_contract': 'one point only; obtain its oracle label and append it to confirmed_train_x/y before requesting the next training-feedback point'}, 'checkpoint': checkpoint}

def _relative_error(a, b, np) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(1e-12, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / scale)

def _central_difference(criterion, x, eps, np):
    x = np.asarray(x, dtype=float)
    gradient = np.empty(x.shape[1], dtype=float)
    for axis in range(x.shape[1]):
        offset = np.zeros_like(x)
        offset[:, axis] = eps
        upper = np.asarray(criterion.evaluate(x + offset), dtype=float)
        lower = np.asarray(criterion.evaluate(x - offset), dtype=float)
        gradient[axis] = float((upper - lower).reshape(-1)[0] / (2.0 * eps))
    return gradient

def _bounded_finite_difference(criterion, x, domain, eps, np):
    """Finite difference that never probes outside the declared domain."""
    x = np.asarray(x, dtype=float)
    domain = np.asarray(domain, dtype=float)
    gradient = np.empty(x.shape[1], dtype=float)
    center = _scalar(criterion.evaluate(x), np)
    for axis in range(x.shape[1]):
        upper = x.copy()
        lower = x.copy()
        upper[:, axis] += eps
        lower[:, axis] -= eps
        if lower[0, axis] < domain[axis, 0]:
            upper_value = _scalar(criterion.evaluate(upper), np)
            gradient[axis] = (upper_value - center) / eps
        elif upper[0, axis] > domain[axis, 1]:
            lower_value = _scalar(criterion.evaluate(lower), np)
            gradient[axis] = (center - lower_value) / eps
        else:
            upper_value = _scalar(criterion.evaluate(upper), np)
            lower_value = _scalar(criterion.evaluate(lower), np)
            gradient[axis] = (upper_value - lower_value) / (2.0 * eps)
    return gradient

def _projected_gradient(gradient, x, domain, np):
    """Projected first-order residual for box-constrained minimization."""
    projected = np.asarray(gradient, dtype=float).copy()
    x = np.asarray(x, dtype=float).reshape(-1)
    domain = np.asarray(domain, dtype=float)
    tolerance = 1e-08 * np.maximum(1.0, np.abs(domain[:, 1] - domain[:, 0]))
    at_lower = x <= domain[:, 0] + tolerance
    at_upper = x >= domain[:, 1] - tolerance
    projected[at_lower & (projected >= 0.0)] = 0.0
    projected[at_upper & (projected <= 0.0)] = 0.0
    return projected

def run_self_test(seed: int=20260907) -> dict:
    api = load_official_api()
    np = api['np']
    GPy = api['GPy']
    np.random.seed(seed)
    x_train = np.linspace(-1.8, 1.8, 15)[:, None]
    y_train = (0.35 * x_train + np.sin(2.1 * x_train)).reshape(-1, 1)
    kernel = api['RBF'](input_dim=1, ARD=True, variance=1.25, lengthscale=np.array([0.65]))
    model = GPy.models.GPRegression(X=x_train, Y=y_train, kernel=kernel, normalizer=True, noise_var=1e-06)
    model.Gaussian_noise.variance.fix(1e-06)
    inputs = api['UniformInputs']([[-2.0, 2.0]])
    original_default_rng = np.random.default_rng

    def seeded_default_rng(*args, **kwargs):
        if not args and (not kwargs):
            return original_default_rng(seed)
        return original_default_rng(*args, **kwargs)
    np.random.default_rng = seeded_default_rng
    try:
        likelihood = api['Likelihood'](model, inputs, kwargs_gmm={'n_components': 2, 'covariance_type': 'full', 'random_state': seed})
    finally:
        np.random.default_rng = original_default_rng
    probe = np.array([[0.37]])
    criteria = {'US_LW': api['US_LW'](model, inputs, likelihood=likelihood), 'IVR_LW': api['IVR_LW'](model, inputs, likelihood=likelihood)}
    results = {}
    passed = True
    for (name, criterion) in criteria.items():
        value = np.asarray(criterion.evaluate(probe), dtype=float)
        analytic = np.asarray(criterion.jacobian(probe), dtype=float).reshape(-1)
        author_forward_difference = np.asarray(api['jacobian_fdiff'](criterion, probe), dtype=float).reshape(-1)
        finite_difference = _central_difference(criterion, probe, 0.0001, np)
        finite = bool(np.all(np.isfinite(value)) and np.all(np.isfinite(analytic)) and np.all(np.isfinite(finite_difference)))
        error = _relative_error(analytic, finite_difference, np)
        criterion_pass = finite and error <= 0.005
        passed = passed and criterion_pass
        results[name] = {'status': 'PASS' if criterion_pass else 'FAIL', 'value': value.tolist(), 'analytic_gradient': analytic.tolist(), 'central_difference_gradient_eps_1e-4': finite_difference.tolist(), 'author_forward_difference_gradient_eps_1e-8': author_forward_difference.tolist(), 'relative_gradient_error': error, 'all_finite': finite}
    return {'status': 'PASS' if passed else 'FAIL', 'third_party_commit': EXPECTED_COMMIT, 'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': api['scipy'].__version__, 'GPy': api['GPy'].__version__, 'KDEpy': api['KDEpy'].__version__, 'scikit_learn': api['sklearn'].__version__, 'reproducibility_shim': 'seed np.random.default_rng used by Likelihood._fit_gmm', 'criteria': results}

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--request', type=Path, help='JSON request for one native sequential acquisition proposal')
    parser.add_argument('--response', type=Path, help='atomic JSON response path (stdout when omitted)')
    parser.add_argument('--checkpoint', type=Path, help='optional official GPy JSON-gzip checkpoint (.zip)')
    args = parser.parse_args(argv)
    if args.self_test == (args.request is not None):
        parser.error('choose exactly one of --self-test or --request')
    if args.self_test:
        report = run_self_test()
    else:
        report = run_sequential_step(args.request, args.checkpoint)
    if args.response is not None:
        _atomic_json(args.response, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['status'] == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
