"""Resumable outer ledger for native sequential US-LW Track-A acquisition."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import numpy as np
MORETEST_ROOT = Path(__file__).resolve().parent.parent
if str(MORETEST_ROOT) not in sys.path:
    sys.path.insert(0, str(MORETEST_ROOT))
from external_adapters.gpsearch_adapter import DEFAULT_RUNTIME_MANIFEST, EXPECTED_COMMIT, _runtime_python
from uncertainty_benchmark.checkpoint import BlockStore, atomic_write_json, canonical_json, safe_artifact_name
from uncertainty_benchmark.ledger import CallLedger
from uncertainty_benchmark.oracle import CachedOracle
from uncertainty_benchmark.seed_blocks import SeedBlock
METHOD = 'US-LW-GP-Seq'
SCHEMA = 'uslw.collection.v1'
SESSION_SCHEMA = 'gpsearch.session-action.v1'
SESSION_WORKER = Path(__file__).resolve().parent / 'gpsearch_session_worker.py'
STABLE_WORKER = Path(__file__).resolve().parent / 'gpsearch_worker.py'
DEFAULT_EXPERIMENT_ROOT = Path(os.environ.get('USLW_EXPERIMENT_ROOT', 'uslw_experiment'))
DEFAULT_SOURCE_ROOT = DEFAULT_EXPERIMENT_ROOT / 'track_a'
DEFAULT_OUTPUT_ROOT = DEFAULT_EXPERIMENT_ROOT / 'track_a_external'
ROOT_SEED = 2026090701
FORMAL_BLOCKS = range(30)
ROUNDS = 4
TRAIN_PER_ROUND = 500
AUDIT_PER_ROUND = 250
TOTAL_TRAIN = ROUNDS * TRAIN_PER_ROUND
AUDIT_POOL_SIZE = 1000000
AUDIT_CHUNK_SIZE = 4096
LATENT_BOUND = 6.0
SAFETY_TIMEOUT_SECONDS = 24.0 * 60.0 * 60.0
CASE_META = {'two-sided-kink': {'dimension': 2, 'problem_name': 'two_sided_kink', 'adverse': 'lower', 'initial_train': 1000}, 'radial-kink': {'dimension': 8, 'problem_name': 'radial_kink', 'adverse': 'lower', 'initial_train': 1000}, 'cantilever_beam_2d': {'dimension': 2, 'problem_name': 'cantilever_beam_2d', 'adverse': 'lower', 'initial_train': 1000}}

class CollectionError(RuntimeError):
    """Fail-closed violation of source, method, ledger or checkpoint state."""

class SafetyTimeout(CollectionError):
    """The pre-registered 24-hour block cap was actually reached."""

    def __init__(self, message: str, process_record: Mapping[str, object] | None=None):
        super().__init__(message)
        self.process_record = dict(process_record or {})

@dataclass(frozen=True)
class SourceBlock:
    root: Path
    case: str
    block: int
    root_seed: int
    dimension: int
    adverse: str
    problem_name: str
    problem_path: Path
    initial_path: Path
    initial_x: np.ndarray
    initial_y: np.ndarray

@dataclass
class InvocationClock:
    base_seconds: float

    def __post_init__(self) -> None:
        self.started = time.monotonic()

    @property
    def active_seconds(self) -> float:
        return float(self.base_seconds + (time.monotonic() - self.started))

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, SAFETY_TIMEOUT_SECONDS - self.active_seconds)

class FrozenProblem:
    """Minimal oracle facade loaded from the source block's frozen case code."""

    def __init__(self, source: SourceBlock):
        module_name = '_uslw_frozen_{}_{}'.format(source.case.replace('-', '_'), source.block)
        spec = importlib.util.spec_from_file_location(module_name, source.problem_path)
        if spec is None or spec.loader is None:
            raise CollectionError(f'cannot load frozen problem source: {source.problem_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, 'response', None)):
            raise CollectionError('frozen problem has no callable response')
        if int(getattr(module, 'DIMENSION', -1)) != source.dimension:
            raise CollectionError('frozen problem dimension differs from source contract')
        if str(getattr(module, 'NAME', '')) != source.problem_name:
            raise CollectionError('frozen problem name differs from source contract')
        self.name = source.problem_name
        self.dimension = source.dimension
        self.adverse = source.adverse
        self._response = module.response

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._response(x), dtype=np.float64).reshape(-1)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _u32(seed: int) -> int:
    return int(seed) & 4294967295

def _seed_record(block: SeedBlock, role: str, *, round_index: int=0, member: int=0) -> dict[str, object]:
    seed64 = block.seed(role, method=METHOD, round_index=int(round_index), member=int(member))
    return {'schema': 'uncertainty-benchmark-seed-v1', 'root_seed': int(block.root_seed), 'block': int(block.block), 'role': role, 'method': METHOD, 'round': int(round_index), 'member': int(member), 'attempt': 0, 'seed_uint64': int(seed64), 'worker_seed_uint32_low_bits': _u32(seed64)}

def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CollectionError(f'required JSON is absent: {path}')
    with path.open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise CollectionError(f'JSON root must be an object: {path}')
    return value

def _resolve_inside(root: Path, relative: str) -> Path:
    target = (root / Path(relative)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise CollectionError(f'source artifact escapes block root: {relative}') from exc
    return target

def _load_initial_artifact(path: Path, *, dimension: int, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    expected_fields = {'x', 'y', 'origin_physical_calls'}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise CollectionError('shared.initial_train NPZ has unsupported fields; refusing possible holdout/test/reference leakage')
        x = np.ascontiguousarray(np.asarray(archive['x'], dtype=np.float64))
        y_original = np.asarray(archive['y'], dtype=np.float64)
        y = np.ascontiguousarray(y_original.reshape(-1))
        origin_calls = int(np.asarray(archive['origin_physical_calls']).item())
    if x.shape != (expected_rows, dimension) or y.shape != (expected_rows,):
        raise CollectionError('shared.initial_train shape differs from frozen contract')
    if origin_calls != expected_rows:
        raise CollectionError('shared.initial_train origin call count is inconsistent')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise CollectionError('shared.initial_train contains non-finite values')
    x.setflags(write=False)
    y.setflags(write=False)
    return (x, y)

def load_source_block(source_track_a_root: Path, case: str, block: int) -> SourceBlock:
    """Read the sole permitted source-label event and frozen problem code."""
    if case not in CASE_META:
        raise CollectionError(f'US-LW formal collection is not applicable to {case!r}')
    if int(block) not in FORMAL_BLOCKS:
        raise CollectionError('formal block must be in 0..29')
    meta = CASE_META[case]
    root = (Path(source_track_a_root) / case / f'block_{block:03d}').resolve()
    request_path = root / 'request.json'
    wrapper = _read_json(request_path)
    if set(wrapper) != {'request'}:
        raise CollectionError('source request wrapper has unexpected fields')
    request = wrapper['request']
    if not isinstance(request, dict):
        raise CollectionError('source request payload must be an object')
    if request.get('schema') != 'uncertainty-benchmark-block-v1':
        raise CollectionError('source block schema is unsupported')
    if request.get('case') != case or int(request.get('block', -1)) != block:
        raise CollectionError('source case/block identity mismatch')
    if int(request.get('root_seed', -1)) != ROOT_SEED:
        raise CollectionError('source root seed differs from frozen 2026090701')
    contract = request.get('contract')
    if not isinstance(contract, dict):
        raise CollectionError('source contract is absent')
    expected_contract = {'slug': case, 'problem_name': meta['problem_name'], 'dimension': meta['dimension'], 'adverse': meta['adverse'], 'train_size': meta['initial_train'], 'rounds': ROUNDS, 'train_per_round': TRAIN_PER_ROUND, 'audit_per_round': AUDIT_PER_ROUND}
    for (key, expected) in expected_contract.items():
        if contract.get(key) != expected:
            raise CollectionError(f'source contract drift for {key}: expected {expected!r}, got {contract.get(key)!r}')
    problem_key = f'cases/{case}/problem.py'
    problem_path = MORETEST_ROOT / problem_key
    if not problem_path.is_file():
        raise CollectionError('problem source is absent')
    ledger_path = root / 'ledger.json'
    ledger = _read_json(ledger_path)
    if ledger.get('schema') != 'uncertainty-benchmark-ledger-v1':
        raise CollectionError('source ledger schema is unsupported')
    events = ledger.get('events')
    if not isinstance(events, list):
        raise CollectionError('source ledger events are absent')
    initial_events = [event for event in events if isinstance(event, dict) and event.get('event_id') == 'shared.initial_train' and (event.get('role') == 'initial_train')]
    if len(initial_events) != 1:
        raise CollectionError('source ledger must contain exactly one shared.initial_train')
    event = initial_events[0]
    expected_event = {'event_id': 'shared.initial_train', 'role': 'initial_train', 'method': 'shared', 'round_index': 0, 'logical_calls': meta['initial_train'], 'unique_inputs': meta['initial_train'], 'physical_calls': meta['initial_train']}
    for (key, expected) in expected_event.items():
        if event.get(key) != expected:
            raise CollectionError(f'source initial event drift for {key}')
    artifact = event.get('artifact')
    if not isinstance(artifact, str):
        raise CollectionError('source initial event has no artifact')
    initial_path = _resolve_inside(root, artifact)
    if not initial_path.is_file():
        raise CollectionError('source shared.initial_train artifact is absent')
    (initial_x, initial_y) = _load_initial_artifact(initial_path, dimension=int(meta['dimension']), expected_rows=int(meta['initial_train']))
    return SourceBlock(root=root, case=case, block=block, root_seed=ROOT_SEED, dimension=int(meta['dimension']), adverse=str(meta['adverse']), problem_name=str(meta['problem_name']), problem_path=problem_path, initial_path=initial_path, initial_x=initial_x, initial_y=initial_y)

def _runtime_record(runtime: Path, runtime_manifest: Path) -> dict[str, object]:
    manifest = _read_json(runtime_manifest)
    return {'python_executable': str(runtime), 'runtime_manifest': str(runtime_manifest.resolve()), 'declared_python_version': manifest.get('python_version'), 'declared_third_party_commit': manifest.get('third_party_commit')}

def _domain(dimension: int) -> list[list[float]]:
    return [[-LATENT_BOUND, LATENT_BOUND] for _ in range(dimension)]

def _truncation_mass_loss(dimension: int) -> float:
    retained = math.erf(LATENT_BOUND / math.sqrt(2.0)) ** dimension
    return 1.0 - retained

def _compute_contract(source: SourceBlock, runtime_record: Mapping[str, object]) -> dict:
    density_rows_per_likelihood = 100000 if source.dimension <= 2 else 1000000
    likelihood_builds = TOTAL_TRAIN + ROUNDS
    return {'schema': SCHEMA, 'method': METHOD, 'case': source.case, 'block': source.block, 'root_seed': source.root_seed, 'source': {'block_root': str(source.root), 'permitted_label_event': 'shared.initial_train', 'initial_artifact': str(source.initial_path), 'initial_rows_reused': len(source.initial_x), 'new_hf_charged_for_initial': 0, 'frozen_problem_source': str(source.problem_path), 'forbidden_reads': ['shared.initial_holdout y', 'test/evaluation y', 'reference manifests, thresholds or chains', "other methods' query y or predictions", 'source summaries/results']}, 'problem': {'name': source.problem_name, 'dimension': source.dimension, 'adverse': source.adverse, 'gp_input': 'raw latent z ~ N(0,I); no physical feature map and no log transform', 'gp_output': 'raw physical response; GPy standardizes internally'}, 'collection': {'rounds': ROUNDS, 'train_per_round': TRAIN_PER_ROUND, 'audit_per_round': AUDIT_PER_ROUND, 'total_train_hf': TOTAL_TRAIN, 'total_audit_hf': ROUNDS * AUDIT_PER_ROUND, 'audit_feedback_to_gp': False, 'common_student_training': False, 'continuous_training_domain': _domain(source.dimension), 'continuous_optimizer': "authors' funmin/L-BFGS-B, 10 restarts + one LHS set", 'audit_nominal_pool_per_round': AUDIT_POOL_SIZE, 'audit_chunk_size': AUDIT_CHUNK_SIZE, 'audit_pool_role': 'method-internal post-round diagnostic only; not the native continuous training search and not the common cross-method primary endpoint', 'audit_exclusion': 'all current GP training x and all prior audit x', 'duplicate_policy': 'fail-closed; no repulsion, pseudo-label or hidden retry'}, 'numerics': {'kernel': "authors' ARD RBF", 'normalize_y': True, 'fixed_nugget_variance': 0.001, 'nugget_units': 'standardized-output variance', 'gp_optimizer': 'BFGS, 10 restarts, max_iters=1000 after every train label', 'acquisition_optimizer': 'L-BFGS-B, 10 restarts, max_iters=1000, ftol=1e-9, gtol=1e-7', 'likelihood': "authors' output-weighted KDE plus 2-component full GMM", 'likelihood_density_rows_per_update': density_rows_per_likelihood, 'likelihood_gmm_resample_rows': 20000, 'prediction_memory_adapter': 'chunk GPy predict at 4096 only; same density points/order and unchanged author KDE/GMM/US_LW', 'checkpoint_adapter': 'GPy 1.9.9 JSON load drops the fixed-parameter flag; verify the serialized nugget is still exactly 1e-3, then restore only that constraint before optimization', 'performance_tuning_after_results': False}, 'domain_disclosure': {'nominal': 'independent standard normal latent coordinates', 'box': f'[-{LATENT_BOUND:g},{LATENT_BOUND:g}]^{source.dimension}', 'theoretical_nominal_mass_lost': _truncation_mass_loss(source.dimension), 'interpretation': "finite box required by the authors' continuous optimiser; audit nominal draws are rejection-truncated to the same box"}, 'zero_hf_compute_lower_bound': {'exact_gp_optimizations': 1 + TOTAL_TRAIN, 'native_continuous_uslw_optimizations': TOTAL_TRAIN, 'uslw_likelihood_rebuilds': likelihood_builds, 'internal_density_prediction_rows': likelihood_builds * density_rows_per_likelihood, 'audit_candidate_score_rows': ROUNDS * AUDIT_POOL_SIZE, 'note': 'These are zero-HF numerical operations and exclude cubic exact-GP factorization cost as n grows from 1000 to 3000. Across 30 blocks, the three low-dimensional cases require 180,000 sequential updates and 72.144 billion internal density-prediction rows.'}, 'safety': {'active_wall_cap_seconds': SAFETY_TIMEOUT_SECONDS, 'formal_block_00_resource_measurement': True, 'resource_observation_may_change_scheduling_only': True, 'mathematical_parameters_may_change_after_block_00': False}, 'runtime': dict(runtime_record), 'sources': {'third_party_commit': EXPECTED_COMMIT}}

def _expected_model(checkpoint: Mapping[str, object]) -> dict[str, object]:
    model = checkpoint.get('model')
    if not isinstance(model, dict):
        raise CollectionError('checkpoint response has no model state')
    return {'rows': int(model['rows']), 'dimension': int(model['dimension']), 'training_data': str(checkpoint['training_data'])}

def _checkpoint_relative(confirmed_count: int) -> Path:
    if confirmed_count == 0:
        return Path('checkpoints') / 'initial.zip'
    if confirmed_count % TRAIN_PER_ROUND == 0:
        return Path('checkpoints') / f'round_{confirmed_count // TRAIN_PER_ROUND:02d}.zip'
    slot = 'a' if confirmed_count % 2 else 'b'
    return Path('checkpoints') / f'latest_{slot}.zip'

def _action_request(action: str, request_id: str, source: SourceBlock, **fields: object) -> dict[str, object]:
    body: dict[str, object] = {'schema_version': SESSION_SCHEMA, 'action': action, 'request_id': request_id, 'expected_third_party_commit': EXPECTED_COMMIT, 'dimension': source.dimension, 'domain': _domain(source.dimension)}
    body.update(fields)
    return body

def _windows_process_ids(root_pid: int) -> list[int]:
    """Return a Toolhelp snapshot of one Windows process tree."""
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD), ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_size_t), ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD), ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG), ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR * 260)]
    kernel32 = ctypes.windll.kernel32
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    create_snapshot.restype = wintypes.HANDLE
    process_first = kernel32.Process32FirstW
    process_first.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    process_first.restype = wintypes.BOOL
    process_next = kernel32.Process32NextW
    process_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    process_next.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    snapshot = create_snapshot(2, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        return [int(root_pid)]
    parents: dict[int, list[int]] = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = process_first(snapshot, ctypes.byref(entry))
        while ok:
            parents.setdefault(int(entry.th32ParentProcessID), []).append(int(entry.th32ProcessID))
            ok = process_next(snapshot, ctypes.byref(entry))
    finally:
        close_handle(snapshot)
    (result, frontier) = ([], [int(root_pid)])
    seen = set()
    while frontier:
        pid = frontier.pop()
        if pid in seen:
            continue
        seen.add(pid)
        result.append(pid)
        frontier.extend(parents.get(pid, ()))
    return result

def _windows_pid_rss(pid: int) -> int | None:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t), ('PrivateUsage', ctypes.c_size_t)]
    open_process = ctypes.windll.kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    close_handle = ctypes.windll.kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    access = 1024 | 16
    handle = open_process(access, False, int(pid))
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        ok = get_memory(handle, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else None
    finally:
        close_handle(handle)

def _process_tree_rss(process: subprocess.Popen) -> int | None:
    try:
        import psutil
        root = psutil.Process(process.pid)
        total = int(root.memory_info().rss)
        for child in root.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total
    except Exception:
        pass
    if os.name == 'nt':
        try:
            values = [value for value in (_windows_pid_rss(pid) for pid in _windows_process_ids(process.pid)) if value is not None]
            return sum(values) if values else None
        except Exception:
            return None
    return None

def _terminate_process_tree(process: subprocess.Popen) -> None:
    try:
        import psutil
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            root.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    except Exception:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False, capture_output=True, text=True)
        elif process.poll() is None:
            process.kill()

def _log_tail(path: Path, *, maximum_bytes: int=8192) -> str:
    size = path.stat().st_size if path.is_file() else 0
    if not size:
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, size - int(maximum_bytes)))
        return stream.read().decode('utf-8', errors='replace')

def _run_process(command: list[str], timeout_seconds: float, *, stdout_path: Path, stderr_path: Path) -> dict[str, object]:
    """Run one scoped child with non-blocking file-backed output capture"""
    if timeout_seconds <= 0:
        raise SafetyTimeout('formal block exhausted its 24-hour active-wall cap')
    stdout_path = Path(stdout_path).resolve()
    stderr_path = Path(stderr_path).resolve()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['OMP_NUM_THREADS'] = '1'
    environment['MKL_NUM_THREADS'] = '1'
    environment['OPENBLAS_NUM_THREADS'] = '1'
    started = time.perf_counter()
    stdout_start = stdout_path.stat().st_size if stdout_path.is_file() else 0
    stderr_start = stderr_path.stat().st_size if stderr_path.is_file() else 0
    process: subprocess.Popen | None = None
    peak_rss = 0
    timed_out = False
    cleanup_on_exception = False
    with stdout_path.open('ab') as stdout, stderr_path.open('ab') as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            while process.poll() is None:
                rss = _process_tree_rss(process)
                if rss is not None:
                    peak_rss = max(peak_rss, rss)
                if time.perf_counter() - started >= timeout_seconds:
                    timed_out = True
                    _terminate_process_tree(process)
                    break
                time.sleep(0.2)
            if process.poll() is None:
                process.wait(timeout=30)
        except BaseException:
            cleanup_on_exception = True
            if process is not None and process.poll() is None:
                _terminate_process_tree(process)
                try:
                    process.wait(timeout=30)
                except Exception:
                    pass
            raise
        finally:
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    elapsed = time.perf_counter() - started
    stdout_end = stdout_path.stat().st_size
    stderr_end = stderr_path.stat().st_size
    record = {'command': command, 'returncode': None if process is None else process.returncode, 'elapsed_seconds': elapsed, 'peak_process_tree_rss_bytes': peak_rss or None, 'stdout_log': str(stdout_path), 'stdout_invocation_byte_range': [stdout_start, stdout_end], 'stdout_tail': _log_tail(stdout_path), 'stderr_log': str(stderr_path), 'stderr_invocation_byte_range': [stderr_start, stderr_end], 'stderr_tail': _log_tail(stderr_path), 'timed_out': timed_out, 'cleanup_on_exception': cleanup_on_exception, 'capture': 'append-only files; no subprocess PIPE', 'thread_environment': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}}
    if timed_out:
        raise SafetyTimeout('gpsearch action reached the remaining formal block wall cap', record)
    return record

def _invoke_action(store: BlockStore, runtime: Path, request: Mapping[str, object], clock: InvocationClock) -> dict[str, Any]:
    request_path = store.path('scratch/action.request.json')
    response_path = store.path('scratch/action.response.json')
    previous_request = store.read_json('scratch/last_completed_request.json')
    atomic_write_json(request_path, request)
    cached: dict[str, Any] | None = None
    if response_path.is_file():
        candidate = _read_json(response_path)
        if previous_request == dict(request) and candidate.get('request_id') == request['request_id'] and candidate.get('schema_version') == SESSION_SCHEMA:
            cached = candidate
    if cached is None:
        log_stem = safe_artifact_name(str(request['request_id']))
        process_record = _run_process([str(runtime), str(SESSION_WORKER), '--request', str(request_path), '--response', str(response_path)], clock.remaining_seconds, stdout_path=store.path(Path('worker_logs') / f'{log_stem}.stdout.log'), stderr_path=store.path(Path('worker_logs') / f'{log_stem}.stderr.log'))
        if not response_path.is_file():
            raise CollectionError('gpsearch session worker produced no atomic response: ' + str(process_record))
        candidate = _read_json(response_path)
        if candidate.get('request_id') != request['request_id'] or candidate.get('schema_version') != SESSION_SCHEMA:
            raise CollectionError('gpsearch session worker left a stale/mismatched response: ' + str(process_record))
        cached = candidate
        cached['launcher'] = process_record
        atomic_write_json(response_path, cached)
        store.write_json('scratch/last_completed_request.json', dict(request))
    if cached.get('third_party', {}).get('commit') != EXPECTED_COMMIT:
        raise CollectionError('gpsearch action returned the wrong third-party provenance')
    launcher = cached.get('launcher')
    if isinstance(launcher, dict) and launcher.get('returncode') not in (None, 0):
        raise CollectionError(f"gpsearch action process failed before HF: {request['request_id']} (returncode={launcher.get('returncode')}, stderr_tail={launcher.get('stderr_tail', '')!r})")
    if cached.get('status') != 'PASS':
        raise CollectionError(f"gpsearch action rejected before HF: {request['request_id']} (status={cached.get('status')})")
    return cached

def _write_state(store: BlockStore, state: dict, clock: InvocationClock) -> None:
    state['active_wall_seconds'] = clock.active_seconds
    state['updated_at_utc'] = _utc_now()
    store.write_json('state.json', state)

def _query_oracle_fail_closed(store: BlockStore, oracle: CachedOracle, event_id: str, x: np.ndarray, *, role: str, method: str, round_index: int):
    """Journal intent before HF and never retry an unknown in-flight call"""
    values = np.ascontiguousarray(np.atleast_2d(np.asarray(x, dtype=np.float64)))
    intent_relative = Path('oracle_intents') / f'{safe_artifact_name(event_id)}.json'
    query_relative = Path('queries') / f'{safe_artifact_name(event_id)}.npz'
    intent = store.read_json(intent_relative)
    if intent is not None:
        expected = {'event_id': event_id, 'role': role, 'method': method, 'round_index': int(round_index), 'rows': len(values)}
        for (key, value) in expected.items():
            if intent.get(key) != value:
                raise CollectionError(f'oracle intent drift for {event_id}: {key}')
        if not store.path(query_relative).is_file():
            raise CollectionError(f'oracle intent exists without an atomic result; the prior physical call is unknown in-flight and will not be repeated: {event_id}')
    else:
        intent = {'schema': 'oracle-intent.v1', 'status': 'request_started', 'event_id': event_id, 'role': role, 'method': method, 'round_index': int(round_index), 'rows': len(values), 'started_at_utc': _utc_now(), 'automatic_retry_without_result': False}
        store.write_json(intent_relative, intent)
    result = oracle.query(event_id, values, role=role, method=method, round_index=round_index)
    intent.update({'status': 'response_persisted', 'artifact': str(query_relative).replace('\\', '/'), 'completed_at_utc': _utc_now()})
    store.write_json(intent_relative, intent)
    return result

def _new_state() -> dict[str, Any]:
    return {'schema': SCHEMA, 'method': METHOD, 'status': 'initializing', 'confirmed_training_count': 0, 'completed_audit_rounds': [], 'checkpoint': None, 'active_wall_seconds': 0.0, 'created_at_utc': _utc_now(), 'updated_at_utc': _utc_now(), 'common_student_trained': False, 'final_common_test_evaluated': False}

def _validate_state(state: Mapping[str, object], source: SourceBlock) -> None:
    if state.get('schema') != SCHEMA or state.get('method') != METHOD:
        raise CollectionError('collection state schema/method mismatch')
    count = int(state.get('confirmed_training_count', -1))
    if count < 0 or count > TOTAL_TRAIN:
        raise CollectionError('collection training count is out of range')
    audits = state.get('completed_audit_rounds')
    if not isinstance(audits, list) or any((int(value) not in range(1, ROUNDS + 1) for value in audits)):
        raise CollectionError('collection audit state is invalid')
    if sorted(set(map(int, audits))) != list(map(int, audits)):
        raise CollectionError('collection audit rounds must be sorted and unique')
    maximum_due = count // TRAIN_PER_ROUND
    if any((int(value) > maximum_due for value in audits)):
        raise CollectionError("audit recorded before its round's 500 training labels")
    checkpoint = state.get('checkpoint')
    if count > 0 and (not isinstance(checkpoint, dict)):
        raise CollectionError('collection state has no model checkpoint')

def _state_checkpoint_path(store: BlockStore, state: Mapping[str, object]) -> Path:
    checkpoint = state.get('checkpoint')
    if not isinstance(checkpoint, dict):
        raise CollectionError('state checkpoint is absent')
    relative = checkpoint.get('relative')
    if not isinstance(relative, str):
        raise CollectionError('state checkpoint relative path is absent')
    path = store.path(relative)
    if not path.is_file():
        raise CollectionError('state checkpoint file is absent')
    return path

def _checkpoint_state(relative: Path, worker_checkpoint: Mapping[str, object]) -> dict:
    return {'relative': str(relative).replace('\\', '/'), 'model': worker_checkpoint['model'], 'format': worker_checkpoint.get('format'), 'training_data': worker_checkpoint['training_data']}

def _initialise_if_needed(store: BlockStore, state: dict[str, Any], source: SourceBlock, runtime: Path, seeds: SeedBlock, clock: InvocationClock) -> None:
    if state.get('checkpoint') is not None:
        _state_checkpoint_path(store, state)
        return
    seed = _seed_record(seeds, 'uslw_gp_initial_optimization')
    relative = _checkpoint_relative(0)
    request = _action_request('initialize', f'{source.case}.block{source.block:03d}.initialize', source, seed=seed['worker_seed_uint32_low_bits'], expected_initial_count=len(source.initial_x), initial_data_npz=str(source.initial_path), checkpoint_out=str(store.path(relative)))
    response = _invoke_action(store, runtime, request, clock)
    record = {'schema': SCHEMA, 'seed_derivation': seed, 'worker_request': request, 'worker_response': response}
    store.write_json('records/initialization.json', record)
    state['checkpoint'] = _checkpoint_state(relative, response['checkpoint'])
    state['status'] = 'running'
    _write_state(store, state, clock)

def _training_record_path(global_step: int) -> Path:
    return Path('records') / f'train_{global_step:04d}.json'

def _ensure_proposal(store: BlockStore, state: Mapping[str, object], source: SourceBlock, runtime: Path, seeds: SeedBlock, clock: InvocationClock, global_step: int) -> tuple[np.ndarray, dict[str, Any]]:
    relative = _training_record_path(global_step)
    existing = store.read_json(relative)
    if existing is not None:
        response = existing.get('proposal_response')
        if not isinstance(response, dict) or response.get('status') != 'PASS':
            raise CollectionError(f'corrupt frozen proposal record: {relative}')
        point = np.asarray(response['proposal']['x'], dtype=np.float64).reshape(1, -1)
        return (point, existing)
    round_index = (global_step - 1) // TRAIN_PER_ROUND + 1
    member = (global_step - 1) % TRAIN_PER_ROUND
    seed = _seed_record(seeds, 'uslw_native_train_proposal', round_index=round_index, member=member)
    checkpoint_path = _state_checkpoint_path(store, state)
    expected = _expected_model(state['checkpoint'])
    if expected['rows'] != len(source.initial_x) + global_step - 1:
        raise CollectionError('GP checkpoint row count is not sequential')
    request = _action_request('propose', f'{source.case}.block{source.block:03d}.train{global_step:04d}.propose', source, seed=seed['worker_seed_uint32_low_bits'], checkpoint_in=str(checkpoint_path), expected_model=expected)
    response = _invoke_action(store, runtime, request, clock)
    point = np.asarray(response['proposal']['x'], dtype=np.float64).reshape(1, -1)
    if point.shape != (1, source.dimension):
        raise CollectionError('native proposal has the wrong shape')
    record = {'schema': SCHEMA, 'global_step': global_step, 'round': round_index, 'step_in_round': member + 1, 'status': 'proposal_frozen_before_oracle', 'proposal_seed_derivation': seed, 'proposal_request': request, 'proposal_response': response, 'audit_test_reference_feedback': False}
    store.write_json(relative, record)
    return (point, record)

def _confirm_training_step(store: BlockStore, ledger: CallLedger, oracle: CachedOracle, state: dict[str, Any], source: SourceBlock, runtime: Path, seeds: SeedBlock, clock: InvocationClock, global_step: int) -> None:
    (point, record) = _ensure_proposal(store, state, source, runtime, seeds, clock, global_step)
    round_index = int(record['round'])
    event_id = f'{METHOD}.round{round_index}.train.step{global_step:04d}'
    result = _query_oracle_fail_closed(store, oracle, event_id, point, role='acquisition_train', method=METHOD, round_index=round_index)
    observe_response = record.get('observe_response')
    if not isinstance(observe_response, dict):
        seed = _seed_record(seeds, 'uslw_gp_post_label_optimization', round_index=round_index, member=global_step - 1)
        checkpoint_in = _state_checkpoint_path(store, state)
        checkpoint_out_relative = _checkpoint_relative(global_step)
        request = _action_request('observe', f'{source.case}.block{source.block:03d}.train{global_step:04d}.observe', source, seed=seed['worker_seed_uint32_low_bits'], checkpoint_in=str(checkpoint_in), checkpoint_out=str(store.path(checkpoint_out_relative)), expected_model=_expected_model(state['checkpoint']), observation_npz=str(result.artifact), expected_proposal_x=point.tolist())
        observe_response = _invoke_action(store, runtime, request, clock)
        record.update({'status': 'confirmed_training_feedback', 'oracle': {'event_id': event_id, 'artifact': str(result.artifact), 'reused_on_this_invocation': result.reused, 'origin_physical_calls': result.origin_physical_calls}, 'observe_seed_derivation': seed, 'observe_request': request, 'observe_response': observe_response, 'feedback_to_gp': True})
        store.write_json(_training_record_path(global_step), record)
    else:
        checkpoint_out_relative = _checkpoint_relative(global_step)
        if observe_response.get('status') != 'PASS':
            raise CollectionError('persisted observe response is not PASS')
    checkpoint_record = observe_response['checkpoint']
    checkpoint_path = store.path(checkpoint_out_relative)
    if not checkpoint_path.is_file():
        raise CollectionError('post-observation checkpoint is absent')
    if int(checkpoint_record['model']['rows']) != len(source.initial_x) + global_step:
        raise CollectionError('post-observation GP row count is incorrect')
    state['checkpoint'] = _checkpoint_state(checkpoint_out_relative, checkpoint_record)
    state['confirmed_training_count'] = global_step
    state['status'] = 'running'
    _write_state(store, state, clock)

def _prior_audit_x(store: BlockStore, completed_rounds: list[int]) -> Path | None:
    if not completed_rounds:
        return None
    pieces = []
    for round_index in completed_rounds:
        body = store.load_npz(f'rounds/round_{round_index:02d}/audit_selection.npz')
        if body is None or set(body) < {'x'}:
            raise CollectionError('prior audit selection is absent')
        pieces.append(np.asarray(body['x'], dtype=np.float64))
    path = store.save_npz('scratch/prior_audit_x.npz', x=np.ascontiguousarray(np.vstack(pieces)))
    return path

def _ensure_round_audit(store: BlockStore, ledger: CallLedger, oracle: CachedOracle, state: dict[str, Any], source: SourceBlock, runtime: Path, seeds: SeedBlock, clock: InvocationClock, round_index: int) -> None:
    if round_index in state['completed_audit_rounds']:
        return
    if int(state['confirmed_training_count']) < round_index * TRAIN_PER_ROUND:
        raise CollectionError('round audit requested before 500 native training labels')
    round_root = Path('rounds') / f'round_{round_index:02d}'
    record_relative = round_root / 'audit.json'
    record = store.read_json(record_relative)
    selection_relative = round_root / 'audit_selection.npz'
    if record is None or not isinstance(record.get('selection_response'), dict):
        likelihood_seed = _seed_record(seeds, 'uslw_audit_likelihood', round_index=round_index)
        candidate_seed = _seed_record(seeds, 'uslw_audit_nominal_pool', round_index=round_index)
        prior_path = _prior_audit_x(store, list(state['completed_audit_rounds']))
        request = _action_request('audit', f'{source.case}.block{source.block:03d}.round{round_index}.audit-select', source, seed=likelihood_seed['worker_seed_uint32_low_bits'], candidate_seed=candidate_seed['worker_seed_uint32_low_bits'], checkpoint_in=str(_state_checkpoint_path(store, state)), expected_model=_expected_model(state['checkpoint']), prior_audit_npz=str(prior_path) if prior_path else None, selection_npz=str(store.path(selection_relative)), pool_size=AUDIT_POOL_SIZE, chunk_size=AUDIT_CHUNK_SIZE, top_k=AUDIT_PER_ROUND)
        response = _invoke_action(store, runtime, request, clock)
        record = {'schema': SCHEMA, 'round': round_index, 'status': 'selection_frozen_before_oracle', 'likelihood_seed_derivation': likelihood_seed, 'candidate_seed_derivation': candidate_seed, 'selection_request': request, 'selection_response': response, 'feedback_to_gp': False, 'diagnostic_scope': 'method-internal acquisition diagnostic; cross-method primary metrics use the common independent tail test and actual training hits'}
        store.write_json(record_relative, record)
    selection = store.load_npz(selection_relative)
    if selection is None or set(selection) != {'x', 'objective_score', 'utility_score', 'candidate_ordinal', 'predicted_mean', 'predicted_variance', 'output_density_at_predicted_mean', 'log_output_density_at_predicted_mean', 'gmm_likelihood_weight'}:
        raise CollectionError('frozen audit selection fields are absent or unsupported')
    selected_x = np.ascontiguousarray(np.asarray(selection['x'], dtype=np.float64))
    if selected_x.shape != (AUDIT_PER_ROUND, source.dimension):
        raise CollectionError('frozen audit selection has the wrong shape')
    response_selection = record['selection_response']['selection']
    event_id = f'{METHOD}.round{round_index}.audit'
    result = _query_oracle_fail_closed(store, oracle, event_id, selected_x, role='acquisition_audit', method=METHOD, round_index=round_index)
    record.update({'status': 'confirmed_diagnostic_audit_no_feedback', 'oracle': {'event_id': event_id, 'artifact': str(result.artifact), 'reused_on_this_invocation': result.reused, 'origin_physical_calls': result.origin_physical_calls, 'observed_y_summary': {'minimum': float(np.min(result.y)), 'median': float(np.median(result.y)), 'maximum': float(np.max(result.y))}}, 'feedback_to_gp': False, 'future_selection_access_to_audit_y': False})
    store.write_json(record_relative, record)
    state['completed_audit_rounds'] = sorted(set(map(int, state['completed_audit_rounds'])) | {round_index})
    _write_state(store, state, clock)

def _validate_final_ledger(ledger: CallLedger) -> dict[str, Any]:
    body = ledger.as_dict()
    by_role = body.get('totals_by_role', {})
    train = by_role.get('acquisition_train', {})
    audit = by_role.get('acquisition_audit', {})
    expected = {'train_physical': TOTAL_TRAIN, 'audit_physical': ROUNDS * AUDIT_PER_ROUND}
    observed = {'train_physical': int(train.get('physical_calls', 0)), 'audit_physical': int(audit.get('physical_calls', 0))}
    if observed != expected:
        raise CollectionError(f'terminal ledger count mismatch: {observed} != {expected}')
    return body

def run_collection(*, source_track_a_root: Path, output_root: Path, case: str, block: int, runtime_manifest: Path=DEFAULT_RUNTIME_MANIFEST, python_executable: Path | None=None) -> dict[str, Any]:
    source = load_source_block(source_track_a_root, case, block)
    runtime_manifest = Path(runtime_manifest).resolve()
    runtime = _runtime_python(python_executable, runtime_manifest)
    runtime_record = _runtime_record(runtime, runtime_manifest)
    if runtime_record['declared_third_party_commit'] != EXPECTED_COMMIT:
        raise CollectionError('runtime manifest third-party commit mismatch')
    request = _compute_contract(source, runtime_record)
    block_root = Path(output_root).resolve() / METHOD / case / f'block_{block:03d}'
    store = BlockStore(block_root, request)
    ledger = CallLedger(store.path('ledger.json'))
    state = store.read_json('state.json') or _new_state()
    _validate_state(state, source)
    if state.get('status') == 'complete':
        return store.read_json('summary.json') or {'status': 'complete', 'block_root': str(block_root)}
    if state.get('status') == 'safety_timeout':
        raise SafetyTimeout('this formal block already reached its frozen 24-hour cap')
    clock = InvocationClock(float(state.get('active_wall_seconds', 0.0)))
    if clock.remaining_seconds <= 0:
        state['status'] = 'safety_timeout'
        state['timeout_record'] = {'occurred': True, 'at_utc': _utc_now(), 'reason': 'active wall cap exhausted before next action'}
        _write_state(store, state, clock)
        raise SafetyTimeout('formal block exhausted its frozen 24-hour cap')
    with store.lock():
        try:
            problem = FrozenProblem(source)
            oracle = CachedOracle(problem, store, ledger)
            seeds = SeedBlock(source.root_seed, source.block)
            _initialise_if_needed(store, state, source, runtime, seeds, clock)
            while int(state['confirmed_training_count']) < TOTAL_TRAIN:
                completed = int(state['confirmed_training_count'])
                due_round = completed // TRAIN_PER_ROUND
                if completed and completed % TRAIN_PER_ROUND == 0:
                    _ensure_round_audit(store, ledger, oracle, state, source, runtime, seeds, clock, due_round)
                if clock.remaining_seconds <= 0:
                    raise SafetyTimeout('formal block exhausted its frozen 24-hour cap')
                _confirm_training_step(store, ledger, oracle, state, source, runtime, seeds, clock, completed + 1)
            for round_index in range(1, ROUNDS + 1):
                _ensure_round_audit(store, ledger, oracle, state, source, runtime, seeds, clock, round_index)
            ledger_body = _validate_final_ledger(ledger)
            state['status'] = 'complete'
            _write_state(store, state, clock)
            summary = {'schema': SCHEMA, 'status': 'complete', 'method': METHOD, 'case': case, 'block': block, 'block_root': str(block_root), 'confirmed_training_labels': TOTAL_TRAIN, 'confirmed_nonfeedback_audit_labels': ROUNDS * AUDIT_PER_ROUND, 'shared_initial_labels_reused': len(source.initial_x), 'new_initial_hf': 0, 'common_student_trained': False, 'final_common_test_evaluated': False, 'active_wall_seconds': state['active_wall_seconds'], 'ledger': ledger_body, 'information_boundary': request['source'], 'scientific_parameters': {'domain': request['collection']['continuous_training_domain'], 'normalize_y': True, 'fixed_nugget_variance_standardized_units': 0.001, 'audit_pool_per_round': AUDIT_POOL_SIZE, 'audit_feedback_to_gp': False}}
            store.write_json('summary.json', summary)
            return summary
        except SafetyTimeout as exc:
            state['status'] = 'safety_timeout'
            state['timeout_record'] = {'occurred': True, 'at_utc': _utc_now(), 'reason': str(exc), 'process': exc.process_record, 'confirmed_training_count': state['confirmed_training_count'], 'completed_audit_rounds': state['completed_audit_rounds'], 'not_a_accuracy_failure': True}
            _write_state(store, state, clock)
            raise
        except Exception as exc:
            state['status'] = 'failed'
            state['failure_record'] = {'occurred': True, 'at_utc': _utc_now(), 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(), 'confirmed_training_count': state['confirmed_training_count'], 'completed_audit_rounds': state['completed_audit_rounds']}
            _write_state(store, state, clock)
            raise

def _inspect_worker(runtime: Path, timeout_seconds: float=120.0) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix='uslw_preflight_') as temporary:
        temporary_root = Path(temporary)
        response_path = temporary_root / 'runtime.json'
        process = _run_process([str(runtime), str(SESSION_WORKER), '--inspect-runtime', '--response', str(response_path)], timeout_seconds, stdout_path=temporary_root / 'runtime.stdout.log', stderr_path=temporary_root / 'runtime.stderr.log')
        report = _read_json(response_path)
        process['stdout_log'] = '<ephemeral zero-HF preflight>'
        process['stderr_log'] = '<ephemeral zero-HF preflight>'
    report['launcher'] = process
    return report

def self_test_file_backed_subprocess_logs() -> dict[str, Any]:
    """Zero-HF regression for output larger than ordinary pipe capacity."""
    payload_bytes = 2 * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix='uslw_log_fixture_') as temporary:
        root = Path(temporary)
        stdout_path = root / 'large.stdout.log'
        stderr_path = root / 'large.stderr.log'
        code = "import sys; n={}; sys.stdout.buffer.write(b'O'*n); sys.stdout.buffer.flush(); sys.stderr.buffer.write(b'E'*n); sys.stderr.buffer.flush()".format(payload_bytes)
        record = _run_process([sys.executable, '-c', code], 30.0, stdout_path=stdout_path, stderr_path=stderr_path)
        checks = {'exit_zero': record.get('returncode') == 0, 'not_timed_out': record.get('timed_out') is False, 'stdout_exceeds_pipe_buffer': payload_bytes >= 2 * 1024 * 1024, 'stderr_exceeds_pipe_buffer': payload_bytes >= 2 * 1024 * 1024, 'stdout_all_bytes_preserved': stdout_path.stat().st_size == payload_bytes, 'stderr_all_bytes_preserved': stderr_path.stat().st_size == payload_bytes, 'no_pipe_capture': record.get('capture') == 'append-only files; no subprocess PIPE'}
        compact_record = {key: value for (key, value) in record.items() if key not in {'stdout_tail', 'stderr_tail'}}
        compact_record['stdout_log'] = '<ephemeral zero-HF fixture>'
        compact_record['stderr_log'] = '<ephemeral zero-HF fixture>'
    return {'schema': SCHEMA, 'status': 'PASS' if all(checks.values()) else 'FAIL', 'mode': 'file_backed_subprocess_output_regression', 'fixture_bytes_per_stream': payload_bytes, 'checks': checks, 'launcher': compact_record, 'hf_calls': 0, 'project_data_opened': False}

def preflight(*, source_track_a_root: Path, case: str, block: int, runtime_manifest: Path=DEFAULT_RUNTIME_MANIFEST, python_executable: Path | None=None) -> dict[str, Any]:
    """Zero-HF source/API check; does not fit a project GP or call a problem."""
    source = load_source_block(source_track_a_root, case, block)
    runtime_manifest = Path(runtime_manifest).resolve()
    runtime = _runtime_python(python_executable, runtime_manifest)
    runtime_record = _runtime_record(runtime, runtime_manifest)
    worker = _inspect_worker(runtime)
    problem = FrozenProblem(source)
    chunk_acceptance = Path(__file__).resolve().parent / 'acceptance' / 'gpsearch_chunked_prediction_equivalence_20260907.json'
    chunk_report = _read_json(chunk_acceptance)
    session_acceptance = Path(__file__).resolve().parent / 'acceptance' / 'gpsearch_session_transaction_20260907.json'
    session_report = _read_json(session_acceptance)
    subprocess_acceptance = Path(__file__).resolve().parent / 'acceptance' / 'uslw_subprocess_file_capture_20260907.json'
    subprocess_report = _read_json(subprocess_acceptance)
    checks = {'source_initial_role_only': True, 'source_initial_rows': len(source.initial_x) == CASE_META[case]['initial_train'], 'frozen_problem_import': problem.dimension == source.dimension, 'runtime_commit': worker.get('third_party_commit') == EXPECTED_COMMIT, 'runtime_api': worker.get('status') == 'PASS', 'chunk_equivalence': chunk_report.get('status') == 'PASS' and chunk_report.get('hf_calls') == 0, 'session_transaction': session_report.get('status') == 'PASS' and session_report.get('hf_calls') == 0, 'file_backed_subprocess_capture': subprocess_report.get('status') == 'PASS' and subprocess_report.get('hf_calls') == 0 and (subprocess_report.get('checks', {}).get('no_pipe_capture') is True), 'formal_block': block in FORMAL_BLOCKS, 'no_project_gp_fit': True, 'no_oracle_call': True}
    return {'schema': SCHEMA, 'status': 'PASS' if all(checks.values()) else 'FAIL', 'mode': 'zero_hf_source_api_preflight', 'case': case, 'block': block, 'checks': checks, 'hf_calls': 0, 'source': {'root': str(source.root), 'permitted_label_event': 'shared.initial_train', 'other_label_artifacts_opened': False}, 'runtime': runtime_record, 'worker_inspection': worker, 'chunk_equivalence_acceptance': {'path': str(chunk_acceptance), 'observed_errors': chunk_report.get('errors'), 'tolerances': chunk_report.get('tolerances')}, 'session_transaction_acceptance': {'path': str(session_acceptance), 'checks': session_report.get('checks'), 'fixture': session_report.get('fixture')}, 'subprocess_file_capture_acceptance': {'path': str(subprocess_acceptance), 'fixture_bytes_per_stream': subprocess_report.get('fixture_bytes_per_stream'), 'checks': subprocess_report.get('checks')}, 'frozen_execution': _compute_contract(source, runtime_record)}

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=sorted(CASE_META))
    parser.add_argument('--block', type=int)
    parser.add_argument('--source-track-a-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--runtime-manifest', type=Path, default=DEFAULT_RUNTIME_MANIFEST)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--preflight', action='store_true', help='zero-HF source/API check only; never starts formal collection')
    parser.add_argument('--self-test-subprocess-logs', action='store_true', help='zero-HF >pipe-buffer file-capture regression only')
    parser.add_argument('--report', type=Path, help='optional atomic JSON result path')
    args = parser.parse_args(argv)
    if args.self_test_subprocess_logs:
        report = self_test_file_backed_subprocess_logs()
    else:
        if args.case is None or args.block is None:
            parser.error('--case and --block are required outside the subprocess-log test')
        if args.preflight:
            report = preflight(source_track_a_root=args.source_track_a_root, case=args.case, block=args.block, runtime_manifest=args.runtime_manifest, python_executable=args.python)
        else:
            report = run_collection(source_track_a_root=args.source_track_a_root, output_root=args.output_root, case=args.case, block=args.block, runtime_manifest=args.runtime_manifest, python_executable=args.python)
    if args.report:
        atomic_write_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get('status') in {'PASS', 'complete'} else 1
if __name__ == '__main__':
    raise SystemExit(main())
