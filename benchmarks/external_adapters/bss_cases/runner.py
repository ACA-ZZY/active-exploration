"""Fail-closed launcher for one budgeted native MATLAB BSS case"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any
from uncertainty_benchmark.checkpoint import BlockStore
from uncertainty_benchmark.seed_blocks import SeedBlock
ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = Path(__file__).resolve().parent
BUDGETED_DIR = ADAPTER_DIR.parent / 'bss_budgeted'
BSS_ROOT = ROOT / 'third_party' / 'bss'
EXPECTED_COMMIT = '574e083634b9d640c4e6bdfc6b94f923e074ff82'
DEFAULT_MATLAB = Path(os.environ.get('MATLAB_EXECUTABLE', 'matlab'))
CASES = ('two-sided-kink', 'radial-kink', 'tail-kink', 'cantilever_beam_2d')
HF_BUDGET = 50000
POPULATION = 1000
P0 = 0.1
DEFAULT_CPU_CORE = 2

class BSSRunError(RuntimeError):
    """Raised when an external BSS run cannot be trusted or safely resumed."""

def _pin_current_process(cpu_core: int) -> dict[str, object]:
    return _pin_pid(os.getpid(), cpu_core)

def _pin_child_process(pid: int, cpu_core: int) -> dict[str, object]:
    return _pin_pid(int(pid), cpu_core)

def _mask_cores(mask: int) -> list[int]:
    return [index for index in range(max(1, int(mask).bit_length())) if mask & 1 << index]

def _pin_pid(pid: int, cpu_core: int) -> dict[str, object]:
    """Pin a process without requiring a non-standard Python dependency."""
    (pid, cpu_core) = (int(pid), int(cpu_core))
    if os.name == 'nt':
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        open_process.restype = ctypes.c_void_p
        get_affinity = kernel32.GetProcessAffinityMask
        get_affinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        get_affinity.restype = ctypes.c_int
        set_affinity = kernel32.SetProcessAffinityMask
        set_affinity.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        set_affinity.restype = ctypes.c_int
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        handle = open_process(512 | 1024, False, pid)
        if not handle:
            raise BSSRunError(f'OpenProcess failed for pid {pid}: winerror={ctypes.get_last_error()}')
        try:
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            if not get_affinity(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
                raise BSSRunError(f'GetProcessAffinityMask failed for pid {pid}: winerror={ctypes.get_last_error()}')
            before_mask = int(process_mask.value)
            requested_mask = 1 << cpu_core
            if not before_mask & requested_mask:
                raise BSSRunError(f'CPU core {cpu_core} is unavailable to pid {pid}; allowed affinity is {_mask_cores(before_mask)}')
            if not set_affinity(handle, ctypes.c_size_t(requested_mask)):
                raise BSSRunError(f'SetProcessAffinityMask failed for pid {pid}: winerror={ctypes.get_last_error()}')
            observed_process = ctypes.c_size_t()
            observed_system = ctypes.c_size_t()
            if not get_affinity(handle, ctypes.byref(observed_process), ctypes.byref(observed_system)):
                raise BSSRunError(f'affinity verification failed for pid {pid}: winerror={ctypes.get_last_error()}')
            after = _mask_cores(int(observed_process.value))
        finally:
            close_handle(handle)
        if after != [cpu_core]:
            raise BSSRunError(f'failed to pin pid {pid} to CPU core {cpu_core}: {after}')
        return {'pid': pid, 'affinity_before': _mask_cores(before_mask), 'affinity': after, 'mechanism': 'SetProcessAffinityMask', 'processor_group_scope': 'current-Windows-processor-group'}
    if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
        raise BSSRunError('this platform has no supported CPU-affinity API')
    before = sorted((int(value) for value in os.sched_getaffinity(pid)))
    if cpu_core not in before:
        raise BSSRunError(f'CPU core {cpu_core} is unavailable to pid {pid}; allowed affinity is {before}')
    os.sched_setaffinity(pid, {cpu_core})
    after = sorted((int(value) for value in os.sched_getaffinity(pid)))
    if after != [cpu_core]:
        raise BSSRunError(f'failed to pin pid {pid} to CPU core {cpu_core}: {after}')
    return {'pid': pid, 'affinity_before': before, 'affinity': after, 'mechanism': 'sched_setaffinity'}

def _kill_job_tree(root_pid: int) -> dict[str, object]:
    """Kill only the processes currently descended from one MATLAB root."""
    root_pid = int(root_pid)
    if os.name == 'nt':
        completed = subprocess.run(['taskkill.exe', '/PID', str(root_pid), '/T', '/F'], check=False, capture_output=True, text=True, errors='replace', timeout=60.0, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return {'root_pid': root_pid, 'scope': 'root PID and descendants only', 'mechanism': 'taskkill /PID <root> /T /F', 'return_code': int(completed.returncode), 'stdout': completed.stdout, 'stderr': completed.stderr}
    try:
        process_group = os.getpgid(root_pid)
        os.killpg(process_group, signal.SIGKILL)
        return {'root_pid': root_pid, 'scope': 'root process group', 'mechanism': 'killpg(SIGKILL)', 'process_group': int(process_group), 'return_code': 0}
    except ProcessLookupError:
        return {'root_pid': root_pid, 'scope': 'root process group', 'mechanism': 'killpg(SIGKILL)', 'return_code': 0, 'already_gone': True}

def _stop_job_process(process: subprocess.Popen[bytes], *, wait_seconds: float=60.0) -> tuple[int, dict[str, object]]:
    """Terminate one owned process tree and report if anything survived."""
    first = _kill_job_tree(process.pid)
    second = None
    try:
        return_code = process.wait(timeout=float(wait_seconds))
    except subprocess.TimeoutExpired:
        second = _kill_job_tree(process.pid)
        try:
            return_code = process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            observed = process.poll()
            return_code = -999 if observed is None else int(observed)
    report = {'first_pass': first, 'second_pass': second, 'return_code_after_kill': int(return_code), 'root_still_running': process.poll() is None}
    return (int(return_code), report)

def _audit_oracle_journal(path: Path) -> dict[str, object]:
    """Recover completed HF rows and any uncertain in-flight request"""
    path = Path(path)
    audit: dict[str, object] = {'schema': 'bss-oracle-journal-audit-v1', 'artifact': path.name, 'exists': path.is_file(), 'complete_lines': 0, 'returned_request_count': 0, 'returned_physical_rows': 0, 'requested_physical_rows': 0, 'in_flight_requests': [], 'unknown_in_flight_physical_rows': 0, 'malformed_final_line': False, 'errors': []}
    if not path.is_file():
        return audit
    payload = path.read_bytes()
    raw_lines = payload.splitlines()
    active: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    errors: list[str] = []
    expected_sequence = 1
    for (index, raw_line) in enumerate(raw_lines):
        try:
            event = json.loads(raw_line.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index == len(raw_lines) - 1:
                audit['malformed_final_line'] = True
            else:
                errors.append(f'malformed interior JSON line {index + 1}')
            continue
        audit['complete_lines'] = int(audit['complete_lines']) + 1
        if not isinstance(event, dict):
            errors.append(f'line {index + 1} is not a JSON object')
            continue
        if event.get('schema') != 'bss-oracle-journal-event-v1':
            errors.append(f'line {index + 1} has an unknown schema')
            continue
        sequence = event.get('sequence')
        if sequence != expected_sequence:
            errors.append(f'line {index + 1} sequence {sequence!r} != {expected_sequence}')
            if isinstance(sequence, int):
                expected_sequence = sequence
        expected_sequence += 1
        request_id = event.get('request_id')
        rows = event.get('physical_rows')
        if not isinstance(request_id, str) or not request_id:
            errors.append(f'line {index + 1} has an invalid request_id')
            continue
        if not isinstance(rows, int) or rows <= 0:
            errors.append(f'line {index + 1} has an invalid physical_rows')
            continue
        kind = event.get('event')
        if kind == 'request_started':
            if request_id in active or request_id in seen:
                errors.append(f'duplicate request start {request_id}')
                continue
            active[request_id] = {'request_id': request_id, 'physical_rows': rows, 'sequence': sequence}
            audit['requested_physical_rows'] = int(audit['requested_physical_rows']) + rows
        elif kind == 'response_returned':
            started = active.pop(request_id, None)
            if started is None:
                errors.append(f'response without active request {request_id}')
                continue
            if int(started['physical_rows']) != rows:
                errors.append(f'row-count mismatch for {request_id}')
                continue
            seen.add(request_id)
            audit['returned_request_count'] = int(audit['returned_request_count']) + 1
            audit['returned_physical_rows'] = int(audit['returned_physical_rows']) + rows
        else:
            errors.append(f'line {index + 1} has unknown event {kind!r}')
    in_flight = sorted(active.values(), key=lambda row: int(row['sequence']))
    audit['in_flight_requests'] = in_flight
    audit['unknown_in_flight_physical_rows'] = sum((int(row['physical_rows']) for row in in_flight))
    audit['errors'] = errors
    return audit

def _git(*arguments: str) -> str:
    process = subprocess.run(['git', '-C', str(BSS_ROOT), *arguments], check=False, capture_output=True, text=True)
    if process.returncode != 0:
        raise BSSRunError(f'BSS git provenance check failed: {process.stderr.strip()}')
    return process.stdout.strip()

def _verify_third_party() -> dict[str, object]:
    if not BSS_ROOT.is_dir():
        raise BSSRunError(f'pinned BSS checkout is absent: {BSS_ROOT}')
    commit = _git('rev-parse', 'HEAD')
    if commit != EXPECTED_COMMIT:
        raise BSSRunError(f'BSS commit mismatch: expected {EXPECTED_COMMIT}, observed {commit}')
    if subprocess.run(['git', '-C', str(BSS_ROOT), 'diff', '--quiet'], check=False).returncode != 0:
        raise BSSRunError('pinned BSS tracked working tree is dirty')
    mex = BSS_ROOT / 'functions' / 'bss' / 'replicate_particles.mexw64'
    archive = BSS_ROOT / 'stk-2.4.0-allpurpose.tar.gz'
    if not mex.is_file() or not archive.is_file():
        raise BSSRunError('compiled BSS helper or pinned STK archive is absent')
    return {'repository': 'https://git.code.sf.net/p/kriging/contrib-bss', 'commit': commit, 'tracked_status': 'clean'}

def _matlab_seed(root_seed: int, block: int) -> tuple[int, int]:
    raw = SeedBlock(int(root_seed), int(block)).seed('BSS-native-SUR-SMC')
    return (raw, raw % 2 ** 32)

def _quote_matlab(value: str | Path) -> str:
    return str(value).replace("'", "''")

def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise BSSRunError(f'invalid BSS JSON artifact: {path}') from exc
    if not isinstance(value, dict):
        raise BSSRunError(f'BSS artifact is not a JSON object: {path}')
    return value

def _validate_result(store: BlockStore, *, slug: str, matlab_seed: int, budget: int, cpu_core: int) -> dict[str, Any]:
    summary_path = store.path('summary.json')
    ledger_path = store.path('ledger.json')
    result_path = store.path('result.mat')
    for path in (summary_path, ledger_path, result_path):
        if not path.is_file():
            raise BSSRunError(f'BSS run omitted required artifact: {path}')
    summary = _read_json(summary_path)
    ledger = _read_json(ledger_path)
    required = {'schema': 'bss-project-run-v1', 'status': 'complete', 'case': slug, 'random_seed': matlab_seed, 'hf_budget': budget, 'requested_cpu_core': cpu_core, 'native_population': POPULATION, 'native_conditional_probability': P0, 'reference_probability_supplied': False, 'reference_threshold_supplied': False, 'third_party_commit': EXPECTED_COMMIT, 'oracle_journal_artifact': 'oracle_journal.jsonl'}
    for (key, expected) in required.items():
        if summary.get(key) != expected:
            raise BSSRunError(f'BSS summary contract drift for {key}: {summary.get(key)!r} != {expected!r}')
    calls = int(summary.get('hf_physical_rows', -1))
    if not 0 < calls <= budget:
        raise BSSRunError(f'BSS physical-call budget drift: {calls}/{budget}')
    if ledger.get('schema') != 'bss-hf-ledger-v1':
        raise BSSRunError('BSS ledger schema drift')
    if ledger.get('initial_shared_data_reused') is not False:
        raise BSSRunError('BSS must count from a new native N0 design')
    if int(ledger.get('totals', {}).get('physical_rows', -1)) != calls:
        raise BSSRunError('BSS summary/ledger HF totals disagree')
    if sum((int(row['physical_rows']) for row in ledger.get('events', []))) != calls:
        raise BSSRunError('BSS ledger event total is inconsistent')
    journal = _audit_oracle_journal(store.path('oracle_journal.jsonl'))
    if not journal['exists']:
        raise BSSRunError('BSS run omitted its durable oracle journal')
    if journal['errors'] or journal['malformed_final_line']:
        raise BSSRunError(f'BSS oracle journal is invalid: {journal}')
    if journal['in_flight_requests']:
        raise BSSRunError('complete BSS run retains an in-flight oracle request')
    if int(journal['returned_physical_rows']) != calls:
        raise BSSRunError('BSS oracle journal/summary HF totals disagree')
    capped = bool(summary.get('stopped_by_hf_budget_cap'))
    native_converged = bool(summary.get('native_converged'))
    expected_reason = 'hf_budget_cap' if capped else 'native_stopping_rule'
    if capped == native_converged:
        raise BSSRunError('BSS termination flags must distinguish cap from convergence')
    if summary.get('termination_reason') != expected_reason:
        raise BSSRunError('BSS termination reason contradicts its termination flags')
    return summary

def _completion(store: BlockStore, summary: dict[str, Any]) -> dict[str, object]:
    return {'schema': 'bss-project-complete-v1', 'status': 'complete', 'case': summary['case'], 'hf_physical_rows': summary['hf_physical_rows']}

def run_bss_case(slug: str, *, block: int, output: str | Path, root_seed: int=2026090701, hf_budget: int=HF_BUDGET, cpu_core: int=DEFAULT_CPU_CORE, matlab_executable: str | Path=DEFAULT_MATLAB, timeout_seconds: float=86400.0) -> dict[str, Any]:
    if slug not in CASES:
        raise ValueError(f'choose a BSS case from {CASES}')
    if int(block) < 0 or int(root_seed) < 0:
        raise ValueError('block and root_seed must be non-negative')
    if int(hf_budget) != HF_BUDGET:
        raise ValueError('formal BSS comparison freezes the HF additional-query ceiling at 50000')
    if int(cpu_core) < 0:
        raise ValueError('cpu_core must be non-negative')
    matlab = Path(matlab_executable).resolve()
    if not matlab.is_file():
        raise FileNotFoundError(f'MATLAB executable is absent: {matlab}')
    third_party = _verify_third_party()
    (raw_seed, matlab_seed) = _matlab_seed(root_seed, block)
    parent_process = _pin_current_process(int(cpu_core))
    additional_budget = int(hf_budget)
    hf_budget = additional_budget + 1000
    request = {'schema': 'bss-project-request-v1', 'case': slug, 'block': int(block), 'root_seed': int(root_seed), 'uint64_named_seed': raw_seed, 'matlab_twister_seed': matlab_seed, 'method': 'BSS author implementation, lazy-oracle instrumentation', 'event': 'g <= 0', 'bss_score_transform': '-g', 'native_initial_design': 1000, 'native_population': POPULATION, 'native_conditional_probability': P0, 'hf_budget': int(hf_budget), 'cpu_core': int(cpu_core), 'matlab_single_computational_thread': True, 'hf_count_starts_at_zero': True, 'shared_initial_inputs': True, 'shared_initial_data_reused': False, 'additional_query_budget': additional_budget, 'reference_probability_supplied': False, 'reference_threshold_supplied': False, 'third_party': third_party}
    store = BlockStore(Path(output), request)
    with store.lock():
        complete = store.read_json('complete.json')
        if complete is not None:
            summary = _validate_result(store, slug=slug, matlab_seed=matlab_seed, budget=hf_budget, cpu_core=int(cpu_core))
            if complete != _completion(store, summary):
                raise BSSRunError('cached BSS completion does not match results')
            return summary
        if store.path('run_started.json').is_file():
            finished = all((store.path(name).is_file() for name in ('summary.json', 'ledger.json', 'result.mat')))
            if finished:
                summary = _validate_result(store, slug=slug, matlab_seed=matlab_seed, budget=hf_budget, cpu_core=int(cpu_core))
                store.write_json('complete.json', _completion(store, summary))
                return summary
            store.write_json('interrupted_audit.json', {'schema': 'bss-interrupted-run-audit-v1', 'status': 'incomplete', 'automatic_retry': False, 'oracle_journal_audit': _audit_oracle_journal(store.path('oracle_journal.jsonl')), 'existing_artifacts': sorted((path.name for path in store.root.iterdir() if path.is_file()))})
            raise BSSRunError('an earlier BSS process started without a valid completion; automatic rerun is refused because it could repeat HF calls; see interrupted_audit.json for returned and unknown rows')
        store.write_json('run_started.json', {'schema': 'bss-project-start-v1', 'matlab_executable': str(matlab), 'runner_process': parent_process, 'cpu_core': int(cpu_core), 'matlab_single_computational_thread': True, 'automatic_rerun_after_interruption': False})
        import numpy as np
        from scipy.io import savemat
        dimension = 8 if slug == "radial-kink" else 50 if slug == "tail-kink" else 2
        seeds = SeedBlock(int(root_seed), int(block))
        initial_x = np.random.default_rng(seeds.seed("initial_inputs")).standard_normal((1250, dimension))[:1000]
        savemat(store.path("initial_training.mat"), {"x0": initial_x})
        expression = f"addpath('{_quote_matlab(ADAPTER_DIR)}'); run_bss_case('{_quote_matlab(slug)}','{_quote_matlab(store.root)}',{matlab_seed},{hf_budget},{int(cpu_core)});"
        environment = os.environ.copy()
        environment.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''})
        stdout_path = store.path('matlab_stdout.log')
        stderr_path = store.path('matlab_stderr.log')
        started = time.monotonic()
        timeout_record = None
        runtime_kill_record = None
        runtime_error: BaseException | None = None
        child_process: dict[str, object] | None = None
        process: subprocess.Popen[bytes] | None = None
        return_code = -998
        with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
            try:
                process = subprocess.Popen([str(matlab), '-singleCompThread', '-batch', expression], cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0), start_new_session=os.name != 'nt')
                child_process = _pin_child_process(process.pid, int(cpu_core))
                store.write_json('process_started.json', {'schema': 'bss-process-start-v1', 'runner': parent_process, 'matlab': child_process, 'command_flags': ['-singleCompThread', '-batch'], 'cpu_core': int(cpu_core)})
                try:
                    return_code = process.wait(timeout=float(timeout_seconds))
                except subprocess.TimeoutExpired:
                    (return_code, timeout_record) = _stop_job_process(process)
            except BaseException as exc:
                runtime_error = exc
                if process is not None and process.poll() is None:
                    (return_code, runtime_kill_record) = _stop_job_process(process)
                elif process is not None and process.poll() is not None:
                    return_code = int(process.returncode)
            finally:
                stdout.flush()
                stderr.flush()
                os.fsync(stdout.fileno())
                os.fsync(stderr.fileno())
        elapsed = time.monotonic() - started
        journal_audit = _audit_oracle_journal(store.path('oracle_journal.jsonl'))
        process_result = {'schema': 'bss-process-result-v1', 'return_code': int(return_code), 'elapsed_seconds': elapsed, 'timed_out': timeout_record is not None, 'timeout_seconds': float(timeout_seconds), 'kill_report': timeout_record, 'runtime_kill_report': runtime_kill_record, 'runner': parent_process, 'matlab': child_process, 'oracle_journal_audit': journal_audit}
        store.write_json('process_result.json', process_result)
        if runtime_error is not None:
            store.write_json('launcher_failure.json', {**process_result, 'status': 'launcher_or_runtime_error', 'error_type': type(runtime_error).__name__, 'error': repr(runtime_error), 'automatic_retry': False})
            if isinstance(runtime_error, (KeyboardInterrupt, SystemExit)):
                raise runtime_error
            raise BSSRunError('MATLAB BSS launcher failed after recording logs, process-tree cleanup, and the durable HF journal') from runtime_error
        if timeout_record is not None:
            store.write_json('timeout_failure.json', {**process_result, 'status': 'timeout', 'automatic_retry': False})
            raise BSSRunError(f'MATLAB BSS exceeded {timeout_seconds:g}s and its job tree was killed; inspect {stdout_path}, {stderr_path}, and timeout_failure.json')
        if return_code != 0:
            store.write_json('launcher_failure.json', {**process_result, 'status': 'matlab_nonzero_exit', 'automatic_retry': False})
            raise BSSRunError(f"MATLAB BSS failed with exit {return_code}; inspect {store.path('matlab_stderr.log')} and launcher_failure.json")
        summary = _validate_result(store, slug=slug, matlab_seed=matlab_seed, budget=hf_budget, cpu_core=int(cpu_core))
        completion = _completion(store, summary)
        store.write_json('complete.json', completion)
        return summary

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--block', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root-seed', type=int, default=2026090701)
    parser.add_argument('--hf-budget', type=int, default=HF_BUDGET)
    parser.add_argument('--cpu-core', type=int, default=DEFAULT_CPU_CORE)
    parser.add_argument('--matlab', type=Path, default=DEFAULT_MATLAB)
    parser.add_argument('--timeout-seconds', type=float, default=86400.0)
    args = parser.parse_args(argv)
    summary = run_bss_case(args.case, block=args.block, output=args.out, root_seed=args.root_seed, hf_budget=args.hf_budget, cpu_core=args.cpu_core, matlab_executable=args.matlab, timeout_seconds=args.timeout_seconds)
    print(json.dumps({'status': summary['status'], 'case': summary['case'], 'pf_estimate': summary['pf_estimate'], 'hf_physical_rows': summary['hf_physical_rows'], 'output': str(args.out.resolve())}, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
