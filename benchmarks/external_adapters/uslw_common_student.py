"""Train the shared Track-A student from completed native US-LW labels"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import io
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import numpy as np
MORETEST_ROOT = Path(__file__).resolve().parents[1]
if str(MORETEST_ROOT) not in sys.path:
    sys.path.insert(0, str(MORETEST_ROOT))
from external_adapters.sequential_uslw_collection import AUDIT_PER_ROUND, CASE_META, DEFAULT_EXPERIMENT_ROOT, METHOD, ROOT_SEED, ROUNDS, TRAIN_PER_ROUND, load_source_block
from protocol.config import SamplerConfig
from protocol.problem import in_conditional_domain
from protocol.sampler import run_subset_simulation
from uncertainty_benchmark.checkpoint import BlockStore, atomic_write_bytes, atomic_write_json
from uncertainty_benchmark.common_student import pack_student, refine_common_student, restore_student
from uncertainty_benchmark.contracts import CaseContract, ResolvedCase
from uncertainty_benchmark.seed_blocks import SeedBlock
SCHEMA = 'uslw.common-student.v1'
DEFAULT_SOURCE_ROOT = DEFAULT_EXPERIMENT_ROOT / 'track_a'
DEFAULT_COLLECTION_ROOT = DEFAULT_EXPERIMENT_ROOT / 'track_a_external'

class StudentBridgeError(RuntimeError):
    """Fail-closed source, lineage or downstream-training violation."""

def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StudentBridgeError(f'required JSON is absent: {path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise StudentBridgeError(f'JSON root must be an object: {path}')
    return value

def _source_request(source_root: Path) -> dict[str, Any]:
    wrapper = _read_json(source_root / 'request.json')
    if set(wrapper) != {'request'}:
        raise StudentBridgeError('source request wrapper has unexpected fields')
    request = wrapper['request']
    if not isinstance(request, dict):
        raise StudentBridgeError('source request payload is not an object')
    return request

def _contract_and_problem(source, request: Mapping[str, Any]) -> tuple[CaseContract, object]:
    payload = request.get('contract')
    if not isinstance(payload, dict):
        raise StudentBridgeError('source contract is absent')
    try:
        contract = CaseContract(**payload)
    except TypeError as exc:
        raise StudentBridgeError(f'source contract cannot construct CaseContract: {exc}') from exc
    expected = CASE_META[source.case]
    if contract.slug != source.case or contract.problem_name != source.problem_name or contract.dimension != source.dimension or (contract.adverse != source.adverse) or (contract.train_size != expected['initial_train']) or (contract.rounds != ROUNDS) or (contract.train_per_round != TRAIN_PER_ROUND) or (contract.audit_per_round != AUDIT_PER_ROUND) or (contract.replay_size != 500):
        raise StudentBridgeError('source CaseContract differs from the US-LW formal contract')
    module_name = f"_uslw_student_problem_{source.case.replace('-', '_')}_{source.block}"
    spec = importlib.util.spec_from_file_location(module_name, source.problem_path)
    if spec is None or spec.loader is None:
        raise StudentBridgeError('cannot import paired frozen problem source')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'build', None)):
        raise StudentBridgeError('paired frozen problem source has no build()')
    problem = module.build()
    identity = (problem.name, int(problem.dimension), problem.adverse, problem.recipe)
    expected_identity = (contract.problem_name, contract.dimension, contract.adverse, contract.base_recipe)
    if identity != expected_identity:
        raise StudentBridgeError(f'paired frozen problem identity/recipe drift: {identity} != {expected_identity}')
    return (contract, problem)

def _reference(source, request: Mapping[str, Any], contract: CaseContract) -> tuple[np.ndarray, Path]:
    relative = contract.reference_manifest.replace('\\', '/')
    path = MORETEST_ROOT / relative
    if not path.is_file():
        raise StudentBridgeError('paired reference manifest is absent')
    body = _read_json(path)
    thresholds = np.asarray(body.get('reference_thresholds'), dtype=np.float64)
    if thresholds.shape != (contract.n_levels,) or not np.all(np.isfinite(thresholds)):
        raise StudentBridgeError('paired reference threshold vector is invalid')
    thresholds.setflags(write=False)
    return (thresholds, path)

def _base_and_replay(source, contract: CaseContract, problem, device: str):
    marker_path = source.root / 'shared' / 'base_complete.json'
    model_path = source.root / 'shared' / 'base_student.pkl'
    marker = _read_json(marker_path)
    if marker.get('status') != 'complete' or int(marker.get('training_rows', -1)) != contract.train_size or int(marker.get('audit_rows_used', -1)) != 0:
        raise StudentBridgeError('paired shared base checkpoint contract is invalid')
    from protocol import surrogate as protocol_surrogate
    protocol_surrogate._torch_modules()
    protocol_surrogate._temporal_attention_module_class()
    packed = pickle.loads(model_path.read_bytes())
    with np.load(source.root / 'shared' / 'base_training.npz', allow_pickle=False) as arrays:
        if not np.array_equal(arrays['x'], source.initial_x) or not np.array_equal(arrays['y'], source.initial_y):
            raise StudentBridgeError('base model training data differ from the source labels')
    student = restore_student(packed, problem, device=device)
    replay_path = source.root / 'shared' / 'replay_indices.npz'
    with np.load(replay_path, allow_pickle=False) as archive:
        if set(archive.files) != {'index'}:
            raise StudentBridgeError('paired replay artifact fields are unsupported')
        index = np.asarray(archive['index'], dtype=np.int64).reshape(-1)
    if index.shape != (contract.replay_size,) or np.unique(index).size != index.size or np.any(index < 0) or np.any(index >= contract.train_size):
        raise StudentBridgeError('paired replay indices are invalid')
    return (student, index, {'base_marker': str(marker_path), 'base_student': str(model_path), 'replay_indices': str(replay_path), 'base_retrained': False})

def _validate_collection(collection_root: Path, source) -> tuple[dict[str, Any], dict[str, Any]]:
    wrapper = _read_json(collection_root / 'request.json')
    request = wrapper.get('request')
    if not isinstance(request, dict):
        raise StudentBridgeError('US-LW collection request payload is absent')
    if request.get('schema') != 'uslw.collection.v1' or request.get('method') != METHOD or request.get('case') != source.case or (int(request.get('block', -1)) != source.block):
        raise StudentBridgeError('US-LW collection does not match the paired source block')
    state = _read_json(collection_root / 'state.json')
    summary = _read_json(collection_root / 'summary.json')
    if state.get('status') != 'complete' or summary.get('status') != 'complete':
        raise StudentBridgeError('US-LW acquisition must be terminal before downstream training')
    if int(summary.get('confirmed_training_labels', -1)) != ROUNDS * TRAIN_PER_ROUND or int(summary.get('confirmed_nonfeedback_audit_labels', -1)) != ROUNDS * AUDIT_PER_ROUND:
        raise StudentBridgeError('terminal US-LW collection has wrong label counts')
    ledger = _read_json(collection_root / 'ledger.json')
    if ledger.get('schema') != 'uncertainty-benchmark-ledger-v1':
        raise StudentBridgeError('terminal US-LW ledger schema is invalid')
    return (request, ledger)

def _event_map(ledger: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    events = ledger.get('events')
    if not isinstance(events, list):
        raise StudentBridgeError('US-LW ledger events are absent')
    result: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get('event_id'), str):
            raise StudentBridgeError('US-LW ledger contains an invalid event')
        if event['event_id'] in result:
            raise StudentBridgeError('US-LW ledger contains a duplicate event id')
        result[event['event_id']] = event
    return result

def _query_from_event(collection_root: Path, events: Mapping[str, Mapping[str, Any]], event_id: str, *, role: str, round_index: int, rows: int, dimension: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    event = events.get(event_id)
    if event is None:
        raise StudentBridgeError(f'US-LW ledger event is absent: {event_id}')
    expected = {'event_id': event_id, 'role': role, 'method': METHOD, 'round_index': int(round_index), 'logical_calls': int(rows), 'unique_inputs': int(rows), 'physical_calls': int(rows)}
    for (key, value) in expected.items():
        if event.get(key) != value:
            raise StudentBridgeError(f'US-LW ledger event drift for {event_id}: {key}')
    relative = event.get('artifact')
    if not isinstance(relative, str):
        raise StudentBridgeError(f'US-LW event artifact path is absent: {event_id}')
    path = (collection_root / relative).resolve()
    try:
        path.relative_to(collection_root.resolve())
    except ValueError as exc:
        raise StudentBridgeError('US-LW query artifact escapes collection root') from exc
    expected_fields = {'x', 'y', 'origin_physical_calls'}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise StudentBridgeError(f'US-LW query fields are unsupported: {event_id}')
        x = np.ascontiguousarray(np.asarray(archive['x'], dtype=np.float64))
        y_original = np.asarray(archive['y'], dtype=np.float64)
        y = np.ascontiguousarray(y_original.reshape(-1))
        origin = int(np.asarray(archive['origin_physical_calls']).item())
    if x.shape != (rows, dimension) or y.shape != (rows,) or origin != rows or (not np.all(np.isfinite(x))) or (not np.all(np.isfinite(y))):
        raise StudentBridgeError(f'US-LW query artifact is invalid: {event_id}')
    return (x, y, {'event_id': event_id, 'artifact': str(path)})

def _round_training(collection_root: Path, events, round_index: int, dimension: int):
    (xs, ys, records) = ([], [], [])
    first = (round_index - 1) * TRAIN_PER_ROUND + 1
    last = round_index * TRAIN_PER_ROUND
    for global_step in range(first, last + 1):
        event_id = f'{METHOD}.round{round_index}.train.step{global_step:04d}'
        (x, y, record) = _query_from_event(collection_root, events, event_id, role='acquisition_train', round_index=round_index, rows=1, dimension=dimension)
        xs.append(x)
        ys.append(y)
        records.append(record)
    x = np.ascontiguousarray(np.vstack(xs))
    y = np.ascontiguousarray(np.concatenate(ys))
    return (x, y, records)

def _round_audit(collection_root: Path, events, round_index: int, dimension: int):
    event_id = f'{METHOD}.round{round_index}.audit'
    (x, y, record) = _query_from_event(collection_root, events, event_id, role='acquisition_audit', round_index=round_index, rows=AUDIT_PER_ROUND, dimension=dimension)
    selection_path = collection_root / 'rounds' / f'round_{round_index:02d}' / 'audit_selection.npz'
    with np.load(selection_path, allow_pickle=False) as archive:
        selected_x = np.ascontiguousarray(np.asarray(archive['x'], dtype=np.float64))
        if 'y' in archive.files:
            raise StudentBridgeError('US-LW audit selection unexpectedly contains labels')
    if selected_x.shape != x.shape or not np.array_equal(selected_x, x):
        raise StudentBridgeError('US-LW audit query differs from frozen audit selection')
    return (x, y, record)

def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def _prediction_audit(student, x: np.ndarray, y: np.ndarray, resolved: ResolvedCase) -> dict[str, Any]:
    prediction = np.asarray(student.predict(x), dtype=np.float64).reshape(-1)
    error = prediction - np.asarray(y, dtype=np.float64).reshape(-1)
    hits = [float(np.mean(in_conditional_domain(y, threshold, resolved.contract.adverse))) for threshold in resolved.reference_thresholds]
    return {'rows': int(y.size), 'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(np.square(error)))), 'pearson': _correlation(prediction, y), 'reference_tail_hit_fraction': hits, 'used_for_training': False}

def _evaluate_student(student, resolved: ResolvedCase, seeds: SeedBlock, round_index: int) -> dict[str, Any]:
    contract = resolved.contract
    cfg = SamplerConfig(population=contract.subset_population, conditional_probability=contract.conditional_probability, n_populations=contract.n_levels)
    run = run_subset_simulation(student, dimension=contract.dimension, adverse=contract.adverse, rng=np.random.default_rng(seeds.seed('paired_evaluation_chain', round_index=round_index)), cfg=cfg, kernel=contract.sampler_kernel)
    proxy = np.asarray(run.thresholds, dtype=np.float64)
    reference = resolved.reference_thresholds
    span = float(abs(reference[-1] - reference[0]))
    direction = 1.0 if contract.adverse == 'upper' else -1.0
    signed = direction * (proxy - reference) / span
    absolute = np.abs(proxy - reference) / span
    return {'schema': 'uncertainty-benchmark-evaluation-v1', 'round': int(round_index), 'proxy_thresholds': proxy.tolist(), 'reference_thresholds': reference.tolist(), 'reference_span': span, 'signed_adverse_threshold_gap': signed.tolist(), 'absolute_normalized_threshold_gap': absolute.tolist(), 'deepest_absolute_normalized_threshold_gap': float(absolute[-1]), 'acceptance_rates': [None if level.acceptance_rate is None else float(level.acceptance_rate) for level in run.levels], 'kernel': contract.sampler_kernel, 'true_model_calls': 0, 'reference_used_by_acquisition': False}

def _load_completed_student(store: BlockStore, relative: Path, problem, device: str):
    marker = store.read_json(relative / 'complete.json')
    if marker is None:
        return (None, None)
    validate_round_artifacts(store.root, int(relative.name.removeprefix('round_')))
    model_path = store.path(relative / 'student.pkl')
    from protocol import surrogate as protocol_surrogate
    protocol_surrogate._torch_modules()
    protocol_surrogate._temporal_attention_module_class()
    packed = pickle.loads(model_path.read_bytes())
    evaluation = store.read_json(relative / 'evaluation.json')
    if evaluation is None:
        raise StudentBridgeError('completed US-LW common-student evaluation is absent')
    return (restore_student(packed, problem, device=device), evaluation)

def validate_round_artifacts(root: Path, round_index: int) -> dict[str, Any]:
    """Validate a sealed refinement without loading a model or observing truth."""
    relative = root / 'rounds' / f'round_{round_index:02d}'
    marker = _read_json(relative / 'complete.json')
    expected = {'schema': SCHEMA, 'status': 'complete', 'method': METHOD, 'round': round_index, 'new_training_labels': TRAIN_PER_ROUND, 'new_audit_labels': AUDIT_PER_ROUND, 'cumulative_training_labels': round_index * TRAIN_PER_ROUND, 'replay_rows': 500, 'audit_rows_used_for_training': 0}
    if any((marker.get(key) != value for (key, value) in expected.items())):
        raise StudentBridgeError('US-LW common-student completion contract drift')
    for name in ('student.pkl', 'lineage.json', 'evaluation.json'):
        path = relative / name
        if not path.is_file():
            raise StudentBridgeError(f'US-LW completed round artifact drift: {name}')
    return marker

def run_student(*, source_track_a_root: Path, collection_root: Path, case: str, block: int, device: str, output: Path | None=None) -> dict[str, Any]:
    if device != 'cuda':
        raise StudentBridgeError("formal shared-student refinement must use device='cuda'")
    source = load_source_block(source_track_a_root, case, block)
    source_request = _source_request(source.root)
    (collection_request, ledger) = _validate_collection(collection_root, source)
    events = _event_map(ledger)
    (contract, problem) = _contract_and_problem(source, source_request)
    (thresholds, reference_path) = _reference(source, source_request, contract)
    resolved = ResolvedCase(contract, problem, thresholds, reference_path)
    output = output or collection_root / 'downstream_common_student'
    request = {'schema': SCHEMA, 'method': METHOD, 'case': case, 'block': int(block), 'root_seed': ROOT_SEED, 'device': device, 'source_block': str(source.root), 'collection_root': str(collection_root.resolve()), 'contract': asdict(contract), 'shared_base_reused': True, 'shared_refinement_seed_role': 'shared_student_refinement', 'audit_labels_used_for_training': False, 'reference_available_only_after_terminal_acquisition': True, 'collection_contract': {'normalize_y': collection_request['numerics']['normalize_y'], 'fixed_nugget_variance': collection_request['numerics']['fixed_nugget_variance'], 'rounds': collection_request['collection']['rounds'], 'train_per_round': collection_request['collection']['train_per_round'], 'audit_per_round': collection_request['collection']['audit_per_round']}}
    store = BlockStore(output, request)
    with store.lock():
        (current, replay_index, base_record) = _base_and_replay(source, contract, problem, device)
        replay_x = source.initial_x[replay_index]
        replay_y = source.initial_y[replay_index]
        seeds = SeedBlock(ROOT_SEED, block)
        cumulative_x: list[np.ndarray] = []
        cumulative_y: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        base_evaluation = _read_json(source.root / 'shared' / 'base_evaluation.json')
        rows.append({'case': case, 'block': block, 'method': 'A-shared', 'round': 0, 'deepest_absolute_normalized_threshold_gap': base_evaluation['deepest_absolute_normalized_threshold_gap'], 'proxy_thresholds': json.dumps(base_evaluation['proxy_thresholds']), 'reference_thresholds': json.dumps(base_evaluation['reference_thresholds'])})
        for round_index in range(1, ROUNDS + 1):
            (train_x, train_y, train_records) = _round_training(collection_root, events, round_index, contract.dimension)
            cumulative_x.append(train_x)
            cumulative_y.append(train_y)
            (audit_x, audit_y, audit_record) = _round_audit(collection_root, events, round_index, contract.dimension)
            relative = Path('rounds') / f'round_{round_index:02d}'
            store.save_npz_once(relative / 'training_data.npz',
                                replay_x=replay_x, replay_y=replay_y,
                                tail_x=np.vstack(cumulative_x), tail_y=np.concatenate(cumulative_y))
            (completed, evaluation) = _load_completed_student(store, relative, problem, device)
            if completed is not None:
                current = completed
            else:
                audit_diagnostic = _prediction_audit(current, audit_x, audit_y, resolved)
                store.write_json(relative / 'acquisition_audit.json', audit_diagnostic)
                tail_x = np.ascontiguousarray(np.vstack(cumulative_x))
                tail_y = np.ascontiguousarray(np.concatenate(cumulative_y))
                current = refine_common_student(current, contract, replay_x, replay_y, tail_x, tail_y, new_labels=TRAIN_PER_ROUND, seed=seeds.seed('shared_student_refinement', round_index=round_index))
                model_path = relative / 'student.pkl'
                store.save_pickle(model_path, pack_student(current))
                evaluation = _evaluate_student(current, resolved, seeds, round_index)
                store.write_json(relative / 'evaluation.json', evaluation)
                lineage = {'train_event_count': len(train_records), 'audit': audit_record, 'audit_y_feedback': False}
                store.write_json(relative / 'lineage.json', lineage)
                store.write_json(relative / 'complete.json', {'schema': SCHEMA, 'status': 'complete', 'method': METHOD, 'round': round_index, 'new_training_labels': TRAIN_PER_ROUND, 'new_audit_labels': AUDIT_PER_ROUND, 'cumulative_training_labels': round_index * TRAIN_PER_ROUND, 'replay_rows': contract.replay_size, 'audit_rows_used_for_training': 0})
            rows.append({'case': case, 'block': block, 'method': METHOD, 'round': round_index, 'deepest_absolute_normalized_threshold_gap': evaluation['deepest_absolute_normalized_threshold_gap'], 'proxy_thresholds': json.dumps(evaluation['proxy_thresholds']), 'reference_thresholds': json.dumps(evaluation['reference_thresholds'])})
        output_csv = io.StringIO(newline='')
        writer = csv.DictWriter(output_csv, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_bytes(store.path('results.csv'), output_csv.getvalue().encode('utf-8'))
        summary = {'schema': SCHEMA, 'status': 'training_complete', 'method': METHOD, 'case': case, 'block': block, 'rounds': ROUNDS, 'shared_base_reused': True, 'base': base_record, 'new_hf_calls': 0, 'training_labels_consumed': ROUNDS * TRAIN_PER_ROUND, 'diagnostic_audit_labels_consumed': ROUNDS * AUDIT_PER_ROUND, 'audit_labels_used_for_training': 0, 'common_student_models': [str(store.path(Path('rounds') / f'round_{r:02d}' / 'student.pkl')) for r in range(1, ROUNDS + 1)], 'terminal_independent_audit_complete': False, 'results_csv': str(store.path('results.csv'))}
        store.write_json('summary.json', summary)
        return summary

def preflight(*, source_track_a_root: Path, case: str, block: int, collection_root: Path | None) -> dict[str, Any]:
    source = load_source_block(source_track_a_root, case, block)
    request = _source_request(source.root)
    (contract, _problem) = _contract_and_problem(source, request)
    base_marker = _read_json(source.root / 'shared' / 'base_complete.json')
    replay_path = source.root / 'shared' / 'replay_indices.npz'
    with np.load(replay_path, allow_pickle=False) as archive:
        replay = np.asarray(archive['index'], dtype=np.int64).reshape(-1)
    checks = {'source_contract': contract.slug == case, 'paired_base_present': base_marker.get('status') == 'complete', 'paired_base_not_retrained': True, 'paired_replay_500': replay.shape == (500,) and np.unique(replay).size == 500, 'training_api_has_no_audit_argument': 'audit' not in refine_common_student.__code__.co_varnames, 'no_hf_call': True}
    acquisition = {'present': False, 'terminal': False}
    if collection_root is not None and collection_root.is_dir():
        (_request, ledger) = _validate_collection(collection_root, source)
        events = _event_map(ledger)
        acquisition = {'present': True, 'terminal': True, 'train_events': sum((event.get('role') == 'acquisition_train' for event in events.values())), 'audit_events': sum((event.get('role') == 'acquisition_audit' for event in events.values()))}
    status = 'PASS' if all(checks.values()) else 'FAIL'
    return {'schema': SCHEMA, 'status': status, 'mode': 'zero_hf_downstream_interface_preflight', 'hf_calls': 0, 'case': case, 'block': block, 'checks': checks, 'acquisition': acquisition, 'formal_device': 'cuda', 'formal_training_started': False}

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(CASE_META), required=True)
    parser.add_argument('--block', type=int, required=True)
    parser.add_argument('--source-track-a-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--collection-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    collection_root = args.collection_root
    if collection_root is None:
        collection_root = DEFAULT_COLLECTION_ROOT / METHOD / args.case / f'block_{args.block:03d}'
    if args.preflight:
        report = preflight(source_track_a_root=args.source_track_a_root, case=args.case, block=args.block, collection_root=collection_root if collection_root.exists() else None)
    else:
        report = run_student(source_track_a_root=args.source_track_a_root, collection_root=collection_root, case=args.case, block=args.block, device=args.device, output=args.output)
    if args.report:
        atomic_write_json(args.report, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get('status') in {'PASS', 'training_complete'} else 1
if __name__ == '__main__':
    raise SystemExit(main())
