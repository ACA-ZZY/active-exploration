"""Read-only, fail-closed adapter from native US-LW to the common terminal audit"""
from __future__ import annotations
from pathlib import Path
from uncertainty_benchmark.checkpoint import canonical_json
from uncertainty_benchmark.strategies import row_key
from . import uslw_common_student as bridge

def prepare_context(shared_block: Path, student_block: Path, target: str, device: str) -> dict:
    from uncertainty_benchmark.posthoc_evaluation import BASE_TARGET, _target_lineage
    (shared_block, student_block) = (shared_block.resolve(), student_block.resolve())
    if target != bridge.METHOD or device != 'cuda':
        raise bridge.StudentBridgeError('external posthoc accepts only US-LW C4 on CUDA')
    request = bridge._source_request(student_block)
    if request.get('schema') != bridge.SCHEMA or request.get('method') != target or request.get('device') != device or (request.get('root_seed') != bridge.ROOT_SEED) or (Path(request.get('source_block', '')).resolve() != shared_block):
        raise bridge.StudentBridgeError('external student request does not match shared block')
    source = bridge.load_source_block(shared_block.parent.parent, request['case'], int(request['block']))
    if source.root.resolve() != shared_block:
        raise bridge.StudentBridgeError('shared block location differs from its case/block')
    shared_request = bridge._source_request(shared_block)
    (contract, _) = bridge._contract_and_problem(source, shared_request)
    if canonical_json(request.get('contract')) != canonical_json(contract.as_dict()):
        raise bridge.StudentBridgeError('external student contract drift')
    collection = Path(request['collection_root']).resolve()
    (_, ledger) = bridge._validate_collection(collection, source)
    summary = bridge._read_json(student_block / 'summary.json')
    expected = {'schema': bridge.SCHEMA, 'status': 'training_complete', 'method': target, 'case': source.case, 'block': source.block, 'rounds': 4, 'shared_base_reused': True, 'new_hf_calls': 0, 'training_labels_consumed': 2000, 'diagnostic_audit_labels_consumed': 1000, 'audit_labels_used_for_training': 0}
    if any((summary.get(k) != v for (k, v) in expected.items())):
        raise bridge.StudentBridgeError('external common student is not valid C4-complete')
    events = bridge._event_map(ledger)
    if len(events) != 2004:
        raise bridge.StudentBridgeError('US-LW collection must have 2000 train and 4 audit events')
    (excluded, lineage) = _target_lineage(shared_block, BASE_TARGET, contract)
    marker = None
    for round_index in range(1, 5):
        marker = bridge.validate_round_artifacts(student_block, round_index)
        (train_x, train_y, train_records) = bridge._round_training(collection, events, round_index, contract.dimension)
        (audit_x, _, audit_record) = bridge._round_audit(collection, events, round_index, contract.dimension)
        round_lineage = bridge._read_json(student_block / 'rounds' / f'round_{round_index:02d}' / 'lineage.json')
        if round_lineage.get('train_event_count') != 500 or round_lineage.get('audit') != audit_record or round_lineage.get('audit_y_feedback') is not False:
            raise bridge.StudentBridgeError('student refinement lineage differs from native collection')
        for (role, x, records) in (('acquisition_train', train_x, train_records), ('acquisition_audit', audit_x, [audit_record])):
            before = len(excluded)
            excluded.update((row_key(row) for row in x))
            lineage.append({'role': role, 'method': target, 'round_index': round_index, 'rows': len(x), 'new_unique_rows': len(excluded) - before, 'source_artifacts': records})
    model_path = student_block / 'rounds' / 'round_04' / 'student.pkl'
    sources = {'external_adapters/' + name: Path(__file__).parent / name for name in ('uslw_posthoc_context.py', 'uslw_common_student.py', 'sequential_uslw_collection.py')}
    sources['uncertainty_benchmark/track_b_oracle.py'] = bridge.MORETEST_ROOT / 'uncertainty_benchmark' / 'track_b_oracle.py'
    return {'training_block': student_block, 'request': request, 'checkpoint': (model_path, 4, marker), 'lineage': (excluded, lineage), 'sources': sources}
