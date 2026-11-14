#!/usr/bin/env python3
"""Run one resumable paired seed block of the uncertainty benchmark."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from protocol.config import SamplerConfig
from protocol.problem import in_conditional_domain
from protocol.sampler import run_subset_simulation

from .checkpoint import (
    BlockStore,
    atomic_write_bytes,
)
from .common_student import (
    NeuralCommittee,
    fit_base_student,
    pack_student,
    refine_common_student,
    restore_student,
)
from .contracts import ROOT, CaseContract, ResolvedCase, available_cases, load_case_contract
from .ledger import CallLedger
from .oracle import CachedOracle
from .seed_blocks import SeedBlock
from .strategies import (
    IMPLEMENTED_METHODS,
    QBC_LCB_K2, MC_DROPOUT_VAR_T30, DIR_LDS_FDS,
    QBCLCBK2Strategy, MCDropoutVarT30Strategy,
    LCMD_TP_KGRAD_SKETCH512,
    NN_QBC_VAR_M5,
    OURS_C,
    RAND_REPLAY,
    AcquisitionProposal,
    LCMDTPKGradSketch512Strategy,
    NNQBCVarM5Strategy,
    OursCStrategy,
    RandReplayStrategy,
    keys_of,
)

SCHEMA = "uncertainty-benchmark-block-v1"
MC_DROPOUT_RATE = 0.1
MC_DROPOUT_PASSES = 30
QBC_POOL_SIZE = 1_000_000
QBC_CHUNK_SIZE = 16_384

def _method_dir(method: str) -> Path:
    return Path("methods") / method.lower().replace("-", "_")

def _round_dir(method: str, round_index: int) -> Path:
    return _method_dir(method) / f"round_{round_index:02d}"

def _request(
    resolved: ResolvedCase,
    seeds: SeedBlock,
    device: str,
) -> dict[str, object]:
    request = {
        "schema": SCHEMA,
        "case": resolved.contract.slug,
        "contract": resolved.contract.as_dict(),
        "root_seed": int(seeds.root_seed),
        "block": int(seeds.block),
        "device": str(device),
    }
    return request

def _ensure_method_request(store: BlockStore, method: str) -> None:
    """Freeze one method independently so new methods can join a block later."""

    lcmd_expected_commit = None
    lcmd_actual_commit = None
    if method == LCMD_TP_KGRAD_SKETCH512:
        from external_adapters.lcmd_tp_adapter import EXPECTED_COMMIT, verify_checkout

        lcmd_expected_commit = EXPECTED_COMMIT
        lcmd_actual_commit = verify_checkout()
    record = {
        "schema": "uncertainty-benchmark-method-request-v1",
        "method": method,
        "qbc_candidate_pool": QBC_POOL_SIZE if method == NN_QBC_VAR_M5 else None,
        "qbc_stream_chunk": QBC_CHUNK_SIZE if method == NN_QBC_VAR_M5 else None,
        "lcmd_sketch_dimension": 512 if method == LCMD_TP_KGRAD_SKETCH512 else None,
        "lcmd_expected_commit": lcmd_expected_commit,
        "lcmd_actual_commit": lcmd_actual_commit,
    }
    relative = _method_dir(method) / "method_request.json"
    observed = store.read_json(relative)
    if observed is not None and observed != record:
        raise RuntimeError(f"method request drift for {method}")
    if observed is None:
        store.write_json(relative, record)

def _validated_array(
    value: np.ndarray,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise RuntimeError(f"invalid persisted {name}: expected {shape}, got {result.shape}")
    return result

def _initial_inputs(
    store: BlockStore,
    contract: CaseContract,
    seeds: SeedBlock,
) -> np.ndarray:
    relative = Path("shared") / "initial_inputs.npz"
    saved = store.load_npz(relative)
    if saved is None:
        rng = np.random.default_rng(seeds.seed("initial_inputs"))
        x = rng.standard_normal((contract.initial_size, contract.dimension))
        store.save_npz_once(relative, x=x)
        saved = store.load_npz(relative)
    if saved is None:  # pragma: no cover - filesystem failure
        raise RuntimeError("initial input checkpoint disappeared")
    x = _validated_array(
        saved["x"], (contract.initial_size, contract.dimension), "initial inputs"
    )
    expected = np.random.default_rng(seeds.seed("initial_inputs")).standard_normal(x.shape)
    if not np.array_equal(x, expected):
        raise RuntimeError("initial input checkpoint does not match its random stream")
    return x

def _replay_indices(
    store: BlockStore,
    contract: CaseContract,
    seeds: SeedBlock,
) -> np.ndarray:
    relative = Path("shared") / "replay_indices.npz"
    saved = store.load_npz(relative)
    if saved is None:
        rng = np.random.default_rng(seeds.seed("replay_indices"))
        index = np.asarray(
            rng.choice(contract.train_size, size=contract.replay_size, replace=False),
            dtype=np.int64,
        )
        store.save_npz_once(relative, index=index)
        saved = store.load_npz(relative)
    if saved is None:  # pragma: no cover
        raise RuntimeError("replay checkpoint disappeared")
    index = np.asarray(saved["index"], dtype=np.int64).reshape(-1)
    if (
        index.shape != (contract.replay_size,)
        or np.unique(index).size != index.size
        or np.any(index < 0)
        or np.any(index >= contract.train_size)
    ):
        raise RuntimeError("invalid replay index checkpoint")
    return index

def _load_base_student(
    store: BlockStore,
    resolved: ResolvedCase,
    train_x: np.ndarray,
    train_y: np.ndarray,
    seeds: SeedBlock,
    device: str,
):
    model_path = Path("shared") / "base_student.pkl"
    marker_path = Path("shared") / "base_complete.json"
    marker = store.read_json(marker_path)
    packed = store.load_pickle(model_path) if marker is not None else None
    if marker is not None:
        stored = store.load_npz(Path("shared") / "base_training.npz")
        if packed is None or stored is None or not np.array_equal(stored["x"], train_x) or not np.array_equal(stored["y"], train_y):
            raise RuntimeError("shared base checkpoint does not match training data")
        return restore_student(packed, resolved.problem, device=device)

    student = fit_base_student(
        resolved.contract,
        resolved.problem,
        train_x,
        train_y,
        seed=seeds.seed("shared_student_initialisation"),
        device=device,
    )
    store.save_pickle(model_path, pack_student(student))
    store.save_npz(Path("shared") / "base_training.npz", x=train_x, y=train_y)
    store.write_json(marker_path, {
        "status": "complete",
        "training_rows": int(train_x.shape[0]),
        "audit_rows_used": 0,
    })
    return student

def _load_qbc_committee(
    store: BlockStore,
    resolved: ResolvedCase,
    train_x: np.ndarray,
    train_y: np.ndarray,
    seeds: SeedBlock,
    device: str,
) -> NeuralCommittee:
    model_path = _method_dir(NN_QBC_VAR_M5) / "base_committee.pkl"
    marker_path = _method_dir(NN_QBC_VAR_M5) / "base_committee_complete.json"
    marker = store.read_json(marker_path)
    packed = store.load_pickle(model_path) if marker is not None else None
    if marker is not None:
        stored = store.load_npz(_method_dir(NN_QBC_VAR_M5) / "base_training.npz")
        if packed is None or stored is None or not np.array_equal(stored["x"], train_x) or not np.array_equal(stored["y"], train_y):
            raise RuntimeError("QBC base committee does not match training data")
        return packed.restored(resolved.problem, device=device)

    committee = NeuralCommittee.fit(
        resolved.contract,
        resolved.problem,
        train_x,
        train_y,
        seeds=[
            seeds.seed("qbc_member_initialisation", member=member)
            for member in range(5)
        ],
        device=device,
    )
    store.save_pickle(model_path, committee.packed())
    store.save_npz(_method_dir(NN_QBC_VAR_M5) / "base_training.npz", x=train_x, y=train_y)
    store.write_json(marker_path, {
        "status": "complete",
        "members": 5,
        "extra_hf_calls": 0,
    })
    return committee

def _mc_dropout_recipe(resolved: ResolvedCase):
    """The case's declared learner with dropout enabled, nothing else changed."""
    import dataclasses

    from protocol.surrogate import RECIPES

    recipe = RECIPES[resolved.contract.base_recipe]
    if not hasattr(recipe, "dropout"):
        raise RuntimeError(
            "recipe {!r} exposes no dropout field, so MC-Dropout-Var-T30 cannot be "
            "defined on it".format(resolved.contract.base_recipe)
        )
    return dataclasses.replace(recipe, dropout=MC_DROPOUT_RATE)

def _load_mc_dropout_committee(
    store: BlockStore,
    resolved: ResolvedCase,
    train_x: np.ndarray,
    train_y: np.ndarray,
    seeds: SeedBlock,
    device: str,
) -> NeuralCommittee:
    """One dropout member, wrapped as a one-member committee.

    Ordinary training only: no physical call is made, and the member is cached per
    block exactly like the five-member QBC committee, so a resume reuses it.
    """
    model_path = _method_dir(MC_DROPOUT_VAR_T30) / "base_committee.pkl"
    marker_path = _method_dir(MC_DROPOUT_VAR_T30) / "base_committee_complete.json"
    marker = store.read_json(marker_path)
    packed = store.load_pickle(model_path) if marker is not None else None
    if marker is not None:
        if packed is None:
            raise RuntimeError("MC-dropout base checkpoint is incomplete or corrupt")
        if int(marker.get("members", 0)) != 1:
            raise RuntimeError("MC-dropout base checkpoint has the wrong member count")
        return packed.restored(resolved.problem, device=device)

    member = fit_base_student(
        resolved.contract,
        resolved.problem,
        train_x,
        train_y,
        seed=seeds.seed("mc_dropout_initialisation"),
        device=device,
        recipe=_mc_dropout_recipe(resolved),
    )
    committee = NeuralCommittee([member])
    store.save_pickle(model_path, committee.packed())
    store.write_json(marker_path, {
        "status": "complete",
        "members": 1,
        "dropout_rate": MC_DROPOUT_RATE,
        "stochastic_passes": MC_DROPOUT_PASSES,
        "extra_hf_calls": 0,
    })
    return committee

def _save_proposal(store: BlockStore, relative: Path, proposal: AcquisitionProposal) -> None:
    store.save_npz(
        relative / "proposal.npz",
        train_x=proposal.train_x,
        audit_x=proposal.audit_x,
        train_score=proposal.train_score,
        audit_score=proposal.audit_score,
    )
    store.write_json(relative / "proposal.json", {
        "schema": "uncertainty-benchmark-proposal-v1",
        "method": proposal.method,
        "round": int(proposal.round_index),
        "metadata": proposal.metadata,
        "candidate_pool_persisted": False,
    })

def _load_proposal(
    store: BlockStore,
    relative: Path,
    method: str,
    round_index: int,
    contract: CaseContract,
    excluded: set[bytes],
) -> AcquisitionProposal | None:
    arrays = store.load_npz(relative / "proposal.npz")
    meta = store.read_json(relative / "proposal.json")
    if arrays is None and meta is None:
        return None
    if arrays is None or meta is None:
        raise RuntimeError(f"partial proposal checkpoint in {relative}")
    if (
        meta.get("schema") != "uncertainty-benchmark-proposal-v1"
        or meta.get("method") != method
        or int(meta.get("round", -1)) != round_index
        or meta.get("candidate_pool_persisted") is not False
    ):
        raise RuntimeError(f"proposal metadata drift in {relative}")
    proposal = AcquisitionProposal(
        method=method,
        round_index=round_index,
        train_x=np.asarray(arrays["train_x"], dtype=np.float64),
        audit_x=np.asarray(arrays["audit_x"], dtype=np.float64),
        train_score=np.asarray(arrays["train_score"], dtype=np.float64),
        audit_score=np.asarray(arrays["audit_score"], dtype=np.float64),
        metadata=dict(meta.get("metadata", {})),
    )
    proposal.validate(contract, excluded)
    return proposal

def _acquire(
    method: str,
    student,
    committee: NeuralCommittee | None,
    contract: CaseContract,
    seeds: SeedBlock,
    round_index: int,
    excluded: set[bytes],
    labelled_train_x: np.ndarray,
) -> AcquisitionProposal:
    if method in (RAND_REPLAY, DIR_LDS_FDS):
        from dataclasses import replace
        proposal = RandReplayStrategy().propose(contract, seeds, round_index, excluded)
        return replace(proposal, method=method)
    if method == OURS_C:
        return OursCStrategy().propose(
            student, contract, seeds, round_index, excluded
        )
    if method == NN_QBC_VAR_M5:
        if committee is None:
            raise RuntimeError("QBC acquisition requires its five-member committee")
        return NNQBCVarM5Strategy().propose(
            committee,
            contract,
            seeds,
            round_index,
            excluded,
            pool_size=QBC_POOL_SIZE,
            chunk_size=QBC_CHUNK_SIZE,
        )
    if method in (QBC_LCB_K2, MC_DROPOUT_VAR_T30):
        if committee is None:
            raise RuntimeError("missing auxiliary acquisition network")
        strategy = QBCLCBK2Strategy() if method == QBC_LCB_K2 else MCDropoutVarT30Strategy()
        return strategy.propose(committee, contract, seeds, round_index, excluded,
                                pool_size=QBC_POOL_SIZE, chunk_size=QBC_CHUNK_SIZE)
    if method == LCMD_TP_KGRAD_SKETCH512:
        return LCMDTPKGradSketch512Strategy().propose(
            student,
            labelled_train_x,
            contract,
            seeds,
            round_index,
            excluded,
            pool_size=QBC_POOL_SIZE,
            chunk_size=QBC_CHUNK_SIZE,
        )
    raise ValueError(f"unknown acquisition method {method!r}")

def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

def _prediction_audit(
    student,
    x: np.ndarray,
    y: np.ndarray,
    resolved: ResolvedCase,
) -> dict[str, object]:
    prediction = np.asarray(student.predict(x), dtype=np.float64).reshape(-1)
    error = prediction - np.asarray(y, dtype=np.float64).reshape(-1)
    hits = [
        float(np.mean(in_conditional_domain(y, threshold, resolved.contract.adverse)))
        for threshold in resolved.reference_thresholds
    ]
    return {
        "rows": int(y.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": _correlation(prediction, y),
        "reference_tail_hit_fraction": hits,
    }

def _level_acceptance_rate(level) -> float | None:
    """Acceptance proxy for one subset-simulation level, whatever it is called.

    ``protocol.sampler.Level`` carries ``acceptance_rate``.  The componentwise-AMH
    kernel (hysteretic-lab only) uses ``ComponentwiseAMHLevel``, which does not
    inherit from ``Level`` and measures acceptance per coordinate instead,
    recording ``mean_coordinate_acceptance``.  Both mean "fraction of proposals
    accepted"; the evaluation record reads whichever the level provides.
    """
    for name in ("acceptance_rate", "mean_coordinate_acceptance"):
        value = getattr(level, name, None)
        if value is not None:
            return float(value)
    return None

def _evaluate_student(
    student,
    resolved: ResolvedCase,
    seeds: SeedBlock,
    round_index: int,
) -> dict[str, object]:
    """Proxy-chain evaluation; frozen reference enters only after acquisition."""

    contract = resolved.contract
    cfg = SamplerConfig(
        population=contract.subset_population,
        conditional_probability=contract.conditional_probability,
        n_populations=contract.n_levels,
    )
    run = run_subset_simulation(
        student,
        dimension=contract.dimension,
        adverse=contract.adverse,
        rng=np.random.default_rng(
            seeds.seed("paired_evaluation_chain", round_index=round_index)
        ),
        cfg=cfg,
        kernel=contract.sampler_kernel,
    )
    proxy = np.asarray(run.thresholds, dtype=np.float64)
    reference = resolved.reference_thresholds
    span = float(abs(reference[-1] - reference[0]))
    direction = 1.0 if contract.adverse == "upper" else -1.0
    signed = direction * (proxy - reference) / span
    absolute = np.abs(proxy - reference) / span
    return {
        "schema": "uncertainty-benchmark-evaluation-v1",
        "round": int(round_index),
        "proxy_thresholds": proxy.tolist(),
        "reference_thresholds": reference.tolist(),
        "reference_span": span,
        "signed_adverse_threshold_gap": signed.tolist(),
        "absolute_normalized_threshold_gap": absolute.tolist(),
        "deepest_absolute_normalized_threshold_gap": float(absolute[-1]),
        "acceptance_rates": [
            _level_acceptance_rate(level)
            for level in run.levels
        ],
        "kernel": contract.sampler_kernel,
        "true_model_calls": 0,
        "reference_used_by_acquisition": False,
    }

def _completed_model(
    store: BlockStore,
    relative: Path,
    resolved: ResolvedCase,
    device: str,
):
    marker = store.read_json(relative / "complete.json")
    if marker is None:
        return None, None
    model_path = relative / "student.pkl"
    packed = store.load_pickle(model_path)
    if packed is None:
        raise RuntimeError(f"completed student checkpoint is corrupt in {relative}")
    evaluation = store.read_json(relative / "evaluation.json")
    if evaluation is None:
        raise RuntimeError(f"completed evaluation is missing in {relative}")
    return restore_student(packed, resolved.problem, device=device), evaluation

def _completed_committee(
    store: BlockStore,
    relative: Path,
    resolved: ResolvedCase,
    device: str,
) -> NeuralCommittee | None:
    marker = store.read_json(relative / "complete.json")
    if marker is None or not store.path(relative / "committee.pkl").is_file():
        return None
    model_path = relative / "committee.pkl"
    packed = store.load_pickle(model_path)
    if packed is None:
        raise RuntimeError(f"completed QBC committee checkpoint is corrupt in {relative}")
    return packed.restored(resolved.problem, device=device)

def _result_row(
    resolved: ResolvedCase,
    seeds: SeedBlock,
    method: str,
    round_index: int,
    evaluation: dict[str, object],
) -> dict[str, object]:
    return {
        "case": resolved.contract.slug,
        "block": int(seeds.block),
        "root_seed": int(seeds.root_seed),
        "method": method,
        "round": int(round_index),
        "deepest_absolute_normalized_threshold_gap": evaluation[
            "deepest_absolute_normalized_threshold_gap"
        ],
        "proxy_thresholds": json.dumps(evaluation["proxy_thresholds"]),
        "reference_thresholds": json.dumps(evaluation["reference_thresholds"]),
    }

def _write_results_csv(store: BlockStore, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(store.path("results.csv"), output.getvalue().encode("utf-8"))

def _collect_completed_results(
    store: BlockStore,
    resolved: ResolvedCase,
    seeds: SeedBlock,
    base_evaluation: dict[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    """Aggregate every already-complete method, not just this invocation."""

    rows = [_result_row(resolved, seeds, "A-shared", 0, base_evaluation)]
    completed_methods: list[str] = []
    for method in IMPLEMENTED_METHODS:
        method_rows: list[dict[str, object]] = []
        for round_index in range(1, resolved.contract.rounds + 1):
            relative = _round_dir(method, round_index)
            if store.read_json(relative / "complete.json") is None:
                method_rows = []
                break
            evaluation = store.read_json(relative / "evaluation.json")
            if evaluation is None:
                raise RuntimeError(
                    f"completed method has no evaluation: {method} round {round_index}"
                )
            method_rows.append(
                _result_row(resolved, seeds, method, round_index, evaluation)
            )
        if method_rows:
            completed_methods.append(method)
            rows.extend(method_rows)
    return rows, completed_methods

def run_seed_block(
    case: str,
    *,
    block: int,
    root_seed: int,
    output: Path,
    methods: Iterable[str] = IMPLEMENTED_METHODS,
    device: str = "auto",
) -> dict[str, object]:
    """Execute or resume one complete four-round paired seed block."""

    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a non-empty sequence without duplicates")
    unknown = sorted(set(methods) - set(IMPLEMENTED_METHODS))
    if unknown:
        raise ValueError(f"methods are not implemented: {unknown}")

    resolved = load_case_contract(case)
    contract = resolved.contract
    seeds = SeedBlock(root_seed=int(root_seed), block=int(block))
    request = _request(resolved, seeds, device)
    store = BlockStore(Path(output), request)

    with store.lock():
        ledger = CallLedger(store.path("ledger.json"))
        oracle = CachedOracle(resolved.problem, store, ledger)

        initial_x = _initial_inputs(store, contract, seeds)
        train_x = initial_x[: contract.train_size]
        holdout_x = initial_x[contract.train_size :]
        train = oracle.query(
            "shared.initial_train", train_x, role="initial_train"
        )
        holdout = oracle.query(
            "shared.initial_holdout", holdout_x, role="initial_holdout"
        )
        replay_index = _replay_indices(store, contract, seeds)
        replay_x, replay_y = train.x[replay_index], train.y[replay_index]

        base = _load_base_student(
            store, resolved, train.x, train.y, seeds, device
        )
        base_audit_path = Path("shared") / "holdout_audit.json"
        if store.read_json(base_audit_path) is None:
            store.write_json(
                base_audit_path,
                _prediction_audit(base, holdout.x, holdout.y, resolved),
            )
        base_eval_path = Path("shared") / "base_evaluation.json"
        base_evaluation = store.read_json(base_eval_path)
        if base_evaluation is None:
            base_evaluation = _evaluate_student(base, resolved, seeds, 0)
            store.write_json(base_eval_path, base_evaluation)

        initial_keys = keys_of(train.x, holdout.x)

        for method in methods:
            _ensure_method_request(store, method)
            packed_base = store.load_pickle(Path("shared") / "base_student.pkl")
            if packed_base is None:  # pragma: no cover
                raise RuntimeError("shared base checkpoint disappeared")
            current = restore_student(packed_base, resolved.problem, device=device)
            committee = (
                _load_qbc_committee(
                    store, resolved, train.x, train.y, seeds, device
                )
                if method in (NN_QBC_VAR_M5, QBC_LCB_K2) else
                _load_mc_dropout_committee(store, resolved, train.x, train.y, seeds, device)
                if method == MC_DROPOUT_VAR_T30 else None
            )
            excluded = set(initial_keys)
            cumulative_x: list[np.ndarray] = []
            cumulative_y: list[np.ndarray] = []

            for round_index in range(1, contract.rounds + 1):
                relative = _round_dir(method, round_index)
                proposal = _load_proposal(
                    store, relative, method, round_index, contract, excluded
                )
                if proposal is None:
                    proposal = _acquire(
                        method, current, committee, contract, seeds,
                        round_index, excluded,
                        np.vstack([train.x, *cumulative_x]),
                    )
                    proposal.validate(contract, excluded)
                    _save_proposal(store, relative, proposal)

                q_train = oracle.query(
                    f"{method}.round{round_index}.train",
                    proposal.train_x,
                    role="acquisition_train",
                    method=method,
                    round_index=round_index,
                )
                q_audit = oracle.query(
                    f"{method}.round{round_index}.audit",
                    proposal.audit_x,
                    role="acquisition_audit",
                    method=method,
                    round_index=round_index,
                )
                cumulative_x.append(q_train.x)
                cumulative_y.append(q_train.y)
                excluded.update(keys_of(q_train.x, q_audit.x))

                tail_x = np.vstack(cumulative_x)
                tail_y = np.concatenate(cumulative_y)
                store.save_npz_once(relative / "training_data.npz",
                                    replay_x=replay_x, replay_y=replay_y,
                                    tail_x=tail_x, tail_y=tail_y)
                completed, evaluation = _completed_model(
                    store, relative, resolved, device
                )
                if completed is not None:
                    current = completed
                    if method in (NN_QBC_VAR_M5, QBC_LCB_K2, MC_DROPOUT_VAR_T30):
                        committee = _completed_committee(
                            store, relative, resolved, device
                        )
                        if committee is None:
                            raise RuntimeError("completed QBC round lacks committee")
                    continue

                store.write_json(
                    relative / "acquisition_audit.json",
                    _prediction_audit(current, q_audit.x, q_audit.y, resolved),
                )
                if method == DIR_LDS_FDS:
                    from .dir_student import dir_fit
                    current = dir_fit(current, contract, np.vstack([replay_x, tail_x]),
                                      np.concatenate([replay_y, tail_y]), train.y,
                                      seeds.seed("shared_student_refinement", round_index=round_index),
                                      round_index, store.path(relative))
                else:
                    current = refine_common_student(
                        current,
                        contract,
                        replay_x,
                        replay_y,
                        tail_x,
                        tail_y,
                        new_labels=contract.train_per_round,
                        seed=seeds.seed("shared_student_refinement", round_index=round_index),
                    )
                if committee is not None:
                    committee = committee.refine(
                        contract,
                        replay_x,
                        replay_y,
                        tail_x,
                        tail_y,
                        new_labels=contract.train_per_round,
                        seeds=[
                            seeds.seed(
                                "qbc_committee_refinement",
                                round_index=round_index,
                                member=member,
                            )
                            for member in range(len(committee.members))
                        ],
                    )

                model_path = relative / "student.pkl"
                store.save_pickle(model_path, pack_student(current))
                if committee is not None:
                    committee_path = relative / "committee.pkl"
                    store.save_pickle(committee_path, committee.packed())
                evaluation = _evaluate_student(
                    current, resolved, seeds, round_index
                )
                store.write_json(relative / "evaluation.json", evaluation)
                complete = {
                    "schema": "uncertainty-benchmark-round-v1",
                    "status": "complete",
                    "method": method,
                    "round": round_index,
                    "new_training_labels": contract.train_per_round,
                    "new_audit_labels": contract.audit_per_round,
                    "cumulative_training_labels": int(tail_y.size),
                    "replay_rows": contract.replay_size,
                    "audit_rows_used_for_training": 0,
                }
                store.write_json(relative / "complete.json", complete)

        ledger_body = ledger.as_dict()
        results, completed_methods = _collect_completed_results(
            store, resolved, seeds, base_evaluation
        )
        expected_acquisition = (
            len(completed_methods) * contract.rounds * contract.acquisition_per_round
        )
        acquisition_physical = sum(
            ledger_body["totals_by_role"].get(role, {}).get("physical_calls", 0)
            for role in ("acquisition_train", "acquisition_audit")
        )
        if acquisition_physical != expected_acquisition:
            raise RuntimeError(
                f"acquisition HF budget drift: expected {expected_acquisition}, "
                f"recorded {acquisition_physical}"
            )
        _write_results_csv(store, results)
        summary = {
            "schema": SCHEMA,
            "status": "training_complete",
            "case": contract.slug,
            "block": int(block),
            "root_seed": int(root_seed),
            "methods": completed_methods,
            "invocation_methods": list(methods),
            "rounds": contract.rounds,
            "hf_per_method_per_round": {
                "acquisition_train": contract.train_per_round,
                "acquisition_audit": contract.audit_per_round,
                "total": contract.acquisition_per_round,
            },
            "shared_initial_hf": contract.initial_size,
            "expected_acquisition_hf": expected_acquisition,
            "audit_labels_used_for_training": 0,
            "shared_replay_rows": contract.replay_size,
            "shared_base_student": True,
            "qbc_candidate_pool_persisted": False,
            "reference_visible_to_acquisition": False,
            "ledger": ledger_body,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "device_request": device,
            },
            "results": results,
        }
        store.write_json("summary.json", summary)
        return summary

def _default_output(case: str, block: int) -> Path:
    return Path(__file__).resolve().parent / "results" / case / f"block_{block:03d}"

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=available_cases())
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--root-seed", type=int, default=2026090701)
    parser.add_argument("--methods", nargs="+", choices=IMPLEMENTED_METHODS,
                        default=list(IMPLEMENTED_METHODS))
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or an explicit CUDA device")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.out or _default_output(args.case, args.block)
    summary = run_seed_block(
        args.case,
        block=args.block,
        root_seed=args.root_seed,
        output=output,
        methods=args.methods,
        device=args.device,
    )
    print(json.dumps({
        "status": summary["status"],
        "case": summary["case"],
        "block": summary["block"],
        "methods": summary["methods"],
        "output": str(Path(output).resolve()),
        "physical_hf_calls": summary["ledger"]["totals"]["physical_calls"],
    }, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
