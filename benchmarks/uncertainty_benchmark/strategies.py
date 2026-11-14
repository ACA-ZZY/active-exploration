"""Leakage-safe acquisition strategies for the first executable slice."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from protocol.config import SamplerConfig
from protocol.problem import adverse_order
from protocol.sampler import run_subset_simulation

from .contracts import CaseContract
from .seed_blocks import SeedBlock

RAND_REPLAY = "RAND-Replay"
OURS_C = "OURS-C"
NN_QBC_VAR_M5 = "NN-QBC-Var-M5"
QBC_LCB_K2 = "QBC-LCB-k2"
MC_DROPOUT_VAR_T30 = "MC-Dropout-Var-T30"
DIR_LDS_FDS = "DIR-LDS-FDS"
LCMD_TP_KGRAD_SKETCH512 = "LCMD-TP-kgrad-sketch512"
IMPLEMENTED_METHODS = (
    RAND_REPLAY,
    OURS_C,
    NN_QBC_VAR_M5,
    LCMD_TP_KGRAD_SKETCH512,
    QBC_LCB_K2,
    MC_DROPOUT_VAR_T30,
    DIR_LDS_FDS,
)

def row_key(row: np.ndarray) -> bytes:
    value = np.asarray(row, dtype=np.float64)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError("candidate rows must be finite one-dimensional vectors")
    value = np.ascontiguousarray(value.copy())
    value[value == 0.0] = 0.0
    return value.tobytes()

def keys_of(*arrays: np.ndarray) -> set[bytes]:
    result: set[bytes] = set()
    for array in arrays:
        value = np.asarray(array, dtype=np.float64)
        if value.size == 0:
            continue
        result.update(row_key(row) for row in np.atleast_2d(value))
    return result

@dataclass(frozen=True)
class AcquisitionProposal:
    method: str
    round_index: int
    train_x: np.ndarray
    audit_x: np.ndarray
    train_score: np.ndarray
    audit_score: np.ndarray
    metadata: dict[str, object]

    def validate(self, contract: CaseContract, excluded: set[bytes]) -> None:
        tx = np.atleast_2d(np.asarray(self.train_x, dtype=np.float64))
        ax = np.atleast_2d(np.asarray(self.audit_x, dtype=np.float64))
        if tx.shape != (contract.train_per_round, contract.dimension):
            raise ValueError(f"invalid acquisition-train shape {tx.shape}")
        if ax.shape != (contract.audit_per_round, contract.dimension):
            raise ValueError(f"invalid acquisition-audit shape {ax.shape}")
        if np.asarray(self.train_score).shape != (contract.train_per_round,):
            raise ValueError("invalid acquisition-train score shape")
        if np.asarray(self.audit_score).shape != (contract.audit_per_round,):
            raise ValueError("invalid acquisition-audit score shape")
        selected = [row_key(row) for row in np.vstack([tx, ax])]
        if len(set(selected)) != contract.acquisition_per_round:
            raise ValueError("proposal contains duplicate or cross-role inputs")
        if any(key in excluded for key in selected):
            raise ValueError("proposal overlaps prior data lineage")

def _split_selected(
    method: str,
    round_index: int,
    x: np.ndarray,
    score: np.ndarray,
    contract: CaseContract,
    rng: np.random.Generator,
    metadata: dict[str, object],
) -> AcquisitionProposal:
    x = np.asarray(x, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if x.shape != (contract.acquisition_per_round, contract.dimension):
        raise ValueError("selected pool has the wrong shape")
    order = rng.permutation(x.shape[0])
    train_idx = order[: contract.train_per_round]
    audit_idx = order[contract.train_per_round :]
    return AcquisitionProposal(
        method=method,
        round_index=round_index,
        train_x=x[train_idx],
        audit_x=x[audit_idx],
        train_score=score[train_idx],
        audit_score=score[audit_idx],
        metadata=metadata,
    )

class RandReplayStrategy:
    method = RAND_REPLAY

    def propose(
        self,
        contract: CaseContract,
        seeds: SeedBlock,
        round_index: int,
        excluded: set[bytes],
    ) -> AcquisitionProposal:
        rng = np.random.default_rng(
            seeds.seed("rand_nominal", method=self.method, round_index=round_index)
        )
        selected: list[np.ndarray] = []
        assigned = set(excluded)
        while len(selected) < contract.acquisition_per_round:
            need = contract.acquisition_per_round - len(selected)
            draw = rng.standard_normal((max(need * 2, 64), contract.dimension))
            for row in draw:
                key = row_key(row)
                if key in assigned:
                    continue
                assigned.add(key)
                selected.append(np.asarray(row, dtype=np.float64).copy())
                if len(selected) == contract.acquisition_per_round:
                    break
        x = np.vstack(selected)
        score = np.full(x.shape[0], np.nan, dtype=np.float64)
        proposal = _split_selected(
            self.method, round_index, x, score, contract,
            np.random.default_rng(
                seeds.seed("role_split", method=self.method, round_index=round_index)
            ),
            {"candidate_source": "independent_nominal_stream"},
        )
        proposal.validate(contract, excluded)
        return proposal

def _unique_indices(
    samples: np.ndarray,
    indices: np.ndarray,
    excluded: set[bytes],
) -> tuple[np.ndarray, list[bytes]]:
    chosen: list[int] = []
    keys: list[bytes] = []
    local: set[bytes] = set()
    for raw in np.asarray(indices, dtype=int):
        index = int(raw)
        key = row_key(samples[index])
        if key in excluded or key in local:
            continue
        chosen.append(index)
        keys.append(key)
        local.add(key)
    return np.asarray(chosen, dtype=int), keys

class OursCStrategy:
    method = OURS_C

    def propose(
        self,
        student,
        contract: CaseContract,
        seeds: SeedBlock,
        round_index: int,
        excluded: set[bytes],
        *,
        max_attempts: int = 8,
    ) -> AcquisitionProposal:
        sampler = SamplerConfig(
            population=contract.subset_population,
            conditional_probability=contract.conditional_probability,
            n_populations=contract.n_levels,
        )
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            chain_seed = seeds.seed(
                "ours_chain", method=self.method, round_index=round_index,
                attempt=attempt,
            )
            navigator = student.clone_frozen(seed=chain_seed)
            run = run_subset_simulation(
                navigator,
                dimension=contract.dimension,
                adverse=contract.adverse,
                rng=np.random.default_rng(chain_seed),
                cfg=sampler,
                kernel=contract.sampler_kernel,
            )
            try:
                proposal = self._from_run(
                    run, contract, seeds, round_index, excluded, attempt
                )
                proposal.validate(contract, excluded)
                return proposal
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(
            f"OURS-C could not form a unique {contract.acquisition_per_round}-row "
            f"proposal after {max_attempts} proxy-only attempts"
        ) from last_error

    def _from_run(
        self,
        run,
        contract: CaseContract,
        seeds: SeedBlock,
        round_index: int,
        excluded: set[bytes],
        attempt: int,
    ) -> AcquisitionProposal:
        assigned = set(excluded)
        plans: list[dict[str, object]] = []
        for level_number, level in enumerate(run.levels):
            samples = np.asarray(level.samples, dtype=np.float64)
            values = np.asarray(level.values, dtype=np.float64).reshape(-1)
            order = adverse_order(values, contract.adverse)
            available, available_keys = _unique_indices(samples, order, assigned)
            if available.size < contract.train_per_level:
                raise ValueError(f"level {level_number} lacks unique training candidates")
            take = available[: contract.train_per_level]
            assigned.update(available_keys[: contract.train_per_level])
            plans.append({"samples": samples, "values": values, "order": order, "train": take})

        audit_rng = np.random.default_rng(
            seeds.seed(
                "ours_audit_split", method=self.method, round_index=round_index,
                attempt=attempt,
            )
        )
        for level_number, plan in enumerate(plans):
            available, available_keys = _unique_indices(
                plan["samples"], plan["order"], assigned
            )
            if available.size < contract.audit_per_level:
                raise ValueError(f"level {level_number} lacks unique audit candidates")
            local = audit_rng.permutation(available.size)[: contract.audit_per_level]
            take = available[local]
            assigned.update(available_keys[int(index)] for index in local)
            plan["audit"] = take

        train_x = np.vstack([plan["samples"][plan["train"]] for plan in plans])
        audit_x = np.vstack([plan["samples"][plan["audit"]] for plan in plans])
        train_score = np.concatenate([plan["values"][plan["train"]] for plan in plans])
        audit_score = np.concatenate([plan["values"][plan["audit"]] for plan in plans])
        return AcquisitionProposal(
            method=self.method,
            round_index=round_index,
            train_x=train_x,
            audit_x=audit_x,
            train_score=train_score,
            audit_score=audit_score,
            metadata={
                "candidate_source": "surrogate_subset_simulation",
                "kernel": contract.sampler_kernel,
                "population_per_level": contract.subset_population,
                "conditional_probability": contract.conditional_probability,
                "attempt": attempt,
                "proxy_thresholds": [float(level.threshold) for level in run.levels],
                "acceptance_rates": [
                    getattr(level, "acceptance_rate", getattr(level, "mean_coordinate_acceptance", None))
                    for level in run.levels
                ],
            },
        )

def nominal_chunks(
    dimension: int,
    size: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[int, np.ndarray]]:
    if size < 1 or chunk_size < 1:
        raise ValueError("candidate pool and chunk sizes must be positive")
    for start in range(0, size, chunk_size):
        count = min(chunk_size, size - start)
        yield start, rng.standard_normal((count, dimension))

def materialize_nominal_pool(
    dimension: int,
    size: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rebuild the shared pool in RAM, matching ``nominal_chunks`` exactly."""

    pool = np.empty((size, dimension), dtype=np.float64)
    cursor = 0
    for start, chunk in nominal_chunks(dimension, size, chunk_size, rng):
        if start != cursor:
            raise RuntimeError("candidate stream is not contiguous")
        pool[start : start + chunk.shape[0]] = chunk
        cursor += chunk.shape[0]
    if cursor != size:  # pragma: no cover - generator invariant
        raise RuntimeError("candidate stream ended early")
    return pool

def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)

def streaming_topk(
    chunks: Iterable[tuple[int, np.ndarray, np.ndarray]],
    k: int,
    excluded: set[bytes],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic score-descending top-k without storing the pool."""

    if k < 1:
        raise ValueError("k must be positive")
    best_x: list[np.ndarray] = []
    best_s = np.empty(0, dtype=np.float64)
    best_i = np.empty(0, dtype=np.int64)
    for start, x, score in chunks:
        x = np.asarray(x, dtype=np.float64)
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or score.shape != (x.shape[0],):
            raise ValueError("candidate chunk and scores do not align")
        valid = np.isfinite(score) & np.all(np.isfinite(x), axis=1)
        x = x[valid]
        score = score[valid]
        indices = int(start) + np.flatnonzero(valid)
        if x.size == 0:
            continue
        local_n = min(x.shape[0], max(4 * k, k + len(excluded)))
        if local_n < x.shape[0]:
            local = np.argpartition(-score, local_n - 1)[:local_n]
            x, score, indices = x[local], score[local], indices[local]
        if best_x:
            x = np.vstack([best_x[0], x])
            score = np.concatenate([best_s, score])
            indices = np.concatenate([best_i, indices])
        order = np.lexsort((indices, -score))
        kept_x: list[np.ndarray] = []
        kept_s: list[float] = []
        kept_i: list[int] = []
        local_keys: set[bytes] = set()
        for index in order:
            key = row_key(x[index])
            if key in excluded or key in local_keys:
                continue
            local_keys.add(key)
            kept_x.append(x[index].copy())
            kept_s.append(float(score[index]))
            kept_i.append(int(indices[index]))
            if len(kept_x) == k:
                break
        best_x = [np.vstack(kept_x)] if kept_x else []
        best_s = np.asarray(kept_s, dtype=np.float64)
        best_i = np.asarray(kept_i, dtype=np.int64)
    if not best_x or best_s.size != k:
        raise RuntimeError(f"candidate stream supplied only {best_s.size} unique valid rows; need {k}")
    return best_x[0], best_s, best_i

class NNQBCVarM5Strategy:
    method = NN_QBC_VAR_M5

    def propose(
        self,
        committee,
        contract: CaseContract,
        seeds: SeedBlock,
        round_index: int,
        excluded: set[bytes],
        *,
        pool_size: int = 1_000_000,
        chunk_size: int = 16_384,
        candidate_chunks: Iterable[tuple[int, np.ndarray]] | None = None,
    ) -> AcquisitionProposal:
        if len(committee.members) != 5:
            raise ValueError("NN-QBC-Var-M5 requires exactly five committee members")
        pool_seed = None
        if candidate_chunks is None:
            pool_seed = seeds.seed("candidate_pool", round_index=round_index)
            rng = np.random.default_rng(
                pool_seed
            )
            candidate_chunks = nominal_chunks(
                contract.dimension, pool_size, chunk_size, rng
            )

        def scored():
            for start, x in candidate_chunks:
                _, variance = committee.mean_variance(x)
                yield start, x, variance

        x, score, source_indices = streaming_topk(
            scored(), contract.acquisition_per_round, excluded
        )
        proposal = _split_selected(
            self.method, round_index, x, score, contract,
            np.random.default_rng(
                seeds.seed("role_split", method=self.method, round_index=round_index)
            ),
            {
                "candidate_source": "shared_iid_nominal_pool",
                "pool_size": int(pool_size),
                "chunk_size": int(chunk_size),
                "candidate_pool_seed": None if pool_seed is None else int(pool_seed),
                "committee_members": 5,
                "variance_ddof": 1,
                "selected_source_indices": [int(value) for value in source_indices],
            },
        )
        proposal.validate(contract, excluded)
        return proposal

class LCMDTPKGradSketch512Strategy:
    """Faithful wrapper around the pinned authors' LCMD-TP implementation."""

    method = LCMD_TP_KGRAD_SKETCH512

    @staticmethod
    def _network_inputs(student, x: np.ndarray) -> np.ndarray:
        if student.recipe.name == "recurrent":
            return student._scale_drive(student._driving(x)[0])
        if student.recipe.name not in ("network", "deep"):
            raise ValueError(
                "LCMD-TP-kgrad-sketch512 is registered only for feed-forward "
                f"network/deep students, got {student.recipe.name!r}"
            )
        if student._model is None or isinstance(student._model, list):
            raise ValueError("LCMD needs one fitted feed-forward PyTorch module")
        raw = student._features(np.atleast_2d(np.asarray(x, dtype=np.float64)))
        if student._scaler is None:
            raise ValueError("LCMD student has no frozen input standardiser")
        return np.asarray(student._scaler.transform(raw), dtype=np.float32)

    def propose(
        self,
        student,
        labelled_train_x: np.ndarray,
        contract: CaseContract,
        seeds: SeedBlock,
        round_index: int,
        excluded: set[bytes],
        *,
        pool_size: int = 1_000_000,
        chunk_size: int = 16_384,
    ) -> AcquisitionProposal:
        from external_adapters.lcmd_tp_adapter import select_lcmd_tp

        pool = materialize_nominal_pool(
            contract.dimension,
            pool_size,
            chunk_size,
            np.random.default_rng(
                seeds.seed("candidate_pool", round_index=round_index)
            ),
        )
        train_features = self._network_inputs(student, labelled_train_x)
        pool_features = self._network_inputs(student, pool)
        selection = select_lcmd_tp(
            student._model[0] if isinstance(student._model, list) else student._model,
            train_features,
            pool_features,
            contract.acquisition_per_round,
            sketch_dim=512,
            linear_layers_only=contract.requires_sequence_map,
            nn_batch_size=256 if contract.requires_sequence_map else 8192,
            seed=seeds.seed(
                "lcmd_random_projection", method=self.method,
                round_index=round_index,
            ),
        )
        source_indices = np.asarray(selection.indices, dtype=np.int64)
        x = pool[source_indices]
        score = np.full(x.shape[0], np.nan, dtype=np.float64)
        proposal = _split_selected(
            self.method,
            round_index,
            x,
            score,
            contract,
            np.random.default_rng(
                seeds.seed("role_split", method=self.method, round_index=round_index)
            ),
            {
                "candidate_source": "shared_iid_nominal_pool",
                "pool_size": int(pool_size),
                "chunk_size": int(chunk_size),
                "candidate_pool_seed": int(
                    seeds.seed("candidate_pool", round_index=round_index)
                ),
                "candidate_pool_persisted": False,
                "labelled_geometry_rows": int(np.asarray(labelled_train_x).shape[0]),
                "input_to_official_code": "case features then frozen student standardisation",
                "selected_source_indices": [int(value) for value in source_indices],
                "selection_order_is_not_a_pointwise_score": True,
                "official_adapter": _jsonable(selection.metadata),
            },
        )
        proposal.validate(contract, excluded)
        return proposal

class MCDropoutVarT30Strategy:
    """Monte-Carlo dropout uncertainty sampling with T = 30 stochastic passes.

    The learner is a single network trained with dropout active.  At acquisition time
    the dropout layers are left switched on, thirty forward passes give a mean and a
    variance per candidate, and the score is that variance - the same score the
    committee sampler uses, so the two arms differ only in where the epistemic spread
    is read from: between independently fitted members, or within one dropout network.
    """

    method = MC_DROPOUT_VAR_T30
    PASSES = 30

    def propose(
        self,
        committee,
        contract: CaseContract,
        seeds: SeedBlock,
        round_index: int,
        excluded: set[bytes],
        *,
        pool_size: int = 1_000_000,
        chunk_size: int = 16_384,
        candidate_chunks: Iterable[tuple[int, np.ndarray]] | None = None,
    ) -> AcquisitionProposal:
        if len(committee.members) != 1:
            raise ValueError("MC-Dropout-Var-T30 requires exactly one dropout member")
        member = committee.members[0]
        if candidate_chunks is None:
            rng = np.random.default_rng(
                seeds.seed("candidate_pool", round_index=round_index)
            )
            candidate_chunks = nominal_chunks(
                contract.dimension, pool_size, chunk_size, rng
            )
        pass_seed = seeds.seed("mc_dropout_passes", round_index=round_index)

        def scored():
            for start, x in candidate_chunks:
                _, variance = member.predict_stochastic(
                    x, passes=self.PASSES, seed=pass_seed
                )
                yield start, x, variance

        x, score, source_indices = streaming_topk(
            scored(), contract.acquisition_per_round, excluded
        )
        proposal = _split_selected(
            self.method, round_index, x, score, contract,
            np.random.default_rng(
                seeds.seed("role_split", method=self.method, round_index=round_index)
            ),
            {
                "candidate_source": "shared_iid_nominal_pool",
                "pool_size": int(pool_size),
                "chunk_size": int(chunk_size),
                "members": 1,
                "acquisition": "mc_dropout_variance",
                "stochastic_passes": int(self.PASSES),
                "pass_seed": int(pass_seed),
                "dropout_rate": float(getattr(member.recipe, "dropout", 0.0)),
                "variance_ddof": 1,
                "selected_source_indices": [int(value) for value in source_indices],
            },
        )
        proposal.validate(contract, excluded)
        return proposal

def confidence_score(mean, variance, adverse, kappa=2.0):
    if adverse not in ('upper', 'lower'):
        raise ValueError(f'Unknown adverse direction: {adverse}')
    mean, variance = np.asarray(mean), np.asarray(variance)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise FloatingPointError('Nonfinite acquisition predictions; no rows removed')
    sd = np.sqrt(np.maximum(variance, 0.0))
    return mean + kappa * sd if adverse == 'upper' else -(mean - kappa * sd)

class DirectionCorrectQBC:
    method = QBC_LCB_K2
    KAPPA = 2.0

    def propose(self, committee, contract, seeds, round_index, excluded, *,
                pool_size=1_000_000, chunk_size=16_384, candidate_chunks=None):
        if len(committee.members) != 5:
            raise ValueError('Exactly five committee members required')
        if candidate_chunks is None:
            rng = np.random.default_rng(seeds.seed('candidate_pool', round_index=round_index))
            candidate_chunks = nominal_chunks(contract.dimension, pool_size, chunk_size, rng)

        def scored():
            for start, x in candidate_chunks:
                mean, variance = committee.mean_variance(x)
                yield start, x, confidence_score(mean, variance, contract.adverse, self.KAPPA)

        x, score, indices = streaming_topk(scored(), contract.acquisition_per_round, excluded)
        result = _split_selected(
            self.method, round_index, x, score, contract,
            np.random.default_rng(seeds.seed('role_split', method=self.method, round_index=round_index)),
            dict(candidate_source='shared_iid_nominal_pool', pool_size=pool_size,
                 chunk_size=chunk_size, committee_members=5, kappa=self.KAPPA,
                 variance_ddof=1, adverse=contract.adverse,
                 acquisition='adverse_confidence_bound', correction='direction_v2',
                 selected_source_indices=[int(v) for v in indices]),
        )
        result.validate(contract, excluded)
        return result

QBCLCBK2Strategy = DirectionCorrectQBC
