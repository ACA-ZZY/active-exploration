"""The shared downstream learner and QBC auxiliary committee."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from protocol.surrogate import Surrogate

from .contracts import CaseContract

def _base_rate(recipe) -> float | None:
    for name in ("learning_rate", "learning_rate_init"):
        if hasattr(recipe, name):
            return float(getattr(recipe, name))
    return None

def fit_base_student(
    contract: CaseContract,
    problem,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    device: str | None = None,
    recipe=None,
) -> Surrogate:
    model = Surrogate(
        seed=seed,
        recipe=contract.base_recipe if recipe is None else recipe,
        adverse=contract.adverse,
        feature_map=problem.feature_map,
        sequence_map=problem.sequence_map,
        device=device,
    )
    expected = contract.base_learning_rate
    observed = _base_rate(model.recipe)
    if expected is not None and (observed is None or not np.isclose(observed, expected)):
        raise RuntimeError(
            f"base learning-rate drift for {contract.slug}: expected {expected}, got {observed}"
        )
    return model.fit(x, y)

def apply_refinement_policy(model: Surrogate, contract: CaseContract) -> Surrogate:
    """Apply the frozen refinement-only settings to a cloned learner."""

    recipe = model.recipe
    updates: dict[str, object] = {}
    if contract.refine_epochs is not None:
        if not hasattr(recipe, "refine_iter"):
            raise RuntimeError("student recipe has no refinement epoch field")
        updates["refine_iter"] = int(contract.refine_epochs)
    if contract.refine_learning_rate is not None:
        if hasattr(recipe, "refine_learning_rate"):
            updates["refine_learning_rate"] = float(contract.refine_learning_rate)
        elif hasattr(recipe, "learning_rate"):
            updates["learning_rate"] = float(contract.refine_learning_rate)
        elif hasattr(recipe, "learning_rate_init"):
            updates["learning_rate_init"] = float(contract.refine_learning_rate)
        else:
            raise RuntimeError("student recipe has no learning-rate field")
    if updates:
        if not dataclasses.is_dataclass(recipe):
            raise TypeError("refinement policy requires a dataclass recipe")
        model.recipe = dataclasses.replace(recipe, **updates)
    return model

def refine_common_student(
    current: Surrogate,
    contract: CaseContract,
    replay_x: np.ndarray,
    replay_y: np.ndarray,
    cumulative_train_x: np.ndarray,
    cumulative_train_y: np.ndarray,
    *,
    new_labels: int,
    seed: int,
) -> Surrogate:
    """Warm-start on replay plus acquired *training* rows only"""

    replay_x = np.atleast_2d(np.asarray(replay_x, dtype=np.float64))
    replay_y = np.asarray(replay_y, dtype=np.float64).reshape(-1)
    tail_x = np.atleast_2d(np.asarray(cumulative_train_x, dtype=np.float64))
    tail_y = np.asarray(cumulative_train_y, dtype=np.float64).reshape(-1)
    if replay_x.shape[0] != replay_y.size or tail_x.shape[0] != tail_y.size:
        raise ValueError("student refinement inputs and labels do not align")
    if replay_x.shape[0] != contract.replay_size:
        raise ValueError(
            f"replay must contain exactly {contract.replay_size} rows, got {replay_x.shape[0]}"
        )
    nxt = current.clone_frozen(seed=seed)
    apply_refinement_policy(nxt, contract)
    nxt.continue_fit(
        np.vstack([replay_x, tail_x]),
        np.concatenate([replay_y, tail_y]),
        new_labels=int(new_labels),
    )
    return nxt

def _modules(model: Surrogate) -> Iterable:
    value = model._model
    return value if isinstance(value, list) else ([] if value is None else [value])

def pack_student(model: Surrogate) -> Surrogate:
    """Portable CPU copy with case-owned callables removed for pickling."""

    packed = copy.deepcopy(model)
    packed._feature_map = None
    packed._sequence_map = None
    packed._device_request = "cpu"
    packed._device = None
    for module in _modules(packed):
        if hasattr(module, "to"):
            module.to("cpu")
    return packed

def restore_student(
    packed: Surrogate,
    problem,
    *,
    device: str | None = None,
) -> Surrogate:
    model = packed
    model._feature_map = problem.feature_map
    model._sequence_map = problem.sequence_map
    model._device_request = device
    model._device = None
    if model.recipe.name in ("network", "deep", "spectral", "recurrent", "temporal_attention"):
        target = model._torch_device()
        for module in _modules(model):
            module.to(target)
    return model

@dataclass
class NeuralCommittee:
    members: list[Surrogate]

    @classmethod
    def fit(
        cls,
        contract: CaseContract,
        problem,
        x: np.ndarray,
        y: np.ndarray,
        *,
        seeds: Iterable[int],
        device: str | None = None,
    ) -> "NeuralCommittee":
        members = [
            fit_base_student(contract, problem, x, y, seed=int(seed), device=device)
            for seed in seeds
        ]
        if len(members) < 2:
            raise ValueError("QBC needs at least two independently fitted members")
        return cls(members)

    def predict_members(self, x: np.ndarray) -> np.ndarray:
        output = np.vstack([member.predict(x) for member in self.members])
        if output.ndim != 2 or output.shape[0] != len(self.members):
            raise RuntimeError("invalid committee prediction matrix")
        return output

    def mean_variance(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictions = self.predict_members(x)
        return predictions.mean(axis=0), predictions.var(axis=0, ddof=1)

    def refine(
        self,
        contract: CaseContract,
        replay_x: np.ndarray,
        replay_y: np.ndarray,
        cumulative_train_x: np.ndarray,
        cumulative_train_y: np.ndarray,
        *,
        new_labels: int,
        seeds: Iterable[int],
    ) -> "NeuralCommittee":
        seeds = list(seeds)
        if len(seeds) != len(self.members):
            raise ValueError("one refinement seed is required per committee member")
        return NeuralCommittee([
            refine_common_student(
                member, contract, replay_x, replay_y,
                cumulative_train_x, cumulative_train_y,
                new_labels=new_labels, seed=int(seed),
            )
            for member, seed in zip(self.members, seeds)
        ])

    def packed(self) -> "NeuralCommittee":
        return NeuralCommittee([pack_student(member) for member in self.members])

    def restored(self, problem, *, device: str | None = None) -> "NeuralCommittee":
        return NeuralCommittee([
            restore_student(member, problem, device=device) for member in self.members
        ])

