"""Frozen case contracts for the new paired comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from protocol.problem import Problem

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class CaseContract:
    slug: str
    problem_name: str
    dimension: int
    adverse: Literal["lower", "upper"]
    train_size: int
    base_recipe: str
    sampler_kernel: str
    reference_manifest: str
    initial_holdout: int = 250
    replay_size: int = 500
    rounds: int = 4
    train_per_round: int = 500
    audit_per_round: int = 250
    n_levels: int = 5
    subset_population: int = 1000
    conditional_probability: float = 0.1
    base_learning_rate: float | None = None
    refine_learning_rate: float | None = None
    refine_epochs: int | None = None
    requires_feature_map: bool = False
    requires_sequence_map: bool = False
    feature_tag: str = ""

    @property
    def initial_size(self) -> int:
        return self.train_size + self.initial_holdout

    @property
    def acquisition_per_round(self) -> int:
        return self.train_per_round + self.audit_per_round

    @property
    def train_per_level(self) -> int:
        if self.train_per_round % self.n_levels:
            raise ValueError("train_per_round must divide evenly across levels")
        return self.train_per_round // self.n_levels

    @property
    def audit_per_level(self) -> int:
        if self.audit_per_round % self.n_levels:
            raise ValueError("audit_per_round must divide evenly across levels")
        return self.audit_per_round // self.n_levels

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True)
class ResolvedCase:
    contract: CaseContract
    problem: Problem
    reference_thresholds: np.ndarray
    reference_manifest_path: Path

_SPECS: dict[str, CaseContract] = {
    "two-sided-kink": CaseContract(
        slug="two-sided-kink", problem_name="two_sided_kink", dimension=2,
        adverse="lower", train_size=1000, base_recipe="network",
        sampler_kernel="pcn",
        reference_manifest="cases/two-sided-kink/sweep/1000/manifest.json",
        base_learning_rate=1e-3, refine_learning_rate=1e-3, refine_epochs=300,
    ),
    "radial-kink": CaseContract(
        slug="radial-kink", problem_name="radial_kink", dimension=8,
        adverse="lower", train_size=1000, base_recipe="deep",
        sampler_kernel="pcn",
        reference_manifest="cases/radial-kink/sweep/1000/manifest.json",
        base_learning_rate=1e-3, refine_learning_rate=1e-3, refine_epochs=300,
    ),
    "tail-kink": CaseContract(
        slug="tail-kink", problem_name="tail_kink", dimension=50,
        adverse="lower", train_size=1000, base_recipe="deep",
        sampler_kernel="pcn",
        reference_manifest="cases/tail-kink/sweep/1000/manifest.json",
        base_learning_rate=1e-3, refine_learning_rate=1e-3, refine_epochs=300,
    ),
    "hysteretic-lab": CaseContract(
        slug="hysteretic-lab", problem_name="hysteretic_lab", dimension=48,
        adverse="upper", train_size=5000, base_recipe="recurrent",
        sampler_kernel="componentwise_amh",
        reference_manifest="cases/hysteretic-lab/sweep/5000/manifest.json",
        base_learning_rate=3e-3, refine_learning_rate=5e-4, refine_epochs=200,
        requires_sequence_map=True,
    ),
    "cantilever_beam_2d": CaseContract(
        slug="cantilever_beam_2d", problem_name="cantilever_beam_2d",
        dimension=2, adverse="lower", train_size=1000,
        base_recipe="network_16x64x16_refine_lr1e3_e200",
        sampler_kernel="joint_rw_amh",
        reference_manifest="cases/cantilever_beam_2d/sweep/1000/manifest.json",
        base_learning_rate=1e-3, refine_learning_rate=1e-3, refine_epochs=200,
        requires_feature_map=True,
    ),
}

def _load_problem(slug: str) -> Problem:
    from run_case import load_case

    return load_case(slug)[2]

def _validate_problem(contract: CaseContract, problem: Problem) -> None:
    observed = (problem.name, problem.dimension, problem.adverse)
    expected = (contract.problem_name, contract.dimension, contract.adverse)
    if observed != expected:
        raise RuntimeError(
            f"case contract drift for {contract.slug}: expected {expected}, got {observed}"
        )
    if problem.recipe != contract.base_recipe:
        raise RuntimeError(
            f"base recipe drift for {contract.slug}: expected "
            f"{contract.base_recipe!r}, got {problem.recipe!r}"
        )
    if contract.requires_feature_map != (problem.feature_map is not None):
        raise RuntimeError(f"feature-map contract drift for {contract.slug}")
    if contract.requires_sequence_map != (problem.sequence_map is not None):
        raise RuntimeError(f"sequence-map contract drift for {contract.slug}")
    if contract.slug != "hysteretic-lab":
        observed_kernel = getattr(problem, "subset_sampler", "pcn")
        if observed_kernel != contract.sampler_kernel:
            raise RuntimeError(
                f"sampler contract drift for {contract.slug}: expected "
                f"{contract.sampler_kernel!r}, got {observed_kernel!r}"
            )
    if contract.slug == "cantilever_beam_2d":
        if getattr(problem, "feature_tag", ""):
            raise RuntimeError("cantilever benchmark must use raw physical coordinates")

def load_case_contract(slug: str) -> ResolvedCase:
    """Resolve and verify one immutable five-case benchmark contract."""

    if slug not in _SPECS:
        raise KeyError(f"unknown benchmark case {slug!r}; choose from {sorted(_SPECS)}")
    contract = _SPECS[slug]
    problem = _load_problem(slug)
    _validate_problem(contract, problem)

    manifest_path = (ROOT / contract.reference_manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing frozen reference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    thresholds = np.asarray(manifest.get("reference_thresholds"), dtype=np.float64)
    if thresholds.shape != (contract.n_levels,) or not np.all(np.isfinite(thresholds)):
        raise RuntimeError(f"invalid frozen reference thresholds for {slug}")
    if float(np.ptp(thresholds)) <= 0:
        raise RuntimeError(f"zero reference span for {slug}")
    thresholds.setflags(write=False)
    return ResolvedCase(contract, problem, thresholds, manifest_path)

def available_cases() -> tuple[str, ...]:
    return tuple(_SPECS)

