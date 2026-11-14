"""Joint-vector isotropic adaptive RW-MH for low-dimensional Subset Simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SAMPLER, SamplerConfig
from .problem import Direction, ResponseFunction, adverse_order, in_conditional_domain
from .sampler import Level, SubsetRun

@dataclass(frozen=True)
class JointRWMHConfig:
    """Adaptation policy for the joint-vector isotropic RW kernel."""

    proposal_geometry: str = "isotropic_gaussian_random_walk"
    accepted_object: str = "complete_vector"
    adapted_quantity: str = "single_global_scale"
    initial_scale_rule: str = "initial_scale / sqrt(dimension)"

    initial_scale: float = 1.0
    scale_adaptation_gain: float = 1.0
    scaling_lower: float = 0.05
    scaling_upper: float = 20.0

    def __post_init__(self) -> None:
        if self.proposal_geometry != "isotropic_gaussian_random_walk":
            raise ValueError("joint RW-AMH proposal geometry must remain isotropic")
        if self.accepted_object != "complete_vector":
            raise ValueError("joint RW-AMH must accept or reject the complete vector")
        if self.adapted_quantity != "single_global_scale":
            raise ValueError("joint RW-AMH may adapt only one global scale")
        if self.initial_scale_rule != "initial_scale / sqrt(dimension)":
            raise ValueError("joint RW-AMH initial scale rule must remain dimension-aware")
        positive = (
            self.initial_scale,
            self.scale_adaptation_gain,
            self.scaling_lower,
            self.scaling_upper,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("all joint RW-AMH settings must be positive and finite")
        if self.scaling_upper < self.scaling_lower:
            raise ValueError("scaling_upper must not be smaller than scaling_lower")

@dataclass
class JointRWMHLevel(Level):
    """One population plus joint-proposal diagnostics."""

    final_scale: float | None
    final_proposal_covariance: np.ndarray | None
    unique_rows: int
    unique_fraction: float

@dataclass(frozen=True)
class _TransitionDiagnostics:
    acceptance_rate: float
    scaling_trace: tuple[float, ...]
    final_scale: float
    final_proposal_covariance: np.ndarray
    evaluations: int

def _validated_scores(
    score: ResponseFunction,
    values: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    """Evaluate one response per proposal and reject invalid outputs."""

    observed = np.asarray(score(values), dtype=float).reshape(-1)
    if observed.shape != (expected_rows,):
        raise ValueError("score must return one value per input row")
    if not np.all(np.isfinite(observed)):
        raise ValueError("score returned a non-finite value")
    return observed

def _unique_count(values: np.ndarray) -> int:
    """Count exact retained states without importing another sampler kernel."""

    return int(np.unique(np.asarray(values, dtype=np.float64), axis=0).shape[0])

def _standard_normal_log_prior_ratio(
    current: np.ndarray,
    proposal: np.ndarray,
) -> np.ndarray:
    """Compute ``log phi_d(proposal) - log phi_d(current)`` row-wise."""

    current_array = np.asarray(current, dtype=float)
    proposal_array = np.asarray(proposal, dtype=float)
    if current_array.shape != proposal_array.shape or current_array.ndim != 2:
        raise ValueError("current and proposal must have the same two-dimensional shape")
    return -0.5 * (
        np.einsum("ij,ij->i", proposal_array, proposal_array)
        - np.einsum("ij,ij->i", current_array, current_array)
    )

def _joint_mh_acceptance(
    current: np.ndarray,
    proposal: np.ndarray,
    in_domain: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Apply the event gate and standard-normal MH ratio to whole vectors."""

    domain = np.asarray(in_domain, dtype=bool).reshape(-1)
    draws = np.asarray(uniforms, dtype=float).reshape(-1)
    log_prior_ratio = _standard_normal_log_prior_ratio(current, proposal)
    if domain.shape != log_prior_ratio.shape or draws.shape != log_prior_ratio.shape:
        raise ValueError("domain mask and uniform draws must have one value per row")
    if np.any((draws < 0.0) | (draws > 1.0) | ~np.isfinite(draws)):
        raise ValueError("uniform draws must be finite values in [0, 1]")
    with np.errstate(divide="ignore"):
        log_uniform = np.log(draws)
    return domain & (log_uniform < np.minimum(0.0, log_prior_ratio))

def _conditional_population_joint_rw(
    seeds: np.ndarray,
    seed_values: np.ndarray,
    threshold: float,
    score: ResponseFunction,
    adverse: Direction,
    rng: np.random.Generator,
    cfg: SamplerConfig,
    rw_cfg: JointRWMHConfig,
    scaling0: float | None,
) -> tuple[np.ndarray, np.ndarray, _TransitionDiagnostics, float]:
    seeds = np.asarray(seeds, dtype=float)
    seed_values = np.asarray(seed_values, dtype=float).reshape(-1)
    if seeds.ndim != 2 or seeds.shape[0] < 1:
        raise ValueError("seeds must be a non-empty two-dimensional array")
    n_chains, dimension = seeds.shape
    if seed_values.shape != (n_chains,):
        raise ValueError("seed_values has the wrong shape")

    order = rng.permutation(n_chains) if cfg.shuffle_seeds else np.arange(n_chains)
    states = seeds[order].copy()
    values = seed_values[order].copy()
    scaling = (
        rw_cfg.initial_scale / np.sqrt(float(dimension))
        if scaling0 is None
        else float(scaling0)
    )
    if not np.isfinite(scaling) or scaling <= 0.0:
        raise ValueError("the inherited joint scaling must be positive and finite")

    out_x = np.empty((n_chains, cfg.chain_length, dimension), dtype=float)
    out_v = np.empty((n_chains, cfg.chain_length), dtype=float)
    out_x[:, 0] = states
    out_v[:, 0] = values
    accepted = 0
    scaling_trace: list[float] = []

    for sweep in range(cfg.n_steps):
        proposal = states + scaling * rng.standard_normal(states.shape)
        proposal_values = _validated_scores(score, proposal, n_chains)
        move = _joint_mh_acceptance(
            states,
            proposal,
            in_conditional_domain(proposal_values, threshold, adverse),
            rng.random(n_chains),
        )
        states[move] = proposal[move]
        values[move] = proposal_values[move]
        accepted += int(np.sum(move))
        out_x[:, sweep + 1] = states
        out_v[:, sweep + 1] = values

        rate = float(np.mean(move))
        zeta = float((sweep + 2) ** (-cfg.decay_exponent))
        scaling *= np.exp(
            rw_cfg.scale_adaptation_gain
            * zeta
            * (rate - cfg.target_acceptance)
        )
        scaling = float(np.clip(
            scaling, rw_cfg.scaling_lower, rw_cfg.scaling_upper
        ))

        scaling_trace.append(scaling)

    attempts = n_chains * cfg.n_steps
    final_proposal_covariance = (scaling * scaling) * np.eye(dimension)
    diagnostics = _TransitionDiagnostics(
        acceptance_rate=float(accepted / max(attempts, 1)),
        scaling_trace=tuple(scaling_trace),
        final_scale=scaling,
        final_proposal_covariance=final_proposal_covariance.astype(float),
        evaluations=int(attempts),
    )
    return (
        out_x.reshape(-1, dimension),
        out_v.reshape(-1),
        diagnostics,
        scaling,
    )

def run_joint_rw_amh_subset_simulation(
    score: ResponseFunction,
    *,
    dimension: int,
    adverse: Direction,
    rng: np.random.Generator,
    cfg: SamplerConfig = SAMPLER,
    rw_cfg: JointRWMHConfig = JointRWMHConfig(),
    initial_population: np.ndarray | None = None,
    initial_values: np.ndarray | None = None,
) -> SubsetRun:
    """Run P0 plus conditional populations using joint-vector RW-AMH."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if initial_population is None:
        population = rng.standard_normal((cfg.population, dimension))
        values = _validated_scores(score, population, cfg.population)
        initial_evaluations = cfg.population
    else:
        population = np.asarray(initial_population, dtype=float)
        if population.shape != (cfg.population, dimension):
            raise ValueError("initial population has the wrong shape")
        values = (
            _validated_scores(score, population, cfg.population)
            if initial_values is None
            else np.asarray(initial_values, dtype=float).reshape(-1)
        )
        if values.shape != (cfg.population,) or not np.all(np.isfinite(values)):
            raise ValueError("initial_values has the wrong shape or is non-finite")
        initial_evaluations = 0 if initial_values is not None else cfg.population

    run = SubsetRun(total_evaluations=initial_evaluations)
    transition: _TransitionDiagnostics | None = None
    new_evaluations = initial_evaluations
    scaling: float | None = None

    for index in range(cfg.n_populations):
        order = adverse_order(values, adverse)
        seed_indices = order[: cfg.n_seed]
        threshold = float(values[order[cfg.n_seed - 1]])
        unique = _unique_count(population)
        run.levels.append(
            JointRWMHLevel(
                index=index,
                nominal_probability=cfg.nominal_levels[index],
                samples=population.copy(),
                values=values.copy(),
                threshold=threshold,
                seed_indices=seed_indices.copy(),
                acceptance_rate=(
                    None if transition is None else transition.acceptance_rate
                ),
                scaling_trace=(
                    () if transition is None else transition.scaling_trace
                ),
                new_evaluations=new_evaluations,
                final_scale=(
                    None if transition is None else transition.final_scale
                ),
                final_proposal_covariance=(
                    None
                    if transition is None
                    else transition.final_proposal_covariance.copy()
                ),
                unique_rows=unique,
                unique_fraction=float(unique / cfg.population),
            )
        )
        if index == cfg.n_populations - 1:
            break

        population, values, transition, scaling = _conditional_population_joint_rw(
            population[seed_indices],
            values[seed_indices],
            threshold,
            score,
            adverse,
            rng,
            cfg,
            rw_cfg,
            scaling,
        )
        new_evaluations = transition.evaluations
        run.total_evaluations += new_evaluations

    return run

__all__ = [
    "JointRWMHConfig",
    "JointRWMHLevel",
    "run_joint_rw_amh_subset_simulation",
]
