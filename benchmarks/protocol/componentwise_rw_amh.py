"""Coordinate-wise adaptive random-walk MH for proxy Subset Simulation"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .componentwise_amh import (
    _initial_coordinate_scale,
    _unique_count,
    _validated_scores,
)
from .config import SAMPLER, SamplerConfig
from .problem import Direction, ResponseFunction, adverse_order, in_conditional_domain

@dataclass(frozen=True)
class ComponentwiseRWMHConfig:
    """Step policy specific to the coordinate-wise random-walk kernel."""

    initial_scale: float = 2.4
    adaptation_gain: float = 1.0
    scaling_lower: float = 0.1
    scaling_upper: float = 20.0

    def __post_init__(self) -> None:
        values = (
            self.initial_scale,
            self.adaptation_gain,
            self.scaling_lower,
            self.scaling_upper,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("all random-walk step settings must be positive and finite")
        if self.scaling_upper < self.scaling_lower:
            raise ValueError("scaling_upper must not be smaller than scaling_lower")

@dataclass
class ComponentwiseRWMHLevel:
    """One population and its coordinate-wise random-walk diagnostics."""

    index: int
    nominal_probability: float
    samples: np.ndarray
    values: np.ndarray
    threshold: float
    seed_indices: np.ndarray
    coordinate_acceptance_rate: np.ndarray | None
    mean_coordinate_acceptance: float | None
    sweep_move_rate: float | None
    mean_changed_dimensions_per_sweep: float | None
    unique_rows: int
    unique_fraction: float
    mean_step_trace: tuple[float, ...]
    final_proposal_std: np.ndarray | None
    new_evaluations: int

    @property
    def seeds(self) -> np.ndarray:
        return self.samples[self.seed_indices]

    @property
    def seed_values(self) -> np.ndarray:
        return self.values[self.seed_indices]

@dataclass
class ComponentwiseRWMHRun:
    """Proxy Subset-Simulation result from the aggressive coordinate kernel."""

    levels: list[ComponentwiseRWMHLevel] = field(default_factory=list)
    total_evaluations: int = 0

    @property
    def thresholds(self) -> np.ndarray:
        return np.asarray([level.threshold for level in self.levels], dtype=float)

@dataclass(frozen=True)
class _TransitionDiagnostics:
    coordinate_acceptance_rate: np.ndarray
    mean_coordinate_acceptance: float
    sweep_move_rate: float
    mean_changed_dimensions_per_sweep: float
    mean_step_trace: tuple[float, ...]
    final_proposal_std: np.ndarray
    evaluations: int

def _conditional_population_componentwise_rw(
    seeds: np.ndarray,
    seed_values: np.ndarray,
    threshold: float,
    score: ResponseFunction,
    adverse: Direction,
    rng: np.random.Generator,
    cfg: SamplerConfig,
    rw_cfg: ComponentwiseRWMHConfig,
    scaling0: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, _TransitionDiagnostics, np.ndarray]:
    seeds = np.asarray(seeds, dtype=float)
    seed_values = np.asarray(seed_values, dtype=float).reshape(-1)
    n_chains, dimension = seeds.shape
    if seed_values.shape != (n_chains,):
        raise ValueError("seed_values has the wrong shape")

    order = rng.permutation(n_chains) if cfg.shuffle_seeds else np.arange(n_chains)
    states = seeds[order].copy()
    values = seed_values[order].copy()
    sigma0 = _initial_coordinate_scale(seeds, cfg)
    scaling = (
        np.full(dimension, rw_cfg.initial_scale, dtype=float)
        if scaling0 is None
        else np.asarray(scaling0, dtype=float).copy()
    )
    if scaling.shape != (dimension,) or not np.all(np.isfinite(scaling)):
        raise ValueError("the inherited coordinate scaling has the wrong shape")

    out_x = np.empty((n_chains, cfg.chain_length, dimension), dtype=float)
    out_v = np.empty((n_chains, cfg.chain_length), dtype=float)
    out_x[:, 0] = states
    out_v[:, 0] = values
    accepted_by_dimension = np.zeros(dimension, dtype=np.int64)
    attempted_by_dimension = np.zeros(dimension, dtype=np.int64)
    moved_sweeps = 0
    changed_dimensions_total = 0
    step_trace: list[float] = []

    for sweep in range(cfg.n_steps):
        accepted_this_sweep = np.zeros((n_chains, dimension), dtype=bool)
        for coordinate in rng.permutation(dimension):
            current = states[:, coordinate].copy()
            proposal_coordinate = (
                current
                + scaling[coordinate]
                * sigma0[coordinate]
                * rng.standard_normal(n_chains)
            )
            proposal = states.copy()
            proposal[:, coordinate] = proposal_coordinate
            proposal_values = _validated_scores(score, proposal, n_chains)
            in_domain = in_conditional_domain(proposal_values, threshold, adverse)
            log_prior_ratio = -0.5 * (
                proposal_coordinate * proposal_coordinate - current * current
            )
            log_uniform = np.log(rng.random(n_chains))
            move = in_domain & (log_uniform < np.minimum(0.0, log_prior_ratio))
            states[move, coordinate] = proposal_coordinate[move]
            values[move] = proposal_values[move]
            accepted_this_sweep[:, coordinate] = move
            accepted_by_dimension[coordinate] += int(np.sum(move))
            attempted_by_dimension[coordinate] += n_chains

        changed = np.sum(accepted_this_sweep, axis=1)
        moved_sweeps += int(np.sum(changed > 0))
        changed_dimensions_total += int(np.sum(changed))
        out_x[:, sweep + 1] = states
        out_v[:, sweep + 1] = values

        rate = np.mean(accepted_this_sweep, axis=0)
        zeta = float((sweep + 1) ** (-cfg.decay_exponent))
        scaling *= np.exp(
            rw_cfg.adaptation_gain * zeta * (rate - cfg.target_acceptance)
        )
        scaling = np.clip(scaling, rw_cfg.scaling_lower, rw_cfg.scaling_upper)
        step_trace.append(float(np.mean(scaling * sigma0)))

    population = out_x.reshape(-1, dimension)
    population_values = out_v.reshape(-1)
    coordinate_rate = accepted_by_dimension / np.maximum(attempted_by_dimension, 1)
    attempted_sweeps = n_chains * cfg.n_steps
    diagnostics = _TransitionDiagnostics(
        coordinate_acceptance_rate=coordinate_rate.astype(float),
        mean_coordinate_acceptance=float(np.mean(coordinate_rate)),
        sweep_move_rate=float(moved_sweeps / max(attempted_sweeps, 1)),
        mean_changed_dimensions_per_sweep=float(
            changed_dimensions_total / max(attempted_sweeps, 1)
        ),
        mean_step_trace=tuple(step_trace),
        final_proposal_std=(scaling * sigma0).astype(float),
        evaluations=int(n_chains * dimension * cfg.n_steps),
    )
    return population, population_values, diagnostics, scaling

def run_componentwise_rw_amh_subset_simulation(
    score: ResponseFunction,
    *,
    dimension: int,
    adverse: Direction,
    rng: np.random.Generator,
    cfg: SamplerConfig = SAMPLER,
    rw_cfg: ComponentwiseRWMHConfig = ComponentwiseRWMHConfig(),
    initial_population: np.ndarray | None = None,
    initial_values: np.ndarray | None = None,
) -> ComponentwiseRWMHRun:
    """Run P0 plus conditional populations using coordinate-wise RW-AMH."""

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

    run = ComponentwiseRWMHRun(total_evaluations=initial_evaluations)
    transition: _TransitionDiagnostics | None = None
    new_evaluations = initial_evaluations
    scaling: np.ndarray | None = None

    for index in range(cfg.n_populations):
        order = adverse_order(values, adverse)
        seed_indices = order[: cfg.n_seed]
        threshold = float(values[order[cfg.n_seed - 1]])
        unique = _unique_count(population)
        run.levels.append(
            ComponentwiseRWMHLevel(
                index=index,
                nominal_probability=cfg.nominal_levels[index],
                samples=population.copy(),
                values=values.copy(),
                threshold=threshold,
                seed_indices=seed_indices.copy(),
                coordinate_acceptance_rate=(
                    None
                    if transition is None
                    else transition.coordinate_acceptance_rate.copy()
                ),
                mean_coordinate_acceptance=(
                    None if transition is None else transition.mean_coordinate_acceptance
                ),
                sweep_move_rate=(
                    None if transition is None else transition.sweep_move_rate
                ),
                mean_changed_dimensions_per_sweep=(
                    None
                    if transition is None
                    else transition.mean_changed_dimensions_per_sweep
                ),
                unique_rows=unique,
                unique_fraction=float(unique / cfg.population),
                mean_step_trace=(
                    () if transition is None else transition.mean_step_trace
                ),
                final_proposal_std=(
                    None
                    if transition is None
                    else transition.final_proposal_std.copy()
                ),
                new_evaluations=new_evaluations,
            )
        )
        if index == cfg.n_populations - 1:
            break
        population, values, transition, scaling = (
            _conditional_population_componentwise_rw(
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
        )
        new_evaluations = transition.evaluations
        run.total_evaluations += new_evaluations

    return run

__all__ = [
    "ComponentwiseRWMHConfig",
    "ComponentwiseRWMHLevel",
    "ComponentwiseRWMHRun",
    "run_componentwise_rw_amh_subset_simulation",
]
