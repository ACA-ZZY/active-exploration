"""Coordinate-wise adaptive Metropolis-Hastings for proxy Subset Simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import SAMPLER, SamplerConfig
from .problem import Direction, ResponseFunction, adverse_order, in_conditional_domain

@dataclass
class ComponentwiseAMHLevel:
    """One registered population and its coordinate-wise AMH diagnostics."""

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
    mean_beta_trace: tuple[float, ...]
    final_beta: np.ndarray | None
    new_evaluations: int

    @property
    def seeds(self) -> np.ndarray:
        return self.samples[self.seed_indices]

    @property
    def seed_values(self) -> np.ndarray:
        return self.values[self.seed_indices]

@dataclass
class ComponentwiseAMHRun:
    """Five-population proxy Subset-Simulation result."""

    levels: list[ComponentwiseAMHLevel] = field(default_factory=list)
    total_evaluations: int = 0

    @property
    def thresholds(self) -> np.ndarray:
        return np.asarray([level.threshold for level in self.levels], dtype=float)

    @property
    def nominal_probabilities(self) -> np.ndarray:
        return np.asarray(
            [level.nominal_probability for level in self.levels], dtype=float
        )

@dataclass(frozen=True)
class _TransitionDiagnostics:
    coordinate_acceptance_rate: np.ndarray
    mean_coordinate_acceptance: float
    sweep_move_rate: float
    mean_changed_dimensions_per_sweep: float
    mean_beta_trace: tuple[float, ...]
    final_beta: np.ndarray
    evaluations: int

def _validated_scores(
    score: ResponseFunction, values: np.ndarray, expected_rows: int
) -> np.ndarray:
    observed = np.asarray(score(values), dtype=float).reshape(-1)
    if observed.shape != (expected_rows,):
        raise ValueError("score must return one value per input row")
    if not np.all(np.isfinite(observed)):
        raise ValueError("score returned a non-finite value")
    return observed

def _initial_coordinate_scale(seeds: np.ndarray, cfg: SamplerConfig) -> np.ndarray:
    spread = np.std(seeds, axis=0)
    usable = np.isfinite(spread) & (spread >= cfg.sigma0_floor)
    return np.where(usable, spread, 1.0)

def _unique_count(values: np.ndarray) -> int:
    return int(np.unique(np.asarray(values, dtype=np.float64), axis=0).shape[0])

def _conditional_population_componentwise(
    seeds: np.ndarray,
    seed_values: np.ndarray,
    threshold: float,
    score: ResponseFunction,
    adverse: Direction,
    rng: np.random.Generator,
    cfg: SamplerConfig,
    scaling0: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, _TransitionDiagnostics, np.ndarray]:
    """Generate one conditional population with random-scan coordinate pCN"""

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
        np.ones(dimension, dtype=float)
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
    beta_trace: list[float] = []

    for sweep in range(cfg.n_steps):
        accepted_this_sweep = np.zeros((n_chains, dimension), dtype=bool)
        for coordinate in rng.permutation(dimension):
            beta = float(min(scaling[coordinate] * sigma0[coordinate], 1.0))
            rho = math.sqrt(max(0.0, 1.0 - beta * beta))
            proposal = states.copy()
            proposal[:, coordinate] = (
                rho * states[:, coordinate]
                + beta * rng.standard_normal(n_chains)
            )
            proposal_values = _validated_scores(score, proposal, n_chains)
            move = in_conditional_domain(proposal_values, threshold, adverse)
            states[move, coordinate] = proposal[move, coordinate]
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
        scaling = np.maximum(
            scaling * np.exp(zeta * (rate - cfg.target_acceptance)),
            cfg.scaling_lower,
        )
        beta_trace.append(float(np.mean(np.minimum(scaling * sigma0, 1.0))))

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
        mean_beta_trace=tuple(beta_trace),
        final_beta=np.minimum(scaling * sigma0, 1.0).astype(float),
        evaluations=int(n_chains * dimension * cfg.n_steps),
    )
    return population, population_values, diagnostics, scaling

def run_componentwise_amh_subset_simulation(
    score: ResponseFunction,
    *,
    dimension: int,
    adverse: Direction,
    rng: np.random.Generator,
    cfg: SamplerConfig = SAMPLER,
    initial_population: np.ndarray | None = None,
    initial_values: np.ndarray | None = None,
) -> ComponentwiseAMHRun:
    """Run P0 plus the configured conditional populations using AMH"""

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

    run = ComponentwiseAMHRun(total_evaluations=initial_evaluations)
    transition: _TransitionDiagnostics | None = None
    new_evaluations = initial_evaluations
    scaling: np.ndarray | None = None

    for index in range(cfg.n_populations):
        order = adverse_order(values, adverse)
        seed_indices = order[: cfg.n_seed]
        threshold = float(values[order[cfg.n_seed - 1]])
        unique = _unique_count(population)
        run.levels.append(
            ComponentwiseAMHLevel(
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
                mean_beta_trace=(
                    () if transition is None else transition.mean_beta_trace
                ),
                final_beta=(
                    None if transition is None else transition.final_beta.copy()
                ),
                new_evaluations=new_evaluations,
            )
        )
        if index == cfg.n_populations - 1:
            break

        population, values, transition, scaling = (
            _conditional_population_componentwise(
                population[seed_indices],
                values[seed_indices],
                threshold,
                score,
                adverse,
                rng,
                cfg,
                scaling,
            )
        )
        new_evaluations = transition.evaluations
        run.total_evaluations += new_evaluations

    return run

__all__ = [
    "ComponentwiseAMHLevel",
    "ComponentwiseAMHRun",
    "run_componentwise_amh_subset_simulation",
]
