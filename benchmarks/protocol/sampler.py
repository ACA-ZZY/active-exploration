"""Subset simulation with adaptive conditional sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import SAMPLER, SamplerConfig
from .problem import Direction, ResponseFunction, adverse_order, in_conditional_domain

@dataclass
class Level:
    """One population and the threshold it defines."""

    index: int
    nominal_probability: float
    samples: np.ndarray
    values: np.ndarray
    threshold: float
    seed_indices: np.ndarray
    acceptance_rate: float | None
    scaling_trace: tuple[float, ...]
    new_evaluations: int

    @property
    def seeds(self) -> np.ndarray:
        return self.samples[self.seed_indices]

    @property
    def seed_values(self) -> np.ndarray:
        return self.values[self.seed_indices]

@dataclass
class SubsetRun:
    levels: list[Level] = field(default_factory=list)
    total_evaluations: int = 0

    @property
    def thresholds(self) -> np.ndarray:
        return np.array([lv.threshold for lv in self.levels], dtype=float)

    @property
    def nominal_probabilities(self) -> np.ndarray:
        return np.array([lv.nominal_probability for lv in self.levels], dtype=float)

    def dynamic_range(self) -> float:
        t = self.thresholds
        return float(t[-1] - t[0])

def _initial_proposal_scale(seeds: np.ndarray, cfg: SamplerConfig) -> np.ndarray:
    """Per-dimension proposal standard deviation from the seed spread."""
    spread = np.std(seeds, axis=0)
    usable = np.isfinite(spread) & (spread >= cfg.sigma0_floor)
    return np.where(usable, spread, 1.0)

def _conditional_population(
    seeds: np.ndarray,
    seed_values: np.ndarray,
    threshold: float,
    score: ResponseFunction,
    adverse: Direction,
    rng: np.random.Generator,
    cfg: SamplerConfig,
    scaling0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, ...], int]:
    n_chains, dimension = seeds.shape
    n_a = max(1, int(round(cfg.adaptation_fraction * n_chains)))

    order = rng.permutation(n_chains) if cfg.shuffle_seeds else np.arange(n_chains)
    states = seeds[order].copy()
    values = seed_values[order].copy()

    sigma0 = _initial_proposal_scale(seeds, cfg)
    scaling = float(scaling0)
    scaling_trace: list[float] = []

    out_x = np.empty((n_chains, cfg.chain_length, dimension), dtype=float)
    out_v = np.empty((n_chains, cfg.chain_length), dtype=float)
    accepted = 0
    attempted = 0
    evaluations = 0

    for batch_index, lo in enumerate(range(0, n_chains, n_a), start=1):
        hi = min(lo + n_a, n_chains)
        chain_x = states[lo:hi]
        chain_v = values[lo:hi]
        out_x[lo:hi, 0] = chain_x          # the seed is state zero, already evaluated
        out_v[lo:hi, 0] = chain_v

        batch_accepted = 0
        batch_attempted = 0
        for step in range(cfg.n_steps):
            sigma = np.minimum(scaling * sigma0, 1.0)
            rho = np.sqrt(1.0 - sigma**2)
            proposal = rho * chain_x + sigma * rng.standard_normal(chain_x.shape)
            proposal_values = np.asarray(score(proposal), dtype=float).reshape(-1)
            evaluations += proposal.shape[0]

            move = in_conditional_domain(proposal_values, threshold, adverse)
            chain_x[move] = proposal[move]
            chain_v[move] = proposal_values[move]
            out_x[lo:hi, step + 1] = chain_x
            out_v[lo:hi, step + 1] = chain_v

            batch_accepted += int(np.sum(move))
            batch_attempted += move.size

        accepted += batch_accepted
        attempted += batch_attempted
        rate = batch_accepted / max(batch_attempted, 1)
        zeta = batch_index ** (-cfg.decay_exponent)
        scaling = max(
            scaling * math.exp(zeta * (rate - cfg.target_acceptance)),
            cfg.scaling_lower,
        )
        scaling_trace.append(float(scaling))

    population = out_x.reshape(-1, dimension)
    population_values = out_v.reshape(-1)
    return (
        population,
        population_values,
        accepted / max(attempted, 1),
        tuple(scaling_trace),
        evaluations,
    )

def run_subset_simulation(
    score: ResponseFunction,
    *,
    dimension: int,
    adverse: Direction,
    rng: np.random.Generator,
    cfg: SamplerConfig = SAMPLER,
    kernel: str = "pcn",
    initial_population: np.ndarray | None = None,
    initial_values: np.ndarray | None = None,
) -> SubsetRun:
    """Run P0 plus four conditional levels and return every population"""

    kernel_name = str(kernel).lower()
    if kernel_name == "joint_rw_amh":
        from .joint_rw_amh import run_joint_rw_amh_subset_simulation

        return run_joint_rw_amh_subset_simulation(
            score, dimension=dimension, adverse=adverse, rng=rng, cfg=cfg,
            initial_population=initial_population, initial_values=initial_values,
        )
    if kernel_name == "componentwise_rw_amh":
        from .componentwise_rw_amh import run_componentwise_rw_amh_subset_simulation

        return run_componentwise_rw_amh_subset_simulation(
            score, dimension=dimension, adverse=adverse, rng=rng, cfg=cfg,
            initial_population=initial_population, initial_values=initial_values,
        )
    if kernel_name == "componentwise_amh":
        from .componentwise_amh import run_componentwise_amh_subset_simulation

        return run_componentwise_amh_subset_simulation(
            score, dimension=dimension, adverse=adverse, rng=rng, cfg=cfg,
            initial_population=initial_population, initial_values=initial_values,
        )
    if kernel_name != "pcn":
        raise ValueError(f"unknown subset-simulation kernel: {kernel!r}")

    if initial_population is None:
        population = rng.standard_normal((cfg.population, dimension))
        values = np.asarray(score(population), dtype=float).reshape(-1)
        evaluations = cfg.population
    else:
        population = np.asarray(initial_population, dtype=float)
        if population.shape != (cfg.population, dimension):
            raise ValueError("initial population has the wrong shape")
        values = (
            np.asarray(score(population), dtype=float).reshape(-1)
            if initial_values is None
            else np.asarray(initial_values, dtype=float).reshape(-1)
        )
        evaluations = 0 if initial_values is not None else cfg.population

    run = SubsetRun(total_evaluations=evaluations)
    acceptance: float | None = None
    trace: tuple[float, ...] = ()
    new_evaluations = evaluations
    scaling = 1.0

    for index in range(cfg.n_populations):
        order = adverse_order(values, adverse)
        seed_indices = order[: cfg.n_seed]
        threshold = float(values[order[cfg.n_seed - 1]])
        run.levels.append(
            Level(
                index=index,
                nominal_probability=cfg.nominal_levels[index],
                samples=population.copy(),
                values=values.copy(),
                threshold=threshold,
                seed_indices=seed_indices.copy(),
                acceptance_rate=acceptance,
                scaling_trace=trace,
                new_evaluations=new_evaluations,
            )
        )
        if index == cfg.n_populations - 1:
            break

        population, values, acceptance, trace, new_evaluations = _conditional_population(
            population[seed_indices],
            values[seed_indices],
            threshold,
            score,
            adverse,
            rng,
            cfg,
            scaling,
        )
        scaling = trace[-1] if trace else scaling
        run.total_evaluations += new_evaluations

    return run
