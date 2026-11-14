"""Frozen protocol constants."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class SamplerConfig:
    """Adaptive conditional sampling."""

    population: int = 1000
    conditional_probability: float = 0.1
    n_populations: int = 5          # P0 plus four conditional levels
    target_acceptance: float = 0.44
    adaptation_fraction: float = 0.1
    decay_exponent: float = 0.5
    sigma0_floor: float = 1e-6
    scaling_lower: float = 0.1
    shuffle_seeds: bool = True

    @property
    def n_seed(self) -> int:
        return int(round(self.population * self.conditional_probability))

    @property
    def chain_length(self) -> int:
        """States per chain, seed included."""
        return self.population // self.n_seed

    @property
    def n_steps(self) -> int:
        """Metropolis-Hastings transitions per chain; the seed is state zero."""
        return self.chain_length - 1

    @property
    def adaptation_batches(self) -> int:
        return int(round(1.0 / self.adaptation_fraction))

    @property
    def nominal_levels(self) -> tuple[float, ...]:
        p0 = self.conditional_probability
        return tuple(p0 ** (level + 1) for level in range(self.n_populations))

@dataclass(frozen=True)
class TailHoldoutConfig:
    """The data organisation"""

    name: str = "tail_holdout"
    train_size: int = 5000              # A trains on all of these
    unconditional_holdout: int = 250    # reserved, never trained on
    audit_per_level: int = 150          # 100 seeds plus 50 extra
    seeds_per_level: int = 100          # go to the refinement training set
    fit_size: int = 350
    validation_size: int = 150
    accumulate_harvests: bool = True
    weight_group: str = "call"
    navigator_ema: float = 0.0

    @property
    def initial_size(self) -> int:
        return self.train_size + self.unconditional_holdout

    @property
    def extra_per_level(self) -> int:
        return self.audit_per_level - self.seeds_per_level

    @property
    def tail_train(self) -> int:
        return self.seeds_per_level * SamplerConfig().n_populations

    @property
    def tail_holdout(self) -> int:
        return self.extra_per_level * SamplerConfig().n_populations

    @property
    def refinement_rows(self) -> int:
        """One to one: every audited seed matched by an unconditional row."""
        return 2 * self.tail_train

    @property
    def holdout_size(self) -> int:
        return self.tail_holdout + self.unconditional_holdout

    @property
    def fit_split(self) -> tuple[int, int]:
        """Tail rows and unconditional rows in the fitting set, one to one."""
        half = self.fit_size // 2
        return half, self.fit_size - half

    @property
    def validation_split(self) -> tuple[int, int]:
        """Same balance again in the distribution-validation set."""
        half = self.validation_size // 2
        return half, self.validation_size - half

    @property
    def acquisition_calls(self) -> int:
        return self.audit_per_level * SamplerConfig().n_populations

    def deployment_calls(self, loops: int = 1) -> int:
        """One practical deployment: the initial dataset plus each acquisition"""
        return self.initial_size + int(loops) * self.acquisition_calls

    @property
    def n_replicates(self) -> int:
        """Independent repetitions, not folds"""
        return 10

@dataclass(frozen=True)
class DataConfig:
    """Initial dataset split and closed-loop acquisition budget."""

    initial_size: int = 1500
    n_folds: int = 10
    fit_size: int = 350             # identical for A, B_N and C by design
    validation_size_fixed: int = 150   # held at 150 whatever the dataset size
    audit_per_level: int = 100
    audit_to_train: int = 50
    audit_to_fit: int = 50
    finetune_unconditional: int = 250

    @property
    def validation_size(self) -> int:
        return self.validation_size_fixed

    @property
    def holdout_size(self) -> int:
        """Never trained on: the error-model fitting set plus the KS set."""
        return self.fit_size + self.validation_size_fixed

    @property
    def train_size(self) -> int:
        return self.initial_size - self.holdout_size

    @property
    def tail_to_train(self) -> int:
        return self.audit_to_train * SamplerConfig().n_populations

    @property
    def tail_to_fit(self) -> int:
        return self.audit_to_fit * SamplerConfig().n_populations

    @property
    def finetune_rows(self) -> int:
        return self.tail_to_train + self.finetune_unconditional

@dataclass(frozen=True)
class BaselineConfig:
    """Unconditional sample-size family"""

    cap_multiple: int = 20

    def cap_for(self, anchor: int) -> int:
        return int(anchor) * self.cap_multiple

SAMPLER = SamplerConfig()
DATA = DataConfig()
def scaled(factor: int, *, sampler: SamplerConfig = SamplerConfig(),
           holdout: TailHoldoutConfig = TailHoldoutConfig()):
    """Both configurations with every count multiplied by ``factor``"""
    import dataclasses

    if factor < 1:
        raise ValueError("factor must be at least 1")
    return (dataclasses.replace(sampler, population=sampler.population * factor),
            dataclasses.replace(
                holdout,
                train_size=holdout.train_size * factor,
                unconditional_holdout=holdout.unconditional_holdout * factor,
                audit_per_level=holdout.audit_per_level * factor,
                seeds_per_level=holdout.seeds_per_level * factor,
                fit_size=holdout.fit_size * factor,
                validation_size=holdout.validation_size * factor))

def sampler_with_population(population: int, *,
                            sampler: SamplerConfig = SamplerConfig()) -> SamplerConfig:
    """Return one sampler with only its per-level population changed"""
    import dataclasses

    population = int(population)
    if population < 1:
        raise ValueError("population must be at least 1")
    configured = dataclasses.replace(sampler, population=population)
    if configured.n_seed < 1:
        raise ValueError("population and p0 must yield at least one seed")
    if configured.population % configured.n_seed:
        raise ValueError(
            "population must be divisible by the number of p0 seeds"
        )
    return configured

def with_subset_population(
    population: int,
    *,
    sampler: SamplerConfig = SamplerConfig(),
    holdout: TailHoldoutConfig = TailHoldoutConfig(),
) -> tuple[SamplerConfig, TailHoldoutConfig]:
    """Change the treatment population while preserving the base experiment"""
    import dataclasses

    configured = sampler_with_population(population, sampler=sampler)
    extra_per_level = holdout.extra_per_level
    organisation = dataclasses.replace(
        holdout,
        seeds_per_level=configured.n_seed,
        audit_per_level=configured.n_seed + extra_per_level,
    )
    return configured, organisation

TAIL_HOLDOUT = TailHoldoutConfig()
BASELINE = BaselineConfig()

COMPUTATION_SOURCES = (
    "arms.py", "audit.py", "checkpoint.py", "componentwise_amh.py",
    "componentwise_rw_amh.py", "config.py", "error_model.py", "folds.py",
    "joint_rw_amh.py", "ledger.py", "metrics.py", "probability.py",
    "problem.py", "resolution.py", "runner.py", "sample_weights.py",
    "sampler.py", "surrogate.py", "tail_holdout.py", "tail_runner.py",
)
