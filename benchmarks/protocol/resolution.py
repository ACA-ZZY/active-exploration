"""Probability resolution of the conditional error model"""

from __future__ import annotations

from dataclasses import dataclass

from .config import DATA, SAMPLER

@dataclass(frozen=True)
class Resolution:
    n_fit: int
    empirical: float
    upper_bound: float
    chain_factor: float
    overall: float

    def describe(self) -> str:
        return (
            f"fitting samples n={self.n_fit}; empirical resolution 1/n="
            f"{self.empirical:.3e}; zero-observation 95% upper bound "
            f"{self.upper_bound:.3e}; chain factor {self.chain_factor:.0e}; "
            f"overall resolution {self.overall:.3e}"
        )

    def is_sub_resolution(self, probability: float) -> bool:
        return probability < self.overall

def declare(n_fit: int = DATA.fit_size, n_populations: int | None = None) -> Resolution:
    populations = SAMPLER.n_populations if n_populations is None else n_populations
    upper = 1.0 - 0.05 ** (1.0 / n_fit)
    chain = SAMPLER.conditional_probability ** (populations - 1)
    return Resolution(
        n_fit=n_fit,
        empirical=1.0 / n_fit,
        upper_bound=upper,
        chain_factor=chain,
        overall=chain * upper,
    )
