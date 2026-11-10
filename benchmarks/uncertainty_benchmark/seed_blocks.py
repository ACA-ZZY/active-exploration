"""Stable, named random streams for paired seed blocks."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class SeedBlock:
    root_seed: int
    block: int

    def __post_init__(self) -> None:
        if int(self.root_seed) < 0 or int(self.block) < 0:
            raise ValueError("root_seed and block must be non-negative")

    def seed(
        self,
        role: str,
        *,
        method: str = "",
        round_index: int = 0,
        member: int = 0,
        attempt: int = 0,
    ) -> int:
        if not role:
            raise ValueError("a named random-stream role is required")
        indices = [int(self.root_seed), int(self.block), int(round_index), int(member), int(attempt)]
        if any(value < 0 for value in indices):
            raise ValueError("random-stream indices must be non-negative")
        role_bytes = str(role).encode("utf-8")
        method_bytes = str(method).encode("utf-8")
        entropy = indices + [len(role_bytes), *role_bytes, len(method_bytes), *method_bytes]
        return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
