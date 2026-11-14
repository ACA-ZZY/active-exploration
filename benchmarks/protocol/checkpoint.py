"""Per-loop checkpoints for the closed-loop arms."""

from __future__ import annotations

import os
import pickle
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROOT = Path(os.environ.get("PROTOCOL_CHECKPOINT_DIR", "/tmp/protocol_checkpoints"))

def _register_classes() -> None:
    """Publish the network classes before unpickling anything that holds one"""
    try:
        from .surrogate import _temporal_attention_module_class, _torch_modules

        _torch_modules()
        _temporal_attention_module_class()
    except Exception:
        pass          # torch absent, or a recipe that needs no network

TORCH_RECIPES = ("network", "deep", "spectral", "recurrent",
                 "temporal_attention")

def _device_part(recipe) -> str:
    """The device, for the recipes whose weights depend on it"""
    recipe_name = str(recipe)
    try:
        from .surrogate import RECIPES

        declared = RECIPES.get(recipe_name)
        family = str(getattr(declared, "name", recipe_name))
    except Exception:
        family = recipe_name
    if family not in TORCH_RECIPES:
        return ""
    try:
        from .surrogate import resolve_device

        return resolve_device()
    except Exception:
        return ""

def _recipe_part(recipe) -> str:
    """The recipe's settings, not merely its name"""
    try:
        from .surrogate import RECIPES

        found = RECIPES.get(str(recipe))
        return repr(found) if found is not None else repr(recipe)
    except Exception:
        return repr(recipe)

def checkpoint_settings(*, problem, recipe, cfg, seeds: dict, sampler=None) -> dict:
    return {
        "problem": str(problem.name),
        "dimension": int(problem.dimension),
        "adverse": str(problem.adverse),
        "feature_tag": str(getattr(problem, "feature_tag", "")),
        "subset_sampler": str(getattr(problem, "subset_sampler", "pcn")),
        "recipe": _recipe_part(recipe),
        "config": repr(cfg),
        "device": _device_part(recipe),
        "sampler": repr(sampler),
        "seeds": {str(k): int(v) for k, v in seeds.items()},
    }

@dataclass
class LoopStore:
    """The checkpoints for one replicate's chain of loops."""

    root: Path
    key: str
    settings: dict
    arrays: tuple[np.ndarray, ...]
    enabled: bool = True

    @classmethod
    def open(cls, *, key: str, settings: dict, arrays: tuple[np.ndarray, ...],
             root: Path | None = None,
             enabled: bool = True) -> "LoopStore":
        base = Path(root) if root is not None else DEFAULT_ROOT
        store = cls(root=base / key, key=key, settings=dict(settings),
                    arrays=tuple(np.array(a, copy=True) for a in arrays), enabled=enabled)
        if store.enabled:
            store.root.mkdir(parents=True, exist_ok=True)
        return store

    def _path(self, loop: int) -> Path:
        return self.root / f"loop{loop}.pkl"

    def load(self, loop: int) -> Any | None:
        """The stored loop, or ``None`` when absent, stale or unreadable."""
        if not self.enabled:
            return None
        path = self._path(loop)
        if not path.exists():
            return None
        _register_classes()
        try:
            with path.open("rb") as fh:
                blob = pickle.load(fh)
        except Exception:
            return None
        if blob.get("settings") != self.settings:
            return None
        previous = blob.get("arrays", ())
        if len(previous) != len(self.arrays) or any(
            not np.array_equal(a, b) for a, b in zip(previous, self.arrays)
        ):
            return None
        return blob.get("payload")

    def save(self, loop: int, payload: Any) -> None:
        """Store one loop"""
        if not self.enabled:
            return
        path = self._path(loop)
        tmp = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}."
            f"{uuid.uuid4().hex}.tmp"
        )
        try:
            with tmp.open("wb") as fh:
                pickle.dump({"settings": self.settings, "arrays": self.arrays, "payload": payload}, fh,
                            protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)          # atomic, so a killed run leaves no half file
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"could not checkpoint loop {loop} under {self.root}: {exc}"
            ) from exc
