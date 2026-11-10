"""The tail-holdout organisation: one refinement, holdout audited in the tail."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SAMPLER, TAIL_HOLDOUT, SamplerConfig, TailHoldoutConfig
from .problem import Direction, Problem, adverse_order
from .checkpoint import LoopStore, checkpoint_settings
from .config import SAMPLER as _SAMPLER
from .sampler import SubsetRun, run_subset_simulation
from .surrogate import Surrogate

@dataclass
class TailHarvest:
    """Everything one acquisition loop brings back, kept by role."""

    train_x: np.ndarray
    train_y: np.ndarray
    train_proxy: np.ndarray
    holdout_x: np.ndarray
    holdout_y: np.ndarray
    holdout_proxy: np.ndarray
    true_calls: int
    per_level_precision: list[float] = field(default_factory=list)
    sampling_attempts: int = 1

class UniqueHarvestError(ValueError):
    """A proxy population cannot supply the pre-declared unique role quotas."""

def _exact_row_key(row: np.ndarray) -> bytes:
    """Identity for one finite floating-point input, with no fuzzy matching"""
    value = np.asarray(row, dtype=np.float64)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError("tail harvest inputs must be finite one-dimensional rows")
    value = np.ascontiguousarray(value.copy())
    value[value == 0.0] = 0.0
    return value.tobytes()

def _unique_representatives(
    samples: np.ndarray,
    indices: np.ndarray,
    excluded: set[bytes],
) -> tuple[np.ndarray, list[bytes]]:
    """First row for each exact input, preserving ``indices`` order."""
    selected: list[int] = []
    keys: list[bytes] = []
    local: set[bytes] = set()
    for raw_index in np.asarray(indices, dtype=int):
        index = int(raw_index)
        key = _exact_row_key(samples[index])
        if key in excluded or key in local:
            continue
        selected.append(index)
        keys.append(key)
        local.add(key)
    return np.asarray(selected, dtype=int), keys

def harvest(
    run: SubsetRun,
    problem: Problem,
    reference_cutoffs: np.ndarray,
    rng: np.random.Generator,
    cfg: TailHoldoutConfig = TAIL_HOLDOUT,
) -> TailHarvest:
    """Audit each level with an exact-unique, role-disjoint split"""
    from .problem import in_conditional_domain

    tr_x, tr_y, tr_p = [], [], []
    ho_x, ho_y, ho_p = [], [], []
    precision: list[float] = []
    calls = 0

    assigned: set[bytes] = set()
    plans: list[dict[str, object]] = []
    for level_number, (level, cutoff) in enumerate(
        zip(run.levels, reference_cutoffs), start=1
    ):
        samples = np.asarray(level.samples, dtype=float)
        values = np.asarray(level.values, dtype=float).reshape(-1)
        if samples.ndim != 2 or values.shape[0] != samples.shape[0]:
            raise ValueError(f"tail harvest level {level_number}: invalid population shape")
        order = adverse_order(values, problem.adverse)
        available, available_keys = _unique_representatives(samples, order, assigned)
        need = int(cfg.seeds_per_level)
        if available.size < need:
            raise UniqueHarvestError(
                f"tail harvest level {level_number}: need {need} unique training "
                f"inputs, found {available.size} after excluding inputs already "
                "assigned in earlier levels"
            )
        seed_idx = available[:need]
        assigned.update(available_keys[:need])
        plans.append({
            "samples": samples,
            "values": values,
            "order": order,
            "cutoff": float(cutoff),
            "seed_idx": seed_idx,
        })

    for level_number, plan in enumerate(plans, start=1):
        samples = plan["samples"]
        order = plan["order"]
        available, available_keys = _unique_representatives(samples, order, assigned)
        need = int(cfg.extra_per_level)
        if available.size < need:
            raise UniqueHarvestError(
                f"tail harvest level {level_number}: need {need} unique holdout "
                f"inputs, found {available.size} after excluding all training "
                "inputs and holdout inputs assigned in earlier levels"
            )
        chosen = rng.permutation(available.size)[:need]
        extra_idx = available[chosen]
        assigned.update(available_keys[int(i)] for i in chosen)
        plan["extra_idx"] = extra_idx

    for plan in plans:
        samples = plan["samples"]
        values = plan["values"]
        seed_idx = plan["seed_idx"]
        extra_idx = plan["extra_idx"]
        cutoff = plan["cutoff"]

        seed_x = samples[seed_idx]
        seed_p = values[seed_idx]
        extra_x = samples[extra_idx]
        extra_p = values[extra_idx]

        seed_y = problem.evaluate(seed_x)          # frozen first, evaluated after
        extra_y = problem.evaluate(extra_x)
        calls += seed_x.shape[0] + extra_x.shape[0]

        tr_x.append(seed_x); tr_y.append(seed_y); tr_p.append(seed_p)
        ho_x.append(extra_x); ho_y.append(extra_y); ho_p.append(extra_p)
        precision.append(
            float(np.mean(in_conditional_domain(seed_y, cutoff, problem.adverse)))
        )

    return TailHarvest(
        train_x=np.vstack(tr_x), train_y=np.concatenate(tr_y),
        train_proxy=np.concatenate(tr_p),
        holdout_x=np.vstack(ho_x), holdout_y=np.concatenate(ho_y),
        holdout_proxy=np.concatenate(ho_p),
        true_calls=calls, per_level_precision=precision,
    )

def _harvest_with_unique_retry(
    model: Surrogate,
    *,
    seed: int,
    problem: Problem,
    reference_cutoffs: np.ndarray,
    sampler: SamplerConfig,
    cfg: TailHoldoutConfig,
    max_attempts: int = 8,
) -> TailHarvest:
    """Repeat only proxy sampling when a chain cannot supply the unique quota"""
    failures: list[str] = []
    for attempt in range(int(max_attempts)):
        attempt_seed = int(seed) + attempt * 1_000_003
        rng = np.random.default_rng(attempt_seed)
        run = run_subset_simulation(
            model, dimension=problem.dimension, adverse=problem.adverse,
            rng=rng, cfg=sampler,
            kernel=getattr(problem, "subset_sampler", "pcn"),
        )
        try:
            got = harvest(run, problem, reference_cutoffs, rng, cfg)
        except UniqueHarvestError as exc:
            failures.append(str(exc))
            continue
        got.sampling_attempts = attempt + 1
        return got
    raise UniqueHarvestError(
        f"unique tail harvest failed after {max_attempts} deterministic proxy-only "
        f"attempts; last failure: {failures[-1]}"
    )

def _refinement_groups(n_unconditional: int, n_tail: int,
                       cfg: TailHoldoutConfig) -> np.ndarray:
    """Which group each refinement row belongs to"""
    if getattr(cfg, "weight_group", "call") == "call":
        return np.zeros(n_unconditional + n_tail, dtype=int)
    per_loop = np.repeat(np.arange(1, cfg_levels(cfg) + 1), cfg.seeds_per_level)
    loops = max(1, n_tail // per_loop.size)
    tail = np.tile(per_loop, loops)[:n_tail]
    return np.concatenate([np.zeros(n_unconditional, dtype=int), tail])

def cfg_levels(cfg: TailHoldoutConfig) -> int:
    return _SAMPLER.n_populations

def _ema_mix(prev, nxt, alpha: float):
    """Convex combination of two fitted surrogates in parameter space"""
    import copy

    mixed = copy.copy(nxt)
    mixed._model = copy.deepcopy(nxt._model)
    prev_state = prev._model.state_dict()
    nxt_state = nxt._model.state_dict()
    mixed_state = {
        key: (1.0 - alpha) * prev_state[key] + alpha * value
        for key, value in nxt_state.items()
    }
    mixed._model.load_state_dict(mixed_state)
    return mixed

def build_arms(
    problem: Problem,
    reference: SubsetRun,
    rng_pool: np.random.Generator,
    seeds: dict[str, int],
    cfg: TailHoldoutConfig = TAIL_HOLDOUT,
    stages: int = 1,
    checkpoints: bool = True,
    accumulate: bool | None = None,
    sampler: SamplerConfig = SAMPLER,
) -> dict:
    """One replicate: draw the dataset, train A, then run ``stages`` closed loops"""
    if accumulate is None:
        accumulate = cfg.accumulate_harvests
    pool = rng_pool.standard_normal((cfg.initial_size, problem.dimension))
    values = problem.evaluate(pool)
    train_x, train_y = pool[: cfg.train_size], values[: cfg.train_size]
    uncond_hold_x = pool[cfg.train_size :]
    uncond_hold_y = values[cfg.train_size :]

    store = LoopStore.open(
        key=f"{problem.name}/{problem.recipe}/seed_{seeds['model']}",
        settings=checkpoint_settings(
            problem=problem, recipe=problem.recipe, cfg=cfg,
            seeds={**seeds, "accumulate": int(bool(accumulate))}, sampler=sampler,
        ),
        arrays=(train_x, train_y, reference.thresholds),
        enabled=bool(checkpoints),
    )

    def _restore(model):
        """Put back what was stripped for storage"""
        model._feature_map = problem.feature_map
        model._sequence_map = problem.sequence_map
        return model

    def _strip(model):
        import copy

        bare = copy.copy(model)
        bare._feature_map = None
        bare._sequence_map = None
        return bare

    reuse_perm = np.random.default_rng(seeds["reuse"]).permutation(cfg.train_size)

    saved = store.load(0)
    if saved is not None:
        base = _restore(saved["model"])
    else:
        base = Surrogate(seed=seeds["model"], recipe=problem.recipe,
                         adverse=problem.adverse,
                         feature_map=problem.feature_map,
                         sequence_map=problem.sequence_map).fit(train_x, train_y)
        store.save(0, {"model": _strip(base)})

    ema = float(getattr(cfg, "navigator_ema", 0.0))
    saved = store.load(1)
    if saved is not None:
        refined, got = _restore(saved["model"]), saved["harvest"]
        nav1 = (_restore(saved["navigator"])
                if saved.get("navigator") is not None else refined)
    else:
        got = _harvest_with_unique_retry(
            base.clone_frozen(seeds["model"]),
            seed=seeds["acquire"], problem=problem,
            reference_cutoffs=reference.thresholds,
            sampler=sampler, cfg=cfg,
        )

        reuse_idx = reuse_perm[: got.train_x.shape[0]]          # one to one
        refine_x = np.vstack([train_x[reuse_idx], got.train_x])
        refine_y = np.concatenate([train_y[reuse_idx], got.train_y])
        if refine_x.shape[0] != cfg.refinement_rows:
            raise ValueError(f"refinement must hold {cfg.refinement_rows} rows")

        refined = base.clone_frozen(seed=seeds["model"])
        refined.continue_fit(
            refine_x, refine_y, new_labels=got.train_x.shape[0],
            groups=_refinement_groups(reuse_idx.size, got.train_x.shape[0], cfg))
        store.save(1, {"model": _strip(refined), "harvest": got,
                       "navigator": _strip(refined)})
        nav1 = refined

    later = []
    current = nav1
    seen_tail_x = [got.train_x]
    seen_tail_y = [got.train_y]
    for loop in range(2, int(stages) + 1):
        saved = store.load(loop)
        if saved is not None:
            model = _restore(saved["model"])
            nav = (_restore(saved["navigator"])
                   if saved.get("navigator") is not None else model)
            later.append({"model": model, "harvest": saved["harvest"]})
            seen_tail_x.append(saved["harvest"].train_x)
            seen_tail_y.append(saved["harvest"].train_y)
            current = nav
            continue
        got_l = _harvest_with_unique_retry(
            current.clone_frozen(seeds["model"]),
            seed=seeds["acquire"] + loop - 1, problem=problem,
            reference_cutoffs=reference.thresholds,
            sampler=sampler, cfg=cfg,
        )
        seen_tail_x.append(got_l.train_x)
        seen_tail_y.append(got_l.train_y)
        tail_x = np.vstack(seen_tail_x) if accumulate else got_l.train_x
        tail_y = np.concatenate(seen_tail_y) if accumulate else got_l.train_y
        idx_l = reuse_perm[: cfg.tail_train]
        nxt = current.clone_frozen(seed=seeds["model"])
        nxt.continue_fit(
            np.vstack([train_x[idx_l], tail_x]),
            np.concatenate([train_y[idx_l], tail_y]),
            new_labels=got_l.train_x.shape[0],
            groups=_refinement_groups(idx_l.size, tail_x.shape[0], cfg))
        nav = _ema_mix(current, nxt, ema) if ema > 0 else nxt
        store.save(loop, {"model": _strip(nxt), "harvest": got_l,
                          "navigator": _strip(nav)})
        later.append({"model": nxt, "harvest": got_l})
        current = nav

    split_rng = np.random.default_rng(seeds.get("holdout", seeds["reuse"] + 1))
    tail_perm = split_rng.permutation(got.holdout_x.shape[0])
    unc_perm = split_rng.permutation(uncond_hold_x.shape[0])
    tail_fit, unc_fit = cfg.fit_split
    tail_val, unc_val = cfg.validation_split
    if tail_fit + tail_val > tail_perm.size or unc_fit + unc_val > unc_perm.size:
        raise ValueError("holdout too small for the requested fit/validation split")

    ti, ui = tail_perm, unc_perm
    fit_x = np.vstack([got.holdout_x[ti[:tail_fit]], uncond_hold_x[ui[:unc_fit]]])
    fit_y = np.concatenate([got.holdout_y[ti[:tail_fit]], uncond_hold_y[ui[:unc_fit]]])
    val_x = np.vstack([got.holdout_x[ti[tail_fit:tail_fit + tail_val]],
                       uncond_hold_x[ui[unc_fit:unc_fit + unc_val]]])
    val_y = np.concatenate([got.holdout_y[ti[tail_fit:tail_fit + tail_val]],
                            uncond_hold_y[ui[unc_fit:unc_fit + unc_val]]])
    out = {
        "base": base, "refined": refined, "harvest": got,
        "train_x": train_x, "train_y": train_y,
        "fit_x": fit_x, "fit_y": fit_y,
        "validation_x": val_x, "validation_y": val_y,
        "fit_composition": (tail_fit, unc_fit),
        "validation_composition": (tail_val, unc_val),
        "true_calls": cfg.initial_size + got.true_calls,
    }
    for k, step in enumerate(later, start=2):
        perm = split_rng.permutation(step["harvest"].holdout_x.shape[0])
        hx, hy = step["harvest"].holdout_x, step["harvest"].holdout_y
        out.update({
            f"refined{k}": step["model"], f"harvest{k}": step["harvest"],
            f"fit{k}_x": np.vstack([hx[perm[:tail_fit]], uncond_hold_x[ui[:unc_fit]]]),
            f"fit{k}_y": np.concatenate([hy[perm[:tail_fit]], uncond_hold_y[ui[:unc_fit]]]),
            f"validation{k}_x": np.vstack([hx[perm[tail_fit:tail_fit + tail_val]],
                                           uncond_hold_x[ui[unc_fit:unc_fit + unc_val]]]),
            f"validation{k}_y": np.concatenate([hy[perm[tail_fit:tail_fit + tail_val]],
                                                uncond_hold_y[ui[unc_fit:unc_fit + unc_val]]]),
        })
        out["true_calls"] += step["harvest"].true_calls
    return out
