from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

def cyclic_block_kfold(n_samples: int, num_slopes: int, k: int, seed: int = 42):
    if n_samples % num_slopes != 0:
        raise ValueError("n_samples must be divisible by num_slopes")
    n_blocks = n_samples // num_slopes
    blocks = np.arange(n_blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    fold_blocks = np.array_split(blocks, k)
    splits = []
    for fold in range(k):
        val_blocks = fold_blocks[fold]
        train_blocks = np.concatenate([fold_blocks[idx] for idx in range(k) if idx != fold])
        val_idx = np.concatenate([np.arange(block * num_slopes, (block + 1) * num_slopes) for block in val_blocks])
        train_idx = np.concatenate([np.arange(block * num_slopes, (block + 1) * num_slopes) for block in train_blocks])
        splits.append((train_idx, val_idx))
    return splits

def load_fos(fos_path: str | Path) -> list[float]:
    with Path(fos_path).open("r", encoding="utf-8") as handle:
        return [float(line.strip()) for line in handle if line.strip()]

def build_index(input_dir: str | Path, label_dir: str | Path) -> List[Dict]:
    input_map: Dict[str, Dict] = {}
    for path in Path(input_dir).glob("*.npy"):
        parts = path.stem.split(",")
        key = parts[0]
        input_map[key] = {
            "path": str(path),
            "slope": int(key.split("-")[1]),
            "input_scalars": tuple(map(float, parts[1:5])),
        }

    records: List[Dict] = []
    label_files = sorted(Path(label_dir).glob("*.npy"), key=lambda item: int(item.name.split("-")[0]))
    for path in label_files:
        parts = path.stem.split(",")
        key = parts[0]
        if key not in input_map:
            raise FileNotFoundError(f"Missing matching Input file for {key}")
        records.append(
            {
                "key": key,
                "slope": input_map[key]["slope"],
                "input_path": input_map[key]["path"],
                "label_path": str(path),
                "input_scalars": input_map[key]["input_scalars"],
                "label_scalars": tuple(map(float, parts[1:7])),
            }
        )
    return records

def build_mask_by_slope(records: list[dict]) -> Dict[int, torch.Tensor]:
    mask_by_slope: Dict[int, torch.Tensor] = {}
    for rec in records:
        slope = int(rec["slope"])
        if slope in mask_by_slope:
            continue
        arr = np.load(rec["input_path"], mmap_mode="r")
        base_mask = arr[0] == -1
        mask_np = np.repeat(base_mask[np.newaxis, ...], 3, axis=0)
        mask_by_slope[slope] = torch.from_numpy(mask_np.copy()).bool().share_memory_()
    return mask_by_slope

class NpyOnDemandDataset(Dataset):
    def __init__(self, records: list[dict], fos_tensor: torch.Tensor, mask_by_slope: Dict[int, torch.Tensor]):
        self.records = records
        self.fos_tensor = fos_tensor
        self.mask_by_slope = mask_by_slope

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        input_raw = np.load(rec["input_path"], mmap_mode="r")
        input_mask = input_raw == -1
        input_arr = input_raw.astype(np.float32, copy=True)
        max1, min1, max2, min2 = rec["input_scalars"]
        input_arr[0] = (input_arr[0] - min1) / (max1 - min1)
        input_arr[1] = (input_arr[1] - min2) / (max2 - min2)
        input_arr[input_mask] = -1

        label_raw = np.load(rec["label_path"], mmap_mode="r")
        label_arr = label_raw.astype(np.float32, copy=True)
        max1_l, min1_l, max2_l, min2_l, max3_l, min3_l = rec["label_scalars"]
        missing_mask = label_raw[0] == min1_l
        label_arr[0] = (label_arr[0] - min1_l) / (max1_l - min1_l)
        label_arr[1] = (label_arr[1] - min2_l) / (max2_l - min2_l)
        label_arr[2] = (label_arr[2] - min3_l) / (max3_l - min3_l)
        label_arr[:, missing_mask] = -1

        x = torch.from_numpy(np.ascontiguousarray(input_arr))
        y = torch.from_numpy(np.ascontiguousarray(label_arr))
        fos = self.fos_tensor[idx]
        mask = self.mask_by_slope[int(rec["slope"])]
        return x, y, fos, mask

class GroupDensitySampler(Sampler[int]):
    def __init__(
        self,
        fos_sub: Sequence[float],
        group_sub: Sequence[int],
        mode: str = "mix",
        beta: float = 1.5,
        epoch_len: int | None = None,
        seed: int = 42,
    ):
        fos = np.asarray(fos_sub, dtype=np.float64)
        groups = np.asarray(group_sub, dtype=np.int64)
        if fos.shape[0] != groups.shape[0]:
            raise ValueError("fos_sub and group_sub length mismatch")
        if mode not in {"density", "uniform", "mix"}:
            raise ValueError("mode must be density, uniform, or mix")
        local = np.zeros(fos.shape[0], dtype=np.float64)
        eps = 1e-12
        for group in np.unique(groups):
            idx = np.where(groups == group)[0]
            if idx.size <= 1:
                local[idx] = 1.0
                continue
            order = np.argsort(fos[idx])
            x = fos[idx][order]
            h = np.zeros_like(x)
            h[0] = x[1] - x[0]
            h[-1] = x[-1] - x[-2]
            if x.size > 2:
                h[1:-1] = 0.5 * (x[2:] - x[:-2])
            h = np.maximum(h, eps)
            h = h / max(float(np.mean(h)), eps)
            local[idx[order]] = h ** float(beta)
        if mode == "density":
            weights = local
        elif mode == "uniform":
            weights = np.ones_like(local)
        else:
            weights = 0.5 * np.ones_like(local) + 0.5 * local
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0:
            raise ValueError("sampler weights sum to zero")
        self.p = weights / weights.sum()
        self.indices = np.arange(fos.shape[0], dtype=np.int64)
        self.epoch_len = int(epoch_len) if epoch_len is not None else int(fos.shape[0])
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.choice(self.indices, size=self.epoch_len, replace=True, p=self.p)
        yield from (int(item) for item in idx)

    def __len__(self) -> int:
        return self.epoch_len
