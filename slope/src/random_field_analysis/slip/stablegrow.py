from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, label

def infer_scalar_shape(values: np.ndarray, vector_data: np.ndarray | None = None) -> tuple[int, int, int]:
    if vector_data is not None:
        return tuple(np.asarray(vector_data).shape[1:4])
    arr = np.asarray(values)
    if arr.ndim == 3:
        return tuple(arr.shape)
    if arr.ndim == 4 and arr.shape[0] <= 8:
        return tuple(arr.shape[1:4])
    if arr.ndim == 4 and arr.shape[-1] <= 8:
        return tuple(arr.shape[:3])
    raise ValueError(f"Unsupported input shape: {arr.shape}")

def scalar_magnitude(values: np.ndarray, vector_data: np.ndarray | None = None) -> np.ndarray:
    if vector_data is not None:
        vec = np.asarray(vector_data, dtype=np.float64)
        if vec.ndim != 4:
            raise ValueError(f"vector_data must be 4D, got {vec.shape}")
        return np.sqrt(np.sum(vec[:3] * vec[:3], axis=0))
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[0] <= 8:
        vec = arr[:3]
        return np.sqrt(np.sum(vec * vec, axis=0))
    if arr.ndim == 4 and arr.shape[-1] <= 8:
        vec = np.moveaxis(arr[..., :3], -1, 0)
        return np.sqrt(np.sum(vec * vec, axis=0))
    raise ValueError(f"Unsupported input shape: {arr.shape}")

def fit_sampled_kmeans_1d(values: np.ndarray, k: int = 2, sample_size: int = 24000, max_iter: int = 24) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(42)
    fit_vals = vals if vals.size <= sample_size else vals[rng.choice(vals.size, size=sample_size, replace=False)]
    centers = np.quantile(fit_vals, np.linspace(0.0, 1.0, k + 2)[1:-1]).astype(np.float64)
    for _ in range(max_iter):
        fit_labels = np.argmin(np.abs(fit_vals[:, None] - centers[None, :]), axis=1).astype(np.int32)
        new_centers = centers.copy()
        for idx in range(k):
            mask = fit_labels == idx
            if np.any(mask):
                new_centers[idx] = float(np.mean(fit_vals[mask]))
        if np.allclose(new_centers, centers, atol=1e-7):
            centers = new_centers
            break
        centers = new_centers
    centers = centers[np.argsort(centers)]
    full_labels = np.argmin(np.abs(vals[:, None] - centers[None, :]), axis=1).astype(np.int32)
    return full_labels, centers

def stablegrow_body_mask(
    values: np.ndarray,
    domain_mask: np.ndarray,
    air_threshold: float = 0.0,
    vertical_axis: int = 1,
    min_stable_region: int = 400,
    vector_data: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scalar = scalar_magnitude(values, vector_data=vector_data)
    domain_mask = np.asarray(domain_mask, dtype=bool)
    if domain_mask.shape != scalar.shape:
        raise ValueError(f"domain_mask shape {domain_mask.shape} != scalar shape {scalar.shape}")

    scalar_flat = scalar.reshape(-1)
    valid_flat = domain_mask.reshape(-1) & np.isfinite(scalar_flat) & (scalar_flat >= float(air_threshold))
    valid_mask = valid_flat.reshape(scalar.shape)

    classify = np.full(scalar.shape, -1, dtype=np.int32)
    labels_3d = np.full(scalar.shape, -1, dtype=np.int32)
    if not np.any(valid_flat):
        return classify, labels_3d, np.zeros(scalar.shape, dtype=bool), valid_mask, np.empty(0, dtype=np.float64)

    labels_1d, centers = fit_sampled_kmeans_1d(scalar_flat[valid_flat], k=2, sample_size=24000, max_iter=24)
    labels_3d.reshape(-1)[np.flatnonzero(valid_flat)] = labels_1d

    low_mask = valid_mask & (labels_3d == 0)
    coord = np.arange(scalar.shape[vertical_axis], dtype=np.int32)
    reshape = [1, 1, 1]
    reshape[vertical_axis] = scalar.shape[vertical_axis]
    seed_mask = low_mask & (coord.reshape(reshape) == 0)

    cc, n_components = label(low_mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    stable_mask = np.zeros_like(valid_mask, dtype=bool)
    if n_components > 0:
        sizes = np.bincount(cc.ravel())[1:]
        seed_ids = np.unique(cc[seed_mask])
        keep_ids = [int(cid) for cid in seed_ids if cid > 0 and sizes[int(cid) - 1] >= int(min_stable_region)]
        if keep_ids:
            stable_mask = np.isin(cc, keep_ids)

    body_mask = valid_mask & (~stable_mask)
    classify[valid_mask] = 0
    classify[body_mask] = 1
    return classify, labels_3d, body_mask, valid_mask, centers

def body_mask_to_edge_label(body_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    body_mask = np.asarray(body_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    edge_label = np.full(body_mask.shape, -1, dtype=np.int32)
    if not np.any(body_mask):
        return edge_label
    edge_label[body_mask] = 0
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = convolve(body_mask.astype(np.uint8), kernel, mode="constant", cval=0)
    boundary = body_mask & (neighbor_count < int(kernel.sum()))
    edge_label[boundary] = 1
    edge_label[~valid_mask] = -1
    return edge_label

def edge_from_body_labels(classify: np.ndarray) -> np.ndarray:
    classify = np.asarray(classify)
    return body_mask_to_edge_label(classify == 1, classify != -1)

def stablegrow_edge_from_values(
    values: np.ndarray,
    domain_mask: np.ndarray,
    air_threshold: float = 0.0,
    vertical_axis: int = 1,
    min_stable_region: int = 400,
    vector_data: np.ndarray | None = None,
) -> np.ndarray:
    _, _, body, valid, _ = stablegrow_body_mask(values, domain_mask, air_threshold, vertical_axis, min_stable_region, vector_data)
    return body_mask_to_edge_label(body, valid)
