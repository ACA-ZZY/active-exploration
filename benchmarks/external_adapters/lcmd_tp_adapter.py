"""Provenance-checked entry point for the authors' LCMD-TP implementation"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "af97eece7a513a2f3603e928c77e9e5d1c9ba7e9"
ADAPTER_DIR = Path(__file__).resolve().parent
MORETEST_ROOT = ADAPTER_DIR.parent
BMDAL_ROOT = MORETEST_ROOT / "third_party" / "bmdal_reg"

class ExternalMethodUnavailable(RuntimeError):
    """Raised when the pinned external method cannot be used faithfully."""

@dataclass(frozen=True)
class LCMDSelection:
    indices: list[int]
    metadata: dict[str, Any]

def _git_head(repo: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExternalMethodUnavailable(
            f"Cannot verify pinned BMDAL checkout at {repo}: {exc}"
        ) from exc
    return proc.stdout.strip()

def verify_checkout() -> str:
    """Return the exact BMDAL commit or fail closed on provenance drift."""

    if not BMDAL_ROOT.is_dir():
        raise ExternalMethodUnavailable(f"BMDAL checkout is absent: {BMDAL_ROOT}")
    head = _git_head(BMDAL_ROOT)
    if head != EXPECTED_COMMIT:
        raise ExternalMethodUnavailable(
            f"BMDAL provenance mismatch: expected {EXPECTED_COMMIT}, found {head}"
        )
    return head

def _apply_runtime_compatibility() -> list[str]:
    """Install API-only shims required by the pinned 2023 code on NumPy 2.x"""

    try:
        import numpy as np
    except ImportError as exc:
        raise ExternalMethodUnavailable("NumPy is required by BMDAL") from exc
    shims: list[str] = []
    numpy_major = int(np.__version__.split(".", 1)[0])
    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    if numpy_major >= 2:
        shims.append("numpy.Inf=numpy.inf")
    return shims

def _load_official_api():
    verify_checkout()
    compatibility_shims = _apply_runtime_compatibility()
    root = str(BMDAL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import torch
        from bmdal_reg.bmdal.algorithms import select_batch
        from bmdal_reg.bmdal.feature_data import TensorFeatureData
    except ImportError as exc:
        raise ExternalMethodUnavailable(
            "The official BMDAL API is not importable. Install torch, numpy, "
            "and dill in the selected Python environment."
        ) from exc
    return torch, select_batch, TensorFeatureData, compatibility_shims

def _as_tensor(torch, values, *, device, dtype):
    if isinstance(values, torch.Tensor):
        return values.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(values, device=device, dtype=dtype)

def _reject_unsupported_recurrence(torch, model) -> None:
    recurrent = [
        type(module).__name__
        for module in model.modules()
        if isinstance(module, torch.nn.modules.rnn.RNNBase)
    ]
    if recurrent:
        names = ", ".join(sorted(set(recurrent)))
        raise ExternalMethodUnavailable(
            "Pinned BMDAL arxiv_v3 has no official LayerGradientComputation "
            f"adapter for recurrent module(s): {names}. Refusing to label an "
            "output-head-only gradient approximation as LCMD-TP."
        )

def select_lcmd_tp(
    model,
    train_x,
    pool_x,
    batch_size: int,
    *,
    sketch_dim: int = 512,
    seed: int = 0,
    precomp_batch_size: int = 32768,
    nn_batch_size: int = 8192,
    verbosity: int = 0,
    linear_layers_only: bool = False,
) -> LCMDSelection:
    """Select pool rows with the authors' LCMD-TP gradient-feature recipe"""

    torch, official_select_batch, TensorFeatureData, compatibility_shims = _load_official_api()
    import numpy as np
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not linear_layers_only:
        _reject_unsupported_recurrence(torch, model)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sketch_dim <= 0:
        raise ValueError("sketch_dim must be positive")

    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise ValueError("model must contain trainable parameters") from exc
    device = parameter.device
    dtype = parameter.dtype
    train_tensor = _as_tensor(torch, train_x, device=device, dtype=dtype)
    pool_tensor = _as_tensor(torch, pool_x, device=device, dtype=dtype)
    if train_tensor.ndim < 2 or pool_tensor.ndim < 2:
        raise ValueError("train_x and pool_x must include sample and feature axes")
    if train_tensor.shape[1:] != pool_tensor.shape[1:]:
        raise ValueError("train_x and pool_x feature shapes differ")
    if batch_size > int(pool_tensor.shape[0]):
        raise ValueError("batch_size exceeds the pool size")

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    was_training = model.training
    parameter_flags = [(p, p.requires_grad) for p in model.parameters()]
    if linear_layers_only:
        linear_ids = {id(p) for layer in model.modules() if isinstance(layer, torch.nn.Linear) for p in layer.parameters()}
        for p, _ in parameter_flags:
            p.requires_grad_(id(p) in linear_ids)
    started = time.perf_counter()
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed_all(int(seed))
            indices, stats = official_select_batch(
                batch_size=int(batch_size),
                models=[model],
                data={
                    "train": TensorFeatureData(train_tensor),
                    "pool": TensorFeatureData(pool_tensor),
                },
                y_train=None,
                selection_method="lcmd",
                sel_with_train=True,
                base_kernel="grad",
                kernel_transforms=[("rp", [int(sketch_dim)])],
                precomp_batch_size=int(precomp_batch_size),
                nn_batch_size=int(nn_batch_size),
                verbosity=int(verbosity),
                use_cuda_synchronize=(device.type == "cuda"),
            )
    finally:
        for p, flag in parameter_flags:
            p.requires_grad_(flag)
        model.train(was_training)

    selected = [int(value) for value in indices.detach().cpu().tolist()]
    if len(selected) != batch_size or len(set(selected)) != batch_size:
        raise RuntimeError("Official LCMD returned a non-unique or incomplete batch")
    if min(selected) < 0 or max(selected) >= int(pool_tensor.shape[0]):
        raise RuntimeError("Official LCMD returned an out-of-range pool index")
    metadata = {
        "method": "LCMD-TP-kgrad-sketch512" if sketch_dim == 512 else "LCMD-TP-kgrad-sketch",
        "third_party_commit": EXPECTED_COMMIT,
        "gradient_scope": "Linear layers only" if linear_layers_only else "all supported layers",
        "selection_method": "lcmd",
        "sel_with_train": True,
        "base_kernel": "grad",
        "kernel_transforms": [["rp", [int(sketch_dim)]]],
        "seed": int(seed),
        "device": str(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "runtime_compatibility_shims": compatibility_shims,
        "elapsed_seconds": time.perf_counter() - started,
        "official_stats": stats,
    }
    return LCMDSelection(indices=selected, metadata=metadata)

def _fixed_model(torch, device):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    ).to(device)
    with torch.no_grad():
        for number, parameter in enumerate(model.parameters(), start=1):
            values = torch.linspace(
                -0.15 * number,
                0.15 * number,
                parameter.numel(),
                device=device,
                dtype=parameter.dtype,
            )
            parameter.copy_(values.reshape_as(parameter))
    return model

def run_self_test(device_name: str = "cpu") -> dict[str, Any]:
    torch, _, _, _ = _load_official_api()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ExternalMethodUnavailable("CUDA requested but torch reports no CUDA device")
    device = torch.device(device_name)
    train = torch.linspace(-1.0, 1.0, 48, device=device).reshape(24, 2)
    pool_axis = torch.linspace(-1.7, 1.7, 64, device=device)
    pool = torch.stack((pool_axis, torch.sin(2.3 * pool_axis)), dim=1)
    model = _fixed_model(torch, device)
    first = select_lcmd_tp(model, train, pool, 8, seed=20260907)
    second = select_lcmd_tp(model, train, pool, 8, seed=20260907)
    recurrence_refused = False
    recurrence_reason = ""
    recurrent_model = torch.nn.GRU(2, 4, batch_first=True).to(device)
    try:
        select_lcmd_tp(
            recurrent_model,
            train.reshape(24, 1, 2),
            pool.reshape(64, 1, 2),
            2,
            seed=20260907,
        )
    except ExternalMethodUnavailable as exc:
        recurrence_refused = True
        recurrence_reason = str(exc)
    checks_pass = (
        first.indices == second.indices
        and len(set(first.indices)) == 8
        and all(0 <= idx < 64 for idx in first.indices)
        and recurrence_refused
    )
    return {
        "status": "PASS" if checks_pass else "FAIL",
        "deterministic_same_seed": first.indices == second.indices,
        "unique_indices": len(set(first.indices)) == 8,
        "indices_in_range": all(0 <= idx < 64 for idx in first.indices),
        "unsupported_recurrence_refused": recurrence_refused,
        "unsupported_recurrence_reason": recurrence_reason,
        "selection": asdict(first),
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args(argv)
    if not args.self_test:
        parser.error("only --self-test is exposed as a CLI; import select_lcmd_tp for experiments")
    report = run_self_test(args.device)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
