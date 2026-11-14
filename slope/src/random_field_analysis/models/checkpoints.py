from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .resunet import ResUNet

def build_model(in_channels: int = 2, out_channels: int = 3, device: str | torch.device = "cpu") -> ResUNet:
    return ResUNet(in_channels=in_channels, out_channels=out_channels).to(device)

def _fos_range(value):
    if value is None:
        return None
    bounds = tuple(float(x) for x in value)
    if len(bounds) != 2 or not all(torch.isfinite(torch.tensor(x)).item() for x in bounds) or bounds[0] >= bounds[1]:
        raise ValueError("FOS normalization requires finite minimum < maximum")
    return bounds

def load_checkpoint(path: str | Path, device: str | torch.device = "cpu", trust_legacy_pickle: bool = False) -> torch.nn.Module:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=not trust_legacy_pickle)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.to(device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected a state-dict checkpoint with model settings")
    model_cfg = checkpoint.get("model", {})
    in_channels = int(model_cfg.get("in_channels", 2))
    out_channels = int(model_cfg.get("out_channels", 3))
    architecture = model_cfg.get("class", "ResUNet")
    if architecture == "nerual.ResUNet":
        from nerual import ResUNet as LegacyResUNet
        model = LegacyResUNet(in_channel=in_channels, out_channel=out_channels).to(device)
    elif architecture == "ResUNet":
        model = build_model(in_channels, out_channels, device)
    else:
        raise ValueError(f"Unsupported architecture {architecture}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    normalization = checkpoint.get("fos_normalization")
    model._fos_normalization = None if normalization is None else _fos_range((normalization["minimum"], normalization["maximum"]))
    return model

def restore_fos(values: torch.Tensor, model: torch.nn.Module, fos_range=None) -> torch.Tensor:
    supplied = _fos_range(fos_range)
    recorded = _fos_range(getattr(model, "_fos_normalization", None))
    if supplied is not None and recorded is not None and supplied != recorded:
        raise ValueError("Requested FOS scale differs from checkpoint metadata")
    bounds = recorded if recorded is not None else supplied
    if bounds is None:
        raise ValueError("This checkpoint has no FOS scale metadata; supply its verified minimum and maximum")
    return values * (bounds[1] - bounds[0]) + bounds[0]

def save_state_dict_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    in_channels: int | None = None,
    out_channels: int | None = None,
    extra: dict[str, Any] | None = None,
    fos_range=None,
) -> None:
    architecture = "nerual.ResUNet" if type(model).__module__ == "nerual" else "ResUNet"
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model": {
            "class": architecture,
            "in_channels": int(model.input_in[0].in_channels if in_channels is None else in_channels),
            "out_channels": int(model.dis_map4.out_channels if out_channels is None else out_channels),
        },
    }
    bounds = _fos_range(fos_range if fos_range is not None else getattr(model, "_fos_normalization", None))
    if bounds is not None:
        payload["fos_normalization"] = {"minimum": bounds[0], "maximum": bounds[1]}
    if extra:
        if set(extra) & set(payload):
            raise ValueError("Extra fields must not replace checkpoint metadata")
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
