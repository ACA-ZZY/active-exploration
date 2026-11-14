from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from random_field_analysis.data.datasets import GroupDensitySampler, NpyOnDemandDataset, build_index, build_mask_by_slope, cyclic_block_kfold, load_fos
from random_field_analysis.models.checkpoints import load_checkpoint, save_state_dict_checkpoint
from random_field_analysis.models.resunet import OHEM, ResUNet

@dataclass
class TrainConfig:
    data_root: Path
    fos_file: Path
    output_root: Path
    kfold: int = 10
    num_slopes: int = 10
    epochs: int = 50
    branch_finetune_epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 1e-4
    fs_scale_min: float = 0.5
    fs_scale_max: float = 3.5
    validation_holdout: int = 1500
    device: str = "cuda:0"
    seed: int = 42
    pretrained_checkpoint: Path | None = None

def mape_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    mask = (label != pred).float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((pred - label) / (label + 1e-6)) * mask
    return torch.where(torch.isnan(loss), torch.zeros_like(loss), loss).mean()

def _save_frame(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

def _make_loader(dataset, indices, batch_size, sampler=None, shuffle=False):
    kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
    return DataLoader(Subset(dataset, indices), **kwargs)

def _load_model(config: TrainConfig, device: torch.device) -> ResUNet:
    if config.pretrained_checkpoint is not None:
        return load_checkpoint(config.pretrained_checkpoint, device=device, trust_legacy_pickle=True)
    return ResUNet(in_channels=2, out_channels=3).to(device)

def run_kfold_training(config: TrainConfig) -> None:
    if not config.fs_scale_min < config.fs_scale_max:
        raise ValueError("FOS normalization minimum must be below its maximum")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    input_dir = config.data_root / "Input"
    label_dir = config.data_root / "Label"
    records = build_index(input_dir, label_dir)
    fos = np.asarray(load_fos(config.fos_file), dtype=np.float32)
    fos = (fos - config.fs_scale_min) / (config.fs_scale_max - config.fs_scale_min)
    if len(fos) != len(records):
        raise ValueError(f"FOS count {len(fos)} does not match records {len(records)}")

    dataset = NpyOnDemandDataset(records, torch.as_tensor(fos, dtype=torch.float32), build_mask_by_slope(records))
    slope_all = np.asarray([rec["slope"] for rec in records], dtype=np.int64)
    device = torch.device(config.device if torch.cuda.is_available() or not config.device.startswith("cuda") else "cpu")
    splits = cyclic_block_kfold(len(records), config.num_slopes, config.kfold, seed=config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx_all, val_idx_base) in enumerate(splits, start=1):
        cut = max(0, int(len(records) / config.kfold * (config.kfold - 1) - config.validation_holdout))
        train_idx = train_idx_all[:cut]
        pre_idx = train_idx_all[cut:]
        val_idx = np.asarray(list(val_idx_base) + list(pre_idx), dtype=np.int64)

        sampler = GroupDensitySampler(fos[train_idx], slope_all[train_idx], mode="mix", beta=1.5, epoch_len=len(train_idx), seed=config.seed)
        train_loader = _make_loader(dataset, train_idx, config.batch_size, sampler=sampler)
        val_loader = _make_loader(dataset, val_idx, 1, shuffle=False)

        model = _load_model(config, device)
        recorded_scale = getattr(model, "_fos_normalization", None)
        current_scale = (config.fs_scale_min, config.fs_scale_max)
        if recorded_scale is not None and tuple(recorded_scale) != current_scale:
            raise ValueError("Pretrained FOS scale differs from training configuration")
        model._fos_normalization = current_scale
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        loss_dis_fn = OHEM().to(device)
        loss_fs_fn = nn.MSELoss().to(device)
        train_rows: list[dict] = []

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss_dis: list[float] = []
            epoch_loss_fs: list[float] = []
            for x, y, fos_target, mask in train_loader:
                x = x.to(device)
                y = y.to(device)
                fos_target = fos_target.to(device)
                mask = mask.to(device)
                fos_pred, disp_pred = model(x)
                loss_dis = loss_dis_fn(disp_pred, y, mask, topk=0.5)
                loss_fs = loss_fs_fn(fos_pred, fos_target)
                optimizer.zero_grad()
                (loss_dis + loss_fs).backward()
                optimizer.step()
                epoch_loss_dis.append(float(loss_dis.detach().cpu()))
                epoch_loss_fs.append(float(loss_fs.detach().cpu()))
            train_rows.append({"epoch": epoch, "loss_dis": np.mean(epoch_loss_dis), "loss_fs": np.mean(epoch_loss_fs), "phase": "joint"})

        for name, param in model.named_parameters():
            if "encoder" in name or "down" in name:
                param.requires_grad = False

        for epoch in range(1, config.branch_finetune_epochs + 1):
            model.train()
            epoch_loss_dis = []
            epoch_loss_fs = []
            for x, y, fos_target, mask in train_loader:
                x = x.to(device)
                y = y.to(device)
                fos_target = fos_target.to(device)
                mask = mask.to(device)
                fos_pred, disp_pred = model(x)
                loss_dis = loss_dis_fn(disp_pred, y, mask, topk=0.5)
                loss_fs = loss_fs_fn(fos_pred, fos_target)
                optimizer.zero_grad()
                (loss_dis + loss_fs).backward()
                optimizer.step()
                epoch_loss_dis.append(float(loss_dis.detach().cpu()))
                epoch_loss_fs.append(float(loss_fs.detach().cpu()))
            train_rows.append({"epoch": config.epochs + epoch, "loss_dis": np.mean(epoch_loss_dis), "loss_fs": np.mean(epoch_loss_fs), "phase": "branch"})

        val_rows = []
        fs_error = []
        fs_true = []
        fs_predicted = []
        model.eval()
        with torch.inference_mode():
            for x, y, fos_target, mask in val_loader:
                x = x.to(device)
                y = y.to(device)
                fos_target = fos_target.to(device)
                mask = mask.to(device)
                fos_pred, disp_pred = model(x)
                loss_dis = loss_dis_fn(disp_pred, y, mask, topk=0.5)
                err = (fos_pred - fos_target) * (config.fs_scale_max - config.fs_scale_min)
                fs_error.append(float(err.detach().cpu()))
                fs_predicted.append(float((fos_pred * (config.fs_scale_max - config.fs_scale_min) + config.fs_scale_min).detach().cpu()))
                fs_true.append(float((fos_target * (config.fs_scale_max - config.fs_scale_min) + config.fs_scale_min).detach().cpu()))
                val_rows.append({"loss_dis": float(loss_dis.detach().cpu()), "fs_mse": float(loss_fs_fn(fos_pred, fos_target).detach().cpu())})

        fold_dir = config.output_root / f"fold_{fold_idx}"
        _save_frame(train_rows, fold_dir / "train_history.csv")
        _save_frame(val_rows, fold_dir / "validation_metrics.csv")
        pd.DataFrame({"Loss Values": fs_error}).to_excel(fold_dir / "fs_error.xlsx", index=False)
        (fold_dir / "fsV.txt").write_text("\n".join(f"{x:.10g}" for x in fs_true) + "\n", encoding="utf-8")
        pd.DataFrame({"row_index": val_idx, "slope": slope_all[val_idx], "y_hat": fs_predicted, "y_true": fs_true, "error": fs_error}).to_csv(fold_dir / "fs_eval_pairs.csv", index=False)
        save_state_dict_checkpoint(model, fold_dir / f"fold_{fold_idx}UCN_state_dict.pt", fos_range=current_scale)
