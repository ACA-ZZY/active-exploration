from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init_orthogonal(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        if module.weight is not None:
            init.orthogonal_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

class OHEM(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        topk: float = 0.5,
    ) -> torch.Tensor:
        diff = pred - target
        sq_err = diff * diff + 1e-8
        batch = sq_err.size(0)
        sq_flat = sq_err.view(batch, -1)
        valid_flat = ~mask.view(batch, -1).bool()
        losses: list[torch.Tensor] = []
        for item in range(batch):
            valid_err = sq_flat[item, valid_flat[item]].sqrt()
            if valid_err.numel() == 0:
                losses.append(torch.zeros((), device=pred.device, dtype=pred.dtype))
                continue
            k = max(1, int(valid_err.numel() * float(topk)))
            losses.append(torch.topk(valid_err, k).values.mean())
        return torch.stack(losses).mean()

class SpatialAttention3D(nn.Module):
    def __init__(self, in_channels: int | None = None, mid_channels: int = 16, **legacy_kwargs):
        super().__init__()
        if in_channels is None:
            in_channels = legacy_kwargs.pop("in_c")
        if "mid" in legacy_kwargs:
            mid_channels = legacy_kwargs.pop("mid")
        if legacy_kwargs:
            raise TypeError(f"Unsupported keyword arguments: {sorted(legacy_kwargs)}")
        mid = max(4, min(int(mid_channels), int(in_channels)))
        self.att = nn.Sequential(
            nn.Conv3d(int(in_channels), mid, kernel_size=3, padding=1, bias=True),
            nn.PReLU(mid),
            nn.Conv3d(mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.att(x).clamp_min(1e-1)
        return x * gate

class SpatialAttentionPool3D(nn.Module):
    def __init__(
        self,
        in_channels: int | None = None,
        mid_channels: int = 16,
        pool: tuple[int, int, int] = (2, 2, 2),
        eps: float = 1e-6,
        **legacy_kwargs,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = legacy_kwargs.pop("in_c")
        if "mid" in legacy_kwargs:
            mid_channels = legacy_kwargs.pop("mid")
        if legacy_kwargs:
            raise TypeError(f"Unsupported keyword arguments: {sorted(legacy_kwargs)}")
        self.prepool = nn.AvgPool3d(pool, pool)
        self.att = nn.Sequential(
            nn.Conv3d(int(in_channels), int(mid_channels), 3, padding=1, bias=True),
            nn.PReLU(int(mid_channels)),
            nn.Conv3d(int(mid_channels), 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.prepool(x)
        gate = self.att(x).clamp_min(1e-6)
        numerator = (gate * x).sum(dim=(2, 3, 4))
        denominator = gate.sum(dim=(2, 3, 4)) + self.eps
        return numerator / denominator, gate

def _conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        nn.PReLU(out_channels),
    )

def _upsample_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
        nn.PReLU(out_channels),
    )

class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        in_channel: int | None = None,
        out_channel: int | None = None,
        training: bool = True,
    ):
        super().__init__()
        del training
        if in_channels is None:
            in_channels = 1 if in_channel is None else in_channel
        if out_channels is None:
            out_channels = 2 if out_channel is None else out_channel

        self.input_in = _conv_block(int(in_channels), 16)

        self.encoder_stage1 = nn.Sequential(_conv_block(16, 16), _conv_block(16, 16), _conv_block(16, 16))
        self.encoder_stage2 = nn.Sequential(_conv_block(32, 32), _conv_block(32, 32), _conv_block(32, 32))
        self.encoder_stage3 = nn.Sequential(_conv_block(64, 64), _conv_block(64, 64), _conv_block(64, 64))
        self.encoder_stage4 = nn.Sequential(_conv_block(128, 128), _conv_block(128, 128), _conv_block(128, 128))

        self.down_conv1 = nn.Sequential(nn.Conv3d(16, 32, 2, 2), nn.PReLU(32))
        self.down_conv2 = nn.Sequential(nn.Conv3d(32, 64, 2, 2), nn.PReLU(64))
        self.down_conv3 = nn.Sequential(nn.Conv3d(64, 128, 2, 2), nn.PReLU(128))

        self.dis_stage1 = nn.Sequential(_conv_block(128, 128), _conv_block(128, 128), _conv_block(128, 128), SpatialAttention3D(128))
        self.dis_stage2 = nn.Sequential(_conv_block(64, 64), _conv_block(64, 64), _conv_block(64, 64), SpatialAttention3D(64))
        self.dis_stage3 = nn.Sequential(_conv_block(32, 32), _conv_block(32, 32), _conv_block(32, 32), SpatialAttention3D(32))
        self.dis_stage4 = nn.Sequential(_conv_block(16, 16), _conv_block(16, 16), _conv_block(16, 16), SpatialAttention3D(16))
        self.dis_map4 = nn.Conv3d(16, int(out_channels), 1, 1)

        self.dis_conv2 = _upsample_block(128, 64)
        self.dis_conv3 = _upsample_block(64, 32)
        self.dis_conv4 = _upsample_block(32, 16)

        self.fs_blocks = nn.ModuleDict(
            {
                "fs1": nn.Sequential(nn.Conv3d(128, 32, 1, bias=True), nn.PReLU(32), SpatialAttention3D(32, mid_channels=16)),
                "fs2": nn.Sequential(nn.AvgPool3d(2, 2), nn.Conv3d(64, 32, 1, bias=True), nn.PReLU(32), SpatialAttention3D(32, mid_channels=16)),
                "fs3": nn.Sequential(nn.AvgPool3d(4, 4), nn.Conv3d(32, 32, 1, bias=True), nn.PReLU(32), SpatialAttention3D(32, mid_channels=16)),
                "fs4": nn.Sequential(nn.AvgPool3d(8, 8), nn.Conv3d(16, 32, 1, bias=True), nn.PReLU(32), SpatialAttention3D(32, mid_channels=16)),
            }
        )
        self.fs_pool = SpatialAttentionPool3D(in_channels=32, mid_channels=16)
        self.fs_fc = nn.Linear(32, 1)
        self.fs_alpha = nn.Parameter(torch.zeros(4))

        self.apply(weights_init_orthogonal)

    def forward(self, inputs: torch.Tensor, training: bool = True, fs: bool = True, dis: bool = True):
        del training, fs, dis
        x0 = self.input_in(inputs)

        long_range1 = self.encoder_stage1(x0)
        short_range1 = self.down_conv1(long_range1 + x0)

        long_range2 = self.encoder_stage2(short_range1)
        short_range2 = self.down_conv2(long_range2 + short_range1)

        long_range3 = self.encoder_stage3(short_range2)
        short_range3 = self.down_conv3(long_range3 + short_range2)

        long_range4 = self.encoder_stage4(short_range3)

        f1 = self.fs_blocks["fs1"](long_range4)
        f2 = self.fs_blocks["fs2"](long_range3)
        f3 = self.fs_blocks["fs3"](long_range2)
        f4 = self.fs_blocks["fs4"](long_range1)
        weights = torch.softmax(self.fs_alpha, dim=0)
        f_stack = torch.stack([f1, f2, f3, f4], dim=1)
        f_weighted = (f_stack * weights.view(1, 4, 1, 1, 1, 1)).sum(dim=1)
        vec, _ = self.fs_pool(f_weighted)
        fs_mu = self.fs_fc(vec).view(-1)

        y = self.dis_stage1(long_range4 + short_range3) + long_range4
        y = self.dis_conv2(y)
        y = self.dis_stage2(y + short_range2) + y
        y = self.dis_conv3(y)
        y = self.dis_stage3(y + short_range1) + y
        y = self.dis_conv4(y)
        outputs = self.dis_map4(self.dis_stage4(y + x0) + y)

        return [fs_mu, outputs]
