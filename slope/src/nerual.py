from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init_orthogonal(m):
    if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
        if m.weight is not None:
            init.orthogonal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Linear):
        if m.weight is not None:
            init.orthogonal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

class SpatialAttention3D(nn.Module):

    def __init__(self, in_c, mid=16):
        super().__init__()
        mid = max(4, min(mid, in_c))
        self.att = nn.Sequential(nn.Conv3d(in_c, mid, kernel_size=3, padding=1, bias=True), nn.PReLU(mid), nn.Conv3d(mid, 1, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x):
        a = self.att(x).clamp_min(0.1)
        return x * a

class SpatialAttentionPool3D(nn.Module):

    def __init__(self, in_c, mid=16, pool=(2, 2, 2), eps=1e-06):
        super().__init__()
        self.prepool = nn.AvgPool3d(pool, pool)
        self.att = nn.Sequential(nn.Conv3d(in_c, mid, 3, padding=1, bias=True), nn.PReLU(mid), nn.Conv3d(mid, 1, 1, bias=True), nn.Sigmoid())
        self.eps = eps

    def forward(self, x):
        x = self.prepool(x)
        a = self.att(x).clamp_min(1e-06)
        num = (a * x).sum(dim=(2, 3, 4))
        den = a.sum(dim=(2, 3, 4)) + self.eps
        return (num / den, a)

class ResUNet(nn.Module):

    def __init__(self, in_channel=1, out_channel=2, training=True):
        super().__init__()
        self.training = training

        def conv_block(in_c, out_c, k=3, s=1, p=1):
            return nn.Sequential(nn.Conv3d(in_c, out_c, k, s, p), nn.PReLU(out_c))
        self.input_in = conv_block(in_channel, 16)
        self.encoder_stage1 = nn.Sequential(conv_block(16, 16), conv_block(16, 16), conv_block(16, 16))
        self.encoder_stage2 = nn.Sequential(conv_block(32, 32), conv_block(32, 32), conv_block(32, 32))
        self.encoder_stage3 = nn.Sequential(conv_block(64, 64), conv_block(64, 64), conv_block(64, 64))
        self.encoder_stage4 = nn.Sequential(conv_block(128, 128), conv_block(128, 128), conv_block(128, 128))
        self.down_conv1 = nn.Sequential(nn.Conv3d(16, 32, 2, 2), nn.PReLU(32))
        self.down_conv2 = nn.Sequential(nn.Conv3d(32, 64, 2, 2), nn.PReLU(64))
        self.down_conv3 = nn.Sequential(nn.Conv3d(64, 128, 2, 2), nn.PReLU(128))
        self.dis_stage1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128), conv_block(128, 128), SpatialAttention3D(128))
        self.dis_stage2 = nn.Sequential(conv_block(64, 64), conv_block(64, 64), conv_block(64, 64), SpatialAttention3D(64))
        self.dis_stage3 = nn.Sequential(conv_block(32, 32), conv_block(32, 32), conv_block(32, 32), SpatialAttention3D(32))
        self.dis_stage4 = nn.Sequential(conv_block(16, 16), conv_block(16, 16), conv_block(16, 16), SpatialAttention3D(16))
        self.dis_map4 = nn.Conv3d(16, out_channel, 1, 1)

        def upsample_block(in_c, out_c):
            return nn.Sequential(nn.ConvTranspose3d(in_c, out_c, 2, 2), nn.PReLU(out_c))
        self.dis_conv2 = upsample_block(128, 64)
        self.dis_conv3 = upsample_block(64, 32)
        self.dis_conv4 = upsample_block(32, 16)
        self.fs_blocks = nn.ModuleDict({'fs1': nn.Sequential(nn.Conv3d(128, 32, 1, bias=True), nn.PReLU(32), SpatialAttention3D(32, mid=16)), 'fs2': nn.Sequential(nn.AvgPool3d(kernel_size=2, stride=2), nn.Conv3d(64, 32, 1, bias=True), nn.PReLU(32), SpatialAttention3D(32, mid=16)), 'fs3': nn.Sequential(nn.AvgPool3d(kernel_size=4, stride=4), nn.Conv3d(32, 32, 1, bias=True), nn.PReLU(32), SpatialAttention3D(32, mid=16)), 'fs4': nn.Sequential(nn.AvgPool3d(kernel_size=8, stride=8), nn.Conv3d(16, 32, 1, bias=True), nn.PReLU(32), SpatialAttention3D(32, mid=16))})
        self.fs_pool = SpatialAttentionPool3D(in_c=32, mid=16)
        self.fs_fc = nn.Linear(32, 1)
        self.fs_alpha = nn.Parameter(torch.zeros(4))
        self.apply(weights_init_orthogonal)

    def forward(self, inputs, training=True, fs=True, dis=True):
        x0 = self.input_in(inputs)
        long_range1 = self.encoder_stage1(x0)
        short_range1 = self.down_conv1(long_range1 + x0)
        long_range2 = self.encoder_stage2(short_range1)
        short_range2 = self.down_conv2(long_range2 + short_range1)
        long_range3 = self.encoder_stage3(short_range2)
        short_range3 = self.down_conv3(long_range3 + short_range2)
        long_range4 = self.encoder_stage4(short_range3)
        if fs:
            f1 = self.fs_blocks['fs1'](long_range4)
            f2 = self.fs_blocks['fs2'](long_range3)
            f3 = self.fs_blocks['fs3'](long_range2)
            f4 = self.fs_blocks['fs4'](long_range1)
            w = torch.softmax(self.fs_alpha, dim=0)
            f_stack = torch.stack([f1, f2, f3, f4], dim=1)
            f_weighted = (f_stack * w.view(1, 4, 1, 1, 1, 1)).sum(dim=1)
            vec, fs_att = self.fs_pool(f_weighted)
            fs_mu = self.fs_fc(vec).view(-1)
        if dis:
            y = self.dis_stage1(long_range4 + short_range3) + long_range4
            y = self.dis_conv2(y)
            y = self.dis_stage2(y + short_range2) + y
            y = self.dis_conv3(y)
            y = self.dis_stage3(y + short_range1) + y
            y = self.dis_conv4(y)
            outputs = self.dis_map4(self.dis_stage4(y + x0) + y)
        return [fs_mu, outputs]
