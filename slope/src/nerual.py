"""Compatibility alias for checkpoints saved under the historical module name `nerual`."""

from legacy_resunet import (
    ResUNet,
    SpatialAttention3D,
    SpatialAttentionPool3D,
    weights_init_orthogonal,
)

# Keep the historical module name in saved architecture metadata so released
# checkpoints continue to round-trip.
for _cls in (ResUNet, SpatialAttention3D, SpatialAttentionPool3D):
    _cls.__module__ = __name__

__all__ = [
    "ResUNet",
    "SpatialAttention3D",
    "SpatialAttentionPool3D",
    "weights_init_orthogonal",
]
