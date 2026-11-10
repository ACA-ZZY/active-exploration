"""Frozen surrogate recipes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class NetworkRecipe:
    """For problems whose tail lies outside the training envelope."""

    name: str = "network"
    sample_weight: str | None = None
    weight_cap: float = 5.0
    weight_gamma: float = 1.5
    hidden_layer_sizes: tuple[int, ...] = (64, 64, 64)
    alpha: float = 1e-5
    dropout: float = 0.0
    batch_size: int = 128
    learning_rate_init: float = 1e-3
    refine_learning_rate: float | None = None
    max_iter: int = 300
    refine_iter: int = 300
    log_shift_target: bool = False

    @property
    def learning_rate(self) -> float:
        return self.learning_rate_init

    @property
    def weight_decay(self) -> float:
        return self.alpha

@dataclass(frozen=True)
class BoostingRecipe:
    """For problems whose difficulty is picking the relevant inputs."""

    name: str = "boosting"
    max_iter: int = 400
    refine_iter: int = 200
    learning_rate: float = 0.1
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0

@dataclass(frozen=True)
class DeepRecipe:
    """A wide pre-norm residual network with GELU activations"""

    name: str = "deep"
    sample_weight: str | None = None
    weight_cap: float = 5.0
    weight_gamma: float = 1.5
    width: int = 512
    depth: int = 3                  # plain layers, or residual blocks if widths is set
    widths: tuple[int, ...] | None = None
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_iter: int = 800             # epochs in the base call
    refine_iter: int = 300          # epochs in the single refinement call

@dataclass(frozen=True)
class SpectralRecipe:
    """For inputs that are spectral coefficient pairs rather than a flat vector"""

    name: str = "spectral"
    sample_weight: str | None = None
    weight_cap: float = 5.0
    weight_gamma: float = 1.5
    token_width: int = 2            # coefficients per frequency
    embed: int = 64
    depth: int = 3
    head_width: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_iter: int = 600
    refine_iter: int = 300

@dataclass(frozen=True)
class RecurrentRecipe:
    """End to end from the input vector: a learned projection, then a recurrence"""

    name: str = "recurrent"
    n_models: int = 1               # >1 averages that many independently seeded fits
    dropout: float = 0.0
    sample_weight: str | None = None
    weight_cap: float = 5.0
    weight_gamma: float = 1.5
    weight_groups: str = "level"
    steps: int = 125
    readout: str = "last"
    log_target: bool = False
    hidden: int = 64
    layers: int = 2
    learning_rate: float = 3e-3
    refine_learning_rate: float | None = None
    grad_clip: float = 1.0
    batch_size: int = 64
    max_iter: int = 400
    refine_iter: int = 200

@dataclass(frozen=True)
class TemporalAttentionRecipe:
    """Causal attention for a discretised first-passage-time problem"""

    name: str = "temporal_attention"
    d_model: int = 64
    n_heads: int = 4
    layers: int = 2
    feedforward: int = 128
    dropout: float = 0.10
    time_step: float = 0.0045
    n_load_steps: int = 300
    n_time_steps: int = 400
    horizon: float = 1.8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    batch_size: int = 32
    max_iter: int = 300
    refine_iter: int = 200
    sample_weight: str | None = None
    weight_cap: float = 5.0
    weight_gamma: float = 1.5
    weight_groups: str = "level"

RECIPES = {"network": NetworkRecipe(),
           "network_refine_lr1e3_e200": NetworkRecipe(
               refine_learning_rate=1e-3, refine_iter=200),
           "network_16x64x16_refine_lr1e3_e200": NetworkRecipe(
               hidden_layer_sizes=(16, 64, 16),
               refine_learning_rate=1e-3, refine_iter=200),
           "network_flood_refine_lr5e4_e200": NetworkRecipe(
               refine_learning_rate=5e-4, refine_iter=200),
           "network_flood_refine_lr1e3_e200": NetworkRecipe(
               refine_learning_rate=1e-3, refine_iter=200),
           "network_flood_refine_lr1e3_e200_log": NetworkRecipe(
               refine_learning_rate=1e-3, refine_iter=200,
               log_shift_target=True),
           "network_flood_refine_lr1e3_e200_wquantile": NetworkRecipe(
               sample_weight="quantile", refine_learning_rate=1e-3,
               refine_iter=200),
           "network_flood_refine_lr1e3_e200_wsparsity": NetworkRecipe(
               sample_weight="sparsity", refine_learning_rate=1e-3,
               refine_iter=200),
           "network_darcy_lr5e4": NetworkRecipe(
               name="network", learning_rate_init=5e-4),
           "boosting": BoostingRecipe(),
           "deep": DeepRecipe(), "spectral": SpectralRecipe(),
           "recurrent": RecurrentRecipe(refine_learning_rate=5e-4),
           "recurrent_lr5e4": RecurrentRecipe(
               name="recurrent", learning_rate=5e-4),
           "shallow_arch_temporal_attention_lr5e4": TemporalAttentionRecipe(),
           "shallow_arch_physics_attention_lr5e4": TemporalAttentionRecipe(
               n_load_steps=400),
           "recurrent_ensemble": RecurrentRecipe(name="recurrent", n_models=3),
           "recurrent_drop05": RecurrentRecipe(name="recurrent", dropout=0.5),
           "recurrent_concat": RecurrentRecipe(name="recurrent", readout="concat"),
           "recurrent_attention": RecurrentRecipe(name="recurrent",
                                                  readout="attention"),
           "recurrent_log": RecurrentRecipe(name="recurrent", log_target=True),
           "recurrent_log_concat": RecurrentRecipe(name="recurrent",
                                                   log_target=True,
                                                   readout="concat"),
           "recurrent_attn_concat": RecurrentRecipe(name="recurrent",
                                                    readout="attention_concat"),
           "recurrent_log_attn_concat": RecurrentRecipe(
               name="recurrent", log_target=True, readout="attention_concat"),
           "recurrent_sparsity": RecurrentRecipe(name="recurrent",
                                                 sample_weight="sparsity",
                                                 weight_groups="level"),
           "recurrent_sparsity_single": RecurrentRecipe(name="recurrent",
                                                        sample_weight="sparsity",
                                                        weight_groups="single"),
           "recurrent_quantile": RecurrentRecipe(name="recurrent",
                                                 sample_weight="quantile"),
           "network_sparsity": NetworkRecipe(name="network",
                                             sample_weight="sparsity"),
           "network_quantile": NetworkRecipe(name="network",
                                             sample_weight="quantile"),
           "deep_sparsity": DeepRecipe(name="deep", sample_weight="sparsity"),
           "deep_quantile": DeepRecipe(name="deep", sample_weight="quantile")}

def _torch_modules():
    """The network classes, defined at module level so a fitted model pickles"""
    import torch
    from torch import nn

    global _ResidualBlock, _Recurrence, _TokenNet
    if _ResidualBlock is not None:
        return _ResidualBlock, _Recurrence, _TokenNet

    class ResidualBlock(nn.Module):
        """Pre-norm residual block: the identity path carries through."""

        def __init__(self, width: int, dropout: float):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.fc1 = nn.Linear(width, width)
            self.fc2 = nn.Linear(width, width)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

        def forward(self, h):
            z = self.norm(h)
            z = self.fc2(self.drop(self.act(self.fc1(z))))
            return h + z

    class TokenNet(nn.Module):
        """A shared per-token network, then pooling across tokens."""

        def __init__(self, token_width: int, n_tokens: int, embed: int,
                     depth: int, head_width: int):
            super().__init__()
            self.token_width, self.n_tokens = token_width, n_tokens
            layers = [nn.Linear(token_width, embed), nn.GELU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(embed, embed), nn.GELU()]
            self.per_token = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.Linear(2 * embed, head_width), nn.GELU(),
                nn.Linear(head_width, head_width), nn.GELU(),
                nn.Linear(head_width, 1),
            )

        def forward(self, flat):
            t = flat.view(flat.shape[0], self.token_width, self.n_tokens)
            h = self.per_token(t.transpose(1, 2))
            return self.head(torch.cat([h.max(dim=1).values, h.mean(dim=1)], -1))

    class Recurrence(nn.Module):
        """A projection when the case hands over a vector, then the recurrence."""

        def __init__(self, n_inputs: int, steps: int, hidden: int, layers: int,
                     projected: bool, dropout: float = 0.0,
                     readout: str = "last", seq_len: int | None = None):
            super().__init__()
            self.project = nn.Linear(n_inputs, steps) if projected else None
            self.gru = nn.GRU(1, hidden, num_layers=layers, batch_first=True,
                              dropout=float(dropout) if layers > 1 else 0.0)
            self.drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()
            self.readout = readout
            if readout == "attention_concat":
                self.score = nn.Linear(hidden, 1)
                length = int(seq_len if seq_len is not None else steps)
                self.head = nn.Linear(hidden * length, 1)
            elif readout == "attention":
                self.score = nn.Linear(hidden, 1)
                self.head = nn.Linear(hidden, 1)
            elif readout == "concat":
                length = int(seq_len if seq_len is not None else steps)
                self.head = nn.Linear(hidden * length, 1)
            elif readout == "last":
                self.head = nn.Linear(hidden, 1)
            else:
                raise ValueError(f"unknown readout {readout!r}")

        def forward(self, x):
            a = self.project(x).unsqueeze(-1) if self.project is not None else x
            self.gru.flatten_parameters()
            h = self.gru(a)[0]
            if self.readout in ("attention", "attention_concat"):
                w = torch.softmax(self.score(h).squeeze(-1), dim=1)
                g = w.unsqueeze(-1) * h
                h = g.flatten(1) if self.readout == "attention_concat" else g.sum(dim=1)
            elif self.readout == "concat":
                h = h.flatten(1)
            else:
                h = h[:, -1]
            return self.head(self.drop(h))

    for cls in (ResidualBlock, Recurrence, TokenNet):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls

    _ResidualBlock, _Recurrence, _TokenNet = ResidualBlock, Recurrence, TokenNet
    return _ResidualBlock, _Recurrence, _TokenNet

_ResidualBlock = None
_Recurrence = None
_TokenNet = None

def _temporal_attention_module_class():
    """Build and publish the causal time-encoded attention module lazily."""
    import math

    import torch
    from torch import nn

    global _TemporalAttention
    if _TemporalAttention is not None:
        return _TemporalAttention

    class TemporalAttention(nn.Module):
        """Physical-time encoding + causal self-attention + discrete hazards."""

        def __init__(self, input_width: int, *, n_time_steps: int,
                     time_step: float, horizon: float, d_model: int,
                     n_heads: int, layers: int, feedforward: int,
                     dropout: float):
            super().__init__()
            if d_model % 2:
                raise ValueError("temporal attention d_model must be even")
            if d_model % n_heads:
                raise ValueError("temporal attention d_model must divide n_heads")
            if not np.isclose(n_time_steps * time_step, horizon):
                raise ValueError("n_time_steps * time_step must equal horizon")
            self.n_time_steps = int(n_time_steps)
            self.time_step = float(time_step)
            self.horizon = float(horizon)
            self.input_projection = nn.Linear(int(input_width), int(d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=int(d_model), nhead=int(n_heads),
                dim_feedforward=int(feedforward), dropout=float(dropout),
                activation="gelu", batch_first=True, norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers),
                                                 norm=nn.LayerNorm(int(d_model)))
            self.hazard_head = nn.Linear(int(d_model), 1)
            nn.init.constant_(self.hazard_head.bias, -7.0)

            times = (torch.arange(self.n_time_steps, dtype=torch.float32) + 1.0) * self.time_step
            n_pairs = int(d_model) // 2
            low = 1.0 / self.horizon
            high = 1.0 / (2.0 * self.time_step)
            frequencies = torch.logspace(math.log10(low), math.log10(high), n_pairs)
            phases = 2.0 * math.pi * times[:, None] * frequencies[None, :]
            encoding = torch.empty(self.n_time_steps, int(d_model), dtype=torch.float32)
            encoding[:, 0::2] = torch.sin(phases)
            encoding[:, 1::2] = torch.cos(phases)
            encoding[:, 0] = times / self.horizon
            self.register_buffer("physical_time_encoding", encoding[None, :, :],
                                 persistent=True)
            causal = torch.triu(
                torch.ones(self.n_time_steps, self.n_time_steps, dtype=torch.bool),
                diagonal=1,
            )
            self.register_buffer("causal_mask", causal, persistent=False)

        def forward(self, drive):
            if drive.ndim != 3:
                raise ValueError("temporal attention expects (batch, time, features)")
            if drive.shape[1] > self.n_time_steps:
                raise ValueError(
                    f"drive has {drive.shape[1]} steps, maximum is {self.n_time_steps}"
                )
            if drive.shape[1] < self.n_time_steps:
                padding = drive.new_zeros(
                    drive.shape[0], self.n_time_steps - drive.shape[1], drive.shape[2]
                )
                drive = torch.cat((drive, padding), dim=1)
            hidden = self.input_projection(drive) + self.physical_time_encoding
            hidden = self.encoder(hidden, mask=self.causal_mask)
            return self.hazard_head(hidden).squeeze(-1)

    TemporalAttention.__module__ = __name__
    TemporalAttention.__qualname__ = TemporalAttention.__name__
    globals()[TemporalAttention.__name__] = TemporalAttention
    _TemporalAttention = TemporalAttention
    return _TemporalAttention

_TemporalAttention = None

def resolve_device(requested: str | None = None) -> str:
    """Where the torch recipes run"""
    import os

    import torch

    name = (requested or os.environ.get("DEVICE") or "auto").lower()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE asked for cuda; torch reports no device")
    return name

class Surrogate:
    """A surrogate under one declared recipe, with an explicit call history"""

    def __init__(self, seed: int, recipe: str | object,
                 feature_map=None, sequence_map=None, device: str | None = None,
                 adverse: str | None = None):
        if isinstance(recipe, str):
            if recipe not in RECIPES:
                raise ValueError(f"unknown recipe: {recipe}")
            recipe = RECIPES[recipe]
        self.recipe = recipe
        self._feature_map = feature_map
        self._sequence_map = sequence_map
        self._seed = int(seed) % (2**31 - 1)
        self._scaler = None
        self._y_scaler = None
        self._target_shift = None     # frozen by the base fit for shifted-log y
        self._model = self._build()
        self._seq_scale = None       # set by fit; a clone must carry it
        self._seq_center = None      # per-channel for temporal attention
        self._device_request = device
        self._device = None          # resolved lazily: only torch recipes need it
        self._adverse = adverse
        self.training_calls: list[int] = []
        self.weight_report: list[dict] = []
        self.lineage: int = 0

    def _encode_target(self, y: np.ndarray) -> np.ndarray:
        """The scale the loss is computed on."""
        if getattr(self.recipe, "log_shift_target", False):
            if self._target_shift is None:
                raise RuntimeError("shifted-log target has not been initialised by fit")
            shifted = np.asarray(y, dtype=float) - float(self._target_shift)
            if np.any(shifted <= 0):
                raise ValueError(
                    "log_shift_target received a response at or below the frozen shift"
                )
            return np.log(shifted)
        if not getattr(self.recipe, "log_target", False):
            return np.asarray(y, dtype=float)
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("log_target needs a strictly positive response")
        return np.log(y)

    def _decode_target(self, z: np.ndarray) -> np.ndarray:
        """Back to physical units"""
        if getattr(self.recipe, "log_shift_target", False):
            if self._target_shift is None:
                raise RuntimeError("shifted-log target has not been initialised by fit")
            return np.exp(z) + float(self._target_shift)
        if not getattr(self.recipe, "log_target", False):
            return z
        return np.exp(z)

    def _torch_device(self):
        """Resolved on first use, so the tree recipes never import torch."""
        import torch

        if self._device is None:
            self._device = torch.device(resolve_device(self._device_request))
        return self._device

    @property
    def device(self) -> str:
        return str(self._torch_device())

    def _build(self):
        if self.recipe.name in ("deep", "spectral", "recurrent", "network",
                                "temporal_attention"):
            return None          # built lazily, once the input width is known
        if self.recipe.name == "boosting":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                max_iter=self.recipe.max_iter,
                learning_rate=self.recipe.learning_rate,
                max_leaf_nodes=self.recipe.max_leaf_nodes,
                min_samples_leaf=self.recipe.min_samples_leaf,
                l2_regularization=self.recipe.l2_regularization,
                early_stopping=False,
                warm_start=False,
                random_state=self._seed,
            )
        raise ValueError(f"no eager estimator for recipe {self.recipe.name!r}")

    def _unused_mlp(self):
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=self.recipe.hidden_layer_sizes,
            activation="relu",
            alpha=self.recipe.alpha,
            batch_size=self.recipe.batch_size,
            learning_rate_init=self.recipe.learning_rate_init,
            max_iter=self.recipe.max_iter,
            tol=0.0,                     # never stop on the loss plateau
            n_iter_no_change=10**9,      # spend the documented budget
            random_state=self._seed,
        )

    def _shallow_module(self, n_features: int):
        """The declared 64-64-64 ReLU stack, built here rather than by sklearn.

        Same widths, same activation, same optimiser and the same documented
        epoch budget; what changes is that the budget is now spent because this
        loop spends it, and that a declared sampling scheme can reach it.
        """
        import torch
        from torch import nn

        torch.manual_seed(self._seed)
        widths = self.recipe.hidden_layer_sizes
        rate = float(getattr(self.recipe, "dropout", 0.0))
        layers: list[nn.Module] = []
        prev = n_features
        for w in widths:
            layers += [nn.Linear(prev, w), nn.ReLU()]
            if rate:
                layers.append(nn.Dropout(rate))
            prev = w
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _deep_module(self, n_features: int):
        import torch
        from torch import nn

        cfg = self.recipe
        torch.manual_seed(self._seed)

        Block, _, _ = _torch_modules()

        if cfg.widths is None:
            layers: list[nn.Module] = [nn.Linear(n_features, cfg.width), nn.GELU()]
            if cfg.dropout:
                layers.append(nn.Dropout(float(cfg.dropout)))
            for _ in range(cfg.depth - 1):
                layers += [nn.Linear(cfg.width, cfg.width), nn.GELU()]
                if cfg.dropout:
                    layers.append(nn.Dropout(float(cfg.dropout)))
            layers.append(nn.Linear(cfg.width, 1))
            return nn.Sequential(*layers)

        layers: list[nn.Module] = [nn.Linear(n_features, cfg.widths[0])]
        for i, w in enumerate(cfg.widths):
            layers.append(Block(w, cfg.dropout))
            if i + 1 < len(cfg.widths) and cfg.widths[i + 1] != w:
                layers += [nn.LayerNorm(w), nn.GELU(),
                           nn.Linear(w, cfg.widths[i + 1])]
        layers += [nn.LayerNorm(cfg.widths[-1]), nn.GELU(),
                   nn.Linear(cfg.widths[-1], 1)]
        return nn.Sequential(*layers)

    def _spectral_module(self, n_features: int):
        import torch
        from torch import nn

        cfg = self.recipe
        torch.manual_seed(self._seed)
        if n_features % cfg.token_width:
            raise ValueError("input width is not a whole number of tokens")
        n_tokens = n_features // cfg.token_width

        _, _, token_net = _torch_modules()
        return token_net(cfg.token_width, n_tokens, cfg.embed, cfg.depth,
                         cfg.head_width)

    def _temporal_attention_module(self, input_width: int):
        import torch

        cfg = self.recipe
        torch.manual_seed(self._seed)
        module = _temporal_attention_module_class()
        return module(
            input_width,
            n_time_steps=cfg.n_time_steps,
            time_step=cfg.time_step,
            horizon=cfg.horizon,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            layers=cfg.layers,
            feedforward=cfg.feedforward,
            dropout=cfg.dropout,
        )

    def _draw_probabilities(self, y: np.ndarray,
                            groups: np.ndarray | None) -> "np.ndarray | None":
        """The declared scheme's draw probabilities, or ``None`` for equal draws"""
        scheme = getattr(self.recipe, "sample_weight", None)
        if not scheme:
            return None
        if groups is None:
            return None
        from .sample_weights import build, effective_sample_size

        mode = getattr(self.recipe, "weight_groups", "level")
        if mode not in ("level", "single"):
            raise ValueError(f"weight_groups must be 'level' or 'single', got {mode!r}")
        w = build(scheme, y, groups if mode == "level" else None,
                  cap=self.recipe.weight_cap,
                  gamma=self.recipe.weight_gamma, adverse=self._adverse)
        self.weight_report.append({
            "call": len(self.training_calls) + 1,
            "rows": int(np.asarray(y).size),
            "scheme": scheme,
            "groups": 1 if mode == "single" else int(np.unique(groups).size),
            "group_mode": mode,
            "effective_rows": round(effective_sample_size(w), 1),
            "max_over_mean": round(float(w.max()), 2),
        })
        return w / w.sum()

    def _deep_train(self, x: np.ndarray, y: np.ndarray, epochs: int,
                    groups: np.ndarray | None = None,
                    learning_rate: float | None = None) -> None:
        import torch
        from torch import nn

        dev = self._torch_device()
        xs = self._scaler.transform(x).astype(np.float32)
        ys = self._y_scaler.transform(
            self._encode_target(y).reshape(-1, 1)).astype(np.float32)
        xt = torch.from_numpy(xs).to(dev)
        yt = torch.from_numpy(ys).to(dev)
        rate = (self.recipe.learning_rate if learning_rate is None
                else float(learning_rate))
        opt = torch.optim.Adam(self._model.parameters(),
                               lr=rate,
                               weight_decay=self.recipe.weight_decay)
        loss_fn = nn.MSELoss()
        n = xt.shape[0]
        gen = torch.Generator().manual_seed(self._seed)
        probs = self._draw_probabilities(y, groups)
        pt = None if probs is None else torch.from_numpy(probs.astype(np.float64))
        self._model.train()
        for _ in range(epochs):
            order = (torch.multinomial(pt, n, replacement=True, generator=gen)
                     if pt is not None
                     else torch.randperm(n, generator=gen)).to(dev)
            for i in range(0, n, self.recipe.batch_size):
                idx = order[i : i + self.recipe.batch_size]
                opt.zero_grad()
                loss_fn(self._model(xt[idx]), yt[idx]).backward()
                opt.step()

    def _recurrent_module(self, n_inputs: int, projected: bool,
                          seq_len: int | None = None):
        """``projected`` builds its own sequence from the input vector"""
        import torch

        torch.manual_seed(self._seed)
        _, recurrence, _ = _torch_modules()
        return recurrence(n_inputs, self.recipe.steps, self.recipe.hidden,
                          self.recipe.layers, projected,
                          getattr(self.recipe, "dropout", 0.0),
                          getattr(self.recipe, "readout", "last"), seq_len)

    def _driving(self, x: np.ndarray):
        """The sequence the recurrence reads, and whether it must build it."""
        if self._sequence_map is None:
            return x, True
        a = np.asarray(self._sequence_map(x), dtype=float)
        return (a[:, :, None] if a.ndim == 2 else a), False

    def _scale_drive(self, drive: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            return self._scaler.transform(drive).astype(np.float32)
        return (drive / self._seq_scale).astype(np.float32)

    def _recurrent_train(self, x: np.ndarray, y: np.ndarray, epochs: int,
                         groups: np.ndarray | None = None,
                         *, learning_rate: float | None = None) -> None:
        import torch
        from torch import nn

        dev = self._torch_device()
        xt = torch.from_numpy(self._scale_drive(x)).to(dev)
        yt = torch.from_numpy(
            self._y_scaler.transform(
                self._encode_target(y).reshape(-1, 1)).astype(np.float32)).to(dev)
        loss_fn = nn.MSELoss()
        n = xt.shape[0]
        p = self._draw_probabilities(y, groups)
        probs = None if p is None else torch.from_numpy(p.astype(np.float64))
        members = self._model if isinstance(self._model, list) else [self._model]
        for k, member in enumerate(members):
            opt = torch.optim.Adam(member.parameters(),
                                   lr=self.recipe.learning_rate if learning_rate is None else learning_rate)
            gen = torch.Generator().manual_seed(self._seed + 1000 * k)
            member.train()
            for _ in range(epochs):
                order = (torch.multinomial(probs, n, replacement=True, generator=gen)
                         if probs is not None else torch.randperm(n, generator=gen)).to(dev)
                for i in range(0, n, self.recipe.batch_size):
                    idx = order[i : i + self.recipe.batch_size]
                    opt.zero_grad()
                    loss_fn(member(xt[idx]), yt[idx]).backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(),
                                                   self.recipe.grad_clip)
                    opt.step()

    def _recurrent_predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        xs = self._scale_drive(self._driving(x)[0])
        members = self._model if isinstance(self._model, list) else [self._model]
        for member in members:
            member.eval()
        out = []
        with torch.no_grad():
            for i in range(0, xs.shape[0], 512):
                chunk = torch.from_numpy(xs[i : i + 512]).to(self._torch_device())
                out.append(np.mean(
                    [self._decode_target(self._y_scaler.inverse_transform(
                        m(chunk).cpu().numpy())) for m in members], axis=0))
        arr = np.concatenate(out) if out else np.zeros((0, 1), dtype=float)
        return np.asarray(arr, dtype=float).reshape(-1)

    def _temporal_attention_drive(self, x: np.ndarray) -> np.ndarray:
        """Return the published forcing tokens read by temporal attention."""
        raw = self._features(np.atleast_2d(np.asarray(x, dtype=float)))
        drive = (raw if self._sequence_map is None
                 else np.asarray(self._sequence_map(raw), dtype=float))
        if drive.ndim == 2:
            drive = drive[:, :, None]
        if drive.ndim != 3:
            raise ValueError("temporal attention sequence_map must return 2D or 3D")
        if drive.shape[1] != int(self.recipe.n_load_steps):
            raise ValueError(
                f"temporal attention expected {self.recipe.n_load_steps} load steps, "
                f"got {drive.shape[1]}"
            )
        if drive.shape[1] > int(self.recipe.n_time_steps):
            raise ValueError("n_load_steps cannot exceed n_time_steps")
        return np.asarray(drive, dtype=float)

    def _temporal_attention_train(self, drive: np.ndarray, y: np.ndarray,
                                  epochs: int,
                                  groups: np.ndarray | None = None) -> None:
        """Optimise the right-censoring-aware discrete survival likelihood."""
        import torch
        from torch.nn import functional as F

        dev = self._torch_device()
        xt = torch.from_numpy(
            ((drive - self._seq_center) / self._seq_scale).astype(np.float32)
        ).to(dev)
        cfg = self.recipe
        y_array = np.asarray(y, dtype=np.float32)
        if (np.any(~np.isfinite(y_array))
                or np.any(y_array < 0.5 * float(cfg.time_step))
                or np.any(y_array > float(cfg.horizon) + 1e-7)):
            raise ValueError(
                "temporal attention targets must be finite first-passage times "
                f"inside [{cfg.time_step}, {cfg.horizon}]"
            )
        yt = torch.from_numpy(y_array).to(dev)
        event = yt < (float(cfg.horizon) - 0.5 * float(cfg.time_step))
        event_index = torch.clamp(
            torch.round(yt / float(cfg.time_step)).long() - 1,
            min=0,
            max=int(cfg.n_time_steps) - 1,
        )
        opt = torch.optim.Adam(
            self._model.parameters(), lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        n = xt.shape[0]
        gen = torch.Generator().manual_seed(self._seed)
        p = self._draw_probabilities(np.asarray(y, dtype=float), groups)
        probabilities = None if p is None else torch.from_numpy(p.astype(np.float64))
        self._model.train()
        for _ in range(int(epochs)):
            order = (torch.multinomial(probabilities, n, replacement=True, generator=gen)
                     if probabilities is not None else torch.randperm(n, generator=gen)).to(dev)
            for start in range(0, n, int(cfg.batch_size)):
                idx = order[start : start + int(cfg.batch_size)]
                logits = self._model(xt[idx])
                log_survival = F.logsigmoid(-logits)
                cumulative_survival = torch.cumsum(log_survival, dim=1)
                local_event = event[idx]
                local_bin = event_index[idx]
                row = torch.arange(idx.numel(), device=dev)
                survival_through_bin = cumulative_survival[row, local_bin]
                survival_before_bin = (
                    survival_through_bin - log_survival[row, local_bin]
                )
                event_nll = -(
                    survival_before_bin + F.logsigmoid(logits[row, local_bin])
                )
                censored_nll = -cumulative_survival[:, -1]
                loss = torch.where(local_event, event_nll, censored_nll).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), cfg.grad_clip)
                opt.step()

    def _temporal_attention_predict(self, drive: np.ndarray) -> np.ndarray:
        """Expected censored event time implied by the learned hazards."""
        import torch

        xs = ((drive - self._seq_center) / self._seq_scale).astype(np.float32)
        self._model.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, xs.shape[0], 256):
                chunk = torch.from_numpy(xs[start : start + 256]).to(self._torch_device())
                hazards = torch.sigmoid(self._model(chunk))
                survival_before = torch.cat((
                    torch.ones(hazards.shape[0], 1, device=hazards.device),
                    torch.cumprod(1.0 - hazards[:, :-1], dim=1),
                ), dim=1)
                expected = float(self.recipe.time_step) * survival_before.sum(dim=1)
                out.append(expected.cpu().numpy())
        return (np.concatenate(out).astype(float) if out
                else np.zeros(0, dtype=float))

    def _features(self, x: np.ndarray) -> np.ndarray:
        return x if self._feature_map is None else np.asarray(
            self._feature_map(x), dtype=float)

    def fit(self, x: np.ndarray, y: np.ndarray,
            groups: np.ndarray | None = None) -> "Surrogate":
        from sklearn.preprocessing import StandardScaler

        y = np.asarray(y, dtype=float).reshape(-1)
        self._target_shift = (
            float(np.min(y) - 1.0)
            if getattr(self.recipe, "log_shift_target", False)
            else None
        )
        if self.recipe.name == "temporal_attention":
            drive = self._temporal_attention_drive(x)
            self._scaler = None
            self._y_scaler = None
            self._seq_center = drive.mean(axis=(0, 1), keepdims=True)
            self._seq_scale = drive.std(axis=(0, 1), keepdims=True)
            self._seq_scale = np.where(self._seq_scale > 0, self._seq_scale, 1.0)
            self._model = self._temporal_attention_module(drive.shape[2]).to(
                self._torch_device()
            )
            self._temporal_attention_train(drive, y, self.recipe.max_iter, groups)
            self.training_calls.append(drive.shape[0])
            self.lineage = drive.shape[0]
            return self
        if self.recipe.name == "recurrent":
            raw = self._features(np.atleast_2d(np.asarray(x, dtype=float)))
            drive, projected = self._driving(raw)
            if projected:
                self._scaler = StandardScaler().fit(drive)
                self._seq_scale = 1.0
            else:
                self._scaler = None
                self._seq_scale = float(drive.std()) or 1.0
            self._y_scaler = StandardScaler().fit(
                self._encode_target(y).reshape(-1, 1))
            n_models = max(1, int(getattr(self.recipe, "n_models", 1)))
            base_seed = self._seed
            self._model = []
            for k in range(n_models):
                self._seed = (base_seed + 1000 * k) % (2**31 - 1)
                member = self._recurrent_module(raw.shape[1], projected,
                                                seq_len=drive.shape[1])
                self._model.append(member.to(self._torch_device()))
            self._seed = base_seed
            self._recurrent_train(drive, y, self.recipe.max_iter, groups)
            self.training_calls.append(raw.shape[0])
            self.lineage = raw.shape[0]
            return self
        x = self._features(np.atleast_2d(np.asarray(x, dtype=float)))
        if self.recipe.name in ("deep", "spectral", "network"):
            self._scaler = StandardScaler().fit(x)
            self._y_scaler = StandardScaler().fit(
                self._encode_target(y).reshape(-1, 1))
            builder = {"spectral": self._spectral_module,
                       "network": self._shallow_module}.get(
                           self.recipe.name, self._deep_module)
            self._model = builder(x.shape[1]).to(self._torch_device())
            self._deep_train(x, y, self.recipe.max_iter, groups)
            self.training_calls.append(x.shape[0])
            self.lineage = x.shape[0]
            return self
        else:
            self._model.fit(x, y)
        self.training_calls.append(x.shape[0])
        self.lineage = x.shape[0]
        return self

    def continue_fit(self, x: np.ndarray, y: np.ndarray, new_labels: int,
                     groups: np.ndarray | None = None) -> "Surrogate":
        """One warm-started refinement call"""
        y = np.asarray(y, dtype=float).reshape(-1)
        if self.recipe.name == "temporal_attention":
            drive = self._temporal_attention_drive(x)
            self._temporal_attention_train(drive, y, self.recipe.refine_iter, groups)
            self.training_calls.append(drive.shape[0])
            self.lineage += int(new_labels)
            return self
        if self.recipe.name == "recurrent":
            raw = self._features(np.atleast_2d(np.asarray(x, dtype=float)))
            self._recurrent_train(self._driving(raw)[0], y, self.recipe.refine_iter,
                                  groups, learning_rate=getattr(self.recipe, "refine_learning_rate", None))
            self.training_calls.append(raw.shape[0])
            self.lineage += int(new_labels)
            return self
        x = self._features(np.atleast_2d(np.asarray(x, dtype=float)))
        if self.recipe.name in ("deep", "spectral", "network"):
            self._deep_train(
                x, y, self.recipe.refine_iter, groups,
                learning_rate=getattr(self.recipe, "refine_learning_rate", None),
            )
            self.training_calls.append(x.shape[0])
            self.lineage += int(new_labels)
            return self
        else:
            self._model.set_params(
                warm_start=True,
                max_iter=self.recipe.max_iter + self.recipe.refine_iter,
            )
            self._model.fit(x, y)
        self.training_calls.append(x.shape[0])
        self.lineage += int(new_labels)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.recipe.name == "temporal_attention":
            return self._temporal_attention_predict(self._temporal_attention_drive(x))
        if self.recipe.name == "recurrent":
            return self._recurrent_predict(
                self._features(np.atleast_2d(np.asarray(x, dtype=float))))
        x = self._features(np.atleast_2d(np.asarray(x, dtype=float)))
        if self.recipe.name in ("deep", "spectral", "network"):
            import torch

            self._model.eval()
            xs = self._scaler.transform(x).astype(np.float32)
            out = []
            with torch.no_grad():
                for i in range(0, xs.shape[0], 4096):
                    chunk = torch.from_numpy(xs[i : i + 4096]).to(self._torch_device())
                    out.append(self._model(chunk).cpu().numpy())
            arr = np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)
            return self._decode_target(
                self._y_scaler.inverse_transform(arr)).reshape(-1)
        return np.asarray(self._model.predict(x), dtype=float).reshape(-1)

    def _require_stochastic_dropout(self) -> None:
        """Reject deterministic legacy selectors instead of ranking roundoff noise."""
        from torch import nn

        members = self._model if isinstance(self._model, list) else [self._model]
        for member in members:
            if not any(isinstance(layer, nn.Dropout) and 0.0 < layer.p < 1.0
                       for layer in member.modules()):
                raise RuntimeError(
                    "MC-Dropout requires active Dropout modules in every selection "
                    "network. A legacy checkpoint without Dropout must be retrained; "
                    "changing its recipe alone does not repair its acquisition history."
                )

    @staticmethod
    def _stochastic_moments(draws: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Float64 sample moments, exactly zero variance for identical draws."""
        stacked = np.asarray(draws, dtype=np.float64)
        if stacked.shape[0] < 2:
            raise ValueError("a variance needs at least two stochastic passes")
        if not np.isfinite(stacked).all():
            raise FloatingPointError("nonfinite MC-Dropout output")
        centered = stacked - stacked[0]
        offset = centered.mean(axis=0, dtype=np.float64)
        variance = np.sum((centered - offset) ** 2, axis=0, dtype=np.float64)
        return stacked[0] + offset, variance / (stacked.shape[0] - 1)

    def _set_dropout_mode(self, enabled: bool) -> None:
        """Switch only the Dropout modules; every other layer stays in eval."""
        from torch import nn

        members = self._model if isinstance(self._model, list) else [self._model]
        for member in members:
            for module in member.modules():
                if isinstance(module, nn.Dropout):
                    module.train(enabled)

    def predict_stochastic(
        self, x: np.ndarray, *, passes: int = 30, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance over ``passes`` dropout-active forward passes.

        This is the acquisition-time estimator of the Monte-Carlo dropout arm.  The
        dropout layers are put in training mode and everything else is left in eval,
        so no running statistic is disturbed; each pass reseeds torch from the frozen
        acquisition seed, which makes the thirty draws reproducible.
        """
        import torch

        if self.recipe.name == "recurrent":
            return self._recurrent_predict_stochastic(
                x, passes=passes, seed=seed
            )
        if self.recipe.name not in ("deep", "spectral", "network"):
            raise RuntimeError(
                f"stochastic prediction is not defined for recipe {self.recipe.name!r}"
            )
        if not float(getattr(self.recipe, "dropout", 0.0)):
            raise RuntimeError(
                "stochastic prediction needs a recipe whose dropout rate is above zero"
            )
        if int(passes) < 2:
            raise ValueError("a variance needs at least two stochastic passes")
        self._require_stochastic_dropout()

        xf = self._features(np.atleast_2d(np.asarray(x, dtype=float)))
        self._model.eval()
        xs = self._scaler.transform(xf).astype(np.float32)
        draws: list[np.ndarray] = []
        try:
            with torch.no_grad():
                for index in range(int(passes)):
                    if seed is not None:
                        torch.manual_seed((int(seed) + index) % (2 ** 31 - 1))
                    self._set_dropout_mode(True)
                    chunks = []
                    for start in range(0, xs.shape[0], 4096):
                        block = torch.from_numpy(xs[start:start + 4096]).to(self._torch_device())
                        chunks.append(self._model(block).cpu().numpy())
                    raw = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)
                    physical = self._y_scaler.inverse_transform(raw.astype(np.float64))
                    draws.append(self._decode_target(physical).reshape(-1))
        finally:
            self._set_dropout_mode(False)
        return self._stochastic_moments(draws)

    def _recurrent_predict_stochastic(
        self, x: np.ndarray, *, passes: int = 30, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance over ``passes`` dropout-active passes of the recurrence.

        Mirrors ``_recurrent_predict`` exactly -- same driving sequence, same
        per-member decoding and the same mean-across-members -- but leaves the
        dropout layers active so each pass is a fresh stochastic draw.  The
        dropout sits on the GRU output before the linear head, which is the
        Gal & Ghahramani (2016) prescription for recurrent MC-dropout: dropout on
        inputs/outputs, never on the recurrent connection itself.
        """
        import torch

        if not float(getattr(self.recipe, "dropout", 0.0)):
            raise RuntimeError(
                "stochastic prediction needs a recipe whose dropout rate is above zero"
            )
        if int(passes) < 2:
            raise ValueError("a variance needs at least two stochastic passes")
        self._require_stochastic_dropout()
        members = self._model if isinstance(self._model, list) else [self._model]
        for member in members:
            member.eval()
        xs = self._scale_drive(self._driving(x)[0])
        draws: list[np.ndarray] = []
        try:
            with torch.no_grad():
                for index in range(int(passes)):
                    if seed is not None:
                        torch.manual_seed((int(seed) + index) % (2 ** 31 - 1))
                    self._set_dropout_mode(True)
                    out = []
                    for start in range(0, xs.shape[0], 512):
                        block = torch.from_numpy(
                            xs[start : start + 512]).to(self._torch_device())
                        out.append(np.mean(
                            [self._decode_target(self._y_scaler.inverse_transform(
                                m(block).cpu().numpy().astype(np.float64))) for m in members], axis=0))
                    arr = np.concatenate(out) if out else np.zeros((0, 1), dtype=float)
                    draws.append(np.asarray(arr, dtype=np.float64).reshape(-1))
        finally:
            self._set_dropout_mode(False)
        return self._stochastic_moments(draws)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)

    def __setstate__(self, state):
        """Tolerate a checkpoint written before an attribute existed"""
        self.__dict__.update(state)
        for name, default in (("_device_request", None), ("_device", None),
                              ("_feature_map", None), ("_sequence_map", None),
                              ("_seq_scale", None), ("_seq_center", None),
                              ("_target_shift", None),
                              ("lineage", 0),
                              ("_adverse", None)):
            if not hasattr(self, name):
                setattr(self, name, default)
        if not hasattr(self, "training_calls"):
            self.training_calls = []

    def clone_frozen(self, seed: int) -> "Surrogate":
        """A deep copy, so acquisition cannot mutate the frozen base learner."""
        import copy

        twin = Surrogate(seed=seed, recipe=self.recipe,
                         feature_map=self._feature_map,
                         sequence_map=self._sequence_map,
                         device=self._device_request,
                         adverse=self._adverse)
        twin._model = copy.deepcopy(self._model)
        twin._scaler = copy.deepcopy(self._scaler)
        twin._y_scaler = copy.deepcopy(self._y_scaler)
        twin._seq_scale = copy.deepcopy(self._seq_scale)
        twin._seq_center = copy.deepcopy(self._seq_center)
        twin._target_shift = self._target_shift
        twin.training_calls = list(self.training_calls)
        twin.weight_report = list(getattr(self, "weight_report", []))
        twin.lineage = self.lineage
        return twin

def __getattr__(name):
    if name in {"ResidualBlock", "Recurrence", "TokenNet"}:
        _torch_modules()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
