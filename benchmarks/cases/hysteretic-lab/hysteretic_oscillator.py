"""Hysteretic oscillator response function for the hysteretic-lab benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class HystereticSpec:
    omega0: float = 5.69            # rad/s, elastic natural frequency
    damping: float = 0.02           # zeta
    alpha: float = -0.20            # post-yield stiffness ratio; negative is P-Delta
    yield_displacement: float = 0.15   # xy, metres
    bouc_A: float = 1.0
    bouc_beta: float = 0.5
    bouc_gamma: float = 0.5
    bouc_n: int = 3
    spectral_intensity: float = 0.015  # S, one-sided, m^2/s^3
    n_frequencies: int = 24         # input dimension is twice this
    duration: float = 10.0          # s
    n_steps: int = 500              # integration steps, not an input dimension

    @property
    def dimension(self) -> int:
        return 2 * self.n_frequencies

    @property
    def dt(self) -> float:
        return self.duration / self.n_steps

    @property
    def frequency_spacing(self) -> float:
        return 1.0 / self.duration

    @property
    def cutoff_frequency(self) -> float:
        return self.n_frequencies * self.frequency_spacing

    @property
    def natural_frequency_hz(self) -> float:
        return self.omega0 / (2.0 * np.pi)

    @property
    def elastic_peak_std(self) -> float:
        """Stationary rms displacement of the purely elastic oscillator."""
        return float(np.sqrt(np.pi * self.spectral_intensity
                             / (2.0 * self.damping * self.omega0**3)))

DEFAULT_SPEC = HystereticSpec()

@dataclass(frozen=True)
class HystereticBatch:
    peak_displacement: Array
    yielded: Array                  # whether |z| ever passed the yield knee
    ductility: Array                # peak |x| / xy
    residual_drift: Array

def excitation(xi: Array, spec: HystereticSpec = DEFAULT_SPEC) -> Array:
    """Spectral representation of the band-limited excitation."""
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    if xi.shape[1] != spec.dimension:
        raise ValueError(f"expected {spec.dimension} inputs per row")
    cosine, sine = xi[:, : spec.n_frequencies], xi[:, spec.n_frequencies :]
    frequencies = np.arange(1, spec.n_frequencies + 1) * spec.frequency_spacing
    time = np.arange(spec.n_steps) * spec.dt
    amplitude = np.sqrt(2.0 * spec.spectral_intensity
                        * (2.0 * np.pi * spec.frequency_spacing))
    phase = 2.0 * np.pi * frequencies[None, :, None] * time[None, None, :]
    return (amplitude * (cosine[:, :, None] * np.cos(phase)
                         + sine[:, :, None] * np.sin(phase))).sum(axis=1)

def evaluate_batch(xi: Array, spec: HystereticSpec = DEFAULT_SPEC) -> HystereticBatch:
    """Integrate the batch with a fourth-order Runge-Kutta scheme."""
    accel = excitation(xi, spec)
    dt = spec.dt
    w2 = spec.omega0**2
    c = 2.0 * spec.damping * spec.omega0
    xy = spec.yield_displacement
    a, b, g, n = spec.bouc_A, spec.bouc_beta, spec.bouc_gamma, spec.bouc_n

    def derivative(state: Array, force: Array) -> Array:
        x, v, z = state[..., 0], state[..., 1], state[..., 2]
        restoring = w2 * (spec.alpha * x + (1.0 - spec.alpha) * xy * z)
        dv = -c * v - restoring - force
        dz = (a * v - b * np.abs(v) * (np.abs(z) ** (n - 1)) * z
              - g * v * (np.abs(z) ** n)) / xy
        return np.stack([v, dv, dz], axis=-1)

    state = np.zeros((accel.shape[0], 3))
    peak = np.zeros(accel.shape[0])
    max_z = np.zeros(accel.shape[0])

    for k in range(spec.n_steps - 1):
        f0, f1 = accel[:, k], accel[:, k + 1]
        fh = 0.5 * (f0 + f1)
        k1 = derivative(state, f0)
        k2 = derivative(state + 0.5 * dt * k1, fh)
        k3 = derivative(state + 0.5 * dt * k2, fh)
        k4 = derivative(state + dt * k3, f1)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        peak = np.maximum(peak, np.abs(state[:, 0]))
        max_z = np.maximum(max_z, np.abs(state[:, 2]))

    return HystereticBatch(
        peak_displacement=peak,
        yielded=max_z > 0.9,
        ductility=peak / xy,
        residual_drift=np.abs(state[:, 0]),
    )

def response(xi: Array, spec: HystereticSpec = DEFAULT_SPEC) -> Array:
    return evaluate_batch(xi, spec).peak_displacement
