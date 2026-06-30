"""Prescribed unsteady Taylor-Green-type carrier flow.

The carrier velocity field on the periodic domain [0, L] x [0, L] is

    u_x(x, y, t) =  A(t) sin(x) cos(y)
    u_y(x, y, t) = -A(t) cos(x) sin(y)
    A(t)         = amp0 + amp_fluct * sin(omega * t)

This field is:
  * smooth and periodic with period 2*pi in both directions,
  * analytically divergence-free (incompressible):
        d(u_x)/dx + d(u_y)/dy
            =  A cos(x) cos(y) - A cos(x) cos(y) = 0,
  * cheap to evaluate anywhere, so no CFD solver is required.

The time-dependent amplitude A(t) makes the flow unsteady, which is what
gives the inertial particles a non-trivial, forecastable evolution.
"""

from __future__ import annotations

import numpy as np


def amplitude(t, amp0: float = 1.0, amp_fluct: float = 0.25, omega: float = 1.0):
    """Time-dependent flow amplitude A(t) = amp0 + amp_fluct * sin(omega t)."""
    return amp0 + amp_fluct * np.sin(omega * t)


def taylor_green_velocity(pos, t, *, amp0=1.0, amp_fluct=0.25, omega=1.0):
    """Carrier-flow velocity sampled at particle locations.

    Parameters
    ----------
    pos : ndarray, shape (..., 2)
        Positions (x, y). Periodicity is built into the sin/cos, so positions
        need not be pre-wrapped into the primary domain.
    t : float
        Time.

    Returns
    -------
    ndarray, shape (..., 2)
        Velocity (u_x, u_y) at each position.
    """
    pos = np.asarray(pos)
    A = amplitude(t, amp0, amp_fluct, omega)
    x = pos[..., 0]
    y = pos[..., 1]
    ux = A * np.sin(x) * np.cos(y)
    uy = -A * np.cos(x) * np.sin(y)
    return np.stack([ux, uy], axis=-1)


def velocity_divergence(flow, pos, t, *, eps: float = 1e-6):
    """Numerically evaluate div(u) of any flow callable by central differences.

    ``flow`` is a callable ``flow(pos, t) -> (..., 2)``. Provided for
    verification only: the analytic divergence of the (incompressible) flows
    here is identically zero, so this should return values at the level of
    finite-difference truncation/round-off error.
    """
    pos = np.asarray(pos, dtype=float)
    dx = np.zeros_like(pos)
    dy = np.zeros_like(pos)
    dx[..., 0] = eps
    dy[..., 1] = eps
    dux_dx = (flow(pos + dx, t)[..., 0] - flow(pos - dx, t)[..., 0]) / (2 * eps)
    duy_dy = (flow(pos + dy, t)[..., 1] - flow(pos - dy, t)[..., 1]) / (2 * eps)
    return dux_dx + duy_dy


class TaylorGreenFlow:
    """Callable wrapper around :func:`taylor_green_velocity` (the default flow)."""

    flow_type = "taylor_green"

    def __init__(self, amp0=1.0, amp_fluct=0.25, omega=1.0):
        self.amp0 = amp0
        self.amp_fluct = amp_fluct
        self.omega = omega

    def __call__(self, pos, t):
        return taylor_green_velocity(
            pos, t, amp0=self.amp0, amp_fluct=self.amp_fluct, omega=self.omega
        )


class RandomFourierFlow:
    """Divergence-free multi-mode carrier flow from a random streamfunction.

    A more turbulence-like (but still prescribed and incompressible) flow built
    from a sum of Fourier modes of a streamfunction on the periodic domain:

        psi(x, y, t) = A(t) * sum_k  c_k sin(kx_k x + ky_k y + phi_k)

    with the incompressible velocity recovered as the curl of the
    streamfunction,

        u_x =  d(psi)/dy =  A(t) sum_k c_k ky_k cos(kx_k x + ky_k y + phi_k)
        u_y = -d(psi)/dx = -A(t) sum_k c_k kx_k cos(kx_k x + ky_k y + phi_k),

    which is divergence-free by construction. Integer wavenumbers keep the
    field 2*pi-periodic. The amplitude A(t) = amp0 + amp_fluct sin(omega t)
    reuses the same unsteady modulation as the Taylor-Green flow.
    """

    flow_type = "fourier"

    def __init__(self, n_modes=8, max_wavenumber=4, seed=0,
                 amp0=1.0, amp_fluct=0.25, omega=1.0):
        self.amp0 = amp0
        self.amp_fluct = amp_fluct
        self.omega = omega
        rng = np.random.default_rng(seed)

        # Draw distinct non-zero integer wavevectors within |k| <= max_wavenumber.
        ks = []
        seen = set()
        K = max_wavenumber
        while len(ks) < n_modes:
            kx = int(rng.integers(-K, K + 1))
            ky = int(rng.integers(-K, K + 1))
            if (kx, ky) == (0, 0) or (kx, ky) in seen:
                continue
            seen.add((kx, ky))
            ks.append((kx, ky))
        self.kx = np.array([k[0] for k in ks], dtype=float)
        self.ky = np.array([k[1] for k in ks], dtype=float)
        self.phase = rng.uniform(0, 2 * np.pi, size=n_modes)
        # Energy decays with wavenumber (~ -5/3-ish steepening) for realism.
        kmag = np.sqrt(self.kx ** 2 + self.ky ** 2)
        coeff = rng.normal(size=n_modes) / kmag ** 1.5
        # Normalise so the rms velocity is O(1) at A = 1.
        self.coeff = coeff / np.sqrt(np.sum((coeff * kmag) ** 2))

    def __call__(self, pos, t):
        pos = np.asarray(pos)
        x = pos[..., 0][..., None]          # (..., 1) broadcast over modes
        y = pos[..., 1][..., None]
        A = amplitude(t, self.amp0, self.amp_fluct, self.omega)
        phase = self.kx * x + self.ky * y + self.phase
        cos = np.cos(phase)
        ux = A * np.sum(self.coeff * self.ky * cos, axis=-1)
        uy = -A * np.sum(self.coeff * self.kx * cos, axis=-1)
        return np.stack([ux, uy], axis=-1)


def build_flow(cfg) -> object:
    """Construct the carrier-flow callable selected by a :class:`SimConfig`."""
    if cfg.flow_type == "taylor_green":
        return TaylorGreenFlow(cfg.amp0, cfg.amp_fluct, cfg.omega)
    if cfg.flow_type == "fourier":
        return RandomFourierFlow(
            n_modes=cfg.n_fourier_modes,
            max_wavenumber=cfg.max_wavenumber,
            seed=cfg.flow_seed,
            amp0=cfg.amp0,
            amp_fluct=cfg.amp_fluct,
            omega=cfg.omega,
        )
    raise ValueError(f"unknown flow_type: {cfg.flow_type!r}")
