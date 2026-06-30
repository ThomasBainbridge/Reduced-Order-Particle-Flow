"""Single-case simulation: integrate particles and emit concentration fields.

``simulate_case`` runs one (Stokes number, seed) case from t = 0 to
``t_final`` and returns the sequence of saved concentration snapshots plus
the corresponding time vector. Optionally it also returns saved particle
positions, used by the visualisation scripts to show preferential
concentration directly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .concentration import concentration_field
from .config import SimConfig
from .particles import STEPPERS, wrap_periodic


def simulate_case(
    stokes: float,
    seed: int,
    cfg: SimConfig,
    return_particles: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Simulate one inertial-particle case.

    Parameters
    ----------
    stokes : float
        Stokes number St = tau_p / tau_f for this case.
    seed : int
        Seed for the (reproducible) initial particle positions.
    cfg : SimConfig
        Simulation configuration.
    return_particles : bool
        If True, also return the saved particle positions, shape
        ``(n_saves, N, 2)``.

    Returns
    -------
    fields : ndarray, shape (n_saves, 1, ny, nx), float32
        Saved concentration fields (channel axis included).
    times : ndarray, shape (n_saves,)
        Snapshot times.
    particles : ndarray or None
        Saved particle positions if ``return_particles`` else None.
    """
    cfg.validate()
    rng = np.random.default_rng(seed)
    L = cfg.domain_size
    tau_p = stokes * cfg.tau_f
    step = STEPPERS[cfg.integrator]
    flow = cfg.make_flow()

    # Initial conditions:
    #   positions uniformly random in the domain,
    #   velocities equal to the local carrier-flow velocity at t = 0.
    pos = rng.uniform(0.0, L, size=(cfg.n_particles, 2))
    vel = flow(pos, 0.0)

    fields = np.empty((cfg.n_saves, 1, cfg.ny, cfg.nx), dtype=np.float32)
    times = np.empty(cfg.n_saves, dtype=np.float64)
    particles = (
        np.empty((cfg.n_saves, cfg.n_particles, 2), dtype=np.float32)
        if return_particles
        else None
    )

    def _record(save_idx, t):
        fields[save_idx, 0] = concentration_field(
            pos, cfg.nx, cfg.ny, L, smoothing=cfg.smoothing
        )
        times[save_idx] = t
        if particles is not None:
            particles[save_idx] = pos.astype(np.float32)

    # Snapshot 0: the initial state.
    pos = wrap_periodic(pos, L)
    _record(0, 0.0)

    save_idx = 1
    t = 0.0
    for i in range(1, cfg.n_steps + 1):
        pos, vel = step(pos, vel, t, cfg.dt, tau_p, flow)
        pos = wrap_periodic(pos, L)
        t = i * cfg.dt
        if i % cfg.save_every == 0:
            _record(save_idx, t)
            save_idx += 1

    return fields, times, particles
