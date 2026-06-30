"""Central simulation configuration.

A single dataclass holds every numerical and physical parameter so that a
run is fully described (and reproducible) by one object. Defaults reproduce
the canonical dataset described in the README:

    4 Stokes numbers x 5 seeds = 20 cases
    201 saved snapshots per case on a 64 x 64 grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass(frozen=True)
class SimConfig:
    # --- Domain (periodic square) -------------------------------------
    domain_size: float = 2.0 * math.pi      # L; domain is [0, L] x [0, L]
    nx: int = 64                             # concentration grid points in x
    ny: int = 64                             # concentration grid points in y

    # --- Carrier flow: A(t) = amp0 + amp_fluct * sin(omega t) ---------
    flow_type: str = "taylor_green"          # "taylor_green" or "fourier"
    amp0: float = 1.0
    amp_fluct: float = 0.25
    omega: float = 1.0
    # Multi-mode random-Fourier streamfunction flow (flow_type="fourier"):
    n_fourier_modes: int = 8
    max_wavenumber: int = 4
    flow_seed: int = 0

    # --- Concentration field ------------------------------------------
    # Gaussian smoothing (kernel density) applied to the histogram, in grid
    # cells. 0 = raw histogram; > 0 suppresses shot noise (periodic wrap).
    smoothing: float = 0.0

    # --- Particle model -----------------------------------------------
    n_particles: int = 5000
    tau_f: float = 1.0                       # flow time scale; St = tau_p / tau_f
    integrator: str = "rk4"                  # "rk4" or "euler"

    # --- Time integration ---------------------------------------------
    dt: float = 0.01
    t_final: float = 20.0
    save_interval: float = 0.1               # snapshot cadence in time units

    # --- Sweep --------------------------------------------------------
    stokes_numbers: Tuple[float, ...] = (0.1, 1.0, 5.0, 10.0)
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def n_steps(self) -> int:
        """Number of integration steps to reach ``t_final``."""
        return round(self.t_final / self.dt)

    @property
    def save_every(self) -> int:
        """Integration steps between saved snapshots."""
        return round(self.save_interval / self.dt)

    @property
    def n_saves(self) -> int:
        """Number of saved snapshots, including the initial state."""
        return self.n_steps // self.save_every + 1

    @property
    def n_cases(self) -> int:
        return len(self.stokes_numbers) * len(self.seeds)

    @property
    def flow_kwargs(self) -> dict:
        """Keyword arguments for the Taylor-Green carrier-flow evaluator."""
        return dict(amp0=self.amp0, amp_fluct=self.amp_fluct, omega=self.omega)

    def make_flow(self):
        """Build the carrier-flow callable selected by this configuration."""
        from .carrier_flow import build_flow

        return build_flow(self)

    def validate(self) -> None:
        """Fail fast on inconsistent settings."""
        if self.integrator not in STEPPER_NAMES:
            raise ValueError(
                f"integrator must be one of {sorted(STEPPER_NAMES)}, "
                f"got {self.integrator!r}"
            )
        if self.n_steps % self.save_every != 0:
            raise ValueError(
                "save_interval must divide t_final evenly in units of dt "
                f"(n_steps={self.n_steps}, save_every={self.save_every})."
            )
        if self.nx <= 0 or self.ny <= 0 or self.n_particles <= 0:
            raise ValueError("Grid sizes and particle count must be positive.")
        if self.flow_type not in {"taylor_green", "fourier"}:
            raise ValueError(
                f"flow_type must be 'taylor_green' or 'fourier', "
                f"got {self.flow_type!r}"
            )
        if self.smoothing < 0:
            raise ValueError("smoothing (Gaussian sigma) must be >= 0.")

    def to_dict(self) -> dict:
        d = asdict(self)
        # Expose derived sizes too, for self-describing dataset metadata.
        d.update(
            n_steps=self.n_steps,
            save_every=self.save_every,
            n_saves=self.n_saves,
            n_cases=self.n_cases,
        )
        return d


# Imported lazily to avoid a circular import at module load.
STEPPER_NAMES = {"rk4", "euler"}
