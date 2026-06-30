"""Reduced-Order Forecasting of Particle-Laden Flow Evolution.

A controlled scientific-machine-learning testbed:

    prescribed carrier flow -> inertial particle dynamics
    -> Eulerian concentration fields -> reduced-order representation
    -> neural reconstruction -> latent forecasting -> error analysis

This package (``ropf``) implements the *physics and dataset-generation*
stage. The machine-learning stages are built on top of the datasets it
produces. See the project README for the full roadmap.
"""

from .config import SimConfig
from .carrier_flow import amplitude, taylor_green_velocity, velocity_divergence
from .particles import particle_rhs, rk4_step, euler_step, wrap_periodic, STEPPERS
from .concentration import concentration_field
from .simulation import simulate_case
from .dataset import build_dataset, save_dataset, load_dataset
from .metrics import rmse, relative_l2_error, per_sample_rmse
from .diagnostics import (
    clustering_index,
    spatial_entropy,
    peak_concentration,
    field_variance,
    all_diagnostics,
)
from .baselines import (
    PODModel,
    flatten_fields,
    train_test_case_masks,
    stack_samples,
    pod_reconstruction_curve,
    persistence_curve,
    persistence_error_per_case,
)

__all__ = [
    "SimConfig",
    "amplitude",
    "taylor_green_velocity",
    "velocity_divergence",
    "particle_rhs",
    "rk4_step",
    "euler_step",
    "wrap_periodic",
    "STEPPERS",
    "concentration_field",
    "simulate_case",
    "build_dataset",
    "save_dataset",
    "load_dataset",
    "rmse",
    "relative_l2_error",
    "per_sample_rmse",
    "clustering_index",
    "spatial_entropy",
    "peak_concentration",
    "field_variance",
    "all_diagnostics",
    "PODModel",
    "flatten_fields",
    "train_test_case_masks",
    "stack_samples",
    "pod_reconstruction_curve",
    "persistence_curve",
    "persistence_error_per_case",
]

__version__ = "0.1.0"
