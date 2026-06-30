"""Dataset assembly and (de)serialisation.

Sweeps over the Cartesian product of Stokes numbers and seeds, runs each
case, and stacks the concentration fields into a single self-describing
array of shape

    (n_cases, n_saves, 1, ny, nx).

The dataset is saved as a compressed ``.npz`` (no extra dependencies). HDF5
can be substituted later without changing the rest of the pipeline.
"""

from __future__ import annotations

import json
import time
from typing import Callable, Optional

import numpy as np

from .config import SimConfig
from .simulation import simulate_case


def build_dataset(
    cfg: SimConfig,
    progress: Optional[Callable[[int, int, float, int], None]] = None,
) -> dict:
    """Run the full (Stokes x seed) sweep and assemble the dataset.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration (defines the sweep).
    progress : callable, optional
        Called as ``progress(case_index, n_cases, stokes, seed)`` before each
        case, for logging.

    Returns
    -------
    dict with keys:
        concentration : (n_cases, n_saves, 1, ny, nx) float32
        stokes        : (n_cases,) float64  -- St for each case
        seed          : (n_cases,) int64    -- seed for each case
        time          : (n_saves,) float64  -- snapshot times
        metadata      : str (JSON)          -- full config + provenance
    """
    cfg.validate()
    cases = [(St, seed) for St in cfg.stokes_numbers for seed in cfg.seeds]
    n_cases = len(cases)

    concentration = np.empty(
        (n_cases, cfg.n_saves, 1, cfg.ny, cfg.nx), dtype=np.float32
    )
    stokes_arr = np.empty(n_cases, dtype=np.float64)
    seed_arr = np.empty(n_cases, dtype=np.int64)
    times_ref: Optional[np.ndarray] = None

    t_start = time.time()
    for c, (St, seed) in enumerate(cases):
        if progress is not None:
            progress(c, n_cases, St, seed)
        fields, times, _ = simulate_case(St, seed, cfg)
        concentration[c] = fields
        stokes_arr[c] = St
        seed_arr[c] = seed
        if times_ref is None:
            times_ref = times

    metadata = {
        "description": (
            "Inertial-particle concentration fields in an unsteady "
            "Taylor-Green-type carrier flow."
        ),
        "config": cfg.to_dict(),
        "axes": ["case", "time", "channel", "y", "x"],
        "stokes_numbers": list(cfg.stokes_numbers),
        "seeds": list(cfg.seeds),
        "wall_time_seconds": round(time.time() - t_start, 2),
        "ropf_version": _version(),
    }

    return {
        "concentration": concentration,
        "stokes": stokes_arr,
        "seed": seed_arr,
        "time": times_ref,
        "metadata": json.dumps(metadata, indent=2),
    }


def save_dataset(path, dataset: dict) -> None:
    """Save a dataset dict to a compressed ``.npz`` file."""
    np.savez_compressed(
        path,
        concentration=dataset["concentration"],
        stokes=dataset["stokes"],
        seed=dataset["seed"],
        time=dataset["time"],
        metadata=np.array(dataset["metadata"]),
    )


def load_dataset(path) -> dict:
    """Load a dataset saved by :func:`save_dataset`.

    Returns a dict with the same keys as :func:`build_dataset`; ``metadata``
    is parsed back into a dict.
    """
    with np.load(path, allow_pickle=False) as npz:
        out = {
            "concentration": npz["concentration"],
            "stokes": npz["stokes"],
            "seed": npz["seed"],
            "time": npz["time"],
            "metadata": json.loads(str(npz["metadata"])),
        }
    return out


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "unknown"
