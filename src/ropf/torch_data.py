"""Tensor preparation for the PyTorch models.

Splits the dataset into train/test by held-out seed (consistent with the POD
baseline), applies a single global amplitude scaling so the tiny
concentration values (order 1/n_cells) become O(1) for stable training, and
exposes both:

  * a flat stack of individual fields (for autoencoder training), and
  * per-case sequences (for latent forecasting and animation).

All physical errors are reported back in the original concentration units by
undoing the scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .baselines import train_test_case_masks


@dataclass
class PreparedData:
    scale: float
    H: int
    W: int
    holdout_kind: str              # "seed" or "stokes"
    holdout_value: float           # the held-out seed or Stokes number
    train_flat: torch.Tensor       # (Ntr, 1, H, W) scaled
    test_flat: torch.Tensor        # (Nte, 1, H, W) scaled
    train_cases: torch.Tensor      # (Ctr, T, 1, H, W) scaled
    test_cases: torch.Tensor       # (Cte, T, 1, H, W) scaled
    train_stokes: np.ndarray       # (Ctr,)
    test_stokes: np.ndarray        # (Cte,)
    time: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.holdout_kind}={self.holdout_value:g}"

    def unscale(self, x):
        """Map scaled fields/tensors back to physical concentration units."""
        return x * self.scale


def prepare(dataset: dict, test_seed: int = None, test_stokes: float = None,
            scale_percentile: float = 99.9) -> PreparedData:
    """Split, scale and tensorise the dataset.

    The held-out set is a seed (default: the last seed) unless ``test_stokes``
    is given, in which case an entire *Stokes number* is held out -- a stronger
    test of generalisation across the inertia parameter.
    """
    conc = dataset["concentration"].astype(np.float32)   # (C, T, 1, H, W)
    seeds = dataset["seed"]
    stokes = dataset["stokes"]
    H, W = conc.shape[-2], conc.shape[-1]

    if test_stokes is not None:
        test_mask = np.isclose(stokes, test_stokes)
        train_mask = ~test_mask
        holdout_kind, holdout_value = "stokes", float(test_stokes)
    else:
        if test_seed is None:
            test_seed = int(max(seeds.tolist()))
        train_mask, test_mask = train_test_case_masks(seeds, test_seed)
        holdout_kind, holdout_value = "seed", float(test_seed)

    train_cases_np = conc[train_mask]
    test_cases_np = conc[test_mask]

    # Single global scale from the training set only (no test leakage).
    scale = float(np.percentile(train_cases_np, scale_percentile))
    if scale <= 0:
        scale = float(train_cases_np.max()) or 1.0

    def to_t(a):
        return torch.from_numpy(a / scale)

    train_cases = to_t(train_cases_np)
    test_cases = to_t(test_cases_np)
    train_flat = train_cases.reshape(-1, 1, H, W)
    test_flat = test_cases.reshape(-1, 1, H, W)

    return PreparedData(
        scale=scale, H=H, W=W,
        holdout_kind=holdout_kind, holdout_value=holdout_value,
        train_flat=train_flat, test_flat=test_flat,
        train_cases=train_cases, test_cases=test_cases,
        train_stokes=stokes[train_mask], test_stokes=stokes[test_mask],
        time=dataset["time"],
    )


def stokes_feature(stokes_values) -> torch.Tensor:
    """Map Stokes numbers to a normalised conditioning feature ~ O(1).

    Uses log10(St); for the default sweep {0.1, 1, 5, 10} this spans roughly
    [-1, 1].
    """
    st = np.asarray(stokes_values, dtype=np.float32).reshape(-1, 1)
    return torch.from_numpy(np.log10(st))
