"""Classical baselines for the concentration-field problem.

These set the bar the later neural models must beat:

* **Persistence** -- the standard forecasting baseline. Predict that the
  field ``h`` snapshots ahead equals the current field:
  ``c_hat(t + h) = c(t)``.

* **POD / PCA** -- a linear reduced-order *reconstruction* baseline. Project
  fields onto the ``r`` leading proper-orthogonal-decomposition modes and
  reconstruct. With ``r = 16`` this is the direct linear counterpart of the
  16-dimensional convolutional autoencoder latent space.

Also included are simple helpers to flatten fields and split cases into
train/test by seed (so the POD basis is evaluated on unseen realisations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .metrics import per_sample_rmse, relative_l2_error, rmse


# --------------------------------------------------------------------------
# Data reshaping / splitting
# --------------------------------------------------------------------------
def flatten_fields(concentration: np.ndarray) -> np.ndarray:
    """(n_cases, n_times, 1, ny, nx) -> (n_cases, n_times, ny*nx)."""
    n_cases, n_times = concentration.shape[:2]
    return concentration.reshape(n_cases, n_times, -1)


def train_test_case_masks(
    seed_per_case: np.ndarray, test_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean masks over cases splitting on a held-out seed.

    The POD basis is fit on the training seeds (all Stokes numbers) and
    evaluated on the held-out seed, measuring how well a linear basis
    generalises to an unseen realisation.
    """
    test_mask = seed_per_case == test_seed
    return ~test_mask, test_mask


def stack_samples(flat: np.ndarray, case_mask: np.ndarray) -> np.ndarray:
    """Stack the selected cases' snapshots into a (n_samples, features) matrix."""
    sel = flat[case_mask]                      # (n_sel_cases, n_times, features)
    return sel.reshape(-1, sel.shape[-1])


# --------------------------------------------------------------------------
# POD / PCA reduced-order model
# --------------------------------------------------------------------------
@dataclass
class PODModel:
    """Proper Orthogonal Decomposition (mean-subtracted PCA) of flattened fields.

    Fit via the (economy) SVD of the mean-subtracted snapshot matrix. Modes are
    the right singular vectors; the captured energy of mode ``k`` is
    proportional to its squared singular value.
    """

    mean_: np.ndarray = None
    components_: np.ndarray = None          # (k, features), orthonormal rows
    singular_values_: np.ndarray = None
    explained_variance_ratio_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "PODModel":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # Economy SVD: Xc = U @ diag(S) @ Vt, rows of Vt are POD modes.
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt
        self.singular_values_ = S
        total = np.sum(S ** 2)
        self.explained_variance_ratio_ = (S ** 2) / total if total > 0 else S * 0
        return self

    @property
    def n_modes(self) -> int:
        return 0 if self.components_ is None else self.components_.shape[0]

    def transform(self, X: np.ndarray, r: int) -> np.ndarray:
        """Project onto the leading ``r`` modes -> latent coefficients (n, r)."""
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        return Xc @ self.components_[:r].T

    def inverse_transform(self, Z: np.ndarray, r: int) -> np.ndarray:
        """Reconstruct fields from ``r`` latent coefficients."""
        return Z @ self.components_[:r] + self.mean_

    def reconstruct(self, X: np.ndarray, r: int) -> np.ndarray:
        return self.inverse_transform(self.transform(X, r), r)

    def cumulative_energy(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio_)

    def modes_for_energy(self, fraction: float) -> int:
        """Smallest number of modes capturing at least ``fraction`` of energy."""
        cum = self.cumulative_energy()
        return int(np.searchsorted(cum, fraction) + 1)


def pod_reconstruction_curve(
    pod: PODModel, X_test: np.ndarray, ranks: Sequence[int]
) -> dict:
    """Reconstruction error on ``X_test`` as a function of the number of modes."""
    rmse_vals, rel_vals = [], []
    for r in ranks:
        recon = pod.reconstruct(X_test, r)
        rmse_vals.append(float(rmse(recon, X_test)))
        rel_vals.append(float(np.mean(relative_l2_error(recon, X_test, axis=1))))
    return {"ranks": list(ranks), "rmse": rmse_vals, "relative_l2": rel_vals}


# --------------------------------------------------------------------------
# Persistence forecast baseline
# --------------------------------------------------------------------------
def persistence_error_per_case(
    concentration: np.ndarray, horizon: int
) -> np.ndarray:
    """Per-case persistence RMSE at a given snapshot ``horizon``.

    For each case the prediction ``c(t+h) = c(t)`` is scored against the truth
    over all valid start times, then averaged. Returns ``(n_cases,)``.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1 snapshot")
    pred = concentration[:, :-horizon]      # c(t)
    true = concentration[:, horizon:]       # c(t+h)
    # Mean over time and space, per case.
    return np.sqrt(np.mean((pred - true) ** 2, axis=(1, 2, 3, 4)))


def persistence_curve(
    concentration: np.ndarray,
    stokes_per_case: np.ndarray,
    horizons: Sequence[int],
) -> dict:
    """Persistence RMSE vs forecast horizon, overall and per Stokes number."""
    unique_st = sorted(set(stokes_per_case.tolist()))
    overall, per_st = [], {st: [] for st in unique_st}
    for h in horizons:
        err = persistence_error_per_case(concentration, h)
        overall.append(float(err.mean()))
        for st in unique_st:
            per_st[st].append(float(err[stokes_per_case == st].mean()))
    return {
        "horizons": list(horizons),
        "overall_rmse": overall,
        "per_stokes_rmse": {str(k): v for k, v in per_st.items()},
    }
