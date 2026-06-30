"""Error metrics for concentration-field reconstruction and forecasting.

All functions operate on NumPy arrays and broadcast over leading (batch /
time / case) axes via the ``axis`` argument, so the same metric serves both
the classical baselines and the later neural models.
"""

from __future__ import annotations

import numpy as np


def rmse(pred, true, axis=None):
    """Root-mean-square error between two arrays."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return np.sqrt(np.mean((pred - true) ** 2, axis=axis))


def relative_l2_error(pred, true, axis=None, eps=1e-12):
    """Relative L2 (Frobenius) error ||pred - true|| / ||true||.

    Scale-invariant, so it is comparable across Stokes numbers whose fields
    have different magnitudes.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    num = np.sqrt(np.sum((pred - true) ** 2, axis=axis))
    den = np.sqrt(np.sum(true ** 2, axis=axis))
    return num / (den + eps)


def per_sample_rmse(pred, true):
    """RMSE computed independently for each sample in a stack.

    ``pred``/``true`` have shape ``(n_samples, ...)``; returns ``(n_samples,)``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    flat_axes = tuple(range(1, pred.ndim))
    return np.sqrt(np.mean((pred - true) ** 2, axis=flat_axes))
