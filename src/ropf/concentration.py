"""Lagrangian -> Eulerian conversion via 2D histogram binning.

Particle positions are binned onto a regular ``ny x nx`` grid covering the
domain and normalised by the total particle count, giving a discrete
particle-concentration (number-density) field whose entries sum to one.

The field is stored in image convention ``(ny, nx)`` with the row index
running over y and the column index over x, ready to be reshaped to the
PyTorch-friendly ``(1, ny, nx)`` used by the downstream models.
"""

from __future__ import annotations

import numpy as np


def concentration_field(pos, nx, ny, L, normalize=True, smoothing=0.0):
    """Bin particle positions into a concentration field.

    Parameters
    ----------
    pos : ndarray, shape (N, 2)
        Particle positions (x, y), assumed already wrapped into [0, L).
    nx, ny : int
        Number of grid cells in x and y.
    L : float
        Domain size (domain is [0, L] x [0, L]).
    normalize : bool
        If True, divide by the number of particles so the field sums to 1.
    smoothing : float
        If > 0, convolve the histogram with a Gaussian of this standard
        deviation (in grid cells) using periodic ('wrap') boundaries -- a
        kernel-density estimate that suppresses the per-cell shot noise while
        preserving the coherent clustering structure. Normalisation (sum = 1)
        is preserved because the periodic Gaussian conserves total mass.

    Returns
    -------
    ndarray, shape (ny, nx), dtype float32
        Concentration field in image convention (row = y, col = x).
    """
    edges_x = np.linspace(0.0, L, nx + 1)
    edges_y = np.linspace(0.0, L, ny + 1)
    # histogram2d returns counts indexed [x_bin, y_bin]; transpose to (y, x).
    counts, _, _ = np.histogram2d(
        pos[:, 0], pos[:, 1], bins=[edges_x, edges_y]
    )
    field = counts.T
    if smoothing and smoothing > 0:
        from scipy.ndimage import gaussian_filter

        field = gaussian_filter(field, sigma=smoothing, mode="wrap")
    if normalize:
        field = field / pos.shape[0]
    return field.astype(np.float32)
