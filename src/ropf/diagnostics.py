"""Physical diagnostics for particle-concentration fields.

These are the quantities a particle-laden-flow audience cares about -- they
characterise *preferential concentration* directly, rather than as pixel
error. All functions act on the trailing ``(ny, nx)`` axes and broadcast over
any leading (case / time) axes.

Conventions: a field ``c`` is the normalised concentration (sums to 1 over the
grid), so the particle count in a cell is ``c * n_particles``.
"""

from __future__ import annotations

import numpy as np


def _ncells(field):
    return field.shape[-1] * field.shape[-2]


def clustering_index(field, n_particles):
    """Preferential-concentration index D = (sigma_n - sigma_Poisson) / <n>.

    ``sigma_n`` is the standard deviation of the per-cell particle counts and
    ``sigma_Poisson = sqrt(<n>)`` is the value expected for a spatially random
    (Poisson) distribution. D = 0 for a random field and grows positive as
    particles cluster. Independent of the shot-noise baseline by construction.
    """
    field = np.asarray(field, dtype=np.float64)
    ncells = _ncells(field)
    mean_counts = n_particles / ncells
    std_counts = field.std(axis=(-2, -1)) * n_particles
    return (std_counts - np.sqrt(mean_counts)) / mean_counts


def spatial_entropy(field, normalized=True, eps=1e-12):
    """Shannon entropy of the concentration field (treated as a distribution).

    Maximal (= 1 when ``normalized``) for a uniform field; decreases as the
    field concentrates into fewer cells. A complementary, noise-robust measure
    of clustering.
    """
    p = np.asarray(field, dtype=np.float64)
    p = np.clip(p, eps, None)
    H = -np.sum(p * np.log(p), axis=(-2, -1))
    if normalized:
        H = H / np.log(_ncells(field))
    return H


def peak_concentration(field, percentile=99.9):
    """High-percentile concentration -- the intensity of the densest regions."""
    field = np.asarray(field, dtype=np.float64)
    return np.percentile(field, percentile, axis=(-2, -1))


def field_variance(field):
    """Spatial variance of the concentration field."""
    return np.asarray(field, dtype=np.float64).var(axis=(-2, -1))


def all_diagnostics(field, n_particles):
    """Return a dict of every scalar diagnostic for the given field(s)."""
    return {
        "clustering_index": clustering_index(field, n_particles),
        "spatial_entropy": spatial_entropy(field),
        "peak_concentration": peak_concentration(field),
        "field_variance": field_variance(field),
    }
