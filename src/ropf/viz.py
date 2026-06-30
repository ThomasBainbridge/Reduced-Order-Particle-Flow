"""Plotting and animation helpers.

These functions produce the figures and GIFs used to visually verify that
the physics behaves as expected -- in particular that increasing the Stokes
number produces stronger *preferential concentration* (inertial particles
being centrifuged out of vortex cores and clustering in strain regions).

The caller is responsible for selecting a non-interactive Matplotlib backend
(e.g. ``matplotlib.use("Agg")``) before importing pyplot.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Static figures (operate on an already-generated dataset)
# --------------------------------------------------------------------------
def _field_extent(L: float):
    return [0.0, L, 0.0, L]


def _nearest_time_indices(times: np.ndarray, wanted: Sequence[float]):
    return [int(np.argmin(np.abs(times - w))) for w in wanted]


def plot_stokes_comparison(
    dataset: dict,
    seed: int = 0,
    snapshot_times: Sequence[float] = (0.0, 5.0, 10.0, 20.0),
    cmap: str = "inferno",
):
    """Grid of concentration fields: rows = Stokes number, cols = time.

    Uses one representative seed. Returns the Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    conc = dataset["concentration"]
    stokes = dataset["stokes"]
    seeds = dataset["seed"]
    times = dataset["time"]
    L = dataset["metadata"]["config"]["domain_size"]

    unique_st = sorted(set(stokes.tolist()))
    t_idx = _nearest_time_indices(times, snapshot_times)

    nrows, ncols = len(unique_st), len(t_idx)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows), squeeze=False
    )

    for r, St in enumerate(unique_st):
        case = int(np.where((stokes == St) & (seeds == seed))[0][0])
        # Shared colour scale per row so St rows are comparable across time.
        row_fields = conc[case, t_idx, 0]
        vmax = np.percentile(row_fields, 99.5) or row_fields.max()
        for c, ti in enumerate(t_idx):
            ax = axes[r][c]
            ax.imshow(
                conc[case, ti, 0],
                origin="lower",
                extent=_field_extent(L),
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {times[ti]:.1f}", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"St = {St:g}", fontsize=11)

    fig.suptitle(
        "Particle concentration fields vs Stokes number (seed "
        f"{seed})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_clustering_diagnostics(dataset: dict):
    """Concentration-field variance vs time, averaged over seeds, per St.

    For a perfectly uniform tracer the normalised concentration would stay
    flat; growing variance is a quantitative signature of clustering. Higher
    Stokes numbers should cluster more strongly (up to a point).
    Returns the Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    conc = dataset["concentration"]
    stokes = dataset["stokes"]
    times = dataset["time"]
    unique_st = sorted(set(stokes.tolist()))

    # Variance over the spatial axes for every (case, time).
    var = conc.var(axis=(2, 3, 4))  # -> (n_cases, n_times)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for St in unique_st:
        mask = stokes == St
        mean_var = var[mask].mean(axis=0)
        ax.plot(times, mean_var, label=f"St = {St:g}", lw=2)

    ax.set_xlabel("time")
    ax.set_ylabel("spatial variance of concentration field")
    ax.set_title("Preferential concentration: field variance vs time")
    ax.legend(title="Stokes number")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pod_reconstruction_panel(
    true_field_flat,
    pod,
    ranks,
    ny: int,
    nx: int,
    L: float,
    cmap: str = "inferno",
):
    """Truth vs POD reconstruction at increasing numbers of modes.

    ``true_field_flat`` is a single flattened test field (length ny*nx).
    Returns the Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    true_field_flat = np.asarray(true_field_flat, dtype=np.float64)[None, :]
    truth = true_field_flat.reshape(ny, nx)
    vmax = np.percentile(truth, 99.5) or truth.max()

    ncols = len(ranks) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(2.5 * ncols, 2.8), squeeze=False)
    axes = axes[0]

    axes[0].imshow(truth, origin="lower", extent=[0, L, 0, L], cmap=cmap,
                   vmin=0, vmax=vmax)
    axes[0].set_title("truth")
    for ax, r in zip(axes[1:], ranks):
        recon = pod.reconstruct(true_field_flat, r).reshape(ny, nx)
        ax.imshow(recon, origin="lower", extent=[0, L, 0, L], cmap=cmap,
                  vmin=0, vmax=vmax)
        ax.set_title(f"r = {r}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("POD reconstruction vs number of modes", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# --------------------------------------------------------------------------
# Animations (re-simulate a single case to also show particles)
# --------------------------------------------------------------------------
def animate_case(
    cfg,
    stokes: float,
    seed: int,
    out_path,
    fps: int = 20,
    max_particles: int = 5000,
    cmap: str = "inferno",
):
    """Side-by-side GIF: particle scatter (left) and concentration (right).

    Re-runs a single case (so it does not need the saved dataset) and writes
    an animated GIF via Matplotlib's PillowWriter.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from .simulation import simulate_case

    fields, times, particles = simulate_case(
        stokes, seed, cfg, return_particles=True
    )
    L = cfg.domain_size
    n_show = min(max_particles, particles.shape[1])
    # Fixed subset of particles for a clean scatter.
    sub = np.random.default_rng(0).choice(
        particles.shape[1], size=n_show, replace=False
    )

    vmax = np.percentile(fields, 99.5)

    fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(9, 4.6))

    scat = ax_p.scatter(
        particles[0, sub, 0], particles[0, sub, 1], s=2, c="#1f77b4", alpha=0.5
    )
    ax_p.set_xlim(0, L)
    ax_p.set_ylim(0, L)
    ax_p.set_aspect("equal")
    ax_p.set_title("particles")
    ax_p.set_xticks([])
    ax_p.set_yticks([])

    im = ax_c.imshow(
        fields[0, 0],
        origin="lower",
        extent=[0, L, 0, L],
        cmap=cmap,
        vmin=0,
        vmax=vmax,
    )
    ax_c.set_title("concentration")
    ax_c.set_xticks([])
    ax_c.set_yticks([])

    suptitle = fig.suptitle("", fontsize=12)

    def update(frame):
        scat.set_offsets(particles[frame, sub])
        im.set_data(fields[frame, 0])
        suptitle.set_text(f"St = {stokes:g}   seed = {seed}   t = {times[frame]:.1f}")
        return scat, im, suptitle

    anim = FuncAnimation(
        fig, update, frames=fields.shape[0], interval=1000 / fps, blit=False
    )
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
