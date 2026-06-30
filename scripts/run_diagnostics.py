#!/usr/bin/env python3
"""Physical diagnostics of the concentration fields vs time and Stokes number.

Computes the preferential-concentration index, spatial entropy, peak
concentration and field variance for every snapshot, then plots each against
time with one curve per Stokes number (mean over seeds, shaded +/- 1 std).

Example
-------
    python scripts/run_diagnostics.py -i data/particle_concentration.npz -o figures
"""

import _bootstrap  # noqa: F401

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ropf import load_dataset
from ropf.diagnostics import (
    clustering_index,
    field_variance,
    peak_concentration,
    spatial_entropy,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default="data/particle_concentration.npz")
    p.add_argument("-o", "--outdir", default="figures")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.input)
    conc = ds["concentration"][:, :, 0]          # (n_cases, n_times, ny, nx)
    stokes = ds["stokes"]
    times = ds["time"]
    n_particles = ds["metadata"]["config"]["n_particles"]
    unique_st = sorted(set(stokes.tolist()))

    diags = {
        "clustering index  D": clustering_index(conc, n_particles),
        "spatial entropy (norm.)": spatial_entropy(conc),
        "peak concentration (p99.9)": peak_concentration(conc),
        "field variance": field_variance(conc),
    }  # each: (n_cases, n_times)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, vals) in zip(axes.ravel(), diags.items()):
        for st in unique_st:
            m = stokes == st
            mean = vals[m].mean(0)
            std = vals[m].std(0)
            line, = ax.plot(times, mean, lw=2, label=f"St = {st:g}")
            ax.fill_between(times, mean - std, mean + std, alpha=0.15,
                            color=line.get_color())
        ax.set_xlabel("time"); ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(title="Stokes number", fontsize=9)
    fig.suptitle("Physical diagnostics of preferential concentration "
                 "(mean +/- 1 std over seeds)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    f = outdir / "physical_diagnostics.png"; fig.savefig(f, dpi=args.dpi); plt.close(fig)
    print(f"  wrote {f}")
    print("Done.")


if __name__ == "__main__":
    main()
