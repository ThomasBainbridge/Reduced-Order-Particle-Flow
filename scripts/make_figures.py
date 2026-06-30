#!/usr/bin/env python3
"""Render static figures from a generated dataset.

Produces:
  * stokes_comparison.png   -- concentration fields, rows=St, cols=time
  * clustering_diagnostics.png -- field variance vs time per Stokes number

Example
-------
    python scripts/make_figures.py -i data/particle_concentration.npz \
        -o figures
"""

import _bootstrap  # noqa: F401

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")  # headless rendering

from ropf import load_dataset
from ropf.viz import plot_clustering_diagnostics, plot_stokes_comparison


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default="data/particle_concentration.npz",
                   help="Input dataset .npz (default: %(default)s).")
    p.add_argument("-o", "--outdir", default="figures",
                   help="Output directory (default: %(default)s).")
    p.add_argument("--seed", type=int, default=0,
                   help="Representative seed for the comparison grid.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    dataset = load_dataset(args.input)
    times = dataset["time"]
    # Pick four evenly spaced snapshot times spanning the run.
    snaps = (times[0], times[len(times) // 3], times[2 * len(times) // 3], times[-1])

    fig1 = plot_stokes_comparison(dataset, seed=args.seed, snapshot_times=snaps)
    f1 = outdir / "stokes_comparison.png"
    fig1.savefig(f1, dpi=args.dpi)
    print(f"  wrote {f1}")

    fig2 = plot_clustering_diagnostics(dataset)
    f2 = outdir / "clustering_diagnostics.png"
    fig2.savefig(f2, dpi=args.dpi)
    print(f"  wrote {f2}")

    print("Done.")


if __name__ == "__main__":
    main()
