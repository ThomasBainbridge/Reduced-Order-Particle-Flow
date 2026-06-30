#!/usr/bin/env python3
"""Render animated GIFs of particle + concentration-field evolution.

Each GIF re-simulates one case (so the saved dataset is not required) and
shows the particle scatter beside its concentration field. By default it
produces one GIF per Stokes number at a single seed.

Example
-------
    python scripts/make_gifs.py -o figures --seed 0
    python scripts/make_gifs.py --stokes 1 10 --t-final 20 -o figures
"""

import _bootstrap  # noqa: F401

import argparse
import dataclasses
import pathlib

import matplotlib
matplotlib.use("Agg")

from ropf import SimConfig
from ropf.viz import animate_case


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--outdir", default="figures",
                   help="Output directory (default: %(default)s).")
    p.add_argument("--stokes", type=float, nargs="+", default=[0.1, 1.0, 5.0, 10.0],
                   help="Stokes numbers to animate.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the case.")
    p.add_argument("--n-particles", type=int, default=5000)
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--save-interval", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max-particles", type=int, default=4000,
                   help="Particles to draw in the scatter (for clarity).")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = dataclasses.replace(
        SimConfig(),
        n_particles=args.n_particles,
        t_final=args.t_final,
        dt=args.dt,
        save_interval=args.save_interval,
    )
    cfg.validate()

    for St in args.stokes:
        tag = f"{St:g}".replace(".", "p")
        out = outdir / f"evolution_St{tag}_seed{args.seed}.gif"
        print(f"Animating St={St:g} seed={args.seed} -> {out} ...", flush=True)
        animate_case(
            cfg, St, args.seed, out,
            fps=args.fps, max_particles=args.max_particles,
        )
        print(f"  wrote {out}")

    print("Done.")


if __name__ == "__main__":
    main()
