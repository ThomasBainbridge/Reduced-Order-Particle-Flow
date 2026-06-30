#!/usr/bin/env python3
"""Generate the inertial-particle concentration-field dataset.

Runs the full sweep over Stokes numbers and seeds and saves a compressed
``.npz`` dataset of shape (n_cases, n_saves, 1, ny, nx).

Examples
--------
    # Canonical dataset (4 Stokes x 5 seeds, 201 snapshots, 64x64):
    python scripts/generate_dataset.py

    # Fast smoke test:
    python scripts/generate_dataset.py --quick -o data/smoke.npz
"""

import _bootstrap  # noqa: F401  (sets up sys.path)

import argparse
import dataclasses
import pathlib
import time

from ropf import SimConfig, build_dataset, save_dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", default="data/particle_concentration.npz",
                   help="Output .npz path (default: %(default)s).")
    p.add_argument("--n-particles", type=int, help="Particles per case.")
    p.add_argument("--t-final", type=float, help="Final time.")
    p.add_argument("--dt", type=float, help="Integration time step.")
    p.add_argument("--save-interval", type=float, help="Snapshot cadence.")
    p.add_argument("--nx", type=int, help="Concentration grid points in x.")
    p.add_argument("--ny", type=int, help="Concentration grid points in y.")
    p.add_argument("--integrator", choices=["rk4", "euler"],
                   help="Time integrator (default: rk4).")
    p.add_argument("--smoothing", type=float,
                   help="Gaussian KDE smoothing of the concentration field, "
                        "in grid cells (0 = raw histogram).")
    p.add_argument("--flow-type", choices=["taylor_green", "fourier"],
                   help="Carrier flow (default: taylor_green).")
    p.add_argument("--n-fourier-modes", type=int,
                   help="Number of Fourier modes (flow-type fourier).")
    p.add_argument("--stokes", type=float, nargs="+",
                   help="Stokes numbers to sweep.")
    p.add_argument("--seeds", type=int, nargs="+", help="Random seeds to sweep.")
    p.add_argument("--quick", action="store_true",
                   help="Small/fast configuration for a smoke test.")
    return p.parse_args()


def build_config(args) -> SimConfig:
    overrides = {}
    if args.quick:
        overrides.update(
            n_particles=1000, t_final=4.0, dt=0.02, save_interval=0.2,
            nx=32, ny=32, stokes_numbers=(0.1, 1.0, 10.0), seeds=(0, 1),
        )
    for name, val in [
        ("n_particles", args.n_particles), ("t_final", args.t_final),
        ("dt", args.dt), ("save_interval", args.save_interval),
        ("nx", args.nx), ("ny", args.ny), ("integrator", args.integrator),
        ("smoothing", args.smoothing), ("flow_type", args.flow_type),
        ("n_fourier_modes", args.n_fourier_modes),
    ]:
        if val is not None:
            overrides[name] = val
    if args.stokes is not None:
        overrides["stokes_numbers"] = tuple(args.stokes)
    if args.seeds is not None:
        overrides["seeds"] = tuple(args.seeds)
    cfg = dataclasses.replace(SimConfig(), **overrides)
    cfg.validate()
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)

    print("=" * 64)
    print("Generating inertial-particle concentration dataset")
    print("=" * 64)
    print(f"  flow / smoothing: {cfg.flow_type} / sigma={cfg.smoothing:g}")
    print(f"  Stokes numbers : {list(cfg.stokes_numbers)}")
    print(f"  seeds          : {list(cfg.seeds)}")
    print(f"  cases          : {cfg.n_cases}")
    print(f"  particles/case : {cfg.n_particles}")
    print(f"  grid           : {cfg.ny} x {cfg.nx}")
    print(f"  integrator     : {cfg.integrator}")
    print(f"  dt / T         : {cfg.dt} / {cfg.t_final}  "
          f"({cfg.n_steps} steps)")
    print(f"  snapshots/case : {cfg.n_saves} (every {cfg.save_interval})")
    print(f"  array shape    : ({cfg.n_cases}, {cfg.n_saves}, 1, "
          f"{cfg.ny}, {cfg.nx})")
    print("-" * 64)

    t0 = time.time()

    def progress(c, n, St, seed):
        print(f"  [{c + 1:>2}/{n}] St={St:<5g} seed={seed}", flush=True)

    dataset = build_dataset(cfg, progress=progress)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(out, dataset)

    size_mb = out.stat().st_size / 1e6
    print("-" * 64)
    print(f"  wall time : {time.time() - t0:.1f} s")
    print(f"  saved     : {out}  ({size_mb:.1f} MB)")
    print("=" * 64)


if __name__ == "__main__":
    main()
