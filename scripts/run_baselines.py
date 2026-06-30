#!/usr/bin/env python3
"""Milestone 2: classical baselines on the concentration-field dataset.

Computes and visualises the two reference models the neural networks must
beat:

  * Persistence forecasting  (c_hat(t+h) = c(t)) -- RMSE vs horizon, per St.
  * POD/PCA reconstruction   -- fit on training seeds, evaluate on a held-out
    seed; reconstruction error vs number of modes, energy spectrum, and a
    truth-vs-reconstruction panel.

Outputs figures to the figures directory and a machine-readable
``baseline_results.json`` summary.

Example
-------
    python scripts/run_baselines.py -i data/particle_concentration.npz -o figures
"""

import _bootstrap  # noqa: F401

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ropf import load_dataset
from ropf.baselines import (
    PODModel,
    flatten_fields,
    persistence_curve,
    pod_reconstruction_curve,
    stack_samples,
    train_test_case_masks,
)
from ropf.viz import plot_pod_reconstruction_panel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default="data/particle_concentration.npz")
    p.add_argument("-o", "--outdir", default="figures")
    p.add_argument("--test-seed", type=int, default=None,
                   help="Seed held out for evaluation (default: last seed).")
    p.add_argument("--ranks", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128],
                   help="POD mode counts to evaluate.")
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 2, 5, 10, 20, 50, 100],
                   help="Forecast horizons in snapshots.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    ds = load_dataset(args.input)
    conc = ds["concentration"]                 # (n_cases, n_times, 1, ny, nx)
    stokes = ds["stokes"]
    seeds = ds["seed"]
    cfg = ds["metadata"]["config"]
    ny, nx, L = cfg["ny"], cfg["nx"], cfg["domain_size"]

    test_seed = args.test_seed
    if test_seed is None:
        test_seed = int(max(seeds.tolist()))
    train_mask, test_mask = train_test_case_masks(seeds, test_seed)
    print(f"  cases: {conc.shape[0]}  | train {train_mask.sum()} "
          f"/ test {test_mask.sum()} (held-out seed = {test_seed})")

    results = {"test_seed": test_seed}

    # ------------------------------------------------------------------
    # POD / PCA reconstruction
    # ------------------------------------------------------------------
    print("Fitting POD on training seeds ...")
    flat = flatten_fields(conc)
    X_train = stack_samples(flat, train_mask)
    X_test = stack_samples(flat, test_mask)

    pod = PODModel().fit(X_train)
    ranks = [r for r in args.ranks if r <= pod.n_modes]
    curve = pod_reconstruction_curve(pod, X_test, ranks)
    energy = pod.cumulative_energy()
    e90, e99 = pod.modes_for_energy(0.90), pod.modes_for_energy(0.99)
    results["pod"] = {
        **curve,
        "modes_for_90pct_energy": e90,
        "modes_for_99pct_energy": e99,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
    }
    r16 = curve["rmse"][ranks.index(16)] if 16 in ranks else None
    print(f"  modes for 90% / 99% energy : {e90} / {e99}")
    if r16 is not None:
        print(f"  test RMSE @ r=16 modes     : {r16:.3e}")

    # Figure: reconstruction error vs number of modes
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogy(curve["ranks"], curve["rmse"], "o-", lw=2)
    if 16 in ranks:
        ax.axvline(16, color="grey", ls="--", lw=1)
        ax.text(16, ax.get_ylim()[1], " autoencoder latent dim (16)",
                va="top", fontsize=9, color="grey")
    ax.set_xlabel("number of POD modes  r")
    ax.set_ylabel("test reconstruction RMSE")
    ax.set_title("POD reduced-order reconstruction (held-out seed)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    f = outdir / "pod_reconstruction_error.png"
    fig.savefig(f, dpi=args.dpi); plt.close(fig)
    print(f"  wrote {f}")

    # Figure: cumulative energy spectrum
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(np.arange(1, len(energy) + 1), energy, lw=2)
    ax.axhline(0.9, color="grey", ls=":", lw=1)
    ax.axhline(0.99, color="grey", ls="--", lw=1)
    ax.set_xlim(0, min(200, len(energy)))
    ax.set_xlabel("number of POD modes")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_title("POD energy spectrum")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f = outdir / "pod_energy_spectrum.png"
    fig.savefig(f, dpi=args.dpi); plt.close(fig)
    print(f"  wrote {f}")

    # Figure: truth vs reconstruction panel for the most-structured test field
    # (highest spatial variance => strongest clustering, clearest illustration).
    sample = X_test[np.argmax(X_test.var(axis=1))]
    panel_ranks = [r for r in (1, 4, 16, 64) if r <= pod.n_modes]
    fig = plot_pod_reconstruction_panel(sample, pod, panel_ranks, ny, nx, L)
    f = outdir / "pod_reconstruction_panel.png"
    fig.savefig(f, dpi=args.dpi); plt.close(fig)
    print(f"  wrote {f}")

    # ------------------------------------------------------------------
    # Persistence forecasting baseline
    # ------------------------------------------------------------------
    print("Computing persistence baseline on test cases ...")
    dt_snap = cfg["save_interval"]
    horizons = [h for h in args.horizons if h < conc.shape[1]]
    pcurve = persistence_curve(conc[test_mask], stokes[test_mask], horizons)
    results["persistence"] = {**pcurve, "snapshot_dt": dt_snap}

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    times = np.array(horizons) * dt_snap
    for st_key, vals in pcurve["per_stokes_rmse"].items():
        ax.plot(times, vals, "o-", lw=2, label=f"St = {float(st_key):g}")
    ax.plot(times, pcurve["overall_rmse"], "k--", lw=2, label="overall")
    ax.set_xlabel("forecast horizon  (time units)")
    ax.set_ylabel("persistence RMSE")
    ax.set_title("Persistence forecast error vs horizon")
    ax.legend(title="Stokes number")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f = outdir / "persistence_error_vs_horizon.png"
    fig.savefig(f, dpi=args.dpi); plt.close(fig)
    print(f"  wrote {f}")

    # ------------------------------------------------------------------
    # Save numeric summary
    # ------------------------------------------------------------------
    out_json = outdir / "baseline_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"  wrote {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
