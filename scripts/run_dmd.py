#!/usr/bin/env python3
"""DMD forecasting baseline: the linear counterpart of the latent forecasters.

Fits POD on the training seeds, projects every case onto the leading ``r``
modes, and fits a linear latent map ``a_{t+1} = A a_t + b`` (DMD in POD
coordinates). Each held-out case is then rolled out recursively from its
t = 0 state -- exactly the protocol of ``train_forecaster.py`` -- and scored
against persistence and against the POD projection floor (the best any
rank-``r`` model could do).

Two variants are fitted:
  * ``global``     -- one operator for all Stokes numbers;
  * ``per-Stokes`` -- a separate operator per Stokes number (the linear
    analogue of Stokes conditioning).

Example
-------
    python scripts/run_dmd.py -i data/particle_concentration_smooth.npz \
        -o figures/smooth --rank 16
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
    DMDModel,
    PODModel,
    flatten_fields,
    stack_samples,
    train_test_case_masks,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default="data/particle_concentration.npz")
    p.add_argument("-o", "--outdir", default="figures")
    p.add_argument("--test-seed", type=int, default=None,
                   help="Seed held out for evaluation (default: last seed).")
    p.add_argument("--rank", type=int, default=16,
                   help="POD rank of the DMD state (default: 16, as the AE).")
    p.add_argument("--rank-sweep", type=int, nargs="+", default=[4, 8, 16, 32],
                   help="Ranks whose final-horizon error is also reported.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def field_rmse(pred, truth):
    """RMSE per snapshot over the spatial axis: (..., T, F) -> (..., T)."""
    return np.sqrt(np.mean((pred - truth) ** 2, axis=-1))


def at_horizons(horizons, curve, marks=(1.0, 5.0, 10.0, 20.0)):
    """Sample an error-vs-horizon curve at a few fixed horizons (time units)."""
    return {f"{h:g}": float(curve[int(np.argmin(np.abs(horizons - h)))])
            for h in marks if h <= horizons[-1] + 1e-9}


def forecast(pod, coeffs, train_mask, test_mask, stokes, r, per_stokes):
    """Roll DMD out from t = 0 for every test case; return (Cte, T, F) fields."""
    a = coeffs[..., :r]
    n_steps = a.shape[1] - 1
    preds = np.empty((test_mask.sum(), a.shape[1], pod.mean_.size))
    radii = {}
    groups = sorted(set(stokes.tolist())) if per_stokes else [None]
    test_idx = np.where(test_mask)[0]
    for st in groups:
        sel_tr = train_mask if st is None else train_mask & (stokes == st)
        dmd = DMDModel().fit(a[sel_tr])
        radii["all" if st is None else f"St={st:g}"] = dmd.spectral_radius
        for k, c in enumerate(test_idx):
            if st is None or stokes[c] == st:
                traj = dmd.rollout(a[c, 0], n_steps)
                preds[k] = np.clip(pod.inverse_transform(traj, r), 0, None)
    return preds, radii


def main():
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    ds = load_dataset(args.input)
    conc, stokes, seeds, times = ds["concentration"], ds["stokes"], ds["seed"], ds["time"]
    test_seed = args.test_seed
    if test_seed is None:
        test_seed = int(max(seeds.tolist()))
    train_mask, test_mask = train_test_case_masks(seeds, test_seed)

    flat = flatten_fields(conc).astype(np.float64)       # (C, T, F)
    pod = PODModel().fit(stack_samples(flat, train_mask))
    r_max = max([args.rank, *args.rank_sweep])
    coeffs = pod.transform(flat, r_max)                  # (C, T, r_max)
    print(f"  POD fitted on {train_mask.sum()} training cases; "
          f"held-out seed = {test_seed}")

    truth = flat[test_mask]                              # (Cte, T, F)
    test_st = stokes[test_mask]
    horizons = times - times[0]
    persistence = field_rmse(truth[:, :1], truth)        # c_hat(t) = c(0)
    floor = field_rmse(pod.reconstruct(truth, args.rank), truth)

    results = {"test_seed": test_seed, "rank": args.rank,
               "holdout": f"seed={test_seed:g}"}
    curves = {}
    for name, per_st in [("global", False), ("per_stokes", True)]:
        preds, radii = forecast(pod, coeffs, train_mask, test_mask, stokes,
                                args.rank, per_st)
        err = field_rmse(preds, truth)
        curves[name] = err
        results[name] = {
            "mean_forecast_rmse_final": float(err.mean(0)[-1]),
            "mean_forecast_rmse_time_avg": float(err.mean()),
            "mean_forecast_rmse_at_t": at_horizons(horizons, err.mean(0)),
            "spectral_radius": radii,
            "per_stokes": {
                f"St={st:g}": {"forecast_final": float(err[test_st == st].mean(0)[-1]),
                               "persistence_final": float(persistence[test_st == st].mean(0)[-1])}
                for st in sorted(set(test_st.tolist()))},
        }
        print(f"  DMD {name:<10} r={args.rank}: final RMSE {err.mean(0)[-1]:.3e}, "
              f"time-avg {err.mean():.3e}  "
              f"(spectral radius max {max(radii.values()):.4f})")

    results["mean_persistence_rmse_final"] = float(persistence.mean(0)[-1])
    results["mean_persistence_rmse_time_avg"] = float(persistence.mean())
    results["mean_persistence_rmse_at_t"] = at_horizons(horizons, persistence.mean(0))
    results["pod_projection_floor_final"] = float(floor.mean(0)[-1])
    print(f"  persistence final RMSE {persistence.mean(0)[-1]:.3e} | "
          f"POD-{args.rank} projection floor {floor.mean(0)[-1]:.3e}")

    sweep = {}
    for r in args.rank_sweep:
        preds, _ = forecast(pod, coeffs, train_mask, test_mask, stokes, r, True)
        sweep[str(r)] = float(field_rmse(preds, truth).mean(0)[-1])
    results["per_stokes_rank_sweep_final"] = sweep
    print("  per-Stokes DMD final RMSE vs rank: "
          + ", ".join(f"r={r}: {v:.3e}" for r, v in sweep.items()))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(horizons, persistence.mean(0), "k--", lw=2.5, label="persistence")
    ax.plot(horizons, curves["global"].mean(0), "C3-", lw=2,
            label=f"DMD, global (r={args.rank})")
    ax.plot(horizons, curves["per_stokes"].mean(0), "C0-", lw=2.5,
            label=f"DMD, per-Stokes (r={args.rank})")
    ax.plot(horizons, floor.mean(0), ":", color="grey", lw=2,
            label=f"POD-{args.rank} projection floor")
    ax.set_xlabel("forecast horizon (time units)")
    ax.set_ylabel("field RMSE (physical units)")
    ax.set_title(f"DMD forecast vs persistence (held-out seed {test_seed})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    f = outdir / "dmd_error_vs_horizon.png"
    fig.savefig(f, dpi=args.dpi); plt.close(fig)
    print(f"  wrote {f}")

    out_json = outdir / "dmd_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"  wrote {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
