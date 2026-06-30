#!/usr/bin/env python3
"""Milestone 4: train the latent-space forecaster and evaluate roll-outs.

Loads the trained autoencoder, encodes every snapshot to its latent vector,
and trains a latent dynamics model z_t -> z_{t+1} on the training cases. It
then recursively forecasts each held-out test case from its initial latent
state, decodes back to fields, and compares against the persistence baseline
as a function of horizon -- including whether the forecast preserves the
physical clustering index.

Two latent models are available (``--model``):
  * ``mlp`` -- residual MLP  z_{t+1} = z_t + f(z_t)        (default)
  * ``ode`` -- neural ODE    dz/dt = f(z), RK4-integrated  (UDE-style)

Multi-step (curriculum) training is enabled with ``--rollout k``.

Example
-------
    python scripts/train_forecaster.py --ckpt checkpoints/autoencoder.pt \
        --model ode --rollout 4 --epochs 200
"""

import _bootstrap  # noqa: F401

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ropf import load_dataset
from ropf.diagnostics import clustering_index
from ropf.models import FORECASTERS, ConvAutoencoder
from ropf.torch_data import prepare, stokes_feature
from ropf.train import (
    encode_cases,
    make_windows,
    recursive_forecast,
    set_seed,
    train_forecaster_seq,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default="data/particle_concentration.npz")
    p.add_argument("-o", "--outdir", default="figures")
    p.add_argument("--ckpt", default="checkpoints/autoencoder.pt")
    p.add_argument("--fc-ckpt", default="checkpoints/forecaster.pt")
    p.add_argument("--model", choices=["mlp", "ode"], default="mlp")
    p.add_argument("--rollout", type=int, default=1,
                   help="Training roll-out length (curriculum); 1 = one-step.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--conditioned", action="store_true",
                   help="Condition the latent map on Stokes number.")
    p.add_argument("--tag", default="", help="Suffix for output filenames.")
    p.add_argument("--gif-stokes", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def decode_latents(ae, latents):
    """Decode (T, L) latents -> (T, H, W) fields (scaled units)."""
    with torch.no_grad():
        return ae.decode(latents).squeeze(1).numpy()


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fc_path = pathlib.Path(args.fc_ckpt); fc_path.parent.mkdir(parents=True, exist_ok=True)
    tag = (f"_{args.tag}" if args.tag else "")

    # --- Load autoencoder + reproduce its train/test split -----------
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ae = ConvAutoencoder(latent_dim=ck["latent_dim"])
    ae.load_state_dict(ck["state_dict"]); ae.eval()
    holdout_kind = ck.get("holdout_kind", "seed")
    holdout_value = ck.get("holdout_value", ck.get("test_seed"))
    data = prepare(
        load_dataset(args.input),
        test_seed=holdout_value if holdout_kind == "seed" else None,
        test_stokes=holdout_value if holdout_kind == "stokes" else None,
    )
    scale, times = data.scale, data.time
    n_particles = load_dataset(args.input)["metadata"]["config"]["n_particles"]
    print(f"Loaded autoencoder (latent={ck['latent_dim']}, held-out {data.label}).")

    # --- Encode all snapshots to latent sequences --------------------
    z_train = encode_cases(ae, data.train_cases)     # (Ctr, T, L)
    z_test = encode_cases(ae, data.test_cases)       # (Cte, T, L)

    # --- Train the latent forecaster ---------------------------------
    z0, targets, st_feat = make_windows(z_train, data.train_stokes, args.rollout)
    fc = FORECASTERS[args.model](latent_dim=ck["latent_dim"], hidden=args.hidden,
                                 conditioned=args.conditioned)
    print(f"Training {args.model} forecaster (rollout={args.rollout}, "
          f"conditioned={args.conditioned}) ...")
    hist = train_forecaster_seq(fc, z0, targets, st_feat, epochs=args.epochs)
    torch.save({"state_dict": fc.state_dict(), "latent_dim": ck["latent_dim"],
                "model": args.model, "conditioned": args.conditioned,
                "rollout": args.rollout, "history": hist}, fc_path)
    print(f"  saved forecaster -> {fc_path}")

    # --- Recursive roll-out from t=0 for every test case -------------
    T = z_test.shape[1]
    st_feat_test = stokes_feature(data.test_stokes) if args.conditioned else None
    z_pred = recursive_forecast(fc, z_test[:, 0], T - 1, st_feat_test)  # (Cte,T,L)

    true_fields = data.test_cases.squeeze(2).numpy()         # (Cte, T, H, W) scaled
    Cte = z_test.shape[0]
    forecast_rmse = np.zeros((Cte, T))
    persistence_rmse = np.zeros((Cte, T))
    ci_true = np.zeros((Cte, T))
    ci_pred = np.zeros((Cte, T))
    for c in range(Cte):
        pred_fields = np.clip(decode_latents(ae, z_pred[c]), 0, None)  # (T,H,W) scaled
        truth = true_fields[c]
        forecast_rmse[c] = np.sqrt(((pred_fields - truth) ** 2).mean(axis=(1, 2))) * scale
        persistence_rmse[c] = np.sqrt(((truth[0][None] - truth) ** 2).mean(axis=(1, 2))) * scale
        # Diagnostics on physical (sum-normalised) fields.
        ci_true[c] = clustering_index(truth / truth.sum(axis=(1, 2), keepdims=True), n_particles)
        ps = pred_fields / pred_fields.sum(axis=(1, 2), keepdims=True)
        ci_pred[c] = clustering_index(ps, n_particles)

    horizons = times - times[0]
    unique_st = sorted(set(data.test_stokes.tolist()))

    # --- Figure: forecast vs persistence (mean +/- std band) ---------
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fm, fs = forecast_rmse.mean(0), forecast_rmse.std(0)
    pm = persistence_rmse.mean(0)
    ax.plot(horizons, fm, "C0-", lw=2.5, label=f"{args.model} forecaster (mean)")
    ax.fill_between(horizons, fm - fs, fm + fs, color="C0", alpha=0.2,
                    label="forecaster +/- 1 std")
    ax.plot(horizons, pm, "k--", lw=2.5, label="persistence (mean)")
    for st in unique_st:
        m = data.test_stokes == st
        ax.plot(horizons, forecast_rmse[m].mean(0), lw=1, alpha=0.6,
                label=f"forecaster St={st:g}")
    ax.set_xlabel("forecast horizon (time units)")
    ax.set_ylabel("field RMSE (physical units)")
    ax.set_title(f"Latent forecast vs persistence (held-out {data.label})")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    f = outdir / f"forecast_error_vs_horizon{tag}.png"; fig.savefig(f, dpi=args.dpi); plt.close(fig)
    print(f"  wrote {f}")

    # --- Figure: clustering-index preservation -----------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(horizons, ci_true.mean(0), "k-", lw=2.5, label="truth")
    ax.plot(horizons, ci_pred.mean(0), "C1--", lw=2.5, label="forecast")
    ax.set_xlabel("forecast horizon (time units)")
    ax.set_ylabel("clustering index  D  (mean over test cases)")
    ax.set_title("Does the latent forecast preserve preferential concentration?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    f = outdir / f"forecast_clustering_index{tag}.png"; fig.savefig(f, dpi=args.dpi); plt.close(fig)
    print(f"  wrote {f}")

    # --- GIF: predicted vs true evolution for one test case ----------
    target = min(unique_st, key=lambda s: abs(s - args.gif_stokes))
    c = int(np.where(data.test_stokes == target)[0][0])
    pred_fields = np.clip(decode_latents(ae, z_pred[c]), 0, None) * scale
    truth = true_fields[c] * scale
    _make_forecast_gif(outdir / f"forecast_vs_true_St{target:g}{tag}.gif",
                       truth, pred_fields, times, target)
    print(f"  wrote forecast_vs_true_St{target:g}{tag}.gif")

    summary = {
        "model": args.model, "rollout": args.rollout,
        "conditioned": args.conditioned, "holdout": data.label,
        "mean_forecast_rmse_final": float(fm[-1]),
        "mean_persistence_rmse_final": float(pm[-1]),
        "clustering_index_final": {"truth": float(ci_true.mean(0)[-1]),
                                   "forecast": float(ci_pred.mean(0)[-1])},
        "per_stokes": {
            f"St={st:g}": {
                "forecast_final": float(forecast_rmse[data.test_stokes == st].mean(0)[-1]),
                "persistence_final": float(persistence_rmse[data.test_stokes == st].mean(0)[-1]),
            } for st in unique_st},
    }
    (outdir / f"forecast_results{tag}.json").write_text(json.dumps(summary, indent=2))
    print("Done.")


def _make_forecast_gif(path, truth, pred, times, stokes, fps=20):
    from matplotlib.animation import FuncAnimation, PillowWriter

    vmax = np.percentile(truth, 99.5) or truth.max()
    fig, (axt, axp) = plt.subplots(1, 2, figsize=(8.6, 4.5))
    im_t = axt.imshow(truth[0], origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    im_p = axp.imshow(pred[0], origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    axt.set_title("truth"); axp.set_title("latent forecast")
    for ax in (axt, axp):
        ax.set_xticks([]); ax.set_yticks([])
    sup = fig.suptitle("")

    def update(k):
        im_t.set_data(truth[k]); im_p.set_data(pred[k])
        sup.set_text(f"St = {stokes:g}   t = {times[k]:.1f}   (recursive forecast)")
        return im_t, im_p, sup

    anim = FuncAnimation(fig, update, frames=truth.shape[0],
                         interval=1000 / fps, blit=False)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    main()
