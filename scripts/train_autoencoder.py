#!/usr/bin/env python3
"""Milestone 3: train the convolutional autoencoder (c_t -> z_t -> c_hat_t).

Trains on all Stokes numbers from the training seeds, validates on a held-out
seed, and reports physical reconstruction RMSE (directly comparable to the
POD baseline). Saves the model checkpoint, a training-curve figure, and a
truth-vs-reconstruction panel.

Example
-------
    python scripts/train_autoencoder.py -i data/particle_concentration.npz \
        --epochs 40 --latent-dim 16
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
from ropf.models import ConvAutoencoder
from ropf.torch_data import prepare
from ropf.train import set_seed, train_autoencoder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default="data/particle_concentration.npz")
    p.add_argument("-o", "--outdir", default="figures")
    p.add_argument("--ckpt", default="checkpoints/autoencoder.pt")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test-seed", type=int, default=None,
                   help="Seed to hold out for validation (default: last seed).")
    p.add_argument("--test-stokes", type=float, default=None,
                   help="Hold out an entire Stokes number instead of a seed.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = pathlib.Path(args.ckpt); ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    data = prepare(load_dataset(args.input), test_seed=args.test_seed,
                   test_stokes=args.test_stokes)
    print(f"  train fields {tuple(data.train_flat.shape)} | "
          f"val fields {tuple(data.test_flat.shape)} | "
          f"held-out {data.label} | scale {data.scale:.3e}")

    model = ConvAutoencoder(latent_dim=args.latent_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training ConvAutoencoder (latent={args.latent_dim}, "
          f"{n_params/1e6:.2f}M params) ...")

    history = train_autoencoder(
        model, data.train_flat, data.test_flat, data.scale,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    best_val = min(history["val_rmse"])
    print(f"Best validation reconstruction RMSE: {best_val:.3e}")

    torch.save(
        {"state_dict": model.state_dict(),
         "latent_dim": args.latent_dim,
         "scale": data.scale,
         "holdout_kind": data.holdout_kind,
         "holdout_value": data.holdout_value,
         "history": history},
        ckpt_path,
    )
    print(f"  saved checkpoint -> {ckpt_path}")

    # Training curve
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogy(history["epoch"], history["train_rmse"], label="train")
    ax.semilogy(history["epoch"], history["val_rmse"], label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("reconstruction RMSE (physical units)")
    ax.set_title(f"Autoencoder training (latent dim {args.latent_dim})")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    f = outdir / "ae_training_curve.png"; fig.savefig(f, dpi=args.dpi); plt.close(fig)
    print(f"  wrote {f}")

    # Truth vs reconstruction panel on the most-structured validation fields
    model.eval()
    with torch.no_grad():
        recon = model(data.test_flat).numpy()[:, 0]
    truth = data.test_flat.numpy()[:, 0]
    var = truth.reshape(truth.shape[0], -1).var(axis=1)
    idx = np.argsort(var)[-4:][::-1]            # 4 most-structured fields
    fig, axes = plt.subplots(2, len(idx), figsize=(2.6 * len(idx), 5.2))
    for col, i in enumerate(idx):
        vmax = np.percentile(truth[i], 99.5) or truth[i].max()
        axes[0, col].imshow(truth[i] * data.scale, origin="lower", cmap="inferno",
                            vmin=0, vmax=vmax * data.scale)
        axes[1, col].imshow(np.clip(recon[i], 0, None) * data.scale, origin="lower",
                            cmap="inferno", vmin=0, vmax=vmax * data.scale)
        for r in (0, 1):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
    axes[0, 0].set_ylabel("truth"); axes[1, 0].set_ylabel("reconstruction")
    fig.suptitle(f"Autoencoder reconstruction (held-out {data.label})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f = outdir / "ae_reconstruction_panel.png"; fig.savefig(f, dpi=args.dpi); plt.close(fig)
    print(f"  wrote {f}")

    (outdir / "ae_results.json").write_text(json.dumps(
        {"best_val_rmse": best_val, "latent_dim": args.latent_dim,
         "epochs": args.epochs, "n_params": int(n_params),
         "holdout": data.label, "history": history}, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
