#!/usr/bin/env python3
"""Overlay the autoencoder's reconstruction error on the POD rank curve.

Reads a ``baseline_results.json`` (POD reconstruction RMSE vs number of modes)
and an ``ae_results.json`` (best validation RMSE of the latent-dim-``d``
convolutional autoencoder) for the *same* dataset and held-out split, and draws
the linear-vs-nonlinear reduced-order comparison: at what POD rank does the
linear basis finally match the nonlinear autoencoder at its own latent size?

This is the figure that answers "does the nonlinear ROM actually beat POD?".
On the smooth, low-rank Taylor-Green fields the answer is no (POD is optimal);
on the multiscale Fourier flow the autoencoder wins at equal latent size.

Example
-------
    python scripts/compare_roms.py \
        --baseline figures/fourier/baseline_results.json \
        --ae figures/fourier/ae_results.json \
        -o figures/fourier/rom_comparison.png --title "Fourier flow"
"""

import _bootstrap  # noqa: F401

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True,
                   help="POD baseline_results.json for the dataset.")
    p.add_argument("--ae", required=True,
                   help="ae_results.json for the same dataset/split.")
    p.add_argument("-o", "--out", required=True, help="Output figure path.")
    p.add_argument("--title", default="", help="Flow name for the title.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def equivalent_pod_rank(ranks, rmse, ae_rmse):
    """First POD rank whose RMSE drops to (or below) the autoencoder's.

    Returns ``None`` if even the largest evaluated POD basis cannot match the
    autoencoder (i.e. the nonlinear model wins outright over the whole range).
    """
    for r, e in zip(ranks, rmse):
        if e <= ae_rmse:
            return r
    return None


def main():
    args = parse_args()
    pod = json.load(open(args.baseline))["pod"]
    ae = json.load(open(args.ae))
    ranks = np.asarray(pod["ranks"], dtype=float)
    rmse = np.asarray(pod["rmse"], dtype=float)
    ae_rmse = float(ae["best_val_rmse"])
    latent = int(ae["latent_dim"])
    e90 = pod["modes_for_90pct_energy"]

    pod_at_latent = float(np.interp(latent, ranks, rmse))
    match = equivalent_pod_rank(pod["ranks"], pod["rmse"], ae_rmse)
    wins = ae_rmse < pod_at_latent

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.semilogy(ranks, rmse, "o-", lw=2, color="#1f77b4", label="POD (linear)")
    ax.axhline(ae_rmse, color="#d62728", ls="--", lw=2,
               label=f"autoencoder, latent {latent} (nonlinear)")
    ax.plot([latent], [ae_rmse], "s", color="#d62728", ms=9, zorder=5)

    # Mark the POD rank that matches the autoencoder, if any. The dotted line
    # locates it; a plain label next to it explains what it means (no arrow).
    if match is not None:
        ax.axvline(match, color="grey", ls=":", lw=1)
        ax.text(match + 3, ae_rmse * 1.05,
                f"POD needs ~{match} modes\nto match the {latent}-dim AE",
                fontsize=9, color="dimgrey", va="bottom", ha="left")

    verdict = ("Nonlinear ROM wins at equal latent size"
               if wins else "Linear POD is not beaten at equal latent size")
    title = "Linear vs nonlinear ROM"
    if args.title:
        title += f"  —  {args.title}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("number of POD modes  r")
    ax.set_ylabel("held-out reconstruction RMSE")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    # Verdict + energy context as a footnote box, clear of the title/curve.
    ax.text(0.02, 0.03,
            f"{verdict}\nPOD needs {e90} modes for 90% energy",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))
    fig.tight_layout()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)

    print(f"  autoencoder (latent {latent}) RMSE : {ae_rmse:.3e}")
    print(f"  POD @ r={latent} RMSE              : {pod_at_latent:.3e}")
    print(f"  equivalent POD rank                : "
          f"{match if match is not None else '>%d' % int(ranks[-1])}")
    print(f"  verdict                            : {verdict}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
