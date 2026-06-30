"""Training loops and latent-forecasting utilities (PyTorch).

Kept backend-agnostic and CPU-friendly. Physical (unscaled) RMSE is tracked
each epoch so the numbers are directly comparable to the POD / persistence
baselines.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import ConvAutoencoder, LatentForecaster


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Autoencoder
# --------------------------------------------------------------------------
def train_autoencoder(
    model: ConvAutoencoder,
    train_flat: torch.Tensor,
    val_flat: torch.Tensor,
    scale: float,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    log: bool = True,
) -> dict:
    """Train the autoencoder; return a history dict with physical RMSE."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(train_flat), batch_size=batch_size,
                        shuffle=True)
    val = val_flat.to(device)

    history = {"epoch": [], "train_rmse": [], "val_rmse": []}
    for ep in range(1, epochs + 1):
        model.train()
        sq_err, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()
            sq_err += loss.item() * xb.shape[0]
            n += xb.shape[0]
        train_rmse = np.sqrt(sq_err / n) * scale     # physical units

        model.eval()
        with torch.no_grad():
            val_rmse = torch.sqrt(((model(val) - val) ** 2).mean()).item() * scale

        history["epoch"].append(ep)
        history["train_rmse"].append(float(train_rmse))
        history["val_rmse"].append(float(val_rmse))
        if log and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print(f"  epoch {ep:>3}/{epochs}  "
                  f"train RMSE {train_rmse:.3e}  val RMSE {val_rmse:.3e}",
                  flush=True)
    return history


@torch.no_grad()
def encode_cases(model: ConvAutoencoder, cases: torch.Tensor,
                 device: str = "cpu") -> torch.Tensor:
    """Encode per-case field sequences to latent sequences.

    ``cases`` has shape (C, T, 1, H, W); returns (C, T, latent).
    """
    model.eval().to(device)
    C, T = cases.shape[:2]
    flat = cases.reshape(C * T, 1, cases.shape[-2], cases.shape[-1]).to(device)
    z = model.encode(flat)
    return z.view(C, T, -1).cpu()


# --------------------------------------------------------------------------
# Latent forecaster
# --------------------------------------------------------------------------
def make_pairs(latent_seqs: torch.Tensor, stokes: np.ndarray):
    """Build one-step training pairs (z_t, z_{t+1}, log10 St) from sequences.

    ``latent_seqs`` has shape (C, T, L). Returns flat tensors over all cases
    and times.
    """
    C, T, L = latent_seqs.shape
    z_t = latent_seqs[:, :-1].reshape(-1, L)
    z_t1 = latent_seqs[:, 1:].reshape(-1, L)
    st = np.repeat(np.log10(stokes).astype(np.float32), T - 1).reshape(-1, 1)
    return z_t, z_t1, torch.from_numpy(st)


def train_forecaster(
    model: LatentForecaster,
    z_t: torch.Tensor,
    z_t1: torch.Tensor,
    st_feat: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    log: bool = True,
) -> dict:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(z_t, z_t1, st_feat)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = {"epoch": [], "loss": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for zb, zb1, sb in loader:
            zb, zb1, sb = zb.to(device), zb1.to(device), sb.to(device)
            opt.zero_grad()
            pred = model(zb, sb if model.conditioned else None)
            loss = loss_fn(pred, zb1)
            loss.backward()
            opt.step()
            tot += loss.item() * zb.shape[0]
            n += zb.shape[0]
        history["epoch"].append(ep)
        history["loss"].append(tot / n)
        if log and (ep == 1 or ep % 25 == 0 or ep == epochs):
            print(f"  epoch {ep:>3}/{epochs}  latent MSE {tot / n:.3e}",
                  flush=True)
    return history


def make_windows(latent_seqs: torch.Tensor, stokes: np.ndarray, rollout: int):
    """Build multi-step training windows from latent sequences.

    Returns ``z0`` (M, L) start states, ``targets`` (M, rollout, L) of the next
    ``rollout`` latent states, and ``st_feat`` (M, 1) = log10 St per window.
    """
    C, T, L = latent_seqs.shape
    z0_list, tgt_list, st_list = [], [], []
    for c in range(C):
        for t0 in range(T - rollout):
            z0_list.append(latent_seqs[c, t0])
            tgt_list.append(latent_seqs[c, t0 + 1: t0 + 1 + rollout])
        st_list.append(np.full(T - rollout, np.log10(stokes[c]), dtype=np.float32))
    z0 = torch.stack(z0_list)
    targets = torch.stack(tgt_list)
    st_feat = torch.from_numpy(np.concatenate(st_list)).unsqueeze(1)
    return z0, targets, st_feat


def train_forecaster_seq(
    model,
    z0: torch.Tensor,
    targets: torch.Tensor,
    st_feat: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    log: bool = True,
) -> dict:
    """Train a latent forecaster over ``rollout``-step recursive windows.

    ``targets`` has shape (M, rollout, L); the model is applied recursively and
    the loss is the mean MSE across all roll-out steps (curriculum-friendly).
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(z0, targets, st_feat),
                        batch_size=batch_size, shuffle=True)
    rollout = targets.shape[1]

    history = {"epoch": [], "loss": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for z, tgt, sf in loader:
            z, tgt, sf = z.to(device), tgt.to(device), sf.to(device)
            opt.zero_grad()
            zc = z
            loss = 0.0
            for k in range(rollout):
                zc = model(zc, sf if model.conditioned else None)
                loss = loss + loss_fn(zc, tgt[:, k])
            loss = loss / rollout
            loss.backward()
            opt.step()
            tot += loss.item() * z.shape[0]
            n += z.shape[0]
        history["epoch"].append(ep)
        history["loss"].append(tot / n)
        if log and (ep == 1 or ep % 25 == 0 or ep == epochs):
            print(f"  epoch {ep:>3}/{epochs}  latent MSE {tot / n:.3e}",
                  flush=True)
    return history


@torch.no_grad()
def recursive_forecast(
    model: LatentForecaster,
    z0: torch.Tensor,
    n_steps: int,
    st_feat: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Roll the latent map forward ``n_steps`` from initial latents ``z0``.

    ``z0`` has shape (B, L); returns (B, n_steps + 1, L) including the start.
    """
    model.eval().to(device)
    z = z0.to(device)
    out = [z]
    sf = st_feat.to(device) if st_feat is not None else None
    for _ in range(n_steps):
        z = model(z, sf if model.conditioned else None)
        out.append(z)
    return torch.stack(out, dim=1).cpu()
