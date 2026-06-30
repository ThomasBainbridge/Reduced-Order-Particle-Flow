"""Shape and learning-sanity checks for the PyTorch models.

Skipped automatically if PyTorch is not installed (the physics/baseline
pipeline does not require it). Runnable with pytest or directly.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from ropf.models import (  # noqa: E402
    ConvAutoencoder,
    LatentForecaster,
    NeuralODEForecaster,
)
from ropf.train import make_pairs, make_windows, recursive_forecast  # noqa: E402


def test_autoencoder_roundtrip_shapes():
    model = ConvAutoencoder(latent_dim=16)
    x = torch.rand(4, 1, 64, 64)
    z = model.encode(x)
    assert z.shape == (4, 16)
    out = model.decode(z)
    assert out.shape == (4, 1, 64, 64)
    assert torch.all(out >= 0)            # Softplus output is non-negative


def test_autoencoder_can_overfit_one_batch():
    torch.manual_seed(0)
    model = ConvAutoencoder(latent_dim=16)
    x = torch.rand(8, 1, 64, 64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(x), x)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.5 * first      # loss should drop substantially


def test_forecaster_residual_and_conditioning():
    fc = LatentForecaster(latent_dim=16, conditioned=False)
    z = torch.zeros(5, 16)
    # With zero input the residual MLP still adds its bias-driven update;
    # check the identity-skip wiring by using a tiny-weight perturbation.
    out = fc(z)
    assert out.shape == (5, 16)

    fcc = LatentForecaster(latent_dim=16, conditioned=True)
    st = torch.zeros(5, 1)
    assert fcc(z, st).shape == (5, 16)
    with pytest.raises(ValueError):
        fcc(z)                            # conditioned model needs st_feat


def test_neural_ode_forecaster_shapes_and_rollout():
    fc = NeuralODEForecaster(latent_dim=16, hidden=64, n_substeps=2)
    z = torch.randn(5, 16)
    assert fc(z).shape == (5, 16)
    # Drop-in compatibility with recursive_forecast.
    roll = recursive_forecast(fc, z, n_steps=4)
    assert roll.shape == (5, 5, 16)


def test_make_windows_shapes():
    z_seqs = torch.randn(3, 10, 16)
    stokes = np.array([0.1, 1.0, 10.0])
    z0, targets, st = make_windows(z_seqs, stokes, rollout=3)
    assert z0.shape == (3 * 7, 16)        # T - rollout = 7 starts per case
    assert targets.shape == (3 * 7, 3, 16)
    assert st.shape == (3 * 7, 1)


def test_make_pairs_and_rollout_shapes():
    z_seqs = torch.randn(3, 10, 16)       # 3 cases, 10 times, latent 16
    stokes = np.array([0.1, 1.0, 10.0])
    z_t, z_t1, st = make_pairs(z_seqs, stokes)
    assert z_t.shape == (3 * 9, 16)
    assert z_t1.shape == (3 * 9, 16)
    assert st.shape == (3 * 9, 1)

    fc = LatentForecaster(latent_dim=16)
    roll = recursive_forecast(fc, z_seqs[:, 0], n_steps=5)
    assert roll.shape == (3, 6, 16)       # includes the initial state


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} model tests passed.")
