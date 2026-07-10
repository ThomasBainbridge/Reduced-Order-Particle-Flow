"""PyTorch models for reduced-order reconstruction and latent forecasting.

* :class:`ConvAutoencoder` -- a convolutional autoencoder that compresses a
  64x64 concentration field to a low-dimensional latent vector and
  reconstructs it: ``c_t -> z_t -> c_hat_t``. Default latent dimension 16,
  the nonlinear counterpart of the 16-mode POD baseline.

* :class:`LatentForecaster` -- a small residual MLP that advances the latent
  state one step, ``z_{t+1} = z_t + f(z_t [, St])``, optionally conditioned on
  the Stokes number. Recursively applied, it forecasts the field evolution
  entirely in latent space.

* :class:`GRUForecaster` -- a gated recurrent one-step map that additionally
  carries a hidden memory ``h`` across the roll-out, so the update can depend
  on the trajectory history rather than the current latent alone.

All forecasters expose ``step(z, st_feat, h) -> (z_next, h_next)``; memoryless
models simply return ``h_next = None``. The training and roll-out loops in
:mod:`ropf.train` thread ``h`` through this interface.

The encoder/decoder use stride-2 (transpose-)convolutions, so the spatial
size halves/doubles cleanly 64 -> 32 -> 16 -> 8 -> 4 and back.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16, base_channels: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        c = base_channels

        def enc_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        # 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            enc_block(1, c),
            enc_block(c, 2 * c),
            enc_block(2 * c, 4 * c),
            enc_block(4 * c, 8 * c),
        )
        self._flat = 8 * c * 4 * 4
        self.to_latent = nn.Linear(self._flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self._flat)
        self._unflat_shape = (8 * c, 4, 4)

        def dec_block(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        # 4 -> 8 -> 16 -> 32 -> 64
        self.decoder = nn.Sequential(
            dec_block(8 * c, 4 * c),
            dec_block(4 * c, 2 * c),
            dec_block(2 * c, c),
            nn.ConvTranspose2d(c, 1, kernel_size=4, stride=2, padding=1),
        )
        # Concentration is non-negative; Softplus enforces it smoothly.
        self.out_activation = nn.Softplus()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.to_latent(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(-1, *self._unflat_shape)
        return self.out_activation(self.decoder(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class LatentForecaster(nn.Module):
    """Residual MLP one-step latent map ``z_{t+1} = z_t + f(z_t [, St_feat])``.

    If ``conditioned`` is True, an extra scalar feature (e.g. log10 St) is
    concatenated to the latent input, enabling Stokes-conditioned forecasting.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden: int = 128,
        depth: int = 2,
        conditioned: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.conditioned = conditioned
        in_dim = latent_dim + (1 if conditioned else 0)

        layers = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, st_feat: torch.Tensor = None) -> torch.Tensor:
        if self.conditioned:
            if st_feat is None:
                raise ValueError("conditioned forecaster requires st_feat")
            inp = torch.cat([z, st_feat], dim=-1)
        else:
            inp = z
        return z + self.net(inp)

    def step(self, z: torch.Tensor, st_feat: torch.Tensor = None, h=None):
        return self.forward(z, st_feat), None


class NeuralODEForecaster(nn.Module):
    """Continuous-time latent dynamics ``dz/dt = f(z [, St])``.

    A neural ODE: a small MLP parameterises the latent vector field, and one
    snapshot interval is advanced by fixed-step RK4 integration over a unit of
    (rescaled) time split into ``n_substeps``. The learned field absorbs the
    physical time scale, so ``forward(z) -> z_{t+1}`` is a drop-in replacement
    for :class:`LatentForecaster`.

    This is the direct fluids analogue of the Universal-Differential-Equation
    machinery: a learned right-hand side integrated by a classical ODE solver.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden: int = 128,
        depth: int = 2,
        conditioned: bool = False,
        n_substeps: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.conditioned = conditioned
        self.n_substeps = n_substeps
        in_dim = latent_dim + (1 if conditioned else 0)

        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, latent_dim)]
        self.func = nn.Sequential(*layers)

    def _rhs(self, z, st_feat):
        if self.conditioned:
            if st_feat is None:
                raise ValueError("conditioned forecaster requires st_feat")
            return self.func(torch.cat([z, st_feat], dim=-1))
        return self.func(z)

    def forward(self, z: torch.Tensor, st_feat: torch.Tensor = None) -> torch.Tensor:
        h = 1.0 / self.n_substeps
        for _ in range(self.n_substeps):
            k1 = self._rhs(z, st_feat)
            k2 = self._rhs(z + 0.5 * h * k1, st_feat)
            k3 = self._rhs(z + 0.5 * h * k2, st_feat)
            k4 = self._rhs(z + h * k3, st_feat)
            z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z

    def step(self, z: torch.Tensor, st_feat: torch.Tensor = None, h=None):
        return self.forward(z, st_feat), None


class GRUForecaster(nn.Module):
    """Recurrent one-step latent map with a persistent hidden memory.

    A single :class:`torch.nn.GRUCell` consumes the current latent state (plus
    the optional Stokes feature) and updates a hidden memory ``h``; a linear
    head turns the memory into a residual update ``z_{t+1} = z_t + W h``.

    Unlike the memoryless residual MLP, the hidden state threaded through a
    roll-out lets the update depend on the trajectory history, not just the
    instantaneous latent -- the longer-memory temporal model. Trained with
    ``--rollout k > 1`` so backpropagation-through-time can shape the memory;
    called without ``h`` (plain ``forward``) it reduces to a memoryless map.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden: int = 128,
        conditioned: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.conditioned = conditioned
        in_dim = latent_dim + (1 if conditioned else 0)
        self.cell = nn.GRUCell(in_dim, hidden)
        self.head = nn.Linear(hidden, latent_dim)

    def step(self, z: torch.Tensor, st_feat: torch.Tensor = None, h=None):
        if self.conditioned:
            if st_feat is None:
                raise ValueError("conditioned forecaster requires st_feat")
            inp = torch.cat([z, st_feat], dim=-1)
        else:
            inp = z
        if h is None:
            h = z.new_zeros(z.shape[0], self.hidden)
        h = self.cell(inp, h)
        return z + self.head(h), h

    def forward(self, z: torch.Tensor, st_feat: torch.Tensor = None) -> torch.Tensor:
        return self.step(z, st_feat)[0]


FORECASTERS = {"mlp": LatentForecaster, "ode": NeuralODEForecaster,
               "gru": GRUForecaster}
