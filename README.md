# Reduced-Order Forecasting of Particle-Laden Flow Evolution

![CI](https://github.com/ThomasBainbridge/Reduced-Order-Particle-Flow/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A compact **scientific-machine-learning** testbed that generates a controlled
2-D inertial-particle dataset, converts Lagrangian particle motion into
Eulerian concentration fields, and trains reduced-order PyTorch models to
**reconstruct and forecast** the concentration-field evolution.

> **Full results write-up with figures: [RESULTS.md](RESULTS.md).**

The full intended workflow is:

```
prescribed carrier flow
   -> inertial particle dynamics
      -> Eulerian concentration fields
         -> reduced-order representation (POD / autoencoder)
            -> neural reconstruction
               -> latent-space forecasting
                  -> physically interpretable error analysis
```

> This is **not** a high-fidelity turbulent particle-laden DNS. It is a
> deliberately controlled proof-of-concept whose carrier flow is prescribed
> analytically, so that the data-generation, reduced-order-modelling and
> forecasting machinery can be developed and validated against clean,
> reproducible physics.

**Status:** complete end-to-end and then some — physics & dataset generation
(M1), persistence + POD baselines (M2), a convolutional autoencoder (M3) and a
latent-space forecaster (M4) are all implemented, tested, and run; plus the
stretch work: a **Neural-ODE** latent forecaster, **Stokes-conditioned** and
**multi-step (curriculum)** training, **held-out-Stokes-number** generalisation,
physical-diagnostics analysis, a **Gaussian-KDE** denoised field option, and a
divergence-free **multi-mode Fourier** carrier flow. Headline numbers are in
[Results](#results) / [RESULTS.md](RESULTS.md); breakdown in the
[roadmap](#roadmap).

---

## Why this project

I am an MSc Computational Fluid Dynamics student whose thesis applies neural
networks and Universal Differential Equations to UAV parameter estimation.
This side project demonstrates the *data-driven reduced-order modelling*
half of that skill set on a fluids problem with genuinely non-trivial physics:
**inertial particle clustering (preferential concentration)** in an unsteady
vortical flow.

It shares a *workflow philosophy* with my earlier OpenFOAM dam-break
surrogate-modelling project (generate a physics database -> build a
reduced-order field surrogate) but is deliberately different in physics
(Lagrangian inertial particles instead of a VOF free surface) and in
ML focus (latent-space *forecasting* of an evolving field, not static
regression).

---

## Physics

### Carrier flow (prescribed, divergence-free)

An unsteady Taylor–Green-type vortex on the periodic square
`[0, 2π] × [0, 2π]`:

```
u_x(x, y, t) =  A(t) · sin(x) · cos(y)
u_y(x, y, t) = -A(t) · cos(x) · sin(y)
A(t)         = 1 + 0.25 · sin(ω t),     ω = 1
```

This field is smooth, `2π`-periodic, and analytically incompressible:

```
∂u_x/∂x + ∂u_y/∂y = A cos x cos y − A cos x cos y ≡ 0
```

so no CFD solver is required — the velocity is evaluated analytically at each
particle location. The time-dependent amplitude `A(t)` makes the flow
**unsteady**, giving the particle field a non-trivial, forecastable evolution.

### Inertial particles (Stokes drag)

Each particle obeys the linear equation of motion

```
dx_p/dt = v_p
dv_p/dt = ( u(x_p, t) − v_p ) / τ_p
```

with particle response time `τ_p`. Using a flow time scale `τ_f = 1`, the
**Stokes number** is simply

```
St = τ_p / τ_f = τ_p
```

| Stokes number | Behaviour |
|---|---|
| `St = 0.1` | particles almost follow the fluid (near-tracer) |
| `St = 1`   | particles respond on the vortex time scale → **strongest clustering** |
| `St = 5`   | particles significantly lag the flow |
| `St = 10`  | highly inertial, weakly follow the instantaneous flow |

Particles are integrated with **classical RK4** (Euler is also available via
`--integrator euler`). Positions are wrapped periodically; because the carrier
flow is `2π`-periodic, wrapping does not affect the sampled velocity.

### Lagrangian → Eulerian concentration

At each saved time, particle positions are binned onto a `64 × 64` grid with a
2-D histogram and normalised by the particle count, giving a
particle-concentration (number-density) field that **sums to one** (discrete
mass conservation). Fields are stored channel-first as `(1, ny, nx)`, ready for
PyTorch.

---

## Dataset

The default sweep is **4 Stokes numbers × 5 seeds = 20 cases**, each with
**201 snapshots** (`t = 0 … 20`, every `0.1`) on a `64 × 64` grid:

```
concentration : (20, 201, 1, 64, 64)  float32   # (case, time, channel, y, x)
stokes        : (20,)   float64                  # St for each case
seed          : (20,)   int64                    # seed for each case
time          : (201,)  float64                  # snapshot times
metadata      : JSON string                      # full config + provenance
```

Saved as a single compressed `.npz` (no extra dependencies; HDF5 can be
swapped in via the `[hdf5]` extra without touching the rest of the pipeline).
Generation is fully reproducible from the random seeds.

| Parameter | Value |
|---|---|
| Domain | `[0, 2π] × [0, 2π]`, periodic |
| Particles / case | 5000 |
| Time step `dt` | 0.01 (2000 steps) |
| Final time `T` | 20 |
| Save interval | 0.1 (201 snapshots) |
| Grid | 64 × 64 |
| Stokes numbers | 0.1, 1, 5, 10 |
| Seeds | 0, 1, 2, 3, 4 |
| Integrator | RK4 |

---

## Repository layout

```
.
├── src/ropf/                 # installable package ("reduced-order particle flow")
│   ├── config.py             # SimConfig dataclass: all numerical/physical params
│   ├── carrier_flow.py       # Taylor–Green velocity + divergence check
│   ├── particles.py          # vectorised RK4/Euler inertial-particle integrators
│   ├── concentration.py      # Lagrangian → Eulerian histogram binning
│   ├── simulation.py         # run one (St, seed) case → concentration snapshots
│   ├── dataset.py            # sweep + (de)serialisation to .npz
│   ├── metrics.py            # RMSE / relative-L2 (shared by baselines & NN)
│   ├── baselines.py          # persistence + POD/PCA reduced-order model
│   ├── diagnostics.py        # clustering index, entropy, peak, variance
│   ├── models.py             # ConvAutoencoder, LatentForecaster, NeuralODE (PyTorch)
│   ├── torch_data.py         # train/test split (seed or Stokes), scaling, tensors
│   ├── train.py              # training loops + multi-step roll-out utilities
│   └── viz.py                # comparison figures, diagnostics, GIFs
├── scripts/                  # command-line entry points (reproducible pipeline)
│   ├── generate_dataset.py   # M1: physics -> dataset
│   ├── make_figures.py       # M1: Stokes comparison + clustering diagnostics
│   ├── make_gifs.py          # M1: particle/concentration evolution GIFs
│   ├── run_baselines.py      # M2: persistence + POD baselines
│   ├── run_diagnostics.py    # physical diagnostics vs time and Stokes number
│   ├── train_autoencoder.py  # M3: convolutional autoencoder
│   └── train_forecaster.py   # M4: latent forecaster (MLP/ODE) + evaluation
├── tests/                    # test_physics.py (M1/M2) + test_models.py (M3/M4)
├── .github/workflows/ci.yml  # pytest + pipeline smoke test on push/PR
├── docs/figures/             # committed showcase figures (for RESULTS.md)
├── data/                     # generated datasets (git-ignored)
├── figures/                  # generated figures/GIFs (git-ignored)
├── checkpoints/              # trained model weights (git-ignored)
├── pyproject.toml            # packaging + optional [ml]/[hdf5]/[dev] extras
└── requirements.txt
```

The code is intentionally script-driven (no notebook-driven workflow), modular,
and fully vectorised over particles — there are **no Python loops over
individual particles**.

---

## Quickstart

```bash
# 1. Environment (Python ≥ 3.9)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (optional, makes `import ropf` work anywhere)
pip install -e .

# 2. Sanity-check the physics
pytest -q                      # or: python tests/test_physics.py

# 3. Generate the canonical dataset  (≈1–2 min on a laptop CPU)
python scripts/generate_dataset.py -o data/particle_concentration.npz

# 4. Render comparison figures
python scripts/make_figures.py -i data/particle_concentration.npz -o figures

# 5. Render evolution GIFs (one per Stokes number, seed 0)
python scripts/make_gifs.py -o figures --seed 0

# --- Machine-learning stages (require PyTorch) -------------------------
# CPU-only PyTorch is sufficient; the whole pipeline runs on a laptop.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 6. M2: persistence + POD/PCA baselines, and physical diagnostics
python scripts/run_baselines.py   -i data/particle_concentration.npz -o figures
python scripts/run_diagnostics.py -i data/particle_concentration.npz -o figures

# 7. M3: train the convolutional autoencoder (latent dim 16)
python scripts/train_autoencoder.py -i data/particle_concentration.npz --epochs 40

# 8. M4: train the latent forecaster and evaluate roll-outs vs persistence
python scripts/train_forecaster.py --ckpt checkpoints/autoencoder.pt --epochs 200
```

### Stretch experiments

```bash
# Denoised (Gaussian-KDE) dataset -> ~3x lower reconstruction floor:
python scripts/generate_dataset.py --smoothing 1.0 -o data/smooth.npz

# Stokes-conditioned, multi-step-trained forecaster (best result):
python scripts/train_autoencoder.py -i data/smooth.npz --ckpt checkpoints/ae.pt
python scripts/train_forecaster.py  --ckpt checkpoints/ae.pt -i data/smooth.npz \
    --conditioned --rollout 4 --tag cond

# Neural-ODE latent forecaster (UDE-style continuous-time dynamics):
python scripts/train_forecaster.py  --ckpt checkpoints/ae.pt -i data/smooth.npz \
    --model ode --conditioned --rollout 8 --tag ode

# Hold out an entire Stokes number to test generalisation:
python scripts/train_autoencoder.py -i data/smooth.npz --test-stokes 5 --ckpt checkpoints/ae5.pt

# More turbulence-like multi-mode Fourier carrier flow:
python scripts/generate_dataset.py --flow-type fourier -o data/fourier.npz
```

For a fast end-to-end check first, use the reduced configuration:

```bash
python scripts/generate_dataset.py --quick -o data/smoke.npz
```

Every script exposes `--help`. Common overrides: `--stokes 1 10`,
`--seeds 0 1 2`, `--n-particles`, `--nx/--ny`, `--integrator {rk4,euler}`;
for the ML scripts, `--latent-dim`, `--epochs`, `--test-seed`,
`--conditioned` (Stokes-conditioned forecasting).

---

## Results

All numbers below are on the **held-out seed** (seed 4, all four Stokes
numbers), with the models trained only on seeds 0–3.

### Physics — preferential concentration

The concentration fields reproduce the textbook inertial-clustering picture
(`stokes_comparison.png`): particles are centrifuged out of the vortex cores
and accumulate along the strain regions between cells.

- **St = 0.1** — near-tracer; clustering builds up slowly over many turnovers.
- **St = 1** — **strongest, fastest clustering** (resonant regime); sharp
  cell-boundary structure with hot spots at stagnation points.
- **St = 5 / 10** — increasingly inertial: the particles lag and decouple from
  the instantaneous flow, leaving noisier, more diffuse structure.

The clustering diagnostic (`clustering_diagnostics.png`) quantifies this as the
growth of spatial concentration variance with time.

### Reduced-order reconstruction (latent dim = 16)

| Model | Test reconstruction RMSE |
|---|---|
| POD / PCA, 16 modes | `2.35e-4` |
| Conv. autoencoder, 16-dim latent | `2.38e-4` |

The two are essentially tied — and both sit near the **histogram shot-noise
floor** (~`2.2e-4` for ~1.2 particles per cell). Crucially, the energy spectrum
shows the *raw* fields need ~930 POD modes for 90 % energy: the per-cell Poisson
noise is high-dimensional and incompressible, so both ROMs instead recover the
smooth **coherent** clustering structure and discard the noise
(`ae_reconstruction_panel.png` shows this denoising clearly). This is the key
physical insight of the reconstruction study.

### Latent-space forecasting vs persistence

Recursively rolling the latent map forward from `t = 0` to `t = 20` and decoding
back to fields, the learned forecaster **beats the persistence baseline at every
Stokes number** (final-time field RMSE):

| St | Persistence | Latent forecaster | Improvement |
|---|---|---|---|
| 0.1 | `6.26e-4` | `5.75e-4` | 8 % |
| 1   | `7.15e-4` | `6.72e-4` | 6 % |
| 5   | `3.26e-4` | `2.82e-4` | 13 % |
| 10  | `3.19e-4` | `2.39e-4` | **25 %** |
| **mean** | `4.97e-4` | `4.42e-4` | **11 %** |

`forecast_error_vs_horizon.png` shows the forecaster tracking below persistence
across medium-to-long horizons, with **St = 1 the hardest regime** (most dynamic
field) — consistent with the persistence-baseline analysis. For very short
horizons persistence wins, since the forecaster inevitably pays the
autoencoder's reconstruction-floor error while persistence is exact at `t = 0`.

> Figures and GIFs are git-ignored (they regenerate from the scripts in
> minutes); force-add a few showcase images with `git add -f figures/<name>`.

---

## Roadmap

The physics dataset and every ML stage are implemented and run end-to-end:

- [x] **M1 — Physics & data:** carrier flow, inertial particles, periodic BCs,
      concentration fields, dataset, comparison figures, GIFs, tests.
- [x] **M2 — Baselines:** persistence (`c_hat(t+1) = c(t)`) and a **POD/PCA**
      reduced-order reconstruction with energy spectrum and error-vs-modes.
- [x] **M3 — Convolutional autoencoder:** `c_t -> z_t -> c_hat_t`, latent dim 16;
      reconstruction RMSE vs epoch, reconstruction panel, comparison to POD.
- [x] **M4 — Latent forecaster:** residual MLP `z_{t+1} = z_t + f(z_t)` trained
      on the frozen latent space; recursive multi-step roll-out; forecast error
      vs horizon and per Stokes number, against the persistence baseline.

**Stretch — implemented and run:**
- [x] **Stokes-conditioned** forecasting `z_{t+1} = f(z_t, St)` (`--conditioned`).
- [x] **Neural-ODE** latent forecaster `dz/dt = f(z, St)` (`--model ode`) — the
      direct UDE analogue; needs conditioning + multi-step training to be stable.
- [x] **Multi-step (curriculum)** training (`--rollout k`).
- [x] **Physical diagnostics**: clustering index, spatial entropy, peak
      concentration, field variance (`run_diagnostics.py`), incl. whether the
      forecast preserves the clustering index.
- [x] **Gaussian-KDE denoising** of the field (`--smoothing`) — 3× lower floor.
- [x] **Held-out-Stokes-number** generalisation split (`--test-stokes`).
- [x] **Multi-mode Fourier** divergence-free carrier flow (`--flow-type fourier`).

**Remaining ideas:** a GRU latent forecaster, a nonlinear-ROM win on the Fourier
flow (where the fields are genuinely multiscale), and two-way coupling.

---

## Limitations

- The carrier flow is **prescribed and laminar** (single-scale Taylor–Green),
  not turbulent — there is no DNS and no two-way coupling or inter-particle
  collisions.
- Particles are **one-way coupled** point particles with linear Stokes drag
  (no gravity, lift, finite-size or Basset history effects).
- Concentration is a normalised **histogram**, so it carries shot noise that
  scales like `1/√(particles per cell)`; the `64 × 64` / 5000-particle choice
  trades resolution against this noise. This noise sets a hard floor on
  reconstruction RMSE that both POD and the autoencoder hit — a kernel-density
  estimate or more particles would lower it.
- The autoencoder and forecaster are deliberately **small and CPU-trainable**;
  they are tuned for a clean demonstration rather than squeezing out the last
  few percent. The forecaster is trained one-step (teacher-forced) and rolled
  out recursively, so it is not explicitly optimised against multi-step drift.
- Explicit integration: very small Stokes numbers (`St ≪ dt`) would become
  stiff; the smallest case here (`St = 0.1`) is comfortably resolved by RK4.

---

## License

MIT — see [LICENSE](LICENSE).
