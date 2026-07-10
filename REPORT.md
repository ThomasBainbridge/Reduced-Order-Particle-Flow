# Reduced-Order Forecasting of Particle-Laden Flow Evolution

**A two-page summary.** Full derivations, every figure, and reproduction
commands are in [RESULTS.md](RESULTS.md); the code and a one-command pipeline
are in [README.md](README.md).

---

## Problem

Inertial particles in a vortical flow do not follow the fluid — they are
centrifuged out of vortex cores and clump along strain regions
(*preferential concentration*). This project builds a small, fully reproducible
testbed for that physics and asks a data-driven-modelling question:

> Can a low-dimensional learned representation **reconstruct and forecast** the
> evolving particle-concentration field, and when does a **nonlinear** model
> (autoencoder) actually beat a **linear** one (POD/PCA)?

The carrier flow is prescribed analytically (a Taylor–Green vortex, and a
divergence-free multi-mode Fourier flow), so the data-generation and
reduced-order-modelling machinery is developed against clean, controlled
physics rather than a noisy DNS.

**Pipeline:** prescribed flow → inertial-particle dynamics → Eulerian 64×64
concentration fields → reduced-order representation (POD / convolutional
autoencoder) → latent-space forecasting → physically interpretable error
analysis.

---

## Key results

### 1. The physics is correct and resonant at St ≈ 1

![Physical diagnostics](docs/figures/physical_diagnostics.png)

Four independent diagnostics (clustering index, spatial entropy, peak
concentration, field variance) all identify **Stokes number St ≈ 1 as the
strongest-clustering regime**. The clustering index
`D = (σ_n − √⟨n⟩)/⟨n⟩` peaks at ≈ `1.85` for St = 1, with tight ±1σ bands over
seeds — the statistics are robust, not artefacts.

### 2. Linear POD vs nonlinear autoencoder — the decisive comparison

At equal latent size (16), whether the nonlinear ROM wins depends entirely on
whether the field is genuinely multiscale:

| Flow | POD modes for 90 % energy | POD @ 16 RMSE | Autoencoder (16-dim) RMSE | Winner |
|------|--------------------------:|--------------:|--------------------------:|--------|
| Smooth Taylor–Green | 24 | `6.69e-5` | `7.42e-5` | **POD** (low-rank → linear is optimal) |
| Raw Taylor–Green | 934 | `2.355e-4` | `2.382e-4` | POD (marginal; shot-noise floor) |
| **Multi-mode Fourier** | **1132** | `2.416e-4` | **`2.371e-4`** | **Autoencoder** |

![Linear vs nonlinear ROM on the Fourier flow](docs/figures/rom_comparison_fourier.png)

On the **Fourier flow** the POD spectrum never collapses (1132 modes for 90 %
energy) and its reconstruction error is nearly flat with mode count. There the
**16-dim autoencoder beats POD at equal latent size and matches a ~32-mode POD
basis** — a ~2× compression at equal accuracy. On the smooth, low-rank fields
POD is unbeatable, exactly as linear theory predicts. The honest conclusion: a
nonlinear ROM is not automatically better; it earns its keep only where the
physics is multiscale.

### 3. Latent-space forecasting beats persistence

A residual model on the frozen latent space, `z_{t+1} = z_t + f(z_t, St)`,
forecasts the field evolution and is rolled out recursively. The best variant —
**Stokes-conditioned, multi-step (curriculum) trained** — **halves the
persistence-baseline error**, with a final-time field RMSE of `1.51e-4`
(50 % below persistence) on the denoised data. A continuous-time **Neural-ODE**
latent model `dz/dt = f(z, St)` is also competitive (`2.32e-4`) once conditioned
and multi-step-trained, and a **GRU** with a hidden memory carried across the
roll-out (`1.64e-4`, trained on 16-step windows) is the only variant that beats
persistence in *every* Stokes regime — its memory wins the near-stationary
high-St cases where the memoryless MLP loses. Persistence only wins at very
short horizons.

### 4. Generalisation across Stokes number

Holding out an entire unseen Stokes number, the **encoder/decoder transfers**
(the learned field representation generalises), while the **forecasting dynamics
require Stokes conditioning** to do the same — a clean separation of what
generalises for free and what needs to be told about the regime.

---

## Honest limitations

- The carrier flow is **prescribed**, not solved — this is a controlled
  proof-of-concept, not a turbulent particle-laden DNS.
- Particles are **one-way coupled** (the flow moves particles; particles do not
  feed back on the flow).
- On the **raw** histogram fields, no low-rank model can beat the per-cell
  Poisson **shot-noise floor** (~`2.2e-4`); the ROM gains require the denoised
  representation.

---

## Reproduce

```bash
make env        # install deps (incl. CPU PyTorch)
make all        # full pipeline: data -> figures -> baselines -> AE -> forecaster
make fourier    # the linear-vs-nonlinear ROM comparison above
```

Everything runs on a laptop CPU. To render this report as a PDF (once `pandoc`
and a LaTeX engine are installed):

```bash
pandoc REPORT.md -o REPORT.pdf --resource-path=.
```
