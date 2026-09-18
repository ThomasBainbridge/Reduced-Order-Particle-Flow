# Results

A walk-through of the experiments, with the headline numbers and the figures
they come from. Unless stated otherwise, models are trained on four Stokes
numbers and seeds 0–3, and **evaluated on the held-out seed 4**. "Physical
units" means the original normalised concentration (each field sums to 1, so
typical values are ~`1/4096 ≈ 2.4e-4`).

Two dataset variants are used:

| Variant | Field generation | Why |
|---|---|---|
| **raw** | 64×64 histogram of 5000 particles | the honest, shot-noise-limited measurement |
| **smooth** | same, then a periodic Gaussian KDE (`sigma = 1` cell) | a denoised, near-low-rank representation for the ROMs |

---

## 1. Physics: preferential concentration

![Concentration fields vs Stokes number](docs/figures/stokes_comparison.png)

Inertial particles are centrifuged out of the vortex cores and accumulate
along the strain regions between cells. Four independent physical diagnostics
all identify **St ≈ 1 as the strongest-clustering (resonant) regime**, with
St = 0.1 accumulating slowly and St = 5, 10 decoupling from the instantaneous
flow:

![Physical diagnostics](docs/figures/physical_diagnostics.png)

The **clustering index** `D = (σ_n − √⟨n⟩)/⟨n⟩` (deviation of per-cell counts
from a random Poisson field) peaks near `1.85` for St = 1; spatial entropy is
correspondingly lowest. Error bands are ±1 std over seeds and are tight — the
clustering statistics are robust.

---

## 2. Reduced-order reconstruction (latent dim = 16)

| Dataset | POD (16 modes) | Conv. autoencoder (16-dim) | Modes for 90 % energy |
|---|---|---|---|
| raw    | `2.35e-4` | `2.38e-4` | **934** |
| smooth | `6.69e-5` | `7.42e-5` | **24** |

**On the raw data**, POD and the autoencoder *tie* — and both sit at the
**histogram shot-noise floor** (~`2.2e-4` for ~1.2 particles/cell). The raw
fields are not compressible by *any* low-rank basis: 934 POD modes are needed
for 90 % energy because the per-cell Poisson noise is high-dimensional.

![Raw POD energy spectrum](docs/figures/pod_energy_spectrum_raw.png)

**Smoothing changes everything**: the denoised fields are nearly low-rank
(24 modes → 90 % energy), so both ROMs improve ~**3×**. On these smooth,
low-rank fields linear POD is *near-optimal* and marginally beats the small
CAE at equal latent budget — a genuine, defensible result (the nonlinear
advantage would be expected to appear on more strongly multiscale fields, e.g.
the Fourier flow in §5). POD reconstructs the clustering structure almost
perfectly by r = 16:

![Smoothed POD reconstruction](docs/figures/pod_panel_smooth.png)

**Generalisation across Stokes number.** An autoencoder trained on
St = {0.1, 1, 10} reconstructs the *completely unseen* St = 5 at `9.7e-5` —
only marginally above the in-distribution `7.4e-5`. The learned field
representation generalises across the inertia parameter.

---

## 3. Latent-space forecasting vs persistence

The forecaster advances the field **entirely in latent space**
(`z_t → z_{t+1}`), is rolled out recursively from `t = 0` to `t = 20`, then
decoded back to fields. Every model is scored on the held-out seed against
**persistence** (`c(t) = c(0)`) and against **DMD** — the linear latent
forecaster, fitted on 16 POD coefficients (§3.1).

Neural forecasters are sensitive to their training seed at long horizons, so
the headline comparison on the denoised data averages **5 training seeds** of
each (mean ± sample std, all on the same trained autoencoder; `make seeds`).
Field RMSE at fixed horizons and averaged over the whole roll-out:

| Model (5 seeds) | `t = 1` | `t = 5` | `t = 10` | `t = 20` (final) | time-avg | beats persistence at every St |
|---|---|---|---|---|---|---|
| conditioned MLP, rollout-4 | **`7.38e-5`** ± `0.4e-5` | **`8.59e-5`** ± `0.5e-5` | `1.15e-4` ± `0.1e-4` | `1.66e-4` ± `0.4e-4` | `1.16e-4` ± `0.03e-4` | 0/5 |
| GRU, rollout-16 | `7.48e-5` ± `0.7e-5` | `9.41e-5` ± `1.7e-5` | `1.39e-4` ± `0.4e-4` | `2.11e-4` ± `0.6e-4` | `1.35e-4` ± `0.35e-4` | 1/5 |
| Neural-ODE, rollout-8 | `7.59e-5` ± `0.7e-5` | `9.96e-5` ± `1.1e-5` | `1.44e-4` ± `0.2e-4` | `2.20e-4` ± `0.3e-4` | `1.39e-4` ± `0.13e-4` | 0/5 |
| **DMD, per-Stokes** (deterministic) | `9.18e-5` | `1.23e-4` | **`1.13e-4`** | **`1.19e-4`** | **`1.09e-4`** | no (St = 10) |
| persistence | `1.36e-4` | `1.99e-4` | `2.43e-4` | `3.02e-4` | `2.29e-4` | – |

- **On average over seeds, every learned model beats persistence** at every
  horizon beyond the first few snapshots, cutting its final-time error by
  27–45 % (though one of the 15 runs, a GRU, finishes just behind it).
- **Short horizons belong to the neural models**: all 15 neural training runs
  beat DMD at `t = 1` and `t = 5`.
- **Long horizons belong to DMD**: it beats all 15 neural runs at `t = 20`,
  and has the lowest time-averaged error.
- **The seed matters most where errors compound.** The conditioned MLP's
  time-averaged error is tight across seeds (±2 %), but its final-time error
  ranges from `1.24e-4` to `2.20e-4`.

Among the neural forecasters, the **Stokes-conditioned, multi-step-trained
MLP** is the strongest. Its original training run (seed 0, final RMSE
`1.51e-4`, 50 % below persistence):

![Conditioned forecast vs persistence](docs/figures/forecast_smooth_conditioned.png)

Single-run final-time results, including the raw (un-denoised) fields, where
the shot-noise floor caps every model:

| Forecaster | Dataset | Forecast | Persistence | Improvement |
|---|---|---|---|---|
| residual MLP, 1-step | raw | `4.42e-4` | `4.97e-4` | 11 % |
| DMD, global (linear) | raw | `3.99e-4` | `4.97e-4` | 20 % |
| DMD, per-Stokes (linear) | raw | `2.88e-4` | `4.97e-4` | 42 % |
| conditioned MLP, rollout-4 (seed 0) | smooth | `1.51e-4` | `3.02e-4` | 50 % |
| Neural-ODE, conditioned, rollout-8 (seed 0) | smooth | `2.32e-4` | `3.02e-4` | 23 % |
| GRU, conditioned, rollout-16 (seed 0) | smooth | `1.64e-4` | `3.02e-4` | 46 % |
| DMD, global (linear) | smooth | `2.68e-4` | `3.02e-4` | 11 % |
| **DMD, per-Stokes (linear)** | smooth | **`1.19e-4`** | `3.02e-4` | **61 %** |

The raw-data one-step forecaster:

![Raw forecast vs persistence](docs/figures/forecast_raw.png)

**Neural ODE (the UDE bridge).** Parameterising the latent dynamics as a
continuous-time field `dz/dt = f(z, St)` integrated by RK4 is the direct
fluids analogue of the Universal-Differential-Equation machinery. Out of the
box it was **unstable** over the 200-step recursive roll-out (the clustering
index blew up to `2.4` vs a true `0.29`). Adding Stokes conditioning and
training on longer (8-step) roll-outs stabilised it: every one of the 5 seeds
beats persistence at the final time, and the original run roughly preserves
the clustering index, slightly under-diffusing it:

![ODE clustering-index preservation](docs/figures/ode_clustering_index.png)

**GRU (longer-memory dynamics).** A gated recurrent forecaster carries a
hidden memory `h` across the roll-out, so the latent update can depend on the
trajectory history rather than the current state alone
(`z_{t+1} = z_t + W h_t`). Like the Neural-ODE, it is **unstable when trained
short**: on 4-step windows the 200-step roll-out blows the clustering index up
to `3.9` (vs a true `0.29`) and loses to persistence. In single runs, the cure
was the **roll-out curriculum, not capacity**: halving the hidden size changed
little, while lengthening the training windows improved it steadily (rollout
4 → 8 → 16 gave `5.16e-4` → `3.73e-4` → `1.64e-4`). Backpropagation through
time only shapes the memory over horizons the training actually visits.

Even at rollout-16 the GRU is the most seed-sensitive model (final RMSE
`1.64e-4` to `3.12e-4` over 5 seeds). Its original run (seed 0) beat
persistence in every Stokes regime — the only neural run to do so — but only
1 of the 5 GRU seeds does, so that is a favourable draw rather than a property
of the model. Per-Stokes final RMSE of the original runs:

| St | conditioned MLP, rollout-4 | GRU, rollout-16 | DMD, per-Stokes | persistence |
|---|---|---|---|---|
| 0.1 | `1.74e-4` | `2.82e-4` | **`0.69e-4`** | `4.58e-4` |
| 1   | **`0.95e-4`** | `2.03e-4` | `2.06e-4` | `5.30e-4` |
| 5   | `2.39e-4` | `0.95e-4` | **`0.93e-4`** | `1.16e-4` |
| 10  | `1.12e-4` | **`0.74e-4`** | `1.10e-4` | `1.04e-4` |
| **mean** | `1.55e-4` | `1.64e-4` | **`1.19e-4`** | `3.02e-4` |

(The MLP column is an independent re-run of the `1.51e-4` configuration above.)
The stabilised GRU errs on the diffusive side — it slightly over-smooths
clustering (final index `-0.14` vs `0.29`), the mirror image of the
short-rollout blow-up:

![GRU forecast vs persistence](docs/figures/forecast_gru.png)

### 3.1 The linear baseline: DMD

Dynamic Mode Decomposition is to the latent forecasters what POD is to the
autoencoder: the same recursive roll-out, with the latent map restricted to be
**linear**. Here it is fitted in POD coordinates, `a_{t+1} = A a_t + b`
(16 modes, least squares on the training seeds), either as one operator for
all Stokes numbers (**global**) or one per Stokes number (**per-Stokes**, the
linear analogue of Stokes conditioning). It is then rolled out from `t = 0`
under exactly the protocol above. It has no training randomness.

![DMD forecast vs persistence](docs/figures/dmd_forecast_smooth.png)

This is the most important caveat in the study: **per-Stokes DMD is a very
strong baseline.** On the denoised data it has the lowest final-time error of
any model — below every one of the 15 neural training runs — and the lowest
time-averaged error. The neural models earn their keep at **short-to-medium
horizons** (`t ≲ 5`, robustly across seeds), where they track the truth more
closely before their errors compound. In the strongly nonlinear, resonant
**St = 1** regime the conditioned MLP *can* do much better than DMD (seed 0:
`0.95e-4` vs `2.06e-4` at the final time), but only 3 of its 5 seeds beat DMD
there.

Where the dynamics are near-linear — slow tracer-like accumulation at
St = 0.1, quasi-stationary lag at St = 5 — a linear operator on 16 POD
coefficients is all that is needed. The per-Stokes operators sit right at the
edge of stability (spectral radii `0.994`–`1.005`); the St = 0.1 operator has
one marginally growing mode, harmless over this 200-step horizon but a
reminder that DMD offers no stability guarantee. A single **global** DMD
operator is far weaker (`2.68e-4`), so for DMD, as for the neural models,
**knowing the Stokes number matters more than model class**. More POD modes
do not help DMD (r = 4 / 8 / 16 / 32 give `1.29e-4` / `1.23e-4` / `1.19e-4` /
`1.42e-4`). On the raw fields, even unconditioned global DMD beats the
unconditioned one-step MLP (`3.99e-4` vs `4.42e-4`). Reproduce with
`python scripts/run_dmd.py -i data/particle_concentration_smooth.npz -o figures/smooth`
(part of `make smooth`).

**An honest limitation — forecasting an unseen Stokes number.** When St = 5 is
held out *entirely*, the unconditioned forecaster loses to persistence
(`4.18e-4` vs `1.18e-4`; `3.44e-4` in a from-scratch re-run): it cannot
extrapolate dynamics to an inertia regime it never saw, and St = 5 is nearly
stationary so persistence is hard to beat. Reconstruction generalises across
St; autoregressive *dynamics* do not, without conditioning. This cleanly
motivates the conditioned model.

### 3.2 Reproducibility

Every result was regenerated from scratch (`make all smooth holdout fourier`
into separate output directories: new datasets, new autoencoders, new
forecasters) and compared with the numbers above:

| Quantity | Reported | From-scratch re-run |
|---|---|---|
| raw / smooth / Fourier datasets | — | **bit-for-bit identical** |
| POD-16 RMSE (raw / smooth / Fourier) | `2.355e-4` / `6.69e-5` / `2.416e-4` | identical |
| AE RMSE, raw | `2.382e-4` | `2.380e-4` |
| AE RMSE, smooth | `7.42e-5` | `7.44e-5` |
| AE RMSE, unseen St = 5 | `9.70e-5` | `9.77e-5` |
| AE RMSE, Fourier (vs POD-16 `2.416e-4`) | `2.371e-4` | `2.374e-4` — still beats POD |
| one-step MLP, raw (final) | `4.42e-4` | `4.42e-4` |
| conditioned MLP, smooth (final) | `1.51e-4` | `1.82e-4` |
| Neural-ODE, smooth (final) | `2.32e-4` | `2.03e-4` |
| GRU, smooth (final) | `1.64e-4` | `3.91e-4` — **worse than persistence** |
| held-out St = 5 forecaster (final) | `4.18e-4` | `3.44e-4` — still loses to persistence |

The physics, the datasets, POD, DMD and the autoencoders reproduce to within
1 %, and every reconstruction conclusion stands. The neural forecasters'
long-horizon errors do not reproduce to that precision: they depend on the
training seed and on the particular autoencoder they were trained on, which is
why §3 reports seed averages. The GRU is the extreme case — one fresh run of
the configuration reported above ends up behind persistence.

---

## 4. Summary of what beats what

- **Reconstruction:** noise floor dominates the raw data (POD = CAE); smoothing
  buys a 3× lower floor and makes the fields low-rank, where POD is near-optimal.
- **Forecasting:** every learned latent forecaster beats persistence, cutting
  its final-time error by 27–45 % on average over training seeds. But a
  **linear per-Stokes DMD** baseline has the lowest long-horizon and
  time-averaged error of all; the neural models win robustly only at short
  horizons (`t ≲ 5`), and their long-horizon errors vary substantially with
  the training seed. Persistence only wins over the first few snapshots or for
  near-stationary, unseen regimes.
- **Generalisation:** the encoder/decoder transfers to an unseen Stokes number;
  the dynamics need conditioning to do the same.

---

## 5. A more turbulence-like flow

Swapping the single-mode Taylor–Green vortex for a divergence-free **multi-mode
random-Fourier streamfunction** flow produces much richer, filamentary
clustering (curved ligaments and voids rather than a regular cell grid):

![Fourier multi-mode flow](docs/figures/fourier_stokes_comparison.png)

Reproduce with `python scripts/generate_dataset.py --flow-type fourier`.

### 5.1 Where the nonlinear ROM finally beats POD

This flow is the testbed that motivates a *nonlinear* autoencoder: its
concentration fields are genuinely high-rank, so the linear POD basis has no
compact representation to exploit. The autoencoder and POD were trained on the
identical dataset and held-out seed; the only difference is linear vs nonlinear
encoding.

![Linear vs nonlinear ROM on the Fourier flow](docs/figures/rom_comparison_fourier.png)

The POD reconstruction error is almost **flat** with mode count — it needs
**1132 modes for 90 % energy** and barely improves from 1 to 128 modes — whereas
the **16-dimensional autoencoder beats POD at equal latent size and matches a
~32-mode POD basis** (a ~2× compression at equal accuracy). This is the
nonlinear-ROM advantage the smooth flow could not show.

The autoencoder reconstructs the filamentary ligaments and voids of the Fourier
fields from just 16 latent numbers:

![Autoencoder reconstruction on the Fourier flow](docs/figures/ae_panel_fourier.png)

The result only appears where the physics is multiscale. Across all three flows,
at equal latent size (16):

| Flow | POD modes for 90 % energy | POD @ 16 RMSE | Autoencoder (16-dim) RMSE | Winner |
|------|--------------------------:|--------------:|--------------------------:|--------|
| Smooth Taylor–Green | 24 | `6.69e-5` | `7.42e-5` | **POD** (low-rank, linear is optimal) |
| Raw Taylor–Green | 934 | `2.355e-4` | `2.382e-4` | POD (marginal) |
| **Multi-mode Fourier** | **1132** | `2.416e-4` | **`2.371e-4`** | **Autoencoder** |

The honest reading: a nonlinear ROM is **not** automatically better — on the
smooth, low-rank fields POD is unbeatable, exactly as linear theory predicts. It
is the **multiscale Fourier flow**, where the POD spectrum never collapses, that
the nonlinear encoder earns its keep. Reproduce with:

```bash
python scripts/run_baselines.py     -i data/particle_concentration_fourier.npz -o figures/fourier
python scripts/train_autoencoder.py -i data/particle_concentration_fourier.npz -o figures/fourier \
    --ckpt checkpoints/autoencoder_fourier.pt --latent-dim 16 --epochs 40
python scripts/compare_roms.py \
    --baseline figures/fourier/baseline_results.json --ae figures/fourier/ae_results.json \
    -o figures/fourier/rom_comparison.png --title "multi-mode Fourier flow"
```
