# Reduced-Order Particle Flow — reproducible pipeline.
#
# Quick start:
#   make help        # list targets
#   make smoke       # fast end-to-end sanity check (tiny dataset, every stage)
#   make all         # full canonical pipeline: data -> figures -> ML stages
#   make smooth      # the headline forecasting results (RESULTS sec 3)
#   make seeds       # seed-to-seed spread of the neural forecasters vs DMD
#   make fourier     # the linear-vs-nonlinear ROM comparison (RESULTS sec 5.1)
#
# Override the interpreter with e.g.  make PYTHON=python all
# Output locations can be redirected, e.g. to reproduce every result without
# touching existing outputs:
#   make all smooth holdout fourier DATADIR=data/rerun FIGS=figures/rerun CKPTS=checkpoints/rerun
PYTHON  ?= python3
DATADIR ?= data
FIGS    ?= figures
CKPTS   ?= checkpoints

DATA        := $(DATADIR)/particle_concentration.npz
DATA_SMOOTH := $(DATADIR)/particle_concentration_smooth.npz
DATA_FOUR   := $(DATADIR)/particle_concentration_fourier.npz
AE_CKPT     := $(CKPTS)/autoencoder.pt
AE_SMOOTH   := $(CKPTS)/autoencoder_smooth.pt
AE_HOLDOUT  := $(CKPTS)/ae_holdoutSt5.pt
AE_FOUR     := $(CKPTS)/autoencoder_fourier.pt

SMOKE_DATA  := $(DATADIR)/smoke.npz
SMOKE_FIGS  := $(FIGS)/smoke
SMOKE_AE    := $(CKPTS)/smoke_ae.pt

.DEFAULT_GOAL := help
.PHONY: help env test smoke data figures baselines autoencoder forecaster \
        smooth seeds holdout fourier all clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

env:  ## Install Python deps, CPU PyTorch and pytest.
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run the physics + model unit tests.
	$(PYTHON) -m pytest -q

# The autoencoder expects 64x64 fields, so the smoke dataset keeps the full
# grid and shrinks everything else.
smoke:  ## Fast end-to-end check of every stage on a tiny dataset.
	$(PYTHON) scripts/generate_dataset.py --quick --nx 64 --ny 64 -o $(SMOKE_DATA)
	$(PYTHON) scripts/make_figures.py   -i $(SMOKE_DATA) -o $(SMOKE_FIGS)
	$(PYTHON) scripts/run_baselines.py  -i $(SMOKE_DATA) -o $(SMOKE_FIGS)
	$(PYTHON) scripts/run_diagnostics.py -i $(SMOKE_DATA) -o $(SMOKE_FIGS)
	$(PYTHON) scripts/run_dmd.py        -i $(SMOKE_DATA) -o $(SMOKE_FIGS)
	$(PYTHON) scripts/train_autoencoder.py -i $(SMOKE_DATA) -o $(SMOKE_FIGS) \
		--ckpt $(SMOKE_AE) --epochs 2
	$(PYTHON) scripts/train_forecaster.py -i $(SMOKE_DATA) -o $(SMOKE_FIGS) \
		--ckpt $(SMOKE_AE) --fc-ckpt $(CKPTS)/smoke_forecaster.pt \
		--model gru --conditioned --rollout 4 --epochs 2

# --- Canonical pipeline ------------------------------------------------------
$(DATA):
	$(PYTHON) scripts/generate_dataset.py -o $(DATA)

data: $(DATA)  ## Generate the canonical concentration-field dataset.

figures: $(DATA)  ## Render comparison figures and evolution GIFs.
	$(PYTHON) scripts/make_figures.py -i $(DATA) -o $(FIGS)
	$(PYTHON) scripts/make_gifs.py -o $(FIGS) --seed 0

baselines: $(DATA)  ## Persistence, POD and DMD baselines; physical diagnostics.
	$(PYTHON) scripts/run_baselines.py   -i $(DATA) -o $(FIGS)
	$(PYTHON) scripts/run_dmd.py         -i $(DATA) -o $(FIGS)
	$(PYTHON) scripts/run_diagnostics.py -i $(DATA) -o $(FIGS)

$(AE_CKPT): $(DATA)
	$(PYTHON) scripts/train_autoencoder.py -i $(DATA) -o $(FIGS) --ckpt $(AE_CKPT) --epochs 40

autoencoder: $(AE_CKPT)  ## Train the convolutional autoencoder (latent dim 16).

forecaster: $(AE_CKPT)  ## Train the latent forecaster and evaluate roll-outs.
	$(PYTHON) scripts/train_forecaster.py -i $(DATA) -o $(FIGS) --ckpt $(AE_CKPT) \
		--fc-ckpt $(CKPTS)/forecaster.pt --epochs 200

all: figures baselines autoencoder forecaster  ## Full canonical pipeline.

# --- Denoised-field forecasting (RESULTS sec 2-3) ----------------------------
$(DATA_SMOOTH):
	$(PYTHON) scripts/generate_dataset.py --smoothing 1.0 -o $(DATA_SMOOTH)

$(AE_SMOOTH): $(DATA_SMOOTH)
	$(PYTHON) scripts/train_autoencoder.py -i $(DATA_SMOOTH) -o $(FIGS)/smooth \
		--ckpt $(AE_SMOOTH) --epochs 40

smooth: $(AE_SMOOTH)  ## DMD + conditioned MLP / Neural-ODE / GRU forecasters on smoothed fields.
	$(PYTHON) scripts/run_baselines.py -i $(DATA_SMOOTH) -o $(FIGS)/smooth
	$(PYTHON) scripts/run_dmd.py       -i $(DATA_SMOOTH) -o $(FIGS)/smooth
	$(PYTHON) scripts/train_forecaster.py -i $(DATA_SMOOTH) -o $(FIGS)/smooth \
		--ckpt $(AE_SMOOTH) --fc-ckpt $(CKPTS)/forecaster_cond.pt \
		--model mlp --conditioned --rollout 4 --tag cond
	$(PYTHON) scripts/train_forecaster.py -i $(DATA_SMOOTH) -o $(FIGS)/smooth \
		--ckpt $(AE_SMOOTH) --fc-ckpt $(CKPTS)/forecaster_ode.pt \
		--model ode --conditioned --rollout 8 --tag ode
	$(PYTHON) scripts/train_forecaster.py -i $(DATA_SMOOTH) -o $(FIGS)/smooth \
		--ckpt $(AE_SMOOTH) --fc-ckpt $(CKPTS)/forecaster_gru.pt \
		--model gru --conditioned --rollout 16 --tag gru

SEEDS ?= 0 1 2 3 4

seeds: $(AE_SMOOTH)  ## Retrain each smoothed-field forecaster over SEEDS; compare with DMD.
	for s in $(SEEDS); do \
		for cfg in "mlp 4 cond" "gru 16 gru" "ode 8 ode"; do \
			set -- $$cfg; \
			$(PYTHON) scripts/train_forecaster.py -i $(DATA_SMOOTH) -o $(FIGS)/seeds \
				--ckpt $(AE_SMOOTH) --fc-ckpt $(CKPTS)/seeds/forecaster_$$3_s$$s.pt \
				--model $$1 --conditioned --rollout $$2 --seed $$s --tag $$3_s$$s \
				|| exit 1; \
		done; \
	done
	$(PYTHON) scripts/run_dmd.py -i $(DATA_SMOOTH) -o $(FIGS)/seeds
	$(PYTHON) scripts/summarise_seeds.py -d $(FIGS)/seeds --tags cond gru ode

holdout: $(DATA_SMOOTH)  ## Hold out St = 5 entirely (Stokes generalisation test).
	$(PYTHON) scripts/train_autoencoder.py -i $(DATA_SMOOTH) -o $(FIGS)/holdout \
		--test-stokes 5 --ckpt $(AE_HOLDOUT) --epochs 40
	$(PYTHON) scripts/train_forecaster.py -i $(DATA_SMOOTH) -o $(FIGS)/holdout \
		--ckpt $(AE_HOLDOUT) --fc-ckpt $(CKPTS)/forecaster_holdoutSt5.pt \
		--rollout 4 --gif-stokes 5 --tag holdoutSt5

# --- Linear-vs-nonlinear ROM comparison (RESULTS sec 5.1) --------------------
$(DATA_FOUR):
	$(PYTHON) scripts/generate_dataset.py --flow-type fourier -o $(DATA_FOUR)

fourier: $(DATA_FOUR)  ## Reproduce the nonlinear-ROM-beats-POD result on Fourier flow.
	$(PYTHON) scripts/run_baselines.py -i $(DATA_FOUR) -o $(FIGS)/fourier
	$(PYTHON) scripts/train_autoencoder.py -i $(DATA_FOUR) -o $(FIGS)/fourier \
		--ckpt $(AE_FOUR) --latent-dim 16 --epochs 40
	$(PYTHON) scripts/compare_roms.py \
		--baseline $(FIGS)/fourier/baseline_results.json \
		--ae $(FIGS)/fourier/ae_results.json \
		-o $(FIGS)/fourier/rom_comparison.png --title "multi-mode Fourier flow"

clean:  ## Remove generated figures, checkpoints, and caches (keeps datasets).
	rm -rf $(FIGS)/*.png $(FIGS)/*.json $(FIGS)/*.gif \
		$(FIGS)/smoke $(FIGS)/smooth $(FIGS)/seeds $(FIGS)/holdout $(FIGS)/fourier \
		$(CKPTS)/*.pt $(CKPTS)/seeds .pytest_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
