# Reduced-Order Particle Flow — reproducible pipeline.
#
# Quick start:
#   make help        # list targets
#   make smoke       # fast end-to-end sanity check (tiny dataset, every stage)
#   make all         # full canonical pipeline: data -> figures -> ML stages
#   make smooth      # the headline forecasting results (RESULTS sec 3)
#   make fourier     # the linear-vs-nonlinear ROM comparison (RESULTS sec 5.1)
#
# Override the interpreter with e.g.  make PYTHON=python all
PYTHON ?= python3
FIGS   ?= figures

DATA        := data/particle_concentration.npz
DATA_SMOOTH := data/particle_concentration_smooth.npz
DATA_FOUR   := data/particle_concentration_fourier.npz
AE_CKPT     := checkpoints/autoencoder.pt
AE_SMOOTH   := checkpoints/autoencoder_smooth.pt
AE_HOLDOUT  := checkpoints/ae_holdoutSt5.pt
AE_FOUR     := checkpoints/autoencoder_fourier.pt

SMOKE_DATA  := data/smoke.npz
SMOKE_FIGS  := $(FIGS)/smoke
SMOKE_AE    := checkpoints/smoke_ae.pt

.DEFAULT_GOAL := help
.PHONY: help env test smoke data figures baselines autoencoder forecaster \
        smooth holdout fourier all clean

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
	$(PYTHON) scripts/train_autoencoder.py -i $(SMOKE_DATA) -o $(SMOKE_FIGS) \
		--ckpt $(SMOKE_AE) --epochs 2
	$(PYTHON) scripts/train_forecaster.py -i $(SMOKE_DATA) -o $(SMOKE_FIGS) \
		--ckpt $(SMOKE_AE) --fc-ckpt checkpoints/smoke_forecaster.pt \
		--model gru --conditioned --rollout 4 --epochs 2

# --- Canonical pipeline ------------------------------------------------------
$(DATA):
	$(PYTHON) scripts/generate_dataset.py -o $(DATA)

data: $(DATA)  ## Generate the canonical concentration-field dataset.

figures: $(DATA)  ## Render comparison figures and evolution GIFs.
	$(PYTHON) scripts/make_figures.py -i $(DATA) -o $(FIGS)
	$(PYTHON) scripts/make_gifs.py -o $(FIGS) --seed 0

baselines: $(DATA)  ## Persistence + POD baselines and physical diagnostics.
	$(PYTHON) scripts/run_baselines.py   -i $(DATA) -o $(FIGS)
	$(PYTHON) scripts/run_diagnostics.py -i $(DATA) -o $(FIGS)

$(AE_CKPT): $(DATA)
	$(PYTHON) scripts/train_autoencoder.py -i $(DATA) --ckpt $(AE_CKPT) --epochs 40

autoencoder: $(AE_CKPT)  ## Train the convolutional autoencoder (latent dim 16).

forecaster: $(AE_CKPT)  ## Train the latent forecaster and evaluate roll-outs.
	$(PYTHON) scripts/train_forecaster.py --ckpt $(AE_CKPT) --epochs 200

all: figures baselines autoencoder forecaster  ## Full canonical pipeline.

# --- Denoised-field forecasting (RESULTS sec 2-3) ----------------------------
$(DATA_SMOOTH):
	$(PYTHON) scripts/generate_dataset.py --smoothing 1.0 -o $(DATA_SMOOTH)

$(AE_SMOOTH): $(DATA_SMOOTH)
	$(PYTHON) scripts/train_autoencoder.py -i $(DATA_SMOOTH) -o $(FIGS)/smooth \
		--ckpt $(AE_SMOOTH) --epochs 40

smooth: $(AE_SMOOTH)  ## Conditioned MLP / Neural-ODE / GRU forecasters on smoothed fields.
	$(PYTHON) scripts/run_baselines.py -i $(DATA_SMOOTH) -o $(FIGS)/smooth
	$(PYTHON) scripts/train_forecaster.py -i $(DATA_SMOOTH) -o $(FIGS)/smooth \
		--ckpt $(AE_SMOOTH) --model mlp --conditioned --rollout 4 --tag cond
	$(PYTHON) scripts/train_forecaster.py -i $(DATA_SMOOTH) -o $(FIGS)/smooth \
		--ckpt $(AE_SMOOTH) --model ode --conditioned --rollout 8 --tag ode
	$(PYTHON) scripts/train_forecaster.py -i $(DATA_SMOOTH) -o $(FIGS)/smooth \
		--ckpt $(AE_SMOOTH) --model gru --conditioned --rollout 16 --tag gru

holdout: $(DATA_SMOOTH)  ## Hold out St = 5 entirely (Stokes generalisation test).
	$(PYTHON) scripts/train_autoencoder.py -i $(DATA_SMOOTH) -o $(FIGS)/holdout \
		--test-stokes 5 --ckpt $(AE_HOLDOUT) --epochs 40
	$(PYTHON) scripts/train_forecaster.py -i $(DATA_SMOOTH) -o $(FIGS)/holdout \
		--ckpt $(AE_HOLDOUT) --rollout 4 --gif-stokes 5 --tag holdoutSt5

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
		$(FIGS)/smoke $(FIGS)/smooth $(FIGS)/holdout $(FIGS)/fourier \
		checkpoints/*.pt .pytest_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
