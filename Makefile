# Reduced-Order Particle Flow — reproducible pipeline.
#
# Quick start:
#   make help        # list targets
#   make smoke       # fast end-to-end sanity check (tiny dataset)
#   make all         # full canonical pipeline: data -> figures -> ML stages
#   make fourier     # the linear-vs-nonlinear ROM comparison (RESULTS sec 5.1)
#
# Override the interpreter with e.g.  make PYTHON=python all
PYTHON ?= python3
FIGS   ?= figures

DATA        := data/particle_concentration.npz
DATA_SMOOTH := data/particle_concentration_smooth.npz
DATA_FOUR   := data/particle_concentration_fourier.npz
AE_CKPT     := checkpoints/autoencoder.pt
AE_FOUR     := checkpoints/autoencoder_fourier.pt

.DEFAULT_GOAL := help
.PHONY: help env test smoke data figures baselines autoencoder forecaster \
        fourier all clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

env:  ## Install Python + CPU PyTorch dependencies.
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:  ## Run the physics + model unit tests.
	$(PYTHON) -m pytest -q

smoke:  ## Fast end-to-end check on a tiny dataset.
	$(PYTHON) scripts/generate_dataset.py --quick -o data/smoke.npz

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
	rm -rf $(FIGS)/*.png $(FIGS)/*.json $(FIGS)/fourier/rom_comparison.png \
		checkpoints/*.pt .pytest_cache **/__pycache__
