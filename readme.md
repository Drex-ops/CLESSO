# CLESSO

CLESSO is a biodiversity modelling workflow for ALA/GBIF-style observation data.
It builds site-level richness (alpha) and turnover/similarity (beta) models from
species occurrence records, environmental covariates, and sampled
observation-pairs.

This repository contains both:

- An R-based CLESSO v2 pipeline (data preparation, pair sampling, TMB model fit, diagnostics)
- A Python neural-network implementation (`src/clesso_nn`) with training, diagnostics, and prediction utilities

## What this repo does

- Aggregates and filters occurrence records into analysis-ready site data
- Samples within-site and between-site observation pairs
- Extracts/joins environmental predictors
- Fits alpha/beta models (classic R/TMB and NN variants)
- Exports diagnostics, plots, and prediction surfaces

## Repository layout

- `src/clesso_v2/`: Core R pipeline scripts (for example `run_clesso.R`)
- `src/clesso_nn/`: Neural-network pipeline (for example `run_clesso_nn.py`, `diagnostics.py`)
- `scripts/`: Small diagnostic and QA scripts
- `tests/`: Python and R tests/smoke checks
- `data/`: Local data staging (large files are intentionally not committed; see `data/INDEX.md`)
- `OLD_RECA/`: Legacy RECA scripts kept for reference

## Quick start

1. Prepare data inputs in `data/` (see `data/INDEX.md`).
2. Choose a workflow:
	- R pipeline: run `src/clesso_v2/run_clesso.R` after configuring `src/clesso_v2/clesso_config.R`.
	- Python NN pipeline: run `src/clesso_nn/run_clesso_nn.py` using exports produced by the R preprocessing steps.
3. Generate diagnostics:
	- R: `src/clesso_v2/clesso_diagnostics.R`
	- Python: `src/clesso_nn/diagnostics.py`

## Dependencies

- R stack for the classical CLESSO workflow (including TMB-related tooling)
- Python 3 for the NN workflow; common packages include `numpy`, `pandas`, `pyarrow`, and `torch`

## Notes

- This repo mixes active pipelines and historical/experimental scripts.
- Start from the entry scripts above and inspect config files before running full workflows.
