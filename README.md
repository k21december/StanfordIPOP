# stanford-ipop-bioage

State-space modeling of biological age from longitudinal multi-omics data, built on the Stanford iPOP (integrative Personal Omics Profiling) study. The project estimates each subject's latent biological-age trajectory from noisy, irregularly sampled biomarker channels and studies how the gap between biological and chronological age evolves over time.

This repository is a self-contained slice of a larger research project on latent health and aging dynamics.

## What it does

The pipeline models biological age as a hidden state observed only through noisy biomarkers, and tracks it over time using state-space filtering:

- Multi-omics ingestion — fetches and aligns multiple modalities (clinical labs, proteome, metabolome, immune/cytokine markers) into a unified longitudinal panel.
- Dimensionality reduction — PCA on each modality (e.g. cytokine PC1, metabolome PCs) to compress high-dimensional omics into interpretable channels.
- State-space filtering — a tuned 2-state Kalman filter fuses metabolic and immune channels to estimate a latent biological-age state, with a particle-filter variant for the nonlinear/non-Gaussian case.
- Biological-age gap analysis — computes the gap between estimated biological age and chronological age, and analyzes how it varies by sex, race, and over time.
- Predictive evaluation — evaluates whether the current latent state predicts future lab values, with uncertainty and error curves as a function of age.
- Downstream analyses — metabolome dynamics, immune volatility, and metabolome-stress interaction models.

## Selected results

Generated into figures/ and results/plots/ :

- bioage_vs_chron.png — estimated biological age vs. chronological age
- bioage_gap_vs_future.png — does the bio-age gap predict future health signals?
- pred_vs_actual_future_labs.png — predicting future labs from the latent state
- uncertainty_vs_age_all.png — estimation uncertainty as a function of age
- spaghetti_bioage_all.png — individual biological-age trajectories

## Pipeline / repository structure

Scripts in src/ are numbered in execution order:

- 00_* — fetch modalities, validate inputs, probe schemas
- 01_* — build the longitudinal panel and per-modality matrices (proteome, etc.)
- 02_select_labs.py — select clinical lab channels
- 10_* / 20_* — fit the bio-age state-space models (Kalman, 1- and 2-state)
- 21_fit_bioage_particle.py — particle-filter variant
- 12_ / 15_* — predictive evaluation, uncertainty and error curves
- 25_ / 27_ / 28_* — metabolome PCA and metage-dynamics models
- 30_ / 31_ / 40_* — final combined-state models and review plots

results/ holds model outputs (CSV) and figures/ holds generated plots.

## Methods

- Latent state: 2-state Kalman filter fusing a metabolic-labs channel and an immune/cytokine channel, with tuned process noise, a mean-reversion term, cross-channel coupling, and per-channel observation noise; particle filter for the nonlinear case.
- Feature extraction: per-modality PCA (scikit-learn) to reduce high-dimensional omics.
- Evaluation: out-of-sample prediction of future labs, calibration, and uncertainty/error curves by age.
- Stack: Python, NumPy, Pandas, SciPy, scikit-learn, statsmodels, Matplotlib, fastparquet/pyarrow.

## Getting started

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python src/20_fit_bioage_ssm_2state_best_kalman.py <input_panel> <output> <params_out>

Scripts are ordered by numeric prefix; run 00_* through 40_* to reproduce the full pipeline once input data is in place.

## Data

This project uses data from the Stanford iPOP (integrative Personal Omics Profiling) study. Raw omics/clinical data is not redistributed in this repository — only derived model outputs and figures are included. To reproduce, obtain the iPOP data from the official source and place it under data/.

---

Part of a larger research project on latent health and aging dynamics; this repository contains a representative, self-contained component. Built by Krishaan Somwanshi.
