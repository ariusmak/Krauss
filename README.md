# Krauss: ML-Driven Statistical Arbitrage

Reproduction and extension of the Krauss et al. (2017) statistical arbitrage strategy using a Python stack built around WRDS data, PyTorch, XGBoost, scikit-learn, and Streamlit.

## Overview

This project studies whether machine learning models can generate tradable cross-sectional equity signals in a realistic daily long-short setting.

The repo has two goals:
1. **Reproduce** the core pipeline from Krauss et al. (2017) as faithfully as possible in Python.
2. **Extend** the original setup with better signal construction and a more complete research/trading workflow.

## Current status

**In progress**

### Completed
- Python project structure with configs, scripts, tests, and modular source code
- Core model-building and training workflow
- Reproduction framing for the original paper’s DNN, gradient-boosted tree, random forest, and ensemble setup
- Clear methodology for no-lookahead, survivor-bias-aware universe construction, and walk-forward evaluation

### In progress
- Full historical data pipeline from WRDS
- End-to-end backtesting and transaction cost evaluation
- Final comparison of reproduction results versus the paper
- Extension experiments and Streamlit research interface

## What this project demonstrates

- Applied machine learning for financial prediction
- Careful leakage-aware and survivor-bias-aware research design
- Translating an academic paper into a structured Python research pipeline
- Model comparison across neural nets, boosted trees, random forests, and ensembles
- Connecting model outputs to trading rules rather than stopping at prediction accuracy

## Method summary

The original paper predicts whether a stock’s next-day return will beat the next-day cross-sectional median, then forms a daily dollar-neutral long-short portfolio by ranking stocks on predicted signal strength.

This project keeps that framing and adds a more explicit research extension: predicting both **direction** and **magnitude** of excess return relative to the next-day cross-sectional median, then testing alternative ranking rules and execution logic.

## Repo structure

```text
app/          Streamlit interface and presentation layer
configs/      Reproducible experiment settings
docs/         Project notes and supporting documentation
notebooks/    Research and exploratory analysis
scripts/      Runnable pipeline entry points
src/krauss/   Core project code
tests/        Validation and regression checks
