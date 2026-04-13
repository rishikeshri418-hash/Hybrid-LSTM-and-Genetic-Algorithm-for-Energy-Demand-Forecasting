# Hybrid LSTM and Genetic Algorithm for Energy Demand Forecasting

## Project Overview
This project investigates a hybrid Genetic Algorithm-optimised Long Short-Term Memory (GA-LSTM) model for short-term energy demand forecasting across three diverse datasets.

## Research Question
To what extent can GA-optimised LSTM hyperparameters reduce forecast error across diverse energy datasets?

## Datasets Used
| Dataset | Source | Records | Period |
|---------|--------|---------|--------|
| PJM (US Regional Grid) | pjm.com | 210,384 | 2018-2026 |
| UK National Grid | neso.energy | 70,120 | 2018-2025 |
| UCI Household | archive.ics.uci.edu | 34,168 | 2006-2010 |

## Results Summary
| Dataset | Baseline MAPE | GA-LSTM MAPE | Improvement |
|---------|---------------|--------------|-------------|
| PJM Regional Grid | 1.57% | 1.49% | **+5.03%** |
| UK National Grid | 4.46% | 5.82% | -30.61% |
| UCI Household | 58.81% | 57.46% | **+2.29%** |

## Key Findings
- GA-LSTM improved PJM forecasting by 5.03% (p = 0.023)
- GA-LSTM underperformed on UK data (dataset-dependent)
- Residential forecasting remains challenging (58.81% baseline MAPE)

## Requirements
- Python 3.11.9
- Install: `pip install -r requirements.txt`

## Ethics Approval
- Reference: P192811
- Date: 24 February 2026
- Risk Level: Low

## Repository Structure
