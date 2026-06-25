# Crude Oil Cross-Hedging with Reinforcement Learning

M.Sc. thesis code — Amirkabir University of Technology (AUT)

This repository contains the full implementation for a study on dynamic cross-hedging of crude oil price exposure (WTI Spot, Brent Spot, OPEC Basket) using NYMEX WTI futures (CL front month). Baseline classical strategies are benchmarked against a Soft Actor-Critic (SAC) reinforcement learning agent trained under a Lower Partial Moment (LPM) risk objective.

---

## Repository Structure

```
.
├── config.yaml               # Central project configuration (assets, data policy, paths)
├── MasterData.csv            # Raw daily price data (WTI, Brent, OPEC, CL1, CL2, exogenous features)
├── MasterData.parquet        # Same data in Parquet format
├── data_adapter.py           # Data loading, alignment, and feature engineering
├── price_engine.py           # Return and price series construction
├── window_engine.py          # Rolling / expanding window logic
├── cost_model.py             # Transaction cost model
├── hedge_simulator.py        # Hedge P&L simulation engine
├── Scenario_Generator.py     # Monte-Carlo / historical scenario generation
├── orchestrator.py           # Pipeline orchestrator (run all strategies end-to-end)
├── Baseline_report.py        # Report generation for classical baseline strategies
├── finalreport.py            # Final consolidated report (baseline + RL comparison)
├── make_rl_reference.py      # Build RL oracle reference scenarios
├── results_schema.py         # Output schema validation
├── Datasanity.py             # Data quality checks
├── validate_pipeline.py      # End-to-end pipeline validation
├── BaseGARCH.py              # Base GARCH model utilities
├── Feature_selector.py       # Feature selection utilities
├── LSTMInput.py              # LSTM input preparation
├── run_sac_lpm_target_grid.sh # Shell script to run SAC LPM hyperparameter grid
│
├── strategy/                 # Classical hedging strategies
│   ├── base.py               # Abstract strategy base class
│   ├── naive.py              # Naive (one-to-one) hedge
│   ├── nohedge.py            # No-hedge benchmark
│   ├── ols_static.py         # OLS static hedge ratio
│   ├── ols_rolling.py        # OLS rolling hedge ratio
│   ├── ccc_garch.py          # CCC-GARCH dynamic hedge ratio
│   └── dcc_garch.py          # DCC-GARCH dynamic hedge ratio
│
├── rl/                       # Reinforcement learning components
│   ├── env_daily.py          # Gymnasium environment (daily oil hedging)
│   ├── SACPortfolioLPMEnv.py # SAC environment with LPM reward
│   ├── precompute.py         # Precompute state features for fast training
│   ├── scenario_loader.py    # Load scenario data for RL training
│   ├── eval_metrics.py       # Evaluation metrics (HE, ES, LPM, Sharpe, etc.)
│   ├── train_sac.py          # SAC training script
│   ├── train_sac_portfolio_lpm.py  # SAC + LPM objective training
│   ├── train_walkforward.py  # Walk-forward training loop
│   ├── SAC_Walkforard_LPM.py # SAC walk-forward with LPM
│   ├── tune_sac_weights_oracle_universe.py  # Hyperparameter tuning
│   ├── configs_RL.yml        # RL hyperparameter configuration
│   └── ...                   # Additional training / sweep scripts
│
├── Tune/                     # Hyperparameter tuning results
│   ├── sac_weight_tuning_oracle_universe/   # Weight-space grid search results
│   └── sac_policy_lpm_target_grid/          # LPM target grid results
│
└── reports/                  # Thesis chapter drafts (Word documents)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

---

## Data

`MasterData.csv` contains daily observations for:

| Column | Description |
|---|---|
| `WTI` | WTI Spot price (USD/bbl) |
| `Brent` | Brent Spot price (USD/bbl) |
| `OPEC` | OPEC Basket price (USD/bbl) |
| `CL1` | NYMEX WTI front-month futures (USD/bbl) |
| `CL2` | NYMEX WTI second-month futures (USD/bbl) |
| `OVX` | Crude Oil Volatility Index |
| `VIX` | CBOE Volatility Index |
| `DXY` | US Dollar Index |
| `DTB3` / `TB3M` | 3-month Treasury bill rate |
| `GEPUCURRENT` | Global Economic Policy Uncertainty |
| `EPUTRADE` | Trade Policy Uncertainty |
| `SP500_ENERGY` | S&P 500 Energy Sector Index |

Negative prices (e.g., 2020-04-20 WTI event) are intentionally preserved.

---

## Hedging Strategies

| Strategy | Type | Description |
|---|---|---|
| No Hedge | Baseline | Unhedged exposure |
| Naive (1:1) | Baseline | One futures contract per unit exposure |
| OLS Static | Classical | Static OLS minimum-variance hedge ratio |
| OLS Rolling | Classical | Rolling OLS hedge ratio (expanding window) |
| CCC-GARCH | Classical | Constant Conditional Correlation GARCH |
| DCC-GARCH | Classical | Dynamic Conditional Correlation GARCH |
| SAC-LPM | RL | Soft Actor-Critic with Lower Partial Moment reward |

---

## Running the Pipeline

```bash
# Run all classical baselines
python orchestrator.py

# Generate baseline report
python Baseline_report.py

# Train SAC agent (single run)
python rl/train_sac_portfolio_lpm.py

# Walk-forward evaluation
python rl/SAC_Walkforard_LPM.py

# Full final report (baseline + RL)
python finalreport.py
```

---

## Evaluation Metrics

- **HE** — Hedging Effectiveness (variance reduction)
- **ES** — Expected Shortfall (CVaR at 95%)
- **LPM** — Lower Partial Moment (downside risk)
- **Sharpe / Sortino** — Risk-adjusted return ratios
- **Turnover** — Transaction cost proxy

---

## Citation

If you use this code, please cite the corresponding thesis (details to be added upon publication).

---

## License

MIT License. See `LICENSE` for details.
