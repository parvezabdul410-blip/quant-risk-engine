# Portfolio Risk Engine + Stress Testing Dashboard

A clean, **end-to-end** mini project you can put on GitHub: a Python **risk engine** (VaR/CVaR + allocation) + **stress testing** + a **Streamlit dashboard**.

## What this does

- Loads:
  - `data/positions.csv` (your portfolio)
  - `data/prices.csv` (daily historical prices)
  - `data/scenarios.csv` (deterministic stress shocks)
- Computes:
  - **VaR / CVaR**: Parametric (Gaussian), Historical, Monte Carlo (Gaussian)
  - **Component VaR (Euler allocation)** for parametric VaR
  - **Exposure** by asset class
  - **Historical window stress**: worst rolling N‑day PnL over a lookback window
  - **Deterministic scenario stress**: per‑asset shock revaluation

> Note: This is a **linear PnL** engine (static positions). It’s the correct “first portfolio risk engine” that you can later extend to options (Greeks / full reval), FX curves, rates DV01, etc.

---

## Repo structure

```
portfolio-risk-engine/
  app/
    streamlit_app.py
  data/
    positions.csv
    prices.csv
    scenarios.csv
  risk_engine/
    __init__.py
    engine.py
    portfolio.py
    var.py
    stress.py
  run_risk.py
  requirements.txt
  README.md
```

---

## Quickstart

### 1) Create venv + install

```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
# .venv\Scripts\activate    # windows

pip install -r requirements.txt
```

### 2) Run CLI

```bash
python run_risk.py --positions data/positions.csv --prices data/prices.csv --scenarios data/scenarios.csv
```

### 3) Run dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL Streamlit prints.

---

## Input formats

### `data/positions.csv`

| column | meaning |
|---|---|
| asset | ticker or identifier (`CASH_USD` supported) |
| quantity | units of the asset |
| asset_class | used for exposure aggregation |
| currency | informational for now |

### `data/prices.csv`

- `date` column + one column per asset ticker
- daily frequency recommended

### `data/scenarios.csv`

- `scenario` column + optional `{ASSET}_shock` columns containing % shocks (e.g., `-0.10` for -10%)

---

## How to customize

- Add assets: put ticker in `positions.csv`, add price column in `prices.csv`
- Change portfolio: update quantities
- Add scenarios: add rows to `scenarios.csv`
- Improve realism:
  - replace synthetic `prices.csv` with real data (yfinance / vendor export)
  - add FX conversion
  - add factor model / GARCH / EVT tails
  - add backtesting (Kupiec / Christoffersen)

---

## Suggested GitHub repo name

- `portfolio-risk-engine`
- `risk-engine-stress-dashboard`
- `var-cvar-stress-testing`

---
