from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from .portfolio import portfolio_valuation

@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    pnl: float
    pnl_pct: float

class StressTester:
    """
    Stress testing:
      1) Deterministic scenario shocks: per-asset % shock applied to spot prices
      2) Historical stress window: worst N-day PnL in lookback window
    """

    def __init__(self, positions: pd.DataFrame, prices: pd.DataFrame):
        self.positions = positions.copy()
        self.prices = prices.copy().sort_index()

    def run_scenarios(self, scenarios: pd.DataFrame) -> pd.DataFrame:
        asof = self.prices.index[-1]
        spot = self.prices.loc[asof]
        port_val, _ = portfolio_valuation(self.positions, spot)

        results: List[ScenarioResult] = []
        for _, row in scenarios.iterrows():
            name = str(row.get("scenario", "Unnamed"))
            shocked = spot.copy()
            for a in shocked.index:
                col = f"{a}_shock"
                if col in row and pd.notna(row[col]):
                    shocked[a] = shocked[a] * (1.0 + float(row[col]))
            shocked_val, _ = portfolio_valuation(self.positions, shocked)
            pnl = shocked_val - port_val
            results.append(ScenarioResult(name, float(pnl), float(pnl / port_val if port_val else 0.0)))

        out = pd.DataFrame([r.__dict__ for r in results]).sort_values("pnl")
        return out

    def worst_window(self, lookback_days: int = 252, window_days: int = 10) -> pd.DataFrame:
        """
        Finds worst rolling window PnL over a lookback.
        """
        px = self.prices.tail(lookback_days + window_days + 1)
        asof = px.index[-1]
        spot = px.loc[asof]
        port_val, _ = portfolio_valuation(self.positions, spot)

        # compute daily portfolio value time series (static positions)
        # vectorized
        pos = self.positions[self.positions["asset"] != "CASH_USD"].copy()
        # align quantities
        qty = pos.set_index("asset")["quantity"].reindex(px.columns).fillna(0.0).values
        val_series = (px.values * qty).sum(axis=1) + float(self.positions.loc[self.positions["asset"] == "CASH_USD", "quantity"].sum())
        val_series = pd.Series(val_series, index=px.index, name="portfolio_value")

        # rolling pnl over window
        rolling_pnl = val_series.diff(window_days)
        worst_date = rolling_pnl.idxmin()
        worst_pnl = float(rolling_pnl.loc[worst_date])
        worst_pct = float(worst_pnl / port_val if port_val else 0.0)

        out = pd.DataFrame([{
            "window_days": window_days,
            "lookback_days": lookback_days,
            "worst_end_date": worst_date,
            "worst_window_pnl": worst_pnl,
            "worst_window_pnl_pct": worst_pct
        }])
        return out
