from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from .stress import StressTester
from .var import (
    parametric_var_cvar,
    historical_var_cvar,
    monte_carlo_var_cvar,
    component_var_parametric,
)
from .portfolio import (
    load_positions,
    load_prices,
    portfolio_valuation,
    portfolio_returns,
    exposures_by_asset_class,
)

@dataclass(frozen=True)
class RiskReport:
    asof: pd.Timestamp
    portfolio_value: float
    var_cvar: pd.DataFrame
    component_var: Optional[pd.DataFrame]
    exposures: pd.DataFrame
    pnl_series: pd.Series

class RiskEngine:
    """
    Portfolio risk engine for linear (delta) instruments.

    Inputs:
      - positions.csv: asset, quantity, asset_class, currency, price_source
      - prices.csv: daily prices indexed by date

    Outputs:
      - VaR/CVaR via parametric (Gaussian), historical, and Monte Carlo (Gaussian)
      - Component VaR (parametric) by asset
      - Asset-class exposures and PnL series
      - Stress test results (scenario and historical window)
    """

    def __init__(self, positions: pd.DataFrame, prices: pd.DataFrame):
        self.positions = positions.copy()
        self.prices = prices.copy().sort_index()
        self._validate()

    @classmethod
    def from_csv(cls, positions_path: str, prices_path: str) -> "RiskEngine":
        positions = load_positions(positions_path)
        prices = load_prices(prices_path)
        return cls(positions, prices)

    def _validate(self) -> None:
        if "asset" not in self.positions.columns or "quantity" not in self.positions.columns:
            raise ValueError("positions must contain columns: asset, quantity")
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise ValueError("prices must have a DatetimeIndex")
        missing = [a for a in self.positions["asset"].unique() if a != "CASH_USD" and a not in self.prices.columns]
        if missing:
            raise ValueError(f"Missing price series for assets: {missing}")

    def build_report(
        self,
        lookback_days: int = 252,
        horizon_days: int = 1,
        alpha: float = 0.99,
        mc_sims: int = 20000,
        include_component_var: bool = True,
    ) -> RiskReport:
        # Align last lookback window
        px = self.prices.tail(lookback_days + 1)
        asof = px.index[-1]

        port_val, asset_values = portfolio_valuation(self.positions, px.loc[asof])
        rets, pnl = portfolio_returns(self.positions, px)

        # VaR/CVaR methods
        var_rows = []
        # Parametric: scale by sqrt(horizon) assuming iid
        mu = rets.mean().values
        cov = rets.cov().values
        weights = (asset_values / port_val).reindex(rets.columns).fillna(0.0).values
        p_var, p_cvar = parametric_var_cvar(mu, cov, weights, port_val, alpha=alpha, horizon_days=horizon_days)

        var_rows.append({"method": "Parametric (Gaussian)", "VaR": p_var, "CVaR": p_cvar})

        h_var, h_cvar = historical_var_cvar(pnl, port_val, alpha=alpha, horizon_days=horizon_days)
        var_rows.append({"method": "Historical", "VaR": h_var, "CVaR": h_cvar})

        mc_var, mc_cvar = monte_carlo_var_cvar(mu, cov, weights, port_val, alpha=alpha, horizon_days=horizon_days, n_sims=mc_sims)
        var_rows.append({"method": f"Monte Carlo (Gaussian, {mc_sims:,} sims)", "VaR": mc_var, "CVaR": mc_cvar})

        var_cvar_df = pd.DataFrame(var_rows)

        component_df = None
        if include_component_var:
            component_df = component_var_parametric(mu, cov, weights, port_val, alpha=alpha, horizon_days=horizon_days)
            component_df["asset"] = list(rets.columns)
            component_df = component_df[["asset", "weight", "marginal_VaR", "component_VaR"]].sort_values("component_VaR", ascending=False)

        exposures = exposures_by_asset_class(self.positions, px.loc[asof])

        return RiskReport(
            asof=asof,
            portfolio_value=float(port_val),
            var_cvar=var_cvar_df,
            component_var=component_df,
            exposures=exposures,
            pnl_series=pnl,
        )

    def stress_tester(self) -> StressTester:
        return StressTester(self.positions, self.prices)
