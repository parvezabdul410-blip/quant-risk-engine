from __future__ import annotations

import pandas as pd
import numpy as np

def load_positions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # enforce types
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    if "asset_class" not in df.columns:
        df["asset_class"] = "Unknown"
    if "currency" not in df.columns:
        df["currency"] = "USD"
    return df

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df

def portfolio_valuation(positions: pd.DataFrame, spot_prices: pd.Series) -> tuple[float, pd.Series]:
    """
    spot_prices: Series indexed by asset tickers (excluding CASH_USD).
    """
    pos = positions.copy()
    pos["price"] = pos["asset"].map(lambda a: 1.0 if a == "CASH_USD" else float(spot_prices.get(a, np.nan)))
    pos["value"] = pos["quantity"] * pos["price"]
    asset_values = pos.groupby("asset")["value"].sum()
    port_val = float(asset_values.sum())
    return port_val, asset_values

def portfolio_returns(positions: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      - asset returns (daily arithmetic)
      - portfolio PnL series (daily) in currency units
    """
    px = prices.copy()
    rets = px.pct_change().dropna()

    # Use last available quantities as fixed (static) for returns window
    asof_prices = px.iloc[-1]
    port_val, asset_values = portfolio_valuation(positions, asof_prices)
    # weights over risky assets only
    risky_assets = [a for a in positions["asset"].unique() if a != "CASH_USD" and a in rets.columns]
    asset_values_risky = asset_values.reindex(risky_assets).fillna(0.0)
    weights = asset_values_risky / port_val if port_val != 0 else asset_values_risky*0.0

    # Portfolio daily return and PnL (linear approximation)
    port_ret = rets[risky_assets].mul(weights, axis=1).sum(axis=1)
    pnl = port_ret * port_val  # currency PnL
    return rets[risky_assets], pnl

def exposures_by_asset_class(positions: pd.DataFrame, spot_prices: pd.Series) -> pd.DataFrame:
    pos = positions.copy()
    pos["price"] = pos["asset"].map(lambda a: 1.0 if a == "CASH_USD" else float(spot_prices.get(a, np.nan)))
    pos["value"] = pos["quantity"] * pos["price"]
    out = pos.groupby("asset_class")["value"].sum().sort_values(ascending=False).reset_index()
    out.rename(columns={"value": "exposure_value"}, inplace=True)
    out["exposure_pct"] = out["exposure_value"] / out["exposure_value"].sum()
    return out
