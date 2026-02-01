from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

def _scale_horizon(x: float, horizon_days: int) -> float:
    # sqrt time scaling for vol; mean scales linearly but typically negligible for VaR horizon=1
    return x * np.sqrt(horizon_days)

def parametric_var_cvar(mu: np.ndarray, cov: np.ndarray, weights: np.ndarray, port_val: float, alpha: float = 0.99, horizon_days: int = 1) -> tuple[float, float]:
    """
    Gaussian VaR/CVaR on portfolio PnL (positive number = loss).
    """
    # portfolio mean & std of returns
    mu_p = float(weights @ mu)
    var_p = float(weights @ cov @ weights)
    sigma_p = np.sqrt(max(var_p, 0.0))

    # horizon scaling
    mu_h = mu_p * horizon_days
    sigma_h = sigma_p * np.sqrt(horizon_days)

    z = norm.ppf(alpha)
    # Loss = -(return)*V. VaR loss at alpha:
    var_loss = (-(mu_h) + z * sigma_h) * port_val
    # CVaR for normal: sigma * phi(z)/(1-alpha) - mu
    cvar_loss = (-(mu_h) + sigma_h * norm.pdf(z) / (1 - alpha)) * port_val
    return float(var_loss), float(cvar_loss)

def historical_var_cvar(pnl: pd.Series, port_val: float, alpha: float = 0.99, horizon_days: int = 1) -> tuple[float, float]:
    """
    Historical VaR/CVaR based on pnl series (daily, currency units).
    For horizon_days>1, aggregate by rolling sum.
    """
    x = pnl.copy().dropna()
    if horizon_days > 1:
        x = x.rolling(horizon_days).sum().dropna()
    # Loss is -PnL
    losses = -x
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return var, cvar

def monte_carlo_var_cvar(mu: np.ndarray, cov: np.ndarray, weights: np.ndarray, port_val: float, alpha: float = 0.99, horizon_days: int = 1, n_sims: int = 20000, seed: int = 123) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    # Draw daily returns ~ N(mu, cov), aggregate to horizon by summing means and scaling cov
    mu_h = mu * horizon_days
    cov_h = cov * horizon_days
    sims = rng.multivariate_normal(mu_h, cov_h, size=n_sims)
    port_rets = sims @ weights
    pnl = port_rets * port_val
    losses = -pnl
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return var, cvar

def component_var_parametric(mu: np.ndarray, cov: np.ndarray, weights: np.ndarray, port_val: float, alpha: float = 0.99, horizon_days: int = 1) -> pd.DataFrame:
    """
    Component VaR for Gaussian model using Euler allocation.
    Returns dataframe with weight, marginal_VaR (per unit weight), component_VaR (currency).
    """
    # Portfolio sigma
    sigma_p = np.sqrt(max(float(weights @ cov @ weights), 0.0))
    if sigma_p == 0.0:
        return pd.DataFrame({"weight": weights, "marginal_VaR": np.zeros_like(weights), "component_VaR": np.zeros_like(weights)})

    z = norm.ppf(alpha)
    # marginal VaR in return space (loss) per d(weight_i)
    # VaR_loss = ( -mu_h + z*sigma_h )*V; derivative wrt w_i is ( -mu_i*h + z*( (cov w)_i / sigma_p )*sqrt(h) )*V
    mu_h = mu * horizon_days
    sigma_scale = np.sqrt(horizon_days)
    covw = cov @ weights
    marginal = (-(mu_h) + z * (covw / sigma_p) * sigma_scale) * port_val
    component = weights * marginal
    return pd.DataFrame({"weight": weights, "marginal_VaR": marginal, "component_VaR": component})
