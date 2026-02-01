import pandas as pd
from risk_engine import RiskEngine

def test_smoke():
    engine = RiskEngine.from_csv("data/positions.csv", "data/prices.csv")
    report = engine.build_report()
    assert report.portfolio_value > 0
    assert not report.var_cvar.empty
