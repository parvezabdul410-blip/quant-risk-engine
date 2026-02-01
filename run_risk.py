import argparse
import pandas as pd

from risk_engine import RiskEngine

def main():
    p = argparse.ArgumentParser(description="Portfolio Risk Engine CLI")
    p.add_argument("--positions", default="data/positions.csv")
    p.add_argument("--prices", default="data/prices.csv")
    p.add_argument("--scenarios", default="data/scenarios.csv")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--mc-sims", type=int, default=20000)
    args = p.parse_args()

    engine = RiskEngine.from_csv(args.positions, args.prices)
    report = engine.build_report(
        lookback_days=args.lookback,
        horizon_days=args.horizon,
        alpha=args.alpha,
        mc_sims=args.mc_sims,
        include_component_var=True,
    )

    print(f"As of: {report.asof.date()}")
    print(f"Portfolio value: ${report.portfolio_value:,.2f}")
    print("\nVaR/CVaR:")
    print(report.var_cvar.to_string(index=False))

    if report.component_var is not None:
        print("\nTop Component VaR contributors:")
        top = report.component_var.head(10)
        print(top.to_string(index=False))

    # Stress
    scen = pd.read_csv(args.scenarios)
    stress = engine.stress_tester()
    scen_res = stress.run_scenarios(scen)
    print("\nScenario PnL:")
    print(scen_res.to_string(index=False))

if __name__ == "__main__":
    main()
