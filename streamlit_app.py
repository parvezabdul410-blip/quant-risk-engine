import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from risk_engine import RiskEngine

st.set_page_config(page_title="Portfolio Risk Engine & Stress Testing", layout="wide")

st.title("Portfolio Risk Engine & Stress Testing Dashboard")

with st.sidebar:
    st.header("Inputs")
    positions_path = st.text_input("Positions CSV path", value="data/positions.csv")
    prices_path = st.text_input("Prices CSV path", value="data/prices.csv")
    scenarios_path = st.text_input("Scenarios CSV path", value="data/scenarios.csv")

    st.divider()
    st.header("Risk settings")
    lookback = st.slider("Lookback (trading days)", min_value=63, max_value=756, value=252, step=21)
    horizon = st.selectbox("VaR horizon (days)", options=[1, 5, 10], index=0)
    alpha = st.selectbox("Confidence level (alpha)", options=[0.95, 0.99], index=1)
    mc_sims = st.selectbox("Monte Carlo sims", options=[5000, 10000, 20000, 50000], index=2)
    show_component = st.checkbox("Show Component VaR (parametric)", value=True)

    st.divider()
    st.header("Stress settings")
    hist_lookback = st.slider("Historical stress lookback (days)", min_value=126, max_value=1000, value=504, step=21)
    window_days = st.selectbox("Historical stress window (days)", options=[5, 10, 20], index=1)

@st.cache_data(show_spinner=False)
def load_engine(positions_path: str, prices_path: str) -> RiskEngine:
    return RiskEngine.from_csv(positions_path, prices_path)

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

try:
    engine = load_engine(positions_path, prices_path)
except Exception as e:
    st.error(f"Failed to load inputs: {e}")
    st.stop()

report = engine.build_report(
    lookback_days=lookback,
    horizon_days=horizon,
    alpha=float(alpha),
    mc_sims=int(mc_sims),
    include_component_var=show_component,
)

# Top KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("As of", str(report.asof.date()))
k2.metric("Portfolio Value", f"${report.portfolio_value:,.0f}")
# pick historical VaR as headline
headline = report.var_cvar.loc[report.var_cvar["method"]=="Historical"].iloc[0]
k3.metric(f"VaR ({int(alpha*100)}%, {horizon}d)", f"${headline['VaR']:,.0f}")
k4.metric(f"CVaR ({int(alpha*100)}%, {horizon}d)", f"${headline['CVaR']:,.0f}")

st.divider()

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("VaR / CVaR Summary")
    st.dataframe(report.var_cvar, use_container_width=True)

    st.subheader("Portfolio PnL Distribution (Lookback)")
    losses = (-report.pnl_series).dropna()
    if horizon > 1:
        losses = losses.rolling(horizon).sum().dropna()
    fig = px.histogram(losses, nbins=60, marginal="box")
    fig.update_layout(xaxis_title="Loss (USD)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Exposures by Asset Class")
    fig2 = px.pie(report.exposures, values="exposure_value", names="asset_class", hole=0.45)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(report.exposures.style.format({"exposure_value":"${:,.0f}","exposure_pct":"{:.1%}"}), use_container_width=True)

st.divider()

st.subheader("Component VaR (Parametric Euler Allocation)")
if report.component_var is None:
    st.info("Enable Component VaR in the sidebar to see allocation.")
else:
    df = report.component_var.copy()
    df["component_VaR_pct"] = df["component_VaR"] / df["component_VaR"].sum() if df["component_VaR"].sum() != 0 else 0.0
    st.dataframe(
        df.style.format({
            "weight":"{:.2%}",
            "marginal_VaR":"${:,.0f}",
            "component_VaR":"${:,.0f}",
            "component_VaR_pct":"{:.1%}"
        }),
        use_container_width=True
    )
    fig3 = px.bar(df.sort_values("component_VaR"), x="component_VaR", y="asset", orientation="h")
    fig3.update_layout(xaxis_title="Component VaR (USD)", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

st.subheader("Stress Testing")

stress = engine.stress_tester()

colA, colB = st.columns(2)

with colA:
    st.markdown("**Deterministic Scenarios**")
    try:
        scen_df = load_csv(scenarios_path)
        scen_res = stress.run_scenarios(scen_df)
        st.dataframe(scen_res.style.format({"pnl":"${:,.0f}","pnl_pct":"{:.2%}"}), use_container_width=True)
        fig4 = px.bar(scen_res.sort_values("pnl"), x="pnl", y="scenario", orientation="h")
        fig4.update_layout(xaxis_title="Scenario PnL (USD)", yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load/run scenarios: {e}")

with colB:
    st.markdown("**Historical Window Stress**")
    worst = stress.worst_window(lookback_days=hist_lookback, window_days=int(window_days))
    st.dataframe(worst.style.format({"worst_window_pnl":"${:,.0f}","worst_window_pnl_pct":"{:.2%}"}), use_container_width=True)

st.divider()
st.caption("Notes: VaR/CVaR are losses (positive numbers). Model assumes static positions and linear PnL. Extend with Greeks for options or full revaluation for nonlinear instruments.")
