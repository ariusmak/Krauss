"""Page 1 — Overview.

Headline narrative, full 1992-2025 equity curve, top-three findings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Overview", page_icon=":bar_chart:", layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning,
    load_equity_curves, load_summary_table, load_pipeline_metadata,
    load_spy_benchmark,
)
from lib.charts import equity_curve_figure  # noqa: E402

if not data_build_is_complete():
    missing_build_warning()

equity = load_equity_curves()
summary = load_summary_table()
meta = load_pipeline_metadata()
spy = load_spy_benchmark()

st.info(
    ":bulb: **New to this?** Read the [Background primer](Background) first — "
    "it explains statistical arbitrage, dollar-neutral long-short, "
    "Sharpe ratios, P̂, Û and alpha decay in plain English."
)

st.title("Overview")
st.caption(
    "A faithful reproduction of Krauss, Do & Huck (2017) plus three extensions: "
    "magnitude-aware scoring (Phase 2), VIX-regime sensitivity, and a 2015-2025 "
    "out-of-sample extension on Refinitiv Datastream."
)

# ----- Top-line narrative ---------------------------------------------------
with st.container(border=True):
    st.subheader("What this project asks")
    st.markdown(
        """
The original paper shows that a daily long-short portfolio built from ML
predictions of *"next-day return above or below the cross-sectional median"*
earned positive pre-cost returns on US equities from 1992 to 2015. Our question
is whether that result **still holds after costs, out-of-sample to 2025, and
under more realistic scoring rules**.

The short answer: partially. The P-only baseline is robust pre-cost but
severely degraded post-cost after 2008. A z-score composite of direction and
magnitude beats the baseline on both sides. Product-composite scoring, which
multiplies direction by magnitude, *destroys* returns — a finding that is
itself informative about how often the two model heads disagree.
        """
    )

# ----- Full equity curve ----------------------------------------------------
st.subheader("ENS1 equity curve, 1992-2025, with SPY for context")

# ENS1 P-only, both cost regimes, both eras — stitched across eras.
ens1 = equity.query("model == 'ENS1' and scheme == 'P-only'").copy()
ens1 = ens1.sort_values(["cost_regime", "date"])
# Compound across era boundaries so the line is continuous through 2015.
ens1["ret"] = ens1["ret"].fillna(0.0)
ens1["stitched_cum"] = ens1.groupby("cost_regime")["ret"].transform(
    lambda s: (1.0 + s).cumprod() - 1.0
)

fig = go.Figure()

ens1_colors = {"no_cost": "#1f77b4", "5bps_half_turn": "#2e7d32"}
ens1_labels = {
    "no_cost": "ENS1 P-only · pre-cost",
    "5bps_half_turn": "ENS1 P-only · post-cost (5 bps/half-turn)",
}
for cr, grp in ens1.groupby("cost_regime"):
    fig.add_trace(go.Scatter(
        x=grp["date"], y=grp["stitched_cum"],
        mode="lines",
        name=ens1_labels[cr],
        line=dict(color=ens1_colors[cr], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>cum ret = %{y:.2f}<extra></extra>",
    ))

# SPY overlay — cumulative total return, rebased to start at 0 from the first
# date the strategy has data for.
if spy is not None and not spy.empty:
    start = ens1["date"].min()
    spy_in = spy[spy["date"] >= start].copy()
    spy_in["ret"] = spy_in["ret"].fillna(0.0)
    spy_in["cum_ret"] = (1.0 + spy_in["ret"]).cumprod() - 1.0
    fig.add_trace(go.Scatter(
        x=spy_in["date"], y=spy_in["cum_ret"],
        mode="lines",
        name="SPY total return (long market exposure — for context only)",
        line=dict(color="#999", width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>SPY cum ret = %{y:.2f}<extra></extra>",
    ))

# Mark the extension-era boundary.
fig.add_shape(
    type="line", x0="2015-10-16", x1="2015-10-16",
    y0=0, y1=1, yref="paper",
    line=dict(color="#666", width=1, dash="dash"),
)
fig.add_annotation(
    x="2015-10-16", y=1.0, yref="paper", yanchor="bottom",
    text="extension era →", showarrow=False,
    font=dict(size=11, color="#444"),
)

fig.update_layout(
    height=500,
    xaxis_title=None,
    yaxis_title="Cumulative return",
    legend=dict(orientation="h", y=-0.2),
    margin=dict(l=50, r=20, t=30, b=60),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "ENS1 is the simple mean of RF, XGB and DNN direction predictions. "
    "Before costs it looks like a straight line up through the CRSP era. "
    "After 5 bps/half-turn, post-2008 returns flatten markedly — the gap is "
    "the cost bill the original paper did not pay. **SPY is plotted as "
    "context, not as a direct benchmark**: this strategy is dollar-neutral, "
    "so the honest comparator is cash plus alpha, not a 100 %-long index."
)

# ----- Top-3 findings callouts ---------------------------------------------
st.subheader("Top three findings")

col1, col2, col3 = st.columns(3)

def _find(model: str, scheme: str, era: str, cost: str) -> float | None:
    row = summary.query(
        "model == @model and scheme == @scheme and era == @era and cost_regime == @cost"
    )
    if row.empty:
        return None
    return float(row["sharpe"].iloc[0])


sharpe_ens1_precost = _find("ENS1", "P-only", "1992-2015 (CRSP)", "no_cost")
sharpe_ens1_postcost = _find("ENS1", "P-only", "1992-2015 (CRSP)", "5bps_half_turn")
sharpe_zcomp_postcost = _find("ENS1", "Z-comp", "1992-2015 (CRSP)", "5bps_half_turn")
sharpe_prod_postcost = _find("ENS1", "Product", "1992-2015 (CRSP)", "5bps_half_turn")

with col1:
    st.metric(
        label="Baseline works, but costs matter a lot",
        value=f"Sharpe {sharpe_ens1_postcost:0.2f}" if sharpe_ens1_postcost else "n/a",
        delta=(f"{sharpe_ens1_postcost - sharpe_ens1_precost:+0.2f} vs pre-cost"
               if sharpe_ens1_postcost and sharpe_ens1_precost else None),
        delta_color="inverse",
    )
    st.caption(
        "ENS1 P-only on 1992-2015, k=10, post-cost. The cost drag is about "
        "half the pre-cost Sharpe."
    )

with col2:
    st.metric(
        label="Z-score composite beats the baseline",
        value=f"Sharpe {sharpe_zcomp_postcost:0.2f}" if sharpe_zcomp_postcost else "n/a",
        delta=(f"{sharpe_zcomp_postcost - sharpe_ens1_postcost:+0.2f} vs P-only"
               if sharpe_zcomp_postcost and sharpe_ens1_postcost else None),
    )
    st.caption(
        "Z-comp averages cross-sectional z-scores of direction and magnitude. "
        "A modest but consistent lift over P-only."
    )

with col3:
    st.metric(
        label="Product composite destroys returns",
        value=f"Sharpe {sharpe_prod_postcost:0.2f}" if sharpe_prod_postcost is not None else "n/a",
        delta=(f"{sharpe_prod_postcost - sharpe_ens1_postcost:+0.2f} vs P-only"
               if sharpe_prod_postcost is not None and sharpe_ens1_postcost else None),
        delta_color="inverse",
    )
    st.caption(
        "Multiplying direction by magnitude amplifies the cases where the two "
        "model heads disagree. We unpack this on the scoring-schemes page."
    )

# ----- Small orientation table --------------------------------------------
st.subheader("Study universe at a glance")

def _era_summary(era: str) -> dict:
    era_spec = meta["eras"].get(
        "crsp" if "CRSP" in era else "extension", {}
    )
    return {
        "era": era,
        "start": era_spec.get("start"),
        "end": era_spec.get("end"),
        "universe": era_spec.get("universe"),
    }

eras_df = pd.DataFrame([
    _era_summary("1992-2015 (CRSP)"),
    _era_summary("2015-2025 (extension)"),
])
st.dataframe(eras_df, use_container_width=True, hide_index=True)

st.caption(
    "CRSP era reproduces the paper's universe. Extension era is an independent "
    "out-of-sample test using Refinitiv Datastream for the US S&P 500 "
    "constituent panel — different vendor, same methodology."
)
