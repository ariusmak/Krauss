"""Page 2 — Reproduction results.

Short, factual page comparing our reproduction of the Krauss et al. (2017)
paper to its published numbers.  Hard cap on length — under 300 words of
prose plus a small comparison table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Reproduction", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning, load_summary_table,
)

if not data_build_is_complete():
    missing_build_warning()

summary = load_summary_table()

st.info(
    ":bulb: **New to this?** The [Background primer](Background) defines "
    "Sharpe, basis points, post-cost and the rest of the vocabulary used "
    "here."
)

st.title("Reproduction results")
st.caption(
    "How closely do our re-runs match the published numbers in "
    "Krauss et al. (2017), Tables 2-3, on the same 1992-2015 sample?"
)

# --- Three-bullet summary --------------------------------------------------

st.markdown(
    """
- **94 % of the paper's pre-cost ENS1 return** is reproduced. Our
  ENS1 averages **0.42 %** per day pre-cost vs the paper's **0.45 %**;
  individual base learners (DNN / GBT / RAF) all land within ±15 % of
  their published per-day means.
- **Post-cost Sharpe exceeds the paper** at **2.18 vs 1.81**, even though
  the per-day return is fractionally lower. The gap is **lower turnover
  than the paper implies** — our walk-forward produces ~2.5 sides
  rebalanced per day vs the ~4.0 implied by the paper's cost arithmetic
  — so the same 5 bps / half-turn convention bites less hard.
- **The 2008-09 sub-period reproduces at ~29 %** of the paper's per-day
  return — the largest gap in our parity table. We attribute this to
  vendor-differential constituent reconstruction around the Lehman
  window: small differences in which delisted names re-enter the panel
  during stress periods compound into a meaningful return spread.
    """
)

# --- Comparison table -------------------------------------------------------

st.subheader("Per-model comparison — 1992-2015, k = 10, pre-cost daily return")

# Pull pre-cost daily returns from summary_table.parquet and compare to
# the paper's published Table 2 values.  These are the headline numbers
# from Krauss et al. 2017 Table 2.
PAPER_VALUES = {
    "DNN":  0.0033,
    "XGB":  0.0037,
    "RF":   0.0043,
    "ENS1": 0.0045,
}
DISPLAY_NAME = {"DNN": "DNN", "XGB": "GBT", "RF": "RAF", "ENS1": "ENS1"}

rows = []
for model, paper in PAPER_VALUES.items():
    sub = summary.query(
        "model == @model and scheme == 'P-only' "
        "and era == '1992-2015 (CRSP)' and cost_regime == 'no_cost'"
    )
    if sub.empty:
        continue
    ours = float(sub["full_sample_return"].iloc[0])
    rows.append({
        "Model": DISPLAY_NAME[model],
        "Paper pre-cost daily ret": f"{paper * 100:.2f}%",
        "Ours pre-cost daily ret":  f"{ours * 100:.2f}%",
        "Ratio (ours / paper)":     f"{ours / paper * 100:.0f}%",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption(
    "Source for paper values: Krauss, Do & Huck (2017), Table 2. "
    "Source for our values: `app/data/summary_table.parquet`, "
    "(model, scheme=P-only, era=1992-2015 CRSP, cost_regime=no_cost) "
    "rows."
)

st.divider()
st.markdown(
    "→ The full results matrix (every model × scheme × era × cost regime "
    "we ran) lives in **[Results matrix](Results_matrix)**. The exact "
    "deviations between our pipeline and the paper's are documented in "
    "[`docs/reproduction_deviations.md`](https://github.com/ariusmak/"
    "krauss-ml-statarb/blob/main/docs/reproduction_deviations.md)."
)
