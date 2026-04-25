"""Entry point for the Krauss ML stat-arb research demo app.

Streamlit auto-discovers the pages under ``app/pages/``.  The landing view
simply renders the Overview page content so the app opens on the headline
narrative.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(
    page_title="Krauss ML stat-arb — research demo",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from lib.data import data_build_is_complete, missing_build_warning  # noqa: E402

if not data_build_is_complete():
    missing_build_warning()

st.title("Krauss ML stat-arb — research demo")
st.caption(
    "A reproduction and extension of Krauss, Do & Huck (2017). "
    "This app summarises findings across the main, vix-regime-analysis and "
    "cost-modeling branches of the research repo."
)

st.markdown(
    """
Use the **sidebar** to navigate through the pages.

0. **Background** — plain-English primer on statistical arbitrage, Sharpe, k = 10, P̂/Û, alpha decay. **Start here** if any of those terms are unfamiliar.
1. **Overview** — headline narrative and equity curve for the full 1992-2025 study, with SPY plotted for context.
2. **Pipeline** — data flow from raw panel to daily long-short portfolio.
3. **Models explained** — RF, XGB, DNN, multi-task DNN, cls+reg pairs, and the three ensembles.
4. **Scoring schemes explained** — P-only, U-only, product, z-score and the P-gate family, plus the interactive directional-disagreement scatter.
5. **Results matrix** — filterable table of every (era, scheme, model, cost regime) we ran, with linked equity curves.
6. **Cost analysis** — turnover vs Sharpe with the no-trade-band Pareto frontier.
7. **What didn't work** — the VIX regime analysis null result.
9. **Conclusions** — what worked, what didn't, what comes next.

The trading demo (page 8) is a planned follow-up and isn't part of this release.

---

Every figure is historical. When the app says *"on 2008-10-10 the model said Y"*
that means Y is exactly the output of the fitted model for that date — not a
prediction for the future.
"""
)

st.info("All data shown here was pre-computed by `scripts/build_app_data.py`. "
        "No backtests run at page-render time.")
