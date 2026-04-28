"""Page 1 — Original paper overview.

Two-paragraph summary of Krauss, Do & Huck (2017) framing the rest of the
deck.  Stays under 250 words by design.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Paper overview", page_icon=":bar_chart:",
                   layout="wide")

st.info(
    ":bulb: **New to this?** The [Background primer](Background) explains "
    "every term used here (statistical arbitrage, k = 10, Sharpe, P̂ / Û, "
    "alpha decay)."
)

st.title("Original paper overview")
st.caption(
    "Krauss, Do & Huck (2017). *Deep neural networks, gradient-boosted trees, "
    "random forests: Statistical arbitrage on the S&P 500.* "
    "European Journal of Operational Research, 259(2), 689–702."
)

st.markdown(
    """
**The strategy.** Every trading day, three machine-learning models —
a feed-forward DNN, a gradient-boosted tree (GBT), and a random forest
(RAF) — predict whether each S&P 500 constituent's next-day return will
beat the next-day cross-sectional median. The 31 input features are
purely lagged returns (R1…R20, R40, R60, …, R240). The top-k stocks by
predicted probability go long; the bottom-k go short, equal-weight
within each leg, dollar-neutral across legs. Three ensembles
(equal / Gini-weighted / rank-weighted averages of the base learners)
sit on top.

**The data and the result.** Universe is every S&P 500 ever-member
between Jan 1990 and Oct 2015 (1,322 names) drawn from Thomson Reuters
Datastream, with strict month-end no-lookahead membership rules.
Twenty-three rolling walk-forward periods of 750 train + 250 trade days
yield 5,750 strictly out-of-sample trading days. At k = 10 the headline
ENS1 ensemble earns **0.45 % per day pre-cost** and **0.25 % per day
after a 5 bps / half-turn assumption**, an annualised post-cost return
near 73 % with Sharpe 1.81. The paper also documents **alpha decay**:
the bulk of the per-day return is concentrated in the first part of the
sample and shrinks substantially after roughly 2001.
    """
)

st.divider()
st.markdown(
    "→ For what we did differently — magnitude predictions, five new "
    "scoring schemes, a 2015-2025 out-of-sample extension — see "
    "**[Methodology](Methodology)**. For our reproduction numbers vs the "
    "paper's, see **[Reproduction](Reproduction)**."
)
