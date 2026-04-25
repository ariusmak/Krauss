"""Reusable Plotly chart builders."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SCHEME_ORDER = [
    "P-only", "U-only", "Z-comp", "Product", "P-gate(0.03)", "P-gate(0.05)"
]
MODEL_ORDER = ["RF", "XGB", "DNN", "ENS1", "ENS2", "ENS3"]


def equity_curve_figure(
    df: pd.DataFrame,
    *,
    group_col: str = "label",
    title: str | None = None,
    height: int = 460,
    log_y: bool = False,
) -> go.Figure:
    """Plotly line chart of cumulative returns.

    ``df`` must include columns ``date``, ``cum_ret`` and whatever ``group_col``
    identifies the distinct lines to draw.
    """
    fig = px.line(
        df,
        x="date",
        y="cum_ret",
        color=group_col,
        hover_data={"cum_ret": ":.3f", "date": True, group_col: True},
    )
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        legend_title=None,
        xaxis_title=None,
        yaxis_title="Cumulative return",
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
    )
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_xaxes(rangeslider_visible=False, showgrid=True)
    return fig
