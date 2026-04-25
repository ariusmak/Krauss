"""Shared data loaders for the Streamlit app.

All loaders wrap a parquet read with ``st.cache_data`` so repeated page renders
do not re-read the files.  The app assumes every file under ``app/data/`` was
produced by ``scripts/build_app_data.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"


def _path(name: str) -> Path:
    return DATA_DIR / name


@st.cache_data(show_spinner=False)
def load_equity_curves() -> pd.DataFrame:
    df = pd.read_parquet(_path("equity_curves.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_summary_table() -> pd.DataFrame:
    return pd.read_parquet(_path("summary_table.parquet"))


@st.cache_data(show_spinner=False)
def load_regime_labels() -> pd.DataFrame:
    df = pd.read_parquet(_path("regime_labels.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_daily_holdings() -> pd.DataFrame | None:
    p = _path("daily_holdings.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_disagreement_panel() -> pd.DataFrame | None:
    p = _path("disagreement_panel.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_pipeline_metadata() -> dict:
    with _path("pipeline_metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_cost_bands() -> pd.DataFrame | None:
    p = _path("cost_bands.parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_regime_k_sensitivity() -> pd.DataFrame | None:
    p = _path("regime_k_sensitivity.parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_spy_benchmark() -> pd.DataFrame | None:
    p = _path("spy_benchmark.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_regime_leg_decomp() -> pd.DataFrame | None:
    p = _path("regime_leg_decomp.parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


def data_build_is_complete() -> bool:
    required = ["equity_curves.parquet", "summary_table.parquet",
                "regime_labels.parquet", "pipeline_metadata.json"]
    return all(_path(p).exists() for p in required)


def missing_build_warning() -> None:
    st.error(
        "App data is missing. Run `python scripts/build_app_data.py` from the "
        "repo root, then reload the page."
    )
    st.stop()
