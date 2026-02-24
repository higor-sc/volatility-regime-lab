import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Volatility Regime Lab", layout="wide")

st.title("Volatility Regime Lab — IBOV Regime Dashboard")

# Path (ajuste se quiser)
DEFAULT_PATH = "data/gold/ibov_scored.parquet"

path = st.text_input("Scored parquet path", DEFAULT_PATH)

@st.cache_data
def load_data(p):
    df = pd.read_parquet(p)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

try:
    df = load_data(path)
except Exception as e:
    st.error(f"Failed to load parquet: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input("Date range", (min_date, max_date))
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]

show_volume = st.sidebar.checkbox("Show volume", True)
show_table = st.sidebar.checkbox("Show data table", False)

# Candlestick + regimes (color by regime)
# We'll create one candlestick trace and overlay regime markers as background rectangles.
fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
    )
)

# Regime coloring (0 low, 1 mid, 2 high)
# Add vertical background bands where regime changes
if "regime" in df.columns:
    regime_colors = {0: "rgba(0,200,0,0.08)", 1: "rgba(255,165,0,0.08)", 2: "rgba(255,0,0,0.08)"}

    # Find regime segments
    seg_start = df.iloc[0]["Date"]
    seg_regime = int(df.iloc[0]["regime"])

    for i in range(1, len(df)):
        r = int(df.iloc[i]["regime"])
        if r != seg_regime:
            seg_end = df.iloc[i - 1]["Date"]
            fig.add_vrect(
                x0=seg_start, x1=seg_end,
                fillcolor=regime_colors.get(seg_regime, "rgba(0,0,0,0.05)"),
                opacity=1,
                line_width=0
            )
            seg_start = df.iloc[i]["Date"]
            seg_regime = r

    # last segment
    fig.add_vrect(
        x0=seg_start, x1=df.iloc[-1]["Date"],
        fillcolor=regime_colors.get(seg_regime, "rgba(0,0,0,0.05)"),
        opacity=1,
        line_width=0
    )

fig.update_layout(height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Optional volume plot
if show_volume and "Volume" in df.columns:
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume"))
    vol_fig.update_layout(height=250)
    st.plotly_chart(vol_fig, use_container_width=True)

# Quick stats
if "regime" in df.columns:
    st.subheader("Regime distribution (in selected range)")
    st.write(df["regime"].value_counts().sort_index())

# Data table
if show_table:
    st.subheader("Scored dataset preview")
    st.dataframe(df.tail(200))