import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide", page_title="Support & Resistance Finder")

FOLDER_PATH = "SymbolWise"
TOLERANCE = 1
MIN_TOUCHES = 2

def load_symbols():
    if not os.path.exists(FOLDER_PATH):
        return []
    return [f for f in os.listdir(FOLDER_PATH) if f.endswith(".csv")]

def compute_avg(df):
    df["avg"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    return df

def find_levels(df):
    avg = df["avg"].values
    volumes = df["volume"].values
    
    levels = []

    for i in range(2, len(avg) - 2):
        level = avg[i]

        # All points touching the level
        touch_indices = np.where(np.abs(avg - level) <= TOLERANCE)[0]
        touches = len(touch_indices)
        if touches < MIN_TOUCHES:
            continue

        vol_at_touches = volumes[touch_indices]
        avg_vol = np.mean(volumes)  # overall avg volume
        mean_touch_vol = np.mean(vol_at_touches)

        duration = touch_indices[-1] - touch_indices[0]

        # Average duration between touches
        if len(touch_indices) > 1:
            gaps = np.diff(touch_indices)
            avg_gap = np.mean(gaps)
        else:
            avg_gap = 0

        # NEW strength rules
        if avg_gap < 30 and mean_touch_vol < avg_vol:
            strength = "Minor"
        elif avg_gap > 30 and mean_touch_vol > avg_vol:
            strength = "Major"
        else:
            strength = "Semi-Major"


        levels.append({
            "price": round(level, 2),
            "touches": touches,
            "strength": strength,
            "mean_volume": round(mean_touch_vol, 0),
            "first_touch": int(touch_indices[0]),
            "last_touch": int(touch_indices[-1]),
            "duration": duration,
            "avg_gap": round(avg_gap, 2),
            "index": i
        })

    # Remove similar levels within tolerance
    filtered = []
    for lvl in levels:
        if not filtered or all(abs(lvl["price"] - x["price"]) > TOLERANCE for x in filtered):
            filtered.append(lvl)

    return filtered


def classify_levels(df, levels):
    close = df["close"].values
    last_price = close[-1]

    supports, resistances = [], []

    for lvl in levels:
        price = lvl["price"]

        if last_price > price:
            lvl["type"] = "Support"
            supports.append(lvl)
        else:
            lvl["type"] = "Resistance"
            resistances.append(lvl)

    supports = sorted(supports, key=lambda x: abs(x["price"] - last_price))[:2]
    resistances = sorted(resistances, key=lambda x: abs(x["price"] - last_price))[:2]

    return supports, resistances

st.title("📊 Automatic Support & Resistance Detector")

files = load_symbols()
if not files:
    st.error("No CSV files found in SymbolWise/")
    st.stop()

selected = st.selectbox("Select Symbol", files)

df = pd.read_csv(f"{FOLDER_PATH}/{selected}")
df.columns = df.columns.str.lower()
df = compute_avg(df)

levels = find_levels(df)
supports, resistances = classify_levels(df, levels)

def filter_level_data(levels):
    rows = []
    for lvl in levels:
        rows.append({
            "Price": lvl["price"],
            "Strength": lvl["strength"],
            "Avg Volume at Touch": lvl["mean_volume"],
            "Avg Duration Between Touches": lvl["avg_gap"]
        })
    return pd.DataFrame(rows)


supports_df = filter_level_data(supports)
resistances_df = filter_level_data(resistances)


col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 Last 2 Supports")
    st.dataframe(supports_df, use_container_width=True) 

with col2:
    st.subheader("🔴 Last 2 Resistances")
    st.dataframe(resistances_df, use_container_width=True)

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["date"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="Price"
))

for s in supports:
    fig.add_hline(
        y=s["price"],
        line_width=2,
        line_color="green"
    )

for r in resistances:
    fig.add_hline(
        y=r["price"],
        line_width=2,
        line_color="red"
    )

fig.update_layout(title=f"Support & Resistance: {selected}", height=600)

st.plotly_chart(fig, use_container_width=True)
