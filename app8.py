import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide", page_title="Support & Resistance Finder")

FOLDER_PATH = "SymbolWise"
TOLERANCE = 1
MIN_TOUCHES = 2
WINDOW = 5  # The window for rolling min/max

def load_symbols():
    if not os.path.exists(FOLDER_PATH):
        return []
    return [f for f in os.listdir(FOLDER_PATH) if f.endswith(".csv")]

def compute_avg(df):
    df["avg"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    return df

def find_levels(df, mode):
    volumes = df["volume"].values
    levels = []

    if mode == "Avg":
        # ORIGINAL LOGIC
        avg = df["avg"].values
        for i in range(2, len(avg) - 2):
            level = avg[i]
            touch_indices = np.where(np.abs(avg - level) <= TOLERANCE)[0]
            if len(touch_indices) >= MIN_TOUCHES:
                levels.append(create_level_dict(level, touch_indices, volumes, i))
    else:
        # NEW HIGH/LOW LOGIC
        # Identify local mins and maxes
        df['min_pivot'] = df['low'][(df['low'] == df['low'].rolling(WINDOW, center=True).min())]
        df['max_pivot'] = df['high'][(df['high'] == df['high'].rolling(WINDOW, center=True).max())]
        
        # Find levels based on local Mins (Support candidates)
        for i in df.index[df['min_pivot'].notna()]:
            level = df.loc[i, 'low']
            touch_indices = np.where(np.abs(df['low'] - level) <= TOLERANCE)[0]
            if len(touch_indices) >= MIN_TOUCHES:
                levels.append(create_level_dict(level, touch_indices, volumes, i))
        
        # Find levels based on local Maxes (Resistance candidates)
        for i in df.index[df['max_pivot'].notna()]:
            level = df.loc[i, 'high']
            touch_indices = np.where(np.abs(df['high'] - level) <= TOLERANCE)[0]
            if len(touch_indices) >= MIN_TOUCHES:
                levels.append(create_level_dict(level, touch_indices, volumes, i))

    # Remove similar levels within tolerance
    filtered = []
    for lvl in sorted(levels, key=lambda x: x["price"]):
        if not filtered or all(abs(lvl["price"] - x["price"]) > TOLERANCE for x in filtered):
            filtered.append(lvl)

    return filtered

def create_level_dict(level, touch_indices, volumes, i):
    vol_at_touches = volumes[touch_indices]
    avg_vol = np.mean(volumes)
    mean_touch_vol = np.mean(vol_at_touches)
    duration = touch_indices[-1] - touch_indices[0]
    avg_gap = np.mean(np.diff(touch_indices)) if len(touch_indices) > 1 else 0

    if avg_gap < 30 and mean_touch_vol < avg_vol:
        strength = "Minor"
    elif avg_gap > 30 and mean_touch_vol > avg_vol:
        strength = "Major"
    else:
        strength = "Semi-Major"

    return {
        "price": round(level, 2),
        "touches": len(touch_indices),
        "strength": strength,
        "mean_volume": round(mean_touch_vol, 0),
        "first_touch": int(touch_indices[0]),
        "last_touch": int(touch_indices[-1]),
        "duration": duration,
        "avg_gap": round(avg_gap, 2),
        "index": i
    }

def classify_levels_with_flip(df, levels):
    close = df["close"].values
    last_close = close[-1]
    supports, resistances = [] , []
    BUFFER = 2

    for lvl in levels:
        price = lvl["price"]
        if last_close <= (price - BUFFER):
            lvl["type"] = "Resistance (Flipped)"
            resistances.append(lvl)
        elif last_close >= (price + BUFFER):
            lvl["type"] = "Support (Flipped)"
            supports.append(lvl)
        else:
            if last_close > price:
                lvl["type"] = "Support"
                supports.append(lvl)
            else:
                lvl["type"] = "Resistance"
                resistances.append(lvl)

    supports = sorted(supports, key=lambda x: abs(x["price"] - last_close))[:2]
    resistances = sorted(resistances, key=lambda x: abs(x["price"] - last_close))[:2]
    return supports, resistances

# --- UI ---
st.title("📊 Automatic Support & Resistance Detector")

# Toggle in Sidebar
mode_choice = st.sidebar.radio("Select Detection Method", ["Avg", "High/Low"])

files = load_symbols()
if not files:
    st.error("No CSV files found in SymbolWise/")
    st.stop()

selected = st.selectbox("Select Symbol", files)

df = pd.read_csv(f"{FOLDER_PATH}/{selected}")
df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'])

max_date = df['date'].max()
one_year_ago = max_date - pd.DateOffset(years=1)
df = df[df['date'] >= one_year_ago].reset_index(drop=True)

df = compute_avg(df)
levels = find_levels(df, mode_choice)
supports, resistances = classify_levels_with_flip(df, levels)

# Table Rendering
def filter_level_data(levels):
    rows = []
    for lvl in levels:
        rows.append({
            "Price": lvl["price"],
            "Strength": lvl["strength"],
            "Avg Volume": lvl["mean_volume"],
            "Type": lvl.get("type", "")
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

# Plotting
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
))

legend_done = {}
for lvl in supports + resistances:
    color = "green" if "Support" in lvl["type"] else "red"
    label = lvl["type"].replace("(Flipped)", "").strip()
    showlegend = label not in legend_done
    fig.add_trace(go.Scatter(
        x=[df["date"].iloc[0], df["date"].iloc[-1]],
        y=[lvl["price"], lvl["price"]],
        mode="lines",
        line=dict(color=color, width=2, dash="dot"),
        name=label,
        showlegend=showlegend
    ))
    legend_done[label] = True

fig.update_layout(title=f"Support & Resistance ({mode_choice}): {selected}", height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
