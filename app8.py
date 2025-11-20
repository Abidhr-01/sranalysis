import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide", page_title="Stock Candlestick Viewer")

# --- Folder path ---
FOLDER_PATH = "SymbolWise"

if os.path.exists(FOLDER_PATH):
    csv_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.csv')]
    if not csv_files:
        st.warning("No CSV files found in the folder.")
    else:
        # Remove '.csv'
        file_names = [os.path.splitext(f)[0] for f in csv_files]
        selected_name = st.selectbox("Select Stock Symbol:", file_names)
        selected_file = csv_files[file_names.index(selected_name)]
        file_path = os.path.join(FOLDER_PATH, selected_file)

        # --- SEARCH FILTER ---
        st.markdown("### 🔍 Search Support / Resistance Lines")
        filter_option = st.selectbox(
            "Select Filter:",
            ["All", "Support Only", "Resistance Only"]
        )
        apply_filter = st.button("Apply Filter")

        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)

            df = df.groupby('date').last().sort_index()

            # Parameters
            WINDOW = 10
            MIN_BOUNCES = 2

            # Local min/max
            df['min'] = df['Low'][(df['Low'] == df['Low'].rolling(WINDOW, center=True).min())]
            df['max'] = df['High'][(df['High'] == df['High'].rolling(WINDOW, center=True).max())]

            # Support detection
            supports = []
            for level in df['min'].dropna().unique():
                bounces = 0
                skip = False
                for i in range(1, len(df)-1):
                    if df['Low'].iloc[i] == level:
                        if skip: continue
                        bounces += 1
                        skip = True
                    else:
                        skip = False
                if bounces >= MIN_BOUNCES:
                    supports.append(level)

            # Resistance detection
            resistances = []
            for level in df['max'].dropna().unique():
                bounces = 0
                skip = False
                for i in range(1, len(df)-1):
                    if df['High'].iloc[i] == level:
                        if skip: continue
                        bounces += 1
                        skip = True
                    else:
                        skip = False
                if bounces >= MIN_BOUNCES:
                    resistances.append(level)

            # Levels
            all_levels = set(supports + resistances)
            both_levels = set(supports).intersection(resistances)

            # --- APPLY FILTER ---
            if apply_filter:
                if filter_option == "Support Only":
                    all_levels = {x for x in all_levels if x in supports or x in both_levels}
                elif filter_option == "Resistance Only":
                    all_levels = {x for x in all_levels if x in resistances or x in both_levels}
                # If "All" → do nothing

            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=selected_name,
                showlegend=False
            ))

            # Add S/R lines with single legend per type
            legend_done = {"Support": False, "Resistance": False, "S/R": False}

            for level in sorted(all_levels):
                if level in both_levels:
                    color = "blue"
                    label = "S/R"
                elif level in supports:
                    color = "green"
                    label = "Support"
                else:
                    color = "red"
                    label = "Resistance"

                fig.add_trace(go.Scatter(
                    x=[df.index[0], df.index[-1]],
                    y=[level, level],
                    mode="lines",
                    line=dict(color=color, width=1.5, dash='dot'),
                    name=label if not legend_done[label] else None,
                    showlegend=not legend_done[label]
                ))

                legend_done[label] = True

            fig.update_layout(
                title=f"{selected_name} Candlestick with S/R Filter: {filter_option}",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                hovermode="x unified",
                height=800,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.error("Folder path does not exist.")
