# app_player_dashboard.py
# Author: Abhinav | System Path: C:/Users/Abhinav/TransferIQ/
# Description: Interactive dashboard for analyzing player market values

import streamlit as st
import pandas as pd

# -------------------------------------------------------------------
# ⚙️ Load Dataset (updated system path)
# -------------------------------------------------------------------
data_file = r"C:\Users\Abhinav\TransferIQ\datasets\Player_Market_Value_Prediction_Dataset.csv"
data = pd.read_csv(data_file)

# Normalize column naming
if "Player Name" in data.columns:
    data.rename(columns={"Player Name": "player"}, inplace=True)

# -------------------------------------------------------------------
# 🎨 Streamlit Page Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="TransferIQ | Player Market Insights",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar setup
st.sidebar.header("🎯 Player Selection")
st.sidebar.info("Browse through players to explore their valuation trends and performance insights.")

# Player list
player_names = sorted(data["player"].dropna().unique().tolist())
chosen_player = st.sidebar.selectbox("Select Player", player_names)

# -------------------------------------------------------------------
# 🏆 Main Page Content
# -------------------------------------------------------------------
st.title("⚽ TransferIQ: Player Market Value Analytics")
st.caption("Developed by Abhinav | System: C:/Users/Abhinav/TransferIQ/")
st.markdown("Get detailed analytics, model predictions, and valuation breakdowns for professional football players.")

if chosen_player:
    player_data = data[data["player"] == chosen_player].iloc[0]

    usd_market_value = player_data["Market Value (M)"]
    inr_rate = 88.73  # USD → INR conversion
    inr_market_value = usd_market_value * 1_000_000 * inr_rate

    st.markdown(f"## 💰 Current Valuation for **{chosen_player}**")
    left, right = st.columns([1, 2])
    left.metric("💵 Market Value (USD)", f"${usd_market_value:.2f} M")
    right.metric("🇮🇳 Market Value (INR)", f"₹{inr_market_value:,.0f}")
    st.divider()

    # -------------------------------------------------------------------
    # 🧠 Player Profile
    # -------------------------------------------------------------------
    st.subheader("🧩 Player Overview")
    profile_keys = ["Age", "Injury Status", "Sentiment Label"]
    profile_cols = st.columns(len(profile_keys))
