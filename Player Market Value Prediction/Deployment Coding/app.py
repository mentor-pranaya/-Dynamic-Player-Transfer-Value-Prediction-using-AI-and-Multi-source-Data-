import streamlit as st
import pandas as pd

# Use the correct dataset
data_path = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\Deployment\Player_Market_Value_Prediction_Dataset.csv"
df = pd.read_csv(data_path)

# Rename column for consistency
if 'Player Name' in df.columns:
    df.rename(columns={'Player Name': 'player'}, inplace=True)

# Streamlit layout
st.set_page_config(
    page_title="TransferIQ: Market Value Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🔍 Player Explorer")
st.sidebar.write("Select a player to view predictions and profile.")
players_list = df['player'].dropna().unique().tolist()
selected_player = st.sidebar.selectbox("🎯 Choose a player:", sorted(players_list))

st.title("⚽ TransferIQ: Player Market Value Dashboard")
st.markdown("Gain insights into player valuation, sentiment, and enriched attributes.")

if selected_player:
    player_row = df[df['player'] == selected_player].iloc[0]
    usd_value = player_row['Market Value (M)']
    exchange_rate = 88.73
    inr_value = usd_value * 1_000_000 * exchange_rate

    st.markdown(f"## 📊 Market Value for **{selected_player}**")
    col1, col2 = st.columns([1, 2])
    col1.metric("💵 USD Value", f"${usd_value:.2f} M")
    col2.metric("🇮🇳 INR Value", f"₹{inr_value:,.0f}")
    st.markdown("---")

    st.markdown("### 🧠 Player Profile")
    profile_fields = ['Age', 'Injury Status', 'Sentiment Label']  # Adjusted to match finally.csv
    profile_cols = st.columns(len(profile_fields))
    for i, field in enumerate(profile_fields):
        if field in player_row:
            profile_cols[i].write(f"**{field}:** {player_row[field]}")

    st.markdown("---")
    st.markdown("### 📈 Prediction Metrics")
    pred_fields = ['y_test', 'lstm_preds', 'ensemble_preds', 'lstm_market_value', 'ensemble_market_value']
    pred_cols = st.columns(2)
    for i, field in enumerate(pred_fields):
        if field in player_row:
            pred_cols[i % 2].write(f"**{field}:** {player_row[field]:.4f}")
    st.markdown("---")