import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# --- Load Model and Preprocessor ---
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load("xgb_model_final.joblib")
    preprocessor = joblib.load("preprocessor_final.joblib")
    return model, preprocessor

xgb_model, preprocessor = load_model_and_preprocessor()

# --- Load Player Data ---
@st.cache_data
def load_player_data():
    df = pd.read_csv("master_list.csv")
    df.columns = df.columns.str.strip()
    return df

player_df = load_player_data()

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="⚽ AI Player Market Value Predictor",
    page_icon="⚽",
    layout="centered"
)

st.markdown("""
<style>
    .main {background-color: #f7f9fb;}
    .stApp {background-color: #f7f9fb;}
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-top: 15px;
    }
    h1, h2, h3 { color: #003399; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ AI Football Player Market Value Predictor")
st.markdown("### Predict a player's **current transfer market value** using AI and historical data!")

# --- Player Selection ---
player_names = sorted(player_df['player'].dropna().unique())
selected_player = st.selectbox("🔍 Select a player:", options=player_names)

if selected_player:
    player_row = player_df[player_df['player'] == selected_player]

    if player_row.empty:
        st.warning("No data found for this player.")
    else:
        player_info = player_row.iloc[0]

        # --- Player Header ---
        st.markdown("## 🧑 Player Profile")
        image_url = player_info.get('image_url', None)
        col1, col2 = st.columns([1, 2])

        with col1:
            if isinstance(image_url, str) and image_url.startswith("http"):
                st.image(image_url, width=220, caption=player_info['player'])
            else:
                st.image("https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg", width=220)

        with col2:
            st.markdown(f"### **{player_info['player']}**")
            st.write(f"🏟️ Club: **{player_info['current_club_name']}**")
            st.write(f"🧩 Position: **{player_info['sub_position']}**")
            
            # Correctly display age from original CSV
            try:
                age_value = float(player_info['age'])
                st.write(f"🎂 Age: **{age_value:.0f}**")
            except:
                st.write("🎂 Age: **Unknown**")

            st.write(f"🌍 Nationality: **{player_info['country_of_citizenship']}**")

        st.markdown("---")

        # --- Prediction Section ---
        with st.spinner("⚙️ Predicting player market value..."):
            X_input = player_row.drop(columns=['market_value_in_euros'])
            try:
                X_processed = preprocessor.transform(X_input)
                predicted_value = xgb_model.predict(X_processed)[0]

                # Display results in cards
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("💰 Predicted Market Value")
                    st.markdown(f"<h2 style='color:#007500;'>€{predicted_value:,.2f}</h2>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col4:
                    try:
                        actual_value = float(player_info['market_value_in_euros'])
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.subheader("📈 Actual Market Value")
                        st.markdown(f"<h2 style='color:#004aad;'>€{actual_value:,.2f}</h2>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    except:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.subheader("📈 Actual Market Value")
                        st.markdown("Unknown", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                # Comparison
                if 'actual_value' in locals():
                    diff = predicted_value - actual_value
                    if diff > 0:
                        st.success(f"✅ Player is predicted to **increase** in value by €{abs(diff):,.2f}")
                    elif diff < 0:
                        st.error(f"📉 Player is predicted to **decrease** in value by €{abs(diff):,.2f}")
                    else:
                        st.info("⚖️ Predicted value matches the current market value!")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

        # --- Value Over Time Chart ---
        if 'time_step' in player_df.columns and 'score' in player_df.columns:
            player_timeseries = player_row[['time_step', 'score']].dropna()
            if not player_timeseries.empty:
                st.markdown("### 📊 Market Value Over Time")
                chart = (
                    alt.Chart(player_timeseries)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("time_step:Q", title="Time Step"),
                        y=alt.Y("score:Q", title="Score / Value Index"),
                        tooltip=["time_step", "score"]
                    )
                    .properties(width=700, height=300)
                )
                st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.caption("✨ Developed by [punyamurthy] — AI Sports Analytics Project (2025)")
