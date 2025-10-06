import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import plotly.graph_objects as go

# Paths
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.join(PROJECT_PATH, "Final_Report")

# Load predictions
preds_path = os.path.join(FINAL_DIR, "Final_Predictions.csv")
df_preds = pd.read_csv(preds_path)

# Load Model
improved_lstm = tf.keras.models.load_model(
    os.path.join(PROJECT_PATH, "Hyperparameter_Tuning", "Improved_Multivariate_lstm_model.h5"),
    compile=False
)

# Encode sentiment into numerical value
def encode_sentiment(sentiment_choice):
    if sentiment_choice == "Good (3–4 sentiments)":
        return 2
    elif sentiment_choice == "Average (1–2 sentiments)":
        return 1
    else:
        return 0

# Streamlit UI
st.set_page_config(page_title=" Player Transfer Value Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.markdown("##  Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to:",
    [" Home", " Final Predictions On Test Split Data", " New Player Prediction"]
)
st.sidebar.markdown("---")
st.sidebar.info(" Developed as part of Infosys Springboard 6.0 Internship Project\n\nAuthor: Aman Singh")

# Home Page
if page == " Home":
    st.title(" Dynamic Player Transfer Value Prediction System")
    st.markdown(
        """
        ###  Project Overview  
        This project predicts **football player transfer values** using:  
        -  Historical performance data  
        -  Injury history  
        -  Sentiment analysis  

        ###  Methodology  
        - Data preprocessing with scaling & log-transform  
        - Improved Multivariate LSTM (final best performing model)  
        - Metrics:  RMSE = 0.6592, MAE = 0.493, R² = 0.3756  

        ###  Deliverables  
        - Final_Predictions.csv with model outputs  
        -  Interactive visualizations  
        - Streamlit app for deployment  

        ###  Impact  
        Enables **clubs, analysts, scouts** to make **data-driven transfer decisions**.
        """
    )

# Predictions Table / Charts
elif page == " Final Predictions On Test Split Data":
    st.title(" Model Predictions & Comparisons")

    option = st.radio("Choose View:", ["Table", "Charts"], horizontal=True)

    if option == "Table":
        st.subheader("Final Predictions (Euro Scale)")
        st.dataframe(df_preds[["True_Euro", "Improved_LSTM_Euro"]])

    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df_preds["True_Euro"], mode="lines+markers",
                                 name="True (€)", line=dict(color="blue")))
        fig.add_trace(go.Scatter(y=df_preds["Improved_LSTM_Euro"], mode="lines+markers",
                                 name="Improved LSTM (€)", line=dict(color="green")))

        fig.update_layout(
            title="Player Transfer Values (Euro Scale)",
            xaxis_title="Sample Index",
            yaxis_title="Market Value (€)",
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# New Player Prediction
elif page == " New Player Prediction":
    st.title(" Predict Transfer Value for a New Player (Improved LSTM)")
    st.markdown("###  Enter Player Details")

    # Feature Inputs
    prev_value = st.number_input("Previous Max Market Value (€)", min_value=0, step=1_000_000)

    total_goals = st.number_input("Total Goals", min_value=0, step=1)
    total_assists = st.number_input("Total Assists", min_value=0, step=1)
    total_minutes_played = st.number_input("Total Minutes Played", min_value=0, step=100)
    matches_played = st.number_input("Matches Played", min_value=0, step=1)
    shots_on_target = st.number_input("Shots on Target", min_value=0, step=1)

    avg_transfer_fee = st.number_input("Average Transfer Fee (€)", min_value=0, step=500_000)
    last_transfer_fee = st.number_input("Last Transfer Fee (€)", min_value=0, step=500_000)
    num_transfers = st.number_input("Number of Transfers", min_value=0, step=1)

    club_prestige = st.slider("Club Prestige Score (1–100)", 1, 100, 50)
    contract_years = st.slider("Contract Years Remaining", 0, 10, 2)

    subreddits_count = st.number_input("Subreddit Mentions", min_value=0, step=10)
    sentiment_choice = st.selectbox(
        "Sentiment",
        ["Good (3–4 sentiments)", "Average (1–2 sentiments)", "Bad (0 sentiments)"]
    )
    sentiment_score = encode_sentiment(sentiment_choice)

    # Prediction
    if st.button("Predict"):
        try:
            timesteps = 4
            features_dim = 13
            features = np.zeros((1, timesteps, features_dim))

            # Repeat values across timesteps
            feature_vector = [
                np.log1p(prev_value), total_goals, total_assists, total_minutes_played,
                matches_played, shots_on_target, np.log1p(avg_transfer_fee),
                np.log1p(last_transfer_fee), num_transfers, club_prestige,
                contract_years, subreddits_count, sentiment_score
            ]

            features[:] = feature_vector  # broadcast same values across all timesteps

            # Predict
            prediction_log = improved_lstm.predict(features, verbose=0).squeeze()
            prediction_euro = np.expm1(prediction_log)

            st.success(f" Predicted Market Value: €{prediction_euro:,.0f}")

            st.metric(label="Previous Value (€)", value=f"€{prev_value:,.0f}")
            st.metric(label="Predicted Value (€)", value=f"€{prediction_euro:,.0f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
