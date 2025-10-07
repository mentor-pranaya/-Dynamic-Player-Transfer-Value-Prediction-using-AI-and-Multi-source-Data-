import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# -----------------------------
# 1️⃣ Load Trained Models (cached for performance)
# -----------------------------
@st.cache_resource
def load_models():
    xgb_model = joblib.load("/Users/aryanupadhyay/Desktop/Intern/Ai project/Mock/xgb_best.save")
    lgb_model = joblib.load("/Users/aryanupadhyay/Desktop/Intern/Ai project/Mock/lgb_best.save")
    lstm_model = tf.keras.models.load_model("/Users/aryanupadhyay/Desktop/Intern/Ai project/Mock/lstm_best.h5", compile=False)
    return xgb_model, lgb_model, lstm_model

xgb_model, lgb_model, lstm_model = load_models()

# -----------------------------
# 2️⃣ Streamlit App UI
# -----------------------------
st.title("⚽ TransferIQ: Dynamic Player Transfer Value Prediction")
st.write("AI-driven prediction using performance, transfer history, and sentiment analysis.")

# Upload dataset
uploaded_file = st.file_uploader("📤 Upload your final feature CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### 📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # Determine which column to use for player identification
    if "player_name" in df.columns:
        player_col = "player_name"
    else:
        player_col = "player_id"
        st.warning("⚠️ 'player_name' column not found — using 'player_id' instead.")

    # Player selection dropdown
    player_names = sorted(df[player_col].unique())
    selected_player = st.selectbox("Select a Player:", player_names)

    # Filter dataset for selected player
    player_df = df[df[player_col] == selected_player].copy()

    if player_df.empty:
        st.error("❌ No data found for selected player.")
    else:
        st.write(f"### Selected Player: {selected_player}")
        st.dataframe(player_df.head())

        # -----------------------------
        # 3️⃣ Feature Selection
        # -----------------------------
        EXCLUDE_COLS = ['player_id', 'player_name', 'date', 'market_value_in_eur_x', 'log_market_value']
        FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

        st.write("Detected Feature Columns:")
        st.write(FEATURE_COLS)

        selected_features = st.multiselect(
            "Select Features for Prediction:",
            FEATURE_COLS,
            default=FEATURE_COLS
        )

        # -----------------------------
        # 4️⃣ Model Selection
        # -----------------------------
        model_choice = st.selectbox("Choose Model:", ["XGBoost", "LightGBM", "LSTM"])

        # -----------------------------
        # 5️⃣ Prediction
        # -----------------------------
        if st.button("🚀 Predict Player Market Value"):
            # Copy selected feature data
            X_df = player_df[selected_features].copy()

            # Encode categorical columns
            for col in X_df.select_dtypes(include=['object']).columns:
                X_df[col] = X_df[col].astype('category').cat.codes

            # Convert to numeric
            X = X_df.apply(pd.to_numeric, errors='coerce').fillna(0).values

            # Perform model-specific prediction
            if model_choice == "XGBoost":
                preds = xgb_model.predict(X)

            elif model_choice == "LightGBM":
                preds = lgb_model.predict(X)

            elif model_choice == "LSTM":
                # LSTM requires 3D input: (samples, timesteps, features)
                X_lstm = np.expand_dims(X, axis=1)
                preds = lstm_model.predict(X_lstm, verbose=0).flatten()

            else:
                st.error("Invalid model choice.")
                preds = None

            if preds is not None:
                # Add predictions to DataFrame
                player_df["Predicted_log_value"] = preds
                player_df["Predicted_Value_EUR"] = np.expm1(preds)

                # -----------------------------
                # 6️⃣ Display Results
                # -----------------------------
                st.success(f"✅ Predictions generated successfully for {selected_player}")
                st.write("### Predicted Market Values (€):")
                st.dataframe(player_df[["date", "Predicted_Value_EUR"]])

                # Plot prediction trend
                st.line_chart(player_df.set_index("date")[["Predicted_Value_EUR"]])

                # Option to download predictions
                csv = player_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"{selected_player}_predicted_values.csv",
                    mime='text/csv'
                )
