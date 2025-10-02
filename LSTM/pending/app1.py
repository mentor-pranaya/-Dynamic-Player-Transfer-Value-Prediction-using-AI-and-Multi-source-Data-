import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import joblib
from helper import prepare_multivariate_sequences, evaluate_model, get_player_predictions  # put helper function in helper.py


st.title("Player Transfer Value Forecasting")

# Upload your merged features CSV or pull from DB
uploaded = st.file_uploader("Upload player_features + transfer history CSV")
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Loaded data:", df.head())

    features_cols = ["market_value","total_injuries","sentiment_mean","avg_cards_per_match"]
    n_steps = st.slider("Past windows (n_steps):", 1, 10, 3)
    n_future = st.slider("Future steps ahead:", 1, 5, 1)

    if st.button("Prepare Sequences"):
        X, y, player_index, scaler = prepare_multivariate_sequences(df, features_cols, n_steps, n_future)
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.scaler = scaler
        st.session_state.player_index = player_index
        st.success(f"Prepared {X.shape[0]} samples for training")

    if "X" in st.session_state and st.button("Train Model"):
        X = st.session_state.X
        y = st.session_state.y
        player_index = st.session_state.player_index
        scaler = st.session_state.scaler
        
        model = Sequential([
            LSTM(64, input_shape=(X.shape[1], X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        
        # Show loss curve
        st.line_chart(pd.DataFrame({
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"]
        }))
        
        # 🔹 Evaluate
        rmse, mae, per_player = evaluate_model(model, X, y, player_index, scaler)
        st.metric("Overall RMSE", f"{rmse:,.2f}")
        st.metric("Overall MAE", f"{mae:,.2f}")
        st.dataframe(per_player.sort_values("rmse"))
        
        # Save model + scaler
        #model.save("lstm_model.h5")
        #joblib.dump(scaler, "scaler.pkl")
        #st.success("Model and scaler saved")
        st.session_state["model"] = model
    
    # 🔹 Player-level actual vs. predicted plot
    if "model" in st.session_state:
        model = st.session_state["model"]
        X = st.session_state.X
        y = st.session_state.y
        player_index = st.session_state.player_index
        scaler = st.session_state.scaler

        player_ids = sorted(set(player_index))
        chosen_player = st.selectbox("Select a player to view predictions", player_ids)

        if chosen_player:
            df_player = get_player_predictions(model, X, y, player_index, scaler, chosen_player)
            st.line_chart(df_player[["actual","predicted"]])
            st.dataframe(df_player)

    if st.button("Load & Predict"):
        model = load_model("lstm_model.h5")
        scaler = joblib.load("scaler.pkl")
        st.success("Model and scaler loaded")

        pid = st.text_input("Enter transfermarkt_id for prediction", "28396")
        player_data = df[df["transfermarkt_id"]==int(pid)].copy()
        arr = player_data[features_cols].fillna(0).values
        arr_scaled = scaler.transform(arr)

        # last n_steps
        if arr_scaled.shape[0] >= n_steps:
            x_input = arr_scaled[-n_steps:].reshape(1, n_steps, len(features_cols))
            pred_scaled = model.predict(x_input)
            dummy = np.zeros((1,len(features_cols)))
            dummy[0,0]=pred_scaled[0,0]
            pred_value = scaler.inverse_transform(dummy)[0,0]
            st.write(f"Predicted future market value for player {pid}: **{pred_value:,.0f}**")
        else:
            st.warning("Not enough data for this player")


