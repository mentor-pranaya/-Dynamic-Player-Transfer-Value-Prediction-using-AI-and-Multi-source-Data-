import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback
import xgboost as xgb
import matplotlib.pyplot as plt
import mysql.connector

# ---------- Helper Functions ----------

def make_multivariate_multistep(array, n_steps, n_future=3):
    X, y = [], []
    for i in range(len(array)-n_steps-n_future+1):
        X.append(array[i:i+n_steps, :])
        y.append(array[i+n_steps:i+n_steps+n_future, 0])  # predict market_value
    return np.array(X), np.array(y)

def inverse_transform_col_series(col_scaled, scaler, n_features_total):
    col_scaled = np.array(col_scaled).reshape(-1, 1)
    expanded = np.zeros((len(col_scaled), n_features_total))
    expanded[:, 0] = col_scaled.flatten()
    return scaler.inverse_transform(expanded)[:, 0]

# ---------- Build Seq2Seq Training Model ----------

def build_seq2seq_train(n_steps, n_features, n_future, latent_dim=64):
    # Encoder
    encoder_inputs = Input(shape=(n_steps, n_features), name="encoder_inputs")
    encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(encoder_inputs)

    # Decoder
    decoder_inputs = Input(shape=(n_future, 1), name="decoder_inputs")
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    decoder_dense = Dense(1, name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], [decoder_outputs, state_h, state_c])
    model.compile(optimizer="adam", loss="mse")
    return model

# ---------- Build Seq2Seq Inference Models ----------
def build_seq2seq_inference_models(train_model, n_steps, n_features, latent_dim=64):
    # Encoder Model
    encoder_inputs = Input(shape=(n_steps, n_features), name="encoder_inputs_inference")
    encoder_lstm = train_model.get_layer("encoder_lstm")
    _, state_h_enc, state_c_enc = encoder_lstm(encoder_inputs)
    encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

    # Decoder Model
    decoder_inputs = Input(shape=(1, 1), name="decoder_inputs_inference")
    decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = train_model.get_layer("decoder_lstm")
    decoder_dense = train_model.get_layer("decoder_dense")

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs, state_h, state_c]
    )

    return encoder_model, decoder_model

# ---------- Recursive Seq2Seq Prediction ----------
def predict_seq2seq_recursive(input_seq_scaled, encoder_model, decoder_model, n_future):
    state_h, state_c = encoder_model.predict(input_seq_scaled)
    start_val = input_seq_scaled[0, -1, 0]
    decoder_input = np.array([[[start_val]]])
    preds = []
    for _ in range(n_future):
        out, state_h, state_c = decoder_model.predict([decoder_input, state_h, state_c])
        pred_val = out[0, 0, 0]
        preds.append(pred_val)
        decoder_input = np.array([[[pred_val]]])
    return np.array(preds)

# ---------- Streamlit UI ----------
st.title("Training Encoder-Decoder LSTM")
st.write("Forecast multi-step player transfer market values using an Encoder-Decoder LSTM.")

n_steps = st.sidebar.slider("Past windows (n_steps)", 2, 10, 3)
n_future = st.sidebar.slider("Future horizons (n_future)", 1, 5, 3)
epochs = st.sidebar.slider("Epochs", 10, 100, 50)

# DB Connection
db = mysql.connector.connect(
    host="localhost", user="root", password="yahoonet", database="AIProject"
)

df_t = pd.read_sql(
    "SELECT transfermarkt_id, transfer_date, market_value FROM player_transfer_history ORDER BY transfermarkt_id, transfer_date", db
)
df_f = pd.read_sql(
    "SELECT distinct p.*, t.name as player_name FROM player_features p, players_trfrmrkt t where p.player_id=t.transfermarkt_id", db
)

df = df_t.merge(
    df_f[
        [
            "player_id", "player_name", "total_injuries", "sentiment_mean",
            "avg_cards_per_match", "avg_days_out", "recent_injury",
            "days_since_last_injury", "minutes_played", "shots_per90", "pressures_per90"
        ]
    ],
    left_on="transfermarkt_id", right_on="player_id", how="left"
)

df = df.sort_values(["transfermarkt_id", "transfer_date"])

player_options = df.sort_values("player_name").apply(
    lambda row: f"{row['player_name']} ({row['player_id']})", axis=1
).unique().tolist()
player_choice = st.sidebar.selectbox("Select Player", player_options)
pid = int(player_choice.split("(")[-1].replace(")", ""))

if st.sidebar.button("Click here to start training and evaluation"):

    # Prepare sequences
    X_list, y_list, player_index = [], [], []
    scaler = MinMaxScaler()
    feature_cols = [
        "market_value", "total_injuries", "sentiment_mean", "avg_cards_per_match",
        "avg_days_out", "recent_injury", "days_since_last_injury",
        "minutes_played", "shots_per90", "pressures_per90"
    ]

    for p, group in df.groupby("transfermarkt_id"):
        features = group[feature_cols].fillna(0).values
        scaled = scaler.fit_transform(features)
        Xp, yp = make_multivariate_multistep(scaled, n_steps, n_future)
        if len(Xp) == 0:
            continue
        X_list.append(Xp)
        y_list.append(yp)
        player_index.extend([p] * len(yp))

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    player_index = np.array(player_index)

    # Train/Validation split
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X, y, player_index, test_size=0.2, shuffle=True, random_state=42
    )

    # Decoder inputs for teacher forcing
    decoder_input_train = np.zeros((y_train.shape[0], n_future, 1))
    decoder_input_train[:, 1:, 0] = y_train[:, :-1]
    decoder_input_val = np.zeros((y_val.shape[0], n_future, 1))
    decoder_input_val[:, 1:, 0] = y_val[:, :-1]

    # Build and train Seq2Seq
    latent_dim = 64
    seq2seq = build_seq2seq_train(n_steps, X.shape[2], n_future, latent_dim=latent_dim)

    st.subheader("Training Seq2Seq Encoder-Decoder LSTM (with teacher forcing)")
    progress_bar = st.progress(0)
    epoch_log = st.empty()
    chart_placeholder = st.empty()
    loss_chart = chart_placeholder.line_chart({"Train Loss": [], "Val Loss": []})

    class StreamlitLogger(Callback):
        def __init__(self, total_epochs):
            super().__init__()
            self.total_epochs = total_epochs
            self.history = {"loss": [], "val_loss": []}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.history["loss"].append(logs.get("loss"))
            self.history["val_loss"].append(logs.get("val_loss"))
            epoch_log.text(
                f"Epoch {epoch+1}/{self.total_epochs} - "
                f"Loss: {logs.get('loss'):.4f}, "
                f"Val Loss: {logs.get('val_loss'):.4f}"
            )
            progress_bar.progress((epoch+1)/self.total_epochs)
            new_data = {
                "Train Loss": self.history["loss"][-1:],
                "Val Loss": self.history["val_loss"][-1:]
            }
            loss_chart.add_rows(new_data)

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # Dummy outputs for state placeholders
    dummy_train = np.zeros((len(y_train), latent_dim))
    dummy_val = np.zeros((len(y_val), latent_dim))

    history = seq2seq.fit(
        [X_train, decoder_input_train],
        [y_train[..., np.newaxis], dummy_train, dummy_train],
        epochs=epochs,
        batch_size=32,
        validation_data=([X_val, decoder_input_val], [y_val[..., np.newaxis], dummy_val, dummy_val]),
        verbose=0,
        callbacks=[es, StreamlitLogger(epochs)]
    )

    # ---------- Build inference models ----------
    encoder_model, decoder_model = build_seq2seq_inference_models(
        seq2seq, n_steps=X.shape[1], n_features=X.shape[2], latent_dim=latent_dim
    )

    st.success("Training completed!")


    # ---------- XGBoost ensemble ----------

    y_train_seq_pred = seq2seq.predict([X_train, decoder_input_train])[0].squeeze(-1)
    y_val_seq_pred = seq2seq.predict([X_val, decoder_input_val])[0].squeeze(-1)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    xgb_models = []
    y_val_pred_xgb_all = np.zeros_like(y_val_seq_pred)

    for step in range(n_future):
        X_meta_train = np.hstack([X_train_flat, y_train_seq_pred[:, step].reshape(-1,1)])
        X_meta_val = np.hstack([X_val_flat, y_val_seq_pred[:, step].reshape(-1,1)])
        model_xgb = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42)
        model_xgb.fit(X_meta_train, y_train[:, step])
        xgb_models.append(model_xgb)
        y_val_pred_xgb_all[:, step] = model_xgb.predict(X_meta_val)
        # Ensemble predictions
        y_val_ensemble = (y_val_seq_pred + y_val_pred_xgb_all) / 2
        st.subheader("Validation Results Comparison")
         # Validation plotting ----------
        def inverse_transform_col(col_scaled, scaler, n_features_total):
            col_scaled = np.array(col_scaled).reshape(-1, 1)
            expanded = np.zeros((len(col_scaled), n_features_total))
            expanded[:, 0] = col_scaled.flatten()
            return scaler.inverse_transform(expanded)[:, 0]

        y_val_true_inv = inverse_transform_col(y_val[:, 0], scaler, X.shape[2])
        y_val_seq_lstm_inv = inverse_transform_col(y_val_seq_pred[:, 0], scaler, X.shape[2])
        y_val_xgb_inv = inverse_transform_col(y_val_pred_xgb_all[:, 0], scaler, X.shape[2])
        y_val_ensemble_inv = inverse_transform_col(y_val_ensemble[:, 0], scaler, X.shape[2])

        fig, ax = plt.subplots(figsize=(12,6))
        n_show = min(50, len(y_val_true_inv))
        ax.plot(y_val_true_inv[:n_show], label="Actual", color="blue")
        ax.plot(y_val_seq_lstm_inv[:n_show], label="Seq2Seq LSTM", color="orange", linestyle="--")
        ax.plot(y_val_xgb_inv[:n_show], label="XGBoost", color="green", linestyle="--")
        ax.plot(y_val_ensemble_inv[:n_show], label="Ensemble (avg)", color="red", linestyle=":")
        ax.set_title(f"Validation Forecast (Step+{step+1}) - Seq2Seq + XGBoost Ensemble")
        ax.set_xlabel("Validation Samples")
        ax.set_ylabel("Transfer Value (€ millions)")
        ax.legend()
        st.pyplot(fig)
        st.write("Validation forecast (first step) comparison for Seq2Seq LSTM, XGBoost, and Ensemble.")

    # ---------- Player-specific recursive forecast ----------
    st.write(f"### {n_future}-Step Forecast for Player: {player_choice}")
    player_data = df[df["player_id"] == pid].sort_values("transfer_date")
    if len(player_data) >= n_steps:
        last_seq = player_data.iloc[-n_steps:][feature_cols].fillna(0).values
        scaled_last_seq = scaler.transform(last_seq)[np.newaxis, :, :]
        preds_seq_scaled = predict_seq2seq_recursive(scaled_last_seq, encoder_model, decoder_model, n_future)
        preds_seq_inv = inverse_transform_col_series(preds_seq_scaled, scaler, X.shape[2])

        X_seq_flat = scaled_last_seq.reshape(1, -1)
        preds_xgb = []
        for step in range(n_future):
            meta_input = np.hstack([X_seq_flat, np.array(preds_seq_scaled[step]).reshape(1,1)])
            preds_xgb.append(xgb_models[step].predict(meta_input)[0])
        preds_xgb_inv = inverse_transform_col_series(np.array(preds_xgb), scaler, X.shape[2])
        preds_ensemble_inv = (preds_seq_inv + preds_xgb_inv)/2

        # Actual last values for comparison
        actual_last = player_data["market_value"].values[-n_future:] if len(player_data) >= n_future else player_data["market_value"].values

        # Plot
        fig, ax = plt.subplots(figsize=(10,5))
        steps = np.arange(1, n_future+1)
        ax.plot(steps[:len(actual_last)], actual_last, marker='o', label="Actual", color="blue")
        ax.plot(steps, preds_seq_inv, marker='o', label="Seq2Seq LSTM", color="orange")
        ax.plot(steps, preds_xgb_inv, marker='o', label="XGBoost refined", color="green")
        ax.plot(steps, preds_ensemble_inv, marker='o', label="Ensemble (avg)", color="red", linestyle=":")
        ax.set_xlabel("Steps Ahead")
        ax.set_ylabel("Transfer Value (€ millions)")
        ax.set_title(f"{n_future}-Step Forecast for Player {player_choice}")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Not enough data for recursive forecasting for this player.")

db.close()
