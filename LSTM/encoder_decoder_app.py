import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
import xgboost as xgb
import matplotlib.pyplot as plt
import mysql.connector
from tensorflow.keras.callbacks import Callback

def make_multivariate_multistep(array, n_steps, n_future=3):
    X, y = [], []
    for i in range(len(array)-n_steps-n_future+1):
        X.append(array[i:i+n_steps, :])
        y.append(array[i+n_steps:i+n_steps+n_future, 0])  # predict market_value
    return np.array(X), np.array(y)

# Build Encoder-Decoder LSTM Model with Streamlit Visualization

def build_encoder_decoder_lstm(n_steps, n_features, n_future):
    # Encoder
    encoder_inputs = Input(shape=(n_steps, n_features))
    encoder_lstm = LSTM(64, activation='tanh', return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = RepeatVector(n_future)(state_h)
    decoder_lstm = LSTM(64, activation='tanh', return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = TimeDistributed(Dense(1))(decoder_outputs)

    model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Streamlit UI
st.title("Training Encoder-Decoder LSTM")
st.write("Forecast multi-step player transfer market values using an Encoder-Decoder LSTM.")
# Parameters
n_steps = st.sidebar.slider("Past windows (n_steps)", 2, 10, 3)
n_future = st.sidebar.slider("Future horizons (n_future)", 1, 5, 3)
epochs = st.sidebar.slider("Epochs", 10, 100, 50)

# DB Connection (⚠️ adapt to your environment)
db = mysql.connector.connect(
    host="localhost", user="root", password="yahoonet", database="AIProject"
)

# Load data
df_t = pd.read_sql(
    "SELECT transfermarkt_id, transfer_date, market_value FROM player_transfer_history ORDER BY transfermarkt_id, transfer_date", db
)
df_f = pd.read_sql("SELECT distinct p.*, t.name as player_name FROM player_features p, players_trfrmrkt t where p.player_id=t.transfermarkt_id", db)

df = df_t.merge(
    df_f[
        [
            "player_id",
            "player_name",
            "total_injuries",
            "sentiment_mean",
            "avg_cards_per_match",
            "avg_days_out",
            "recent_injury",
            "days_since_last_injury",
            "minutes_played",
            "shots_per90",
            "pressures_per90",
        ]
    ],
    left_on="transfermarkt_id",
    right_on="player_id",
    how="left",
)

#df = df.sort_values(["player_name", "transfer_date"])
df = df.sort_values(["transfermarkt_id", "transfer_date"])

# Choose player
#players = df["transfermarkt_id"].unique()
#pid = st.selectbox("Select Player ID", players)

player_options = df.sort_values("player_name").apply(
    lambda row: f"{row['player_name']} ({row['player_id']})", axis=1
).unique().tolist()

# Display in sidebar
player_choice = st.sidebar.selectbox("Select Player", player_options)

# Extract numeric player_id from choice
pid = int(player_choice.split("(")[-1].replace(")", ""))

if st.sidebar.button("Click here to start training and evaluation"):
    # Prepare sequences
    X_list, y_list, player_index = [], [], []
    scaler = MinMaxScaler()

    for p, group in df.groupby("transfermarkt_id"):
        features = group[
            [
                "market_value",
                "total_injuries",
                "sentiment_mean",
                "avg_cards_per_match",
                "avg_days_out",
                "recent_injury",
                "days_since_last_injury",
                "minutes_played",
                "shots_per90",
                "pressures_per90",
            ]
        ].fillna(0).values

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
    
    # Inverse transform helper 
    def inverse_transform_column(scaled_col, scaler, col_index=0, total_features=None):
        scaled_col = np.array(scaled_col).reshape(-1, 1)
        if total_features is None:
            total_features = X_val.shape[2]
        expanded = np.zeros((len(scaled_col), total_features))
        expanded[:, col_index] = scaled_col.flatten()
        return scaler.inverse_transform(expanded)[:, col_index]

    # Prepare LSTM data
    y_train_encdec = y_train[..., np.newaxis]  # (samples, n_future, 1)
    y_val_encdec = y_val[..., np.newaxis]

    # Train Encoder-Decoder LSTM
    n_features = X.shape[2]
    model_encdec = build_encoder_decoder_lstm(n_steps, n_features, n_future)

    st.subheader("Training Encoder-Decoder LSTM")
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

    history = model_encdec.fit(
        X_train,
        y_train_encdec,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val_encdec),
        verbose=0,
        callbacks=[es, StreamlitLogger(epochs)]
    )

    # Predictions on Validation Set
    y_val_pred_encdec = model_encdec.predict(X_val).squeeze(-1)  # (samples, n_future)

    # Inverse transform multi-step predictions
    y_val_true = inverse_transform_column(y_val[:, 0], scaler)
    y_val_pred_encdec_inv = inverse_transform_column(y_val_pred_encdec[:, 0], scaler)

    # Plot Actual vs Predicted (Validation Set)
    fig, ax = plt.subplots(figsize=(12, 6))
    n_show = 50
    ax.plot(y_val_true[:n_show], label="Actual", color="blue", linewidth=2)
    ax.plot(y_val_pred_encdec_inv[:n_show], label="Encoder-Decoder LSTM", color="orange", linestyle="--")
    ax.set_title(f"Validation Set Forecast (Step+1)")
    ax.set_xlabel("Validation Samples")
    ax.set_ylabel("Transfer Value (€ millions)")
    ax.legend()
    st.pyplot(fig)

    # Player-specific 3-step forecast
    st.subheader("Player-Specific Forecast")
    player_id = pid  # from sidebar selection
    st.write(f"### {n_future}-Step Forecast for Player: {player_choice}")

    # Last sequence for this player
    player_data = df[df["player_id"] == player_id].sort_values("transfer_date")
    feature_columns = [
        "market_value", "total_injuries", "sentiment_mean",
        "avg_cards_per_match", "avg_days_out", "recent_injury",
        "days_since_last_injury", "minutes_played", "shots_per90", "pressures_per90"
    ]

    if len(player_data) >= n_steps:
        last_seq = player_data.iloc[-n_steps:][feature_columns].values
        scaled_last_seq = scaler.transform(last_seq)[np.newaxis, :, :]
        
        # Predict next n_future steps
        preds_encdec = model_encdec.predict(scaled_last_seq).squeeze(0)
        preds_encdec_inv = inverse_transform_column(preds_encdec, scaler)

        actual_last = player_data["market_value"].values[-n_future:]

        fig, ax = plt.subplots(figsize=(10, 5))
        steps = range(1, n_future+1)
        ax.plot(steps, actual_last, marker='o', label="Actual", color="blue")
        ax.plot(steps, preds_encdec_inv[:n_future], marker='o', label="Encoder-Decoder LSTM", color="orange")
        ax.set_title(f"{n_future}-Step Forecast for Player {player_choice}")
        ax.set_xlabel("Steps Ahead")
        ax.set_ylabel("Transfer Value (€ millions)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Not enough data for this player to generate a multi-step forecast.")


db.close()
