import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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

def build_lstm(n_steps, n_features, n_future):
    model = Sequential()
    model.add(LSTM(64, activation="tanh", input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(n_future))
    model.compile(optimizer="adam", loss="mse")
    return model

# --------------------------
# Streamlit UI
# --------------------------
st.title("Market Value Forecasting: LSTM vs Ensemble")
st.write("Compare **LSTM** and **Ensemble (LSTM + XGBoost)** for player transfer market values.")

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

    # LSTM Training
    st.subheader("🔹 Training LSTM")
    n_features = X.shape[2]
    model = build_lstm(n_steps, n_features, n_future)
    
    progress_bar = st.progress(0)
    epoch_log = st.empty()
    chart_placeholder = st.empty()
    # Initialize line chart
    loss_chart = chart_placeholder.line_chart({"Train Loss": [], "Val Loss": []})

    class StreamlitLogger(Callback):
        def __init__(self, total_epochs):
            super().__init__()
            self.total_epochs = total_epochs
            self.history = {"loss": [], "val_loss": []}
            
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            #loss = logs.get("loss")
            #val_loss = logs.get("val_loss")
            self.history["loss"].append(logs.get("loss"))
            self.history["val_loss"].append(logs.get("val_loss"))
            #epoch_log.text(f"Epoch {epoch+1}/{self.total_epochs} - Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
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

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0,
        callbacks=[es, StreamlitLogger(epochs)],
    )
    # Final static chart (for reference after training finishes)
    st.line_chart(history.history)

    # Plot loss curve
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.legend()
    ax.set_title("Loss Curves")
    st.pyplot(fig)

    # Predictions
    y_val_pred_lstm = model.predict(X_val)

    # Ensemble with XGBoost
    st.subheader("🔹 Ensemble: LSTM + XGBoost")

    # Flatten sequences for meta-features
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    y_train_pred_lstm = model.predict(X_train)

    train_meta = np.hstack([X_train_flat, y_train_pred_lstm])
    val_meta = np.hstack([X_val_flat, y_val_pred_lstm])

    rmses = []
    for step in range(n_future):
        y_train_step = y_train[:, step]
        y_val_step = y_val[:, step]

        model_xgb = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model_xgb.fit(train_meta, y_train_step)

        y_val_pred_xgb = model_xgb.predict(val_meta)

        rmse_lstm = np.sqrt(mean_squared_error(y_val_step, y_val_pred_lstm[:, step]))
        rmse_xgb = np.sqrt(mean_squared_error(y_val_step, y_val_pred_xgb))

        rmses.append((rmse_lstm, rmse_xgb))

        fig, ax = plt.subplots()
        ax.plot(y_val_step[:50], label="True", color="black")
        ax.plot(y_val_pred_lstm[:50, step], label="LSTM", alpha=0.7)
        ax.plot(y_val_pred_xgb[:50], label="Ensemble", alpha=0.7)
        ax.set_title(f"Step+{step+1} Forecast")
        ax.legend()
        st.pyplot(fig)

    # RMSE Summary
    rmse_df = pd.DataFrame(rmses, columns=["RMSE_LSTM", "RMSE_Ensemble"])
    rmse_df.index = [f"Step+{i+1}" for i in range(n_future)]
    st.write("### RMSE Comparison")
    st.dataframe(rmse_df)
    
    # Actual vs Predicted Comparison (LSTM, XGBoost, Ensemble)
    st.subheader("Comparison (LSTM, XGBoost, Ensemble)")

    # Inverse transform helper 
    # inverse transform only market_value
    def inverse_transform_column(scaled_col, scaler, col_index=0, total_features=None):
        """Inverse transform a single column from scaled data."""
        scaled_col = np.array(scaled_col).reshape(-1, 1)
        if total_features is None:
            total_features = X_val.shape[2]
        expanded = np.zeros((len(scaled_col), total_features))
        expanded[:, col_index] = scaled_col.flatten()
        return scaler.inverse_transform(expanded)[:, col_index]

    # Inverse transform validation set predictions
    
    # 1. LSTM predictions
    y_val_pred_lstm_inv = np.zeros_like(y_val_pred_lstm)
    for step in range(n_future):
        y_val_pred_lstm_inv[:, step] = inverse_transform_column(y_val_pred_lstm[:, step], scaler)

    # 2. XGBoost predictions (per step)
    y_val_pred_xgb_list = []
    for step in range(n_future):
        model_xgb = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model_xgb.fit(train_meta, y_train[:, step])
        y_val_pred_step = model_xgb.predict(val_meta)
        y_val_pred_xgb_list.append(y_val_pred_step)

    # Stack XGBoost predictions
    y_val_pred_xgb = np.column_stack(y_val_pred_xgb_list)

    # Inverse transform
    y_val_pred_xgb_inv = np.zeros_like(y_val_pred_xgb)
    for step in range(n_future):
        y_val_pred_xgb_inv[:, step] = inverse_transform_column(y_val_pred_xgb[:, step], scaler)

    # Ensemble predictions
    y_val_pred_ensemble_inv = (y_val_pred_lstm_inv + y_val_pred_xgb_inv) / 2

    # Actual market values
    y_val_true_inv = np.zeros_like(y_val)
    for step in range(n_future):
        y_val_true_inv[:, step] = inverse_transform_column(y_val[:, step], scaler)

    # Plot Actual vs Predicted for first 50 samples
    st.subheader("Actual vs Predicted Transfer Values (Validation Set)")

    fig, ax = plt.subplots(figsize=(12, 6))
    n_show = min(50, len(y_val_true_inv))  # first 50 samples

    for step in range(n_future):
        ax.plot(y_val_true_inv[:n_show, step], label=f"Actual Step+{step+1}", linestyle="-", linewidth=2)
        ax.plot(y_val_pred_lstm_inv[:n_show, step], label=f"LSTM Step+{step+1}", linestyle="--")
        ax.plot(y_val_pred_xgb_inv[:n_show, step], label=f"XGBoost Step+{step+1}", linestyle=":")
        ax.plot(y_val_pred_ensemble_inv[:n_show, step], label=f"Ensemble Step+{step+1}", linestyle="-.")

    ax.set_title("Actual vs Predicted Transfer Values")
    ax.set_xlabel("Validation Samples")
    ax.set_ylabel("Market Value (€ millions)")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    # Error metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse_lstm = np.sqrt(mean_squared_error(y_val_true_inv, y_val_pred_lstm_inv))
    mae_lstm = mean_absolute_error(y_val_true_inv, y_val_pred_lstm_inv)

    rmse_xgb = np.sqrt(mean_squared_error(y_val_true_inv, y_val_pred_xgb_inv))
    mae_xgb = mean_absolute_error(y_val_true_inv, y_val_pred_xgb_inv)

    rmse_ens = np.sqrt(mean_squared_error(y_val_true_inv, y_val_pred_ensemble_inv))
    mae_ens = mean_absolute_error(y_val_true_inv, y_val_pred_ensemble_inv)

    col1, col2, col3 = st.columns(3)
    col1.metric("LSTM RMSE", f"{rmse_lstm:.2f}")
    col1.metric("LSTM MAE", f"{mae_lstm:.2f}")
    col2.metric("XGBoost RMSE", f"{rmse_xgb:.2f}")
    col2.metric("XGBoost MAE", f"{mae_xgb:.2f}")
    col3.metric("Ensemble RMSE", f"{rmse_ens:.2f}")
    col3.metric("Ensemble MAE", f"{mae_ens:.2f}")

    # Player-specific 3-step forecast
    st.subheader("Player-Specific Forecast")
    player_id = pid  # from sidebar selection
    st.write(f"### 3-Step Forecast for Player: {player_choice}")

    # Last sequence for this player
    player_data = df[df["player_id"] == player_id].sort_values("transfer_date")
    feature_columns = [
        "market_value", "total_injuries", "sentiment_mean",
        "avg_cards_per_match", "avg_days_out", "recent_injury",
        "days_since_last_injury", "minutes_played", "shots_per90", "pressures_per90"
    ]

    seq_length = n_steps
    
    if len(player_data) >= seq_length:
        # Last sequence for the player
        last_seq = player_data.iloc[-seq_length:][feature_columns].values
        scaled_last_seq = scaler.transform(last_seq)
        scaled_last_seq = np.expand_dims(scaled_last_seq, axis=0)

        # ---------- LSTM rolling prediction ----------
        preds_lstm = []
        lstm_input = scaled_last_seq.copy()
        for _ in range(n_future):
            p = model.predict(lstm_input, verbose=0)[0, 0]
            preds_lstm.append(p)
            # Roll the input sequence and append predicted market value
            lstm_input = np.roll(lstm_input, -1, axis=1)
            lstm_input[0, -1, 0] = p  # update market_value

        # ---------- XGBoost sequential prediction ----------
        # Flatten last sequence
        X_seq_flat = scaled_last_seq.reshape(1, -1)
        preds_xgb = []
        xgb_meta_preds = []
        n_xgb_features = train_meta.shape[1]  # expected features

        for step in range(n_future):
            if len(xgb_meta_preds) == 0:
                meta_input = X_seq_flat
                # Pad if needed
                if meta_input.shape[1] < n_xgb_features:
                    meta_input = np.hstack([meta_input, np.zeros((1, n_xgb_features - meta_input.shape[1]))])
            else:
                meta_pred_array = np.array(xgb_meta_preds).reshape(1, -1)
                # Pad if needed
                n_missing = n_xgb_features - X_seq_flat.shape[1] - meta_pred_array.shape[1]
                if n_missing > 0:
                    meta_pred_array = np.hstack([meta_pred_array, np.zeros((1, n_missing))])
                meta_input = np.hstack([X_seq_flat, meta_pred_array])

            p = model_xgb.predict(meta_input)[0]
            preds_xgb.append(p)
            xgb_meta_preds.append(p)

        # ---------- Inverse transform to actual € ----------
        def inverse_transform_column(scaled_col, scaler, col_index=0, total_features=None):
            scaled_col = np.array(scaled_col).reshape(-1, 1)
            if total_features is None:
                total_features = X.shape[2]
            expanded = np.zeros((len(scaled_col), total_features))
            expanded[:, col_index] = scaled_col.flatten()
            return scaler.inverse_transform(expanded)[:, col_index]

        preds_lstm_inv = inverse_transform_column(np.array(preds_lstm), scaler)
        preds_xgb_inv = inverse_transform_column(np.array(preds_xgb), scaler)
        preds_ensemble_inv = (preds_lstm_inv + preds_xgb_inv) / 2

        # Get last actual values
        actual_last = player_data["market_value"].values[-n_future:]

        # ---------- Plot ----------
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, n_future+1), actual_last, marker="o", label="Actual", color="blue")
        ax.plot(range(1, n_future+1), preds_lstm_inv, marker="o", label="LSTM Predicted", color="orange")
        ax.plot(range(1, n_future+1), preds_xgb_inv, marker="o", label="XGBoost Predicted", color="green")
        ax.plot(range(1, n_future+1), preds_ensemble_inv, marker="o", label="Ensemble", color="red", linestyle="--")

        ax.set_title(f"{n_future}-Step Forecast for Player {player_choice}")
        ax.set_xlabel("Steps Ahead")
        ax.set_ylabel("Transfer Value (€ millions)")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("Not enough data for this player to generate a forecast.")

db.close()
