import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping


# -------------------------------------------------------------------
# 🔧 Helper Functions
# -------------------------------------------------------------------
def create_multistep_sequences(data, past_window, future_horizon=3):
    """Convert timeseries into past/future sequences for supervised learning."""
    X, y = [], []
    for i in range(len(data) - past_window - future_horizon + 1):
        X.append(data[i:i + past_window, :])
        y.append(data[i + past_window:i + past_window + future_horizon, 0])
    return np.array(X), np.array(y)


def init_lstm(past_window, feature_count, future_horizon):
    """Build a simple LSTM network."""
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(past_window, feature_count)),
        Dropout(0.25),
        Dense(future_horizon)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# -------------------------------------------------------------------
# 🧭 Streamlit Interface
# -------------------------------------------------------------------
st.title("⚽ TransferIQ: LSTM + XGBoost Market Value Forecasting")
st.caption("Developed by Abhinav | System: C:/Users/Abhinav/TransferIQ/")
st.write("This dashboard allows you to train **LSTM** and **hybrid ensemble models** "
         "to predict player transfer market values.")

# Sidebar controls
st.sidebar.header("Model Parameters")
n_steps = st.sidebar.slider("Past Window (n_steps)", 2, 12, 4)
n_future = st.sidebar.slider("Forecast Horizon (n_future)", 1, 5, 3)
epochs = st.sidebar.slider("Epochs", 20, 150, 60)

# -------------------------------------------------------------------
# 🗃 Database Connection (adjust for your environment)
# -------------------------------------------------------------------
st.info("Connecting to local MySQL database...")
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yahoonet",
    database="AIProject"
)
st.success("✅ Database connection established!")

# -------------------------------------------------------------------
# 🧩 Data Loading
# -------------------------------------------------------------------
query_transfers = """
SELECT transfermarkt_id, transfer_date, market_value
FROM player_transfer_history
ORDER BY transfermarkt_id, transfer_date
"""
query_features = """
SELECT DISTINCT p.*, t.name AS player_name
FROM player_features p
JOIN players_trfrmrkt t ON p.player_id = t.transfermarkt_id
"""
df_transfers = pd.read_sql(query_transfers, db)
df_features = pd.read_sql(query_features, db)

# Merge both dataframes
merged = df_transfers.merge(
    df_features[
        [
            "player_id", "player_name", "total_injuries", "sentiment_mean",
            "avg_cards_per_match", "avg_days_out", "recent_injury",
            "days_since_last_injury", "minutes_played",
            "shots_per90", "pressures_per90"
        ]
    ],
    left_on="transfermarkt_id",
    right_on="player_id",
    how="left"
).sort_values(["transfermarkt_id", "transfer_date"])

# -------------------------------------------------------------------
# 👤 Player Selection
# -------------------------------------------------------------------
players_display = merged.sort_values("player_name").apply(
    lambda x: f"{x['player_name']} (ID: {x['player_id']})", axis=1
).unique().tolist()

selected_player = st.sidebar.selectbox("Select Player", players_display)
selected_pid = int(selected_player.split("ID:")[-1].replace(")", "").strip())

# -------------------------------------------------------------------
# 🚀 Training Trigger
# -------------------------------------------------------------------
if st.sidebar.button("🚀 Run Training and Evaluation"):
    scaler = MinMaxScaler()
    X_all, y_all, ids = [], [], []

    # Sequence creation for each player
    for pid, grp in merged.groupby("transfermarkt_id"):
        subset = grp[
            [
                "market_value", "total_injuries", "sentiment_mean",
                "avg_cards_per_match", "avg_days_out", "recent_injury",
                "days_since_last_injury", "minutes_played", "shots_per90",
                "pressures_per90"
            ]
        ].fillna(0).values

        scaled = scaler.fit_transform(subset)
        X_seq, y_seq = create_multistep_sequences(scaled, n_steps, n_future)
        if len(X_seq) == 0:
            continue
        X_all.append(X_seq)
        y_all.append(y_seq)
        ids.extend([pid] * len(y_seq))

    X = np.vstack(X_all)
    y = np.vstack(y_all)
    ids = np.array(ids)

    # Train/Test split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42, shuffle=True
    )

    # -------------------------------------------------------------------
    # 🧠 Train LSTM Model
    # -------------------------------------------------------------------
    st.subheader("🔹 LSTM Model Training")
    model = init_lstm(n_steps, X.shape[2], n_future)

    progress = st.progress(0)
    log_box = st.empty()
    loss_chart = st.line_chart({"Training Loss": [], "Validation Loss": []})

    class StreamlitMonitor(Callback):
        def __init__(self, total_epochs):
            super().__init__()
            self.total_epochs = total_epochs
            self.losses = {"loss": [], "val_loss": []}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.losses["loss"].append(logs.get("loss"))
            self.losses["val_loss"].append(logs.get("val_loss"))
            log_box.text(f"Epoch {epoch+1}/{self.total_epochs} | "
                         f"Loss: {logs.get('loss'):.4f} | Val: {logs.get('val_loss'):.4f}")
            progress.progress((epoch + 1) / self.total_epochs)
            loss_chart.add_rows({"Training Loss": [logs.get("loss")],
                                 "Validation Loss": [logs.get("val_loss")]})

    early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[early_stop, StreamlitMonitor(epochs)]
    )

    # Plot training curves
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="Train Loss", color="orange")
    ax.plot(history.history["val_loss"], label="Val Loss", color="blue")
    ax.legend()
    ax.set_title("LSTM Loss Over Epochs")
    st.pyplot(fig)

    # -------------------------------------------------------------------
    # 🧩 Ensemble Training (LSTM + XGBoost)
    # -------------------------------------------------------------------
    st.subheader("🔸 Ensemble Model (LSTM + XGBoost)")
    y_pred_train_lstm = model.predict(X_train)
    y_pred_test_lstm = model.predict(X_test)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    meta_train = np.hstack([X_train_flat, y_pred_train_lstm])
    meta_test = np.hstack([X_test_flat, y_pred_test_lstm])

    rmse_results = []
    for step in range(n_future):
        y_train_step = y_train[:, step]
        y_test_step = y_test[:, step]

        xgb_model = xgb.XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(meta_train, y_train_step)

        preds_xgb = xgb_model.predict(meta_test)
        rmse_lstm = np.sqrt(mean_squared_error(y_test_step, y_pred_test_lstm[:, step]))
        rmse_xgb = np.sqrt(mean_squared_error(y_test_step, preds_xgb))
        rmse_results.append((rmse_lstm, rmse_xgb))

        # Plot for each step
        fig, ax = plt.subplots()
        ax.plot(y_test_step[:40], label="True", color="black")
        ax.plot(y_pred_test_lstm[:40, step], label="LSTM", alpha=0.7)
        ax.plot(preds_xgb[:40], label="Ensemble", alpha=0.7)
        ax.set_title(f"Forecast Horizon +{step+1}")
        ax.legend()
        st.pyplot(fig)

    # RMSE summary table
    rmse_table = pd.DataFrame(rmse_results, columns=["LSTM_RMSE", "Ensemble_RMSE"])
    rmse_table.index = [f"Step+{i+1}" for i in range(n_future)]
    st.dataframe(rmse_table.style.highlight_min(color='lightgreen', axis=1))

    # -------------------------------------------------------------------
    # 🎯 Final Evaluation Metrics
    # -------------------------------------------------------------------
    st.subheader("Model Evaluation Metrics")
    y_test_true = scaler.inverse_transform(np.hstack([
        y_test[:, [0]], np.zeros((y_test.shape[0], X_test.shape[2] - 1))
    ]))[:, 0]
    y_test_pred = scaler.inverse_transform(np.hstack([
        y_pred_test_lstm[:, [0]], np.zeros((y_pred_test_lstm.shape[0], X_test.shape[2] - 1))
    ]))[:, 0]

    rmse_final = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mae_final = mean_absolute_error(y_test_true, y_test_pred)

    col1, col2 = st.columns(2)
    col1.metric("RMSE (LSTM)", f"{rmse_final:.3f}")
    col2.metric("MAE (LSTM)", f"{mae_final:.3f}")

    # Plot final comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test_true[:50], label="Actual", color="navy")
    ax.plot(y_test_pred[:50], label="Predicted", linestyle="--", color="orange")
    ax.set_title("Actual vs Predicted Market Values")
    ax.legend()
    st.pyplot(fig)

db.close()
