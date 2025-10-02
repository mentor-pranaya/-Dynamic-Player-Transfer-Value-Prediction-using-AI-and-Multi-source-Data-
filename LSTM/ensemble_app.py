import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import matplotlib.pyplot as plt
import mysql.connector

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
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0,
    callbacks=[es],
)

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
