"""
seq2seq_marketvalue_pipeline.py

One-file pipeline:
- Loads 2021_22, 2022_23, 2023_24 season csvs (handles slight column-name differences)
- Builds per-player sequences
- Prepares multivariate input sequences and multistep target (market_value_eur for next `horizon` windows)
- Trains an encoder-decoder LSTM (seq2seq)
- Saves model/scaler/splits
- Includes regression metrics and visualization

Edit the FILE_* constants below if your filenames are different.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------
# USER CONFIG
# ---------------------------
FILE_2021 = "2021_22_features_with_prior_injuries_and_sentiment.csv"
FILE_2022 = "2022_23_features_with_prior_injuries_and_sentiment.csv"
FILE_2023 = "2023_24_features_with_prior_injuries_and_sentiment.csv"

PROJECT_PATH = os.getcwd()
SAVE_DIR = os.path.join(PROJECT_PATH, "models", "LSTM_Models")
os.makedirs(SAVE_DIR, exist_ok=True)

horizon = 2    # predict next 2 transfer windows
max_len = 4    # max input sequence length (seasons)
random_state = 42

# ---------------------------
# helper functions
# ---------------------------
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, low_memory=False)

def pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---------------------------
# 1) Read CSVs
# ---------------------------
print("Loading CSVs...")
df21 = safe_read_csv(FILE_2021)
df22 = safe_read_csv(FILE_2022)
df23 = safe_read_csv(FILE_2023)

# Add season labels
df21["season"] = "2021_22"
df22["season"] = "2022_23"
df23["season"] = "2023_24"

# ---------------------------
# 2) Normalize column names & unify
# ---------------------------
col_player_id_candidates = ["player_id", "player id", "id"]
col_player_name_candidates = ["player_name", "player name", "player"]
col_market_candidates = ["market_value_eur", "market_value", "market_value"]
col_goals_candidates = ["goals", "gls", "total_goals", "g"]
col_assists_candidates = ["assists", "ast", "total_assists"]
col_mp_candidates = ["mp", "appearances", "matches_played"]
col_min_candidates = ["min", "minutes", "minutes_played", "min_played", "mins"]
col_sot_candidates = ["sot", "shots_on_target", "shots_on_target_per90", "sot%"]
col_compound_candidates = ["compound", "mean_compound", "mean_compound_score"]
col_prior_injury_candidates = ["prior_injury_count", "prior_injury_count.1"]
col_days_missed_candidates = ["total_days_missed_prior", "total_days_missed_prior.1"]

def standardize_df(df):
    d = df.copy()

    pid_col = pick_first_existing_col(d, col_player_id_candidates)
    pname_col = pick_first_existing_col(d, col_player_name_candidates)
    if pid_col is None and pname_col is None:
        raise ValueError("No player id/name column found in a CSV.")

    d["player_id"] = d[pid_col].astype(str) if pid_col else d[pname_col].astype(str)
    d["player_name"] = d[pname_col].astype(str) if pname_col else d["player_id"].astype(str)

    def safe_num(col_candidates):
        col = pick_first_existing_col(d, col_candidates)
        return pd.to_numeric(d[col], errors="coerce") if col else np.nan

    d["market_value_eur"] = safe_num(col_market_candidates)
    d["goals"] = safe_num(col_goals_candidates)
    d["assists"] = safe_num(col_assists_candidates)
    d["mp"] = safe_num(col_mp_candidates)
    d["min"] = safe_num(col_min_candidates)
    d["sot"] = safe_num(col_sot_candidates)
    d["compound"] = safe_num(col_compound_candidates)
    d["prior_injury_count"] = safe_num(col_prior_injury_candidates)
    d["total_days_missed_prior"] = safe_num(col_days_missed_candidates)

    # Fill missing numeric values
    d.fillna({
        "market_value_eur": 0, "goals": 0, "assists": 0, "mp": 0,
        "min": 0, "sot": 0, "compound": 0,
        "prior_injury_count": 0, "total_days_missed_prior": 0
    }, inplace=True)

    # Derived stats
    d["goals_per90"] = d.apply(lambda r: (r["goals"] / (r["min"]/90)) if r["min"] > 0 else 0.0, axis=1)
    d["assists_per90"] = d.apply(lambda r: (r["assists"] / (r["min"]/90)) if r["min"] > 0 else 0.0, axis=1)

    return d[[
        "season", "player_id", "player_name", "market_value_eur",
        "goals", "assists", "mp", "min", "sot",
        "compound", "prior_injury_count", "total_days_missed_prior",
        "goals_per90", "assists_per90"
    ]]

print("Standardizing CSVs...")
d21s, d22s, d23s = map(standardize_df, [df21, df22, df23])
df_all = pd.concat([d21s, d22s, d23s], ignore_index=True)

# Ensure clean data
numeric_cols = [
    "market_value_eur","goals","assists","mp","min","sot",
    "compound","prior_injury_count","total_days_missed_prior",
    "goals_per90","assists_per90"
]
df_all[numeric_cols] = df_all[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
df_all = df_all[df_all["player_id"].notna() & (df_all["player_id"].astype(str).str.strip() != "")]
if df_all[numeric_cols].isna().sum().sum() > 0:
    print("Warning: NaNs remain, filling with 0.")
    df_all[numeric_cols] = df_all[numeric_cols].fillna(0)

print(f"Combined rows: {len(df_all)}")
print(df_all.head(4).to_string(index=False))

# ---------------------------
# 3) Create per-player sequences
# ---------------------------
print("Building per-player sequences...")
player_sequences = defaultdict(list)
season_order_map = {s:i for i,s in enumerate(sorted(df_all["season"].unique()))}

for _, row in df_all.iterrows():
    pid = str(row["player_id"])
    vals = np.array([
        row["market_value_eur"], row["goals"], row["assists"], row["mp"],
        row["min"], row["sot"], row["compound"], row["prior_injury_count"],
        row["total_days_missed_prior"], row["goals_per90"], row["assists_per90"]
    ], dtype=float)
    player_sequences[pid].append((season_order_map[row["season"]], row["season"], vals))

for pid in player_sequences:
    player_sequences[pid] = sorted(player_sequences[pid], key=lambda x: x[0])

# ---------------------------
# 4) Assemble X, y
# ---------------------------
print("Assembling samples (X, y)...")
X_list, y_list = [], []

for pid, seq in player_sequences.items():
    vals = np.array([v for (_, _, v) in seq])
    if vals.shape[0] <= horizon:
        continue
    input_seq = vals[:-horizon]
    target_seq = vals[-horizon:, 0]
    padded = np.zeros((max_len, vals.shape[1]), dtype=float)
    if input_seq.shape[0] > max_len:
        input_seq = input_seq[-max_len:]
    padded[-input_seq.shape[0]:, :] = input_seq
    X_list.append(padded)
    y_list.append(target_seq)

if not X_list:
    raise RuntimeError("No training samples found. Check horizon/max_len.")

X = np.array(X_list)
y = np.array(y_list)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------------------------
# 5) Scale and split
# ---------------------------
n_samples, seq_len, n_features = X.shape
X_2d = X.reshape(-1, n_features)
scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)
X_scaled = X_2d_scaled.reshape(n_samples, seq_len, n_features)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=random_state)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

y_train_seq = np.log1p(y_train).reshape((y_train.shape[0], y_train.shape[1], 1))
y_val_seq = np.log1p(y_val).reshape((y_val.shape[0], y_val.shape[1], 1))
y_test_seq = np.log1p(y_test).reshape((y_test.shape[0], y_test.shape[1], 1))

# ---------------------------
# 6) Model (Optimized)
# ---------------------------
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization

timesteps_in, timesteps_out = X_train.shape[1], y_train_seq.shape[1]
latent_dim = 128   # 🔼 increased for better capacity

def build_seq2seq(latent_dim=128, dropout_rate=0.3, lr=1e-3):
    encoder_inputs = Input(shape=(timesteps_in, n_features))
    encoder_lstm = LSTM(latent_dim, activation="tanh", return_sequences=True)(encoder_inputs)
    encoder_lstm = Dropout(dropout_rate)(encoder_lstm)
    encoder_out = LSTM(latent_dim, activation="tanh")(encoder_lstm)
    encoder_out = BatchNormalization()(encoder_out)

    decoder_inputs = RepeatVector(timesteps_out)(encoder_out)
    decoder_lstm = LSTM(latent_dim, activation="tanh", return_sequences=True)(decoder_inputs)
    decoder_lstm = Dropout(dropout_rate)(decoder_lstm)
    decoder_out = TimeDistributed(Dense(64, activation="relu"))(decoder_lstm)
    decoder_out = TimeDistributed(Dense(1))(decoder_out)

    model = Model(encoder_inputs, decoder_out)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

model = build_seq2seq(latent_dim=128, dropout_rate=0.35, lr=5e-4)
model.summary()

# ---------------------------
# 7) Train (with better patience & LR scheduling)
# ---------------------------
from tensorflow.keras.callbacks import ReduceLROnPlateau

epochs = 100
batch_size = 8  # 🔽 smaller batches help LSTMs converge better
ckpt_path = os.path.join(SAVE_DIR, "best_seq2seq_tuned.h5")

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
    ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1)
]

print("Training tuned model...")
history = model.fit(
    X_train, y_train_seq,
    validation_data=(X_val, y_val_seq),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=2
)


# ---------------------------
# 8) Evaluate & Predict
# ---------------------------
print("Evaluating...")
test_loss, test_mae = model.evaluate(X_test, y_test_seq, verbose=1)
print(f"Test Loss (MSE, log-space): {test_loss:.4f}")
print(f"Test MAE (log-space): {test_mae:.4f}")

y_pred_seq_log = model.predict(X_test).squeeze(-1)
y_true_seq_log = y_test_seq.squeeze(-1)

y_pred_eur = np.expm1(y_pred_seq_log)
y_true_eur = np.expm1(y_true_seq_log)

# ---------------------------
# 9) Performance metrics
# ---------------------------
y_true_flat = y_true_eur.flatten()
y_pred_flat = y_pred_eur.flatten()

mae = mean_absolute_error(y_true_flat, y_pred_flat)
rmse = mean_squared_error(y_true_flat, y_pred_flat, squared=False)
r2 = r2_score(y_true_flat, y_pred_flat)
mape = np.mean(np.abs((y_true_flat - y_pred_flat) / np.maximum(y_true_flat, 1))) * 100
corr = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]

print("\n--- Model Performance (Original € Scale) ---")
print(f"MAE  : {mae:,.0f} €")
print(f"RMSE : {rmse:,.0f} €")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.3f}")
print(f"Correlation: {corr:.3f}")

# sample
print("\nExample true / pred for first test sample:")
print("True (EUR):", y_true_eur[0])
print("Pred (EUR):", y_pred_eur[0])

# ---------------------------
# 10) Visualization
# ---------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
plt.xlabel("True Market Value (€)")
plt.ylabel("Predicted Market Value (€)")
plt.title("True vs Predicted Market Value")
plt.plot([0, max(y_true_flat)], [0, max(y_true_flat)], 'r--')
plt.tight_layout()
plt.show()

# ---------------------------
# 11) Save everything
# ---------------------------
print("Saving model, scaler and splits...")
model.save(os.path.join(SAVE_DIR, "encoder_decoder_lstm_model.h5"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_seq2seq.pkl"))

np.save(os.path.join(SAVE_DIR, "X_train_seq2seq.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "X_val_seq2seq.npy"), X_val)
np.save(os.path.join(SAVE_DIR, "X_test_seq2seq.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "y_train_seq2seq.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "y_val_seq2seq.npy"), y_val)
np.save(os.path.join(SAVE_DIR, "y_test_seq2seq.npy"), y_test)

print("All saved to:", SAVE_DIR)
print("Done.")
