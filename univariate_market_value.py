# requirements:
# pip install pandas numpy scikit-learn matplotlib tensorflow joblib

import os, re, glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ---------- USER PARAMETERS ----------
DATA_FOLDER = "./"   # folder containing your Player_Feature_*.csv files
FILE_GLOB = "Player_Feature_2021_22_Enhanced*.csv"  # adjust if your filenames differ
PLAYER_COL = "player"
VALUE_COL = "market_value_eur"
WINDOW_SIZE = 3      # number of past snapshots to use (if season-level, try 2-4)
TEST_RATIO = 0.15
VAL_RATIO = 0.15
USE_LOG = True       # strongly recommended for monetary values
BATCH_SIZE = 32
EPOCHS = 80
MODEL_SAVE = "univariate_lstm_pooled.h5"
SCALER_X_SAVE = "scaler_x.pkl"
SCALER_Y_SAVE = "scaler_y.pkl"

# 1. Load & stack CSVs 
files = sorted(glob.glob(os.path.join(DATA_FOLDER, FILE_GLOB)))
if not files:
    raise RuntimeError(f"No files found with pattern {os.path.join(DATA_FOLDER, FILE_GLOB)}")

def extract_season_label(fname):
    b = os.path.basename(fname)
    m = re.search(r'(\d{4})[_-]?(\d{2})', b)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m2 = re.search(r'(\d{4})', b)
    return m2.group(1) if m2 else b

dfs = []
for f in files:
    # Try latin1 then utf-8 to avoid mojibake loss
    try:
        d = pd.read_csv(f, encoding="latin1")
    except Exception:
        d = pd.read_csv(f, encoding="utf-8", errors="replace")
    # rename some columns if they have weird whitespace
    d.columns = [c.strip() for c in d.columns]
    d['_source_file'] = os.path.basename(f)
    d['_season_label'] = extract_season_label(f)
    dfs.append(d)

df_all = pd.concat(dfs, ignore_index=True, sort=False)
print("Loaded files:", files)
print("Total rows:", len(df_all))

# 2. Fix textual mojibake in object columns (player, squad, comp, etc.)
text_cols = [c for c in df_all.columns if df_all[c].dtype == object]
def fix_text_col(s):
    if not isinstance(s, str):
        return s
    try:
        # reinterpret bytes from latin1 -> utf-8
        return s.encode('latin1').decode('utf-8')
    except Exception:
        return s

for c in text_cols:
    df_all[c] = df_all[c].apply(fix_text_col)

# 3. Keep only necessary columns & cast value to numeric 
keep_cols = [PLAYER_COL, VALUE_COL, '_season_label']
for c in keep_cols:
    if c not in df_all.columns:
        raise RuntimeError(f"Required column '{c}' not found. Available columns: {df_all.columns.tolist()}")

df_subset = df_all[keep_cols].copy()
# convert market value to numeric (some commas or € might exist)
df_subset[VALUE_COL] = (
    df_subset[VALUE_COL]
    .astype(str)
    .str.replace('[^0-9.-]', '', regex=True)
    .replace('', np.nan)
    .astype(float)
)
df_subset = df_subset.dropna(subset=[PLAYER_COL, VALUE_COL])

# 4. Create numeric season index so we can order snapshots 
# Prefer using first 4-digit year of season label as ordering key
def season_key(label):
    m = re.search(r'(\d{4})', str(label))
    return int(m.group(1)) if m else 0

season_order = sorted(df_subset['_season_label'].unique(), key=season_key)
season_to_idx = {s: i for i, s in enumerate(season_order)}
df_subset['season_idx'] = df_subset['_season_label'].map(season_to_idx)

# ---------- 5. Build per-player ordered series ----------
df_subset = df_subset.sort_values([PLAYER_COL, 'season_idx'])
grouped = df_subset.groupby(PLAYER_COL)

# we will collect sequences and the season index of the target (useful for splitting)
X_list, y_list, target_season_idx_list = [], [], []

min_history = WINDOW_SIZE + 1
for player, g in grouped:
    vals = g[VALUE_COL].values
    seasons = g['season_idx'].values
    if len(vals) < min_history:
        continue
    # create sliding windows by season order (g is already sorted)
    for i in range(len(vals) - WINDOW_SIZE):
        X_seq = vals[i : i + WINDOW_SIZE]
        y_val = vals[i + WINDOW_SIZE]
        target_season_idx = seasons[i + WINDOW_SIZE]
        X_list.append(X_seq)
        y_list.append(y_val)
        target_season_idx_list.append(int(target_season_idx))

X = np.array(X_list)  # shape (samples, WINDOW_SIZE)
y = np.array(y_list)  # shape (samples,)
targets_seasons = np.array(target_season_idx_list)

print("Created sequences:", X.shape, "Targets:", y.shape)
if X.shape[0] == 0:
    raise RuntimeError("No sequences created. Try reducing WINDOW_SIZE or supply more seasons per player.")

# ---------- 6. Apply log transform if desired ----------
if USE_LOG:
    y = np.log1p(y)
    X = np.log1p(X)

# 7. Split into train/val/test by target season distribution (time-aware) 
# Use season index quantiles so sequences whose prediction target is in last seasons go to test
test_thresh = np.quantile(targets_seasons, 1 - TEST_RATIO)
val_thresh  = np.quantile(targets_seasons, 1 - TEST_RATIO - VAL_RATIO)

train_idx = targets_seasons <= val_thresh
val_idx = (targets_seasons > val_thresh) & (targets_seasons <= test_thresh)
test_idx = targets_seasons > test_thresh

X_train = X[train_idx]; y_train = y[train_idx]
X_val   = X[val_idx];   y_val   = y[val_idx]
X_test  = X[test_idx];  y_test  = y[test_idx]

print("Split sizes: train", len(X_train), "val", len(X_val), "test", len(X_test))

#  8. Scale X (fit on train only) 
# Univariate -> scale flattened values
scaler_x = MinMaxScaler()
scaler_x.fit(X_train.reshape(-1, 1))

def scale_X(X_arr, scaler):
    s = scaler.transform(X_arr.reshape(-1,1)).reshape(X_arr.shape)
    return s

X_train_s = scale_X(X_train, scaler_x)
X_val_s = scale_X(X_val, scaler_x)
X_test_s = scale_X(X_test, scaler_x)

# reshape for LSTM: (samples, timesteps, features=1)
X_train_s = X_train_s.reshape((X_train_s.shape[0], X_train_s.shape[1], 1))
X_val_s   = X_val_s.reshape((X_val_s.shape[0], X_val_s.shape[1], 1))
X_test_s  = X_test_s.reshape((X_test_s.shape[0], X_test_s.shape[1], 1))

# 9. Scale y (target)
scaler_y = MinMaxScaler()
scaler_y.fit(y_train.reshape(-1,1))
y_train_s = scaler_y.transform(y_train.reshape(-1,1)).reshape(-1)
y_val_s   = scaler_y.transform(y_val.reshape(-1,1)).reshape(-1)
y_test_s  = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1)

# 10. Build the LSTM model 
tf.keras.backend.clear_session()
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE,1)),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
model.summary()

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6),
    ModelCheckpoint(MODEL_SAVE, save_best_only=True, monitor='val_loss')
]

# 11. Train 
history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=False
)

#  12. Evaluate 
def invert_y(y_scaled):
    raw = scaler_y.inverse_transform(y_scaled.reshape(-1,1)).reshape(-1)
    if USE_LOG:
        return np.expm1(raw)
    return raw

y_pred_s = model.predict(X_test_s).reshape(-1)
y_pred = invert_y(y_pred_s)
y_true = invert_y(y_test_s)

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)
print("Test MAE:", mae, "RMSE:", rmse, "R2:", r2)

# Plot actual vs predicted (sampled)
plt.figure(figsize=(12,5))
plt.plot(y_true, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend(); plt.title("Test: Actual vs Predicted"); plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title("Loss curves"); plt.show()

#13. Save model & scalers
model.save(MODEL_SAVE)
joblib.dump(scaler_x, SCALER_X_SAVE)
joblib.dump(scaler_y, SCALER_Y_SAVE)
print("Saved model and scalers.")
