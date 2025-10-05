import os
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 1) User-editable paths & hyperparameters
# -----------------------------
DATA_DIR = '.' # folder containing the CSV files
TRANSFERS_CSV = os.path.join(DATA_DIR, '/content/drive/MyDrive/Data/Preprocessed_Transferedvalue copy/engineered_transfers.csv')
SENTIMENT_CSV = os.path.join(DATA_DIR, '/content/drive/MyDrive/Data/Preprocessed_Transferedvalue copy/final_featured_dataset (3).csv')
PERFORMANCE_CSV = os.path.join(DATA_DIR, '/content/drive/MyDrive/Data/Preprocessed_Transferedvalue copy/7298.csv')


MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)


# sequence settings
N_STEPS_IN = 12 
N_STEPS_OUT = 1 
FREQ = 'M' 


# training
BATCH_SIZE = 64
EPOCHS = 100
VAL_SPLIT = 0.1
PATIENCE = 10


# 2) Load CSVs
# -----------------------------
print('Loading CSVs...')
transfers_df = pd.read_csv(TRANSFERS_CSV, parse_dates=['transfer_date'], dayfirst=False)
sentiment_df = pd.read_csv(SENTIMENT_CSV, parse_dates=['date'])
perf_df = pd.read_csv(PERFORMANCE_CSV, parse_dates=['timestamp'])


print('Transfers rows:', len(transfers_df))
print('Sentiment rows:', len(sentiment_df))
print('Performance rows:', len(perf_df))

# 5) Build a unified time series per player (resample to monthly/weekly)
# -----------------------------
print('Building unified timeseries and resampling to', FREQ)


# Merge transfers + sentiment + perf_agg
# For transfers, there may be multiple market values per player; we will align them by date
merged = transfers_df[['player_id','date','market_value_in_eur','log_market_value']].copy()


# merge sentiment
if not sentiment_df.empty:
 sent_cols = [c for c in sentiment_df.columns if c not in ['player_name','player_id'][:0]]
# keep only numerical sentiment columns + player_id/date
 keep_cols = ['player_id','date'] + [c for c in sentiment_df.columns if c not in ['player_id','date'] and pd.api.types.is_numeric_dtype(sentiment_df[c])]
 sent_small = sentiment_df[keep_cols].drop_duplicates(subset=['player_id','date'])
 merged = merged.merge(sent_small, on=['player_id','date'], how='left')


# merge performance aggregated
if not perf_agg.empty:
 merged = merged.merge(perf_agg, on=['player_id','date'], how='left')


# set index for resampling
# We'll create a per-player, regularly spaced series (monthly) and forward/back fill features
all_players = merged['player_id'].unique()
rows = []
for pid in all_players:
 tmp = merged[merged['player_id']==pid].set_index('date').sort_index()
# create monthly index from min to max
 idx = pd.date_range(start=tmp.index.min(), end=tmp.index.max(), freq=FREQ)
 tmp = tmp[~tmp.index.duplicated(keep='last')]

 tmp2 = tmp.reindex(idx).rename_axis('date').reset_index()
 tmp2['player_id'] = pid
# forward fill past values (market values often updated at transfers; forward-fill to have continuous series)
 tmp2 = tmp2.sort_values('date').ffill().bfill()
 rows.append(tmp2)


if rows:
 ts = pd.concat(rows, ignore_index=True)
else:
 raise ValueError('No rows to build timeseries from — check your merges')


print('Timeseries shape after resample:', ts.shape)

# 6) Choose features and target
# -----------------------------
# We'll predict log_market_value by default (more stable). If you prefer raw, switch to market_value_in_eur
TARGET_COL = 'log_market_value'
FEATURE_COLS = [c for c in ts.columns if c not in ['player_id','date','market_value_in_eur','log_market_value']]
# often drop columns with many NaNs
FEATURE_COLS = [c for c in FEATURE_COLS if ts[c].notna().sum() > 0]


print('Using features:', FEATURE_COLS[:20])


# Keep only necessary columns
ts = ts[['player_id','date'] + FEATURE_COLS + [TARGET_COL]]


def create_sequences_playerwise(df, feature_cols, target_col, n_steps_in, n_steps_out=1):
    """
    Create sequences by sliding a window over each player's timeseries.
    Returns X (samples, n_steps_in, n_features), y (samples, n_steps_out).
    """
    X_list, y_list = [], []
    players = df['player_id'].unique()

    for pid in players:
        p = df[df['player_id'] == pid].sort_values('date')
        values = p[feature_cols + [target_col]].values
        n_features = len(feature_cols)   # <-- correctly indented

        for i in range(len(values) - n_steps_in - n_steps_out + 1):
            X_list.append(values[i:i+n_steps_in, :n_features])
            y_list.append(values[i+n_steps_in:i+n_steps_in+n_steps_out, n_features])

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y
# 9) Split into train/val by time (player-wise holdout of last portion)
# -----------------------------
# I'll create sequences then split last X% of each player's sequences for validation to avoid leakage.

# create a copy and drop rows with missing target
ts = ts.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# Temporarily fill features NaN with 0 (we will scale after split). You might want smarter imputation.
ts_filled = ts.copy()
for col in FEATURE_COLS:
    if pd.api.types.is_numeric_dtype(ts_filled[col]):
        ts_filled[col] = ts_filled[col].fillna(0)
    else:
        ts_filled[col] = ts_filled[col].fillna(0)

# Create sequences
print('Creating sequences... (this may take a while)')
X_all, y_all = create_sequences_playerwise(ts_filled, FEATURE_COLS, TARGET_COL, N_STEPS_IN, N_STEPS_OUT)
print('Total sequences:', X_all.shape[0])

# Train/val split by temporal holdout — we'll split sequences by player timeline using an index per-player
# Simpler approach: shuffle and use a validation split (not ideal for strict time-series, but workable when sequences are already time-ordered per player)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=VAL_SPLIT, random_state=RANDOM_SEED, shuffle=True)

print('Train sequences:', X_train.shape[0], 'Val sequences:', X_val.shape[0])

# 10) Scaling — fit on training data then apply to val & test
# -----------------------------
# Flatten X_train for scaler fit
n_train = X_train.shape[0]
n_tim = X_train.shape[1]
n_feat = X_train.shape[2]

scaler_X = MinMaxScaler(feature_range=(0,1))
scaler_y = MinMaxScaler(feature_range=(0,1))

X_train_flat = X_train.reshape(-1, n_feat)
X_val_flat = X_val.reshape(-1, n_feat)

scaler_X.fit(X_train_flat)
X_train_scaled = scaler_X.transform(X_train_flat).reshape(n_train, n_tim, n_feat)
X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape[0], n_tim, n_feat)

# y scaling (target is 1D)
y_train_flat = y_train.reshape(-1, 1)
y_val_flat = y_val.reshape(-1, 1)
scaler_y.fit(y_train_flat)
y_train_scaled = scaler_y.transform(y_train_flat).reshape(y_train.shape)
y_val_scaled = scaler_y.transform(y_val_flat).reshape(y_val.shape)
