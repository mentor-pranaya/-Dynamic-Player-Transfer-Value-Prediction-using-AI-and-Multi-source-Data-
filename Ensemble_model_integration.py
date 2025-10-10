import os, joblib
import tensorflow as tf
import pandas as pd

MODEL_DIR = '/content/drive/MyDrive/TransferIQ_Models'

multi_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'multivariate_lstm_final.h5'),
    compile=False
)

uni_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'univariate_lstm_final.h5'),
    compile=False
)

# Re-compile manually
multi_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
uni_model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# -----------------------------
# Milestone : Safe Ensemble Model Integration
# -----------------------------
print('
Starting Milestone 5 — Ensemble model integration (safe version)...')

import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

# Flatten sequences for tree-based models
X_tree = X_all.reshape(X_all.shape[0], -1)
y_tree = y_all.reshape(-1, 1)

VAL_SPLIT = 0.2   # 20% validation split
RANDOM_SEED = 42 # for reproducibility


# Train/test split
from sklearn.model_selection import train_test_split
X_train_tree, X_val_tree, y_train_tree, y_val_tree = train_test_split(
    X_tree, y_tree, test_size=VAL_SPLIT, random_state=RANDOM_SEED
)

# -----------------------------
# 1) Train XGBoost safely (no early stopping)
# -----------------------------
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_SEED
)
xgb_model.fit(X_train_tree, y_train_tree)
print('XGBoost trained.')

# -----------------------------
# 2) Train LightGBM
# -----------------------------
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=RANDOM_SEED
)
lgb_model.fit(X_train_tree, y_train_tree)
print('LightGBM trained.')


# -----------------------------
# 3) Stacking Ensemble
# -----------------------------
estimators = [('xgb', xgb_model), ('lgb', lgb_model)]
stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    n_jobs=-1,
    passthrough=True
)
stack_model.fit(X_train_tree, y_train_tree)
print('Stacking ensemble trained.')

# -----------------------------
# 4) Evaluation function
# -----------------------------
def evaluate_model(name, model, X_t, y_t):
    y_pred = model.predict(X_t)
    mae = mean_absolute_error(y_t, y_pred)
    rmse = np.sqrt(mean_squared_error(y_t, y_pred))
    mape = np.mean(np.abs((y_t - y_pred) / np.where(y_t==0, 1e-8, y_t))) * 100
    print(f"{name} — MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")


import numpy as np

# -----------------------------
# Function to safely sample validation data
# -----------------------------
def get_val_sample(X_val, y_val, frac=0.2, seed=RANDOM_SEED):
    np.random.seed(seed)
    val_size = int(frac * len(X_val))
    idx = np.random.choice(len(X_val), size=val_size, replace=False)
    return X_val[idx], y_val[idx]

# -----------------------------
# Take 20% subset of validation data
# -----------------------------
X_val_sample, y_val_sample = get_val_sample(X_val_tree, y_val_tree, frac=0.2)

# -----------------------------
# Evaluate all models on same sample
# -----------------------------
print("
Evaluating XGBoost:")
evaluate_model("XGBoost", xgb_model, X_val_sample, y_val_sample)

print("
Evaluating LightGBM:")
evaluate_model("LightGBM", lgb_model, X_val_sample, y_val_sample)

print("
Evaluating Stacking Ensemble:")
evaluate_model("Stacking Ensemble", stack_model, X_val_sample, y_val_sample)

# -----------------------------
# 5) Save models
# -----------------------------
joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model.save'))
joblib.dump(lgb_model, os.path.join(MODEL_DIR, 'lgb_model.save'))
joblib.dump(stack_model, os.path.join(MODEL_DIR, 'stacking_model.save'))
print('
All ensemble models saved in', MODEL_DIR)