"""
lstm_univariate_model.py
------------------------
Trains a univariate LSTM model using only player market value
history to predict future market value trends.
Part of the player performance prediction pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ================================================================
# 1. Configuration
# ================================================================
BASE_DIR = r"C:\Users\ghans\OneDrive\Desktop\filemanaging"

# File paths (consistent with your project)
FILE_PATH = os.path.join(BASE_DIR, "fifa_players_data_no_duplicates.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "univariate_lstm_model.h5")
RESULTS_SAVE_PATH = os.path.join(BASE_DIR, "results", "univariate_lstm_results.csv")

# Create folders if missing
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_SAVE_PATH), exist_ok=True)

# Model configuration
TIME_STEPS = 12
TRAIN_RATIO = 0.8
TARGET_COLUMN = "value_euro"

# ================================================================
# 2. Data Loading and Preparation
# ================================================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at: {FILE_PATH}")

df = pd.read_csv(FILE_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Sort by player and date (important for time-based learning)
sort_cols = [col for col in ["player", "date"] if col in df.columns]
if sort_cols:
    df = df.sort_values(by=sort_cols)

# Select the target column
if TARGET_COLUMN not in df.columns:
    raise KeyError(f"❌ Column '{TARGET_COLUMN}' not found in dataset!")

target_values = df[TARGET_COLUMN].fillna(method="ffill").fillna(0).values.reshape(-1, 1)

# Normalize (scaling to 0–1 range)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(target_values)

# ================================================================
# 3. Sequence Creation
# ================================================================
def create_sequences(data, time_steps):
    """Create sequential samples for univariate LSTM."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_values, TIME_STEPS)

# Reshape for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train/Validation Split
split_idx = int(TRAIN_RATIO * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"📊 Sequences prepared: {len(X)} total")
print(f"   Training shape: {X_train.shape}")
print(f"   Validation shape: {X_val.shape}\n")

# ================================================================
# 4. Model Definition
# ================================================================
model = Sequential([
    LSTM(64, activation='relu', input_shape=(TIME_STEPS, 1), return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("🧠 Model Summary:")
model.summary()

# ================================================================
# 5. Model Training
# ================================================================
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1)

print("\n🚀 Training Univariate LSTM Model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print(f"\n✅ Training complete. Model saved to: {MODEL_SAVE_PATH}")

# ================================================================
# 6. Evaluation
# ================================================================
# Predict and reverse scaling
y_pred_scaled = model.predict(X_val)
dummy_true = np.zeros((len(y_val), 1))
dummy_pred = np.zeros((len(y_pred_scaled), 1))

dummy_true[:, 0] = y_val[:, 0]
dummy_pred[:, 0] = y_pred_scaled[:, 0]

y_true_eur = scaler.inverse_transform(dummy_true).flatten()
y_pred_eur = scaler.inverse_transform(dummy_pred).flatten()

# Calculate RMSE and MAE (in euros)
rmse = np.sqrt(np.mean((y_true_eur - y_pred_eur) ** 2))
mae = np.mean(np.abs(y_true_eur - y_pred_eur))

# Save results
results_df = pd.DataFrame({
    "Metric": ["RMSE (EUR)", "MAE (EUR)"],
    "Value": [rmse, mae]
})
results_df.to_csv(RESULTS_SAVE_PATH, index=False)

print("\n📈 Evaluation Results:")
print(f"   RMSE: {rmse:,.2f} EUR")
print(f"   MAE: {mae:,.2f} EUR")
print(f"💾 Results saved to: {RESULTS_SAVE_PATH}")

# ================================================================
# 7. Visualization
# ================================================================
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Univariate LSTM Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_true_eur[:200], label='Actual Market Value', linewidth=2)
plt.plot(y_pred_eur[:200], label='Predicted Market Value', linestyle='--')
plt.title('Actual vs Predicted Market Value (Sample)')
plt.xlabel('Time Steps')
plt.ylabel('Market Value (EUR)')
plt.legend()
plt.tight_layout()
plt.show()
