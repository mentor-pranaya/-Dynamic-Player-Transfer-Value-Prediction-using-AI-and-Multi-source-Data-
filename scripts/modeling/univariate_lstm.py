import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# --- Configuration ---
FILE_PATH = "/Users/veerababu/Downloads/sample_master_list.csv"
TIME_STEPS = 12  # Look-back window
TRAIN_RATIO = 0.8  # 80% training
TARGET_COLUMN = 'market_value_in_eur'
MODEL_SAVE_PATH = '/Users/veerababu/Downloads/univariate_lstm_model.h5' 
# -----------------------

def create_sequences(data, time_steps):
    """Convert univariate data into sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# =============================================================
# 1. Load and Prepare Data
# =============================================================
try:
    df = pd.read_csv(FILE_PATH)
    if 'date' in df.columns and 'player' in df.columns:
        df = df.sort_values(by=['player', 'date'])
    else:
        print("Warning: Data may not be properly sorted for time-series if 'date' or 'player' are missing.")

except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()

# Ensure target is numeric
target_values = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').fillna(0).values.reshape(-1, 1)

# Scale target
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(target_values)

# Create sequences
X, y = create_sequences(scaled_data, TIME_STEPS)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train/Validation split
split_index = int(TRAIN_RATIO * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

print(f"Total sequences: {len(X)}")
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

# =============================================================
# 2. Define LSTM Model
# =============================================================
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(TIME_STEPS, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# =============================================================
# 3. Callbacks and Training
# =============================================================
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# =============================================================
# 4. Evaluation
# =============================================================
loss, mae = model.evaluate(X_val, y_val, verbose=0)
rmse_scaled = np.sqrt(loss)
dummy = np.zeros((X_val.shape[0], 1))
dummy[:, 0] = rmse_scaled
rmse_eur = scaler.inverse_transform(dummy)[0, 0]

print(f"Validation MSE (scaled): {loss:.6f}")
print(f"Validation MAE (scaled): {mae:.6f}")
print(f"Estimated Validation RMSE (EUR): {rmse_eur:,.2f}")

# =============================================================
# 5. Plot Loss
# =============================================================
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Univariate LSTM Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()
