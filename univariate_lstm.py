import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# --- Configuration ---
FILE_PATH = "/Users/veerababu/Downloads/cleaned/master_list_final_features.csv"
TIME_STEPS = 12  # Look-back window: uses the last 12 data points
TRAIN_RATIO = 0.8  # 80% for training
TARGET_COLUMN = 'market_value_in_eur'
# --- MODEL SAVE PATH ---
MODEL_SAVE_PATH = '/Users/veerababu/Downloads/univariate_lstm_model.h5' 
# -----------------------

def create_sequences(data, time_steps):
    """Convert time-series data (single feature) into sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        # Input sequence (X): T past values
        X.append(data[i:(i + time_steps), 0])
        # Output value (y): The single value immediately following the sequence
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# =============================================================
# 1. Data Loading and Preparation
# =============================================================
try:
    df = pd.read_csv(FILE_PATH)
    
    # Ensure data is sorted by player and date (Crucial for correct sequencing)
    # You must verify 'date' and 'player' column names are correct in your CSV.
    if 'date' in df.columns and 'player' in df.columns:
        df = df.sort_values(by=['player', 'date'])
    else:
        print("Warning: Data may not be properly sorted for time-series if 'date' or 'player' are missing.")

except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()

# Select the target column, fill NaNs (assuming 0 is appropriate), and reshape
target_values = df[TARGET_COLUMN].fillna(0).values.reshape(-1, 1)

# Scale the market values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(target_values)

# Create sequences
X, y = create_sequences(scaled_data, TIME_STEPS)

# Reshape input to [samples, time_steps, features] (Features = 1 for univariate)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Chronological Train/Validation Split
split_index = int(TRAIN_RATIO * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

print(f"Total sequences created: {len(X)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# =============================================================
# 2. Define Univariate LSTM Model
# =============================================================
model = Sequential()

# LSTM Layer: 50 units
model.add(LSTM(
    units=50, 
    activation='relu', 
    input_shape=(TIME_STEPS, 1) # Time-steps (12) and 1 feature
))

# Dense Output Layer: 1 unit for single-step prediction
model.add(Dense(units=1))

# Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("\nModel Summary:")
model.summary()

# =============================================================
# 3. Callbacks and Training
# =============================================================
# Ensure the directory for the save path exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10, 
    restore_best_weights=True
)

# Model Checkpoint: Saves the best performing model to the Downloads folder
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

print("\nStarting Univariate LSTM Training...")
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\nUnivariate LSTM Training Complete.")
print(f"Model saved to: {MODEL_SAVE_PATH}")

# =============================================================
# 4. Evaluation and Visualization
# =============================================================

# Final Evaluation
loss, mae = model.evaluate(X_val, y_val, verbose=0)

# Calculate RMSE in original currency (EUR)
rmse_scaled = np.sqrt(loss) 
dummy_array = np.zeros(shape=(X_val.shape[0], 1))
dummy_array[:, 0] = rmse_scaled 
rmse_eur = scaler.inverse_transform(dummy_array)[0, 0]

print(f"\nFinal Validation MSE (Scaled): {loss:.6f}")
print(f"Final Validation MAE (Scaled): {mae:.6f}")
print(f"Estimated Validation RMSE (in EUR): {rmse_eur:,.2f}") 

# Plotting Loss Curves (Part of Deliverables)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Univariate LSTM Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()
