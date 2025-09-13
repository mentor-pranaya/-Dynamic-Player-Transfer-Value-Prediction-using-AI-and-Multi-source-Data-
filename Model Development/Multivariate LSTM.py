import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ...rest of your code...
# Load your data
df = pd.read_csv(r'Data\Market value data.csv', encoding='latin1')

# Print columns to see what's available
print("Columns in your CSV:", df.columns.tolist())

# Use only columns that exist in your data
features = ['market_value']  # Update this list after you see the printed columns

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=features)

# Prepare data for LSTM
def create_sequences_mv(data, seq_length, target_idx=0):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_idx])
    return np.array(X), np.array(y)

sequence_length = 10
data = df[features].values
X, y = create_sequences_mv(data, sequence_length)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, input_shape=(sequence_length, len(features))),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)