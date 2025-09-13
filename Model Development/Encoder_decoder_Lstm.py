import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# ...rest of your code...

# Load your data
df = pd.read_csv(r'Data\Market value data.csv', encoding='latin1')

# Define your features (update as needed)
features = ['market_value']  # Add more features if available
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=features)

data = df[features].values  # <-- This defines 'data'

# Encoder-decoder sequence creation
def create_sequences_ed(data, n_steps_in, n_steps_out, target_idx=0):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out):
        X.append(data[i:i+n_steps_in])
        y.append(data[i+n_steps_in:i+n_steps_in+n_steps_out, target_idx])
    return np.array(X), np.array(y)

n_steps_in = 10
n_steps_out = 3
X, y = create_sequences_ed(data, n_steps_in, n_steps_out)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, activation='relu', input_shape=(n_steps_in, len(features))),
    RepeatVector(n_steps_out),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)