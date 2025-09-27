import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout

data_dir = r"C:/Users/Abhinav/TransferIQ/data"
file_path = data_dir + "/market_value_timeseries.csv"
df = pd.read_csv(file_path, encoding="latin1")

print("✅ Data loaded successfully from:", file_path)

selected_features = ['transfer_value']
df[selected_features] = df[selected_features].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=selected_features)

series = df[selected_features].values.astype(float)

def make_sequences(data, in_steps, out_steps, target_index=0):
    X_seq, y_seq = [], []
    for i in range(len(data) - in_steps - out_steps):
        X_seq.append(data[i:i+in_steps])
        y_seq.append(data[i+in_steps:i+in_steps+out_steps, target_index])
    return np.array(X_seq), np.array(y_seq)

n_input_steps = 12
n_output_steps = 4
X, y = make_sequences(series, n_input_steps, n_output_steps)

print(f"📊 Sequence shapes → X: {X.shape}, y: {y.shape}")

train_size = int(0.75 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(80, activation="tanh", input_shape=(n_input_steps, len(selected_features))))
model.add(Dropout(0.2))
model.add(RepeatVector(n_output_steps))
model.add(LSTM(80, activation="tanh", return_sequences=True))
model.add(TimeDistributed(Dense(1)))

model.compile(optimizer="adam", loss="mse")
print("🛠️ Model compiled and ready.")

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=8,
    validation_split=0.15,
    verbose=1
)

print("✅ Training complete.")
