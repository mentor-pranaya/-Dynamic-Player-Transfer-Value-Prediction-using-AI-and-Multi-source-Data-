import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split


df = pd.read_csv(r'Data\Market value data.csv', encoding='latin1')

print("Columns in DataFrame:", df.columns.tolist())  

df['market_value'] = pd.to_numeric(df['market_value'], errors='coerce')

df_numeric = df.select_dtypes(include=[np.number])

df_numeric = df_numeric.dropna()

def create_sequences(df, seq_length, target_col):
    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(df.iloc[i:i+seq_length].drop(target_col, axis=1, errors='ignore').values)
        y.append(df.iloc[i+seq_length][target_col])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(df_numeric, sequence_length, 'market_value')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)