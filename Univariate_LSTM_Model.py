import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('/content/Feature Engineering/player_valuations_featured.csv', parse_dates=['date'])

# Sort data by date
df = df.sort_values('date')

# Plot the data
plt.plot(df['date'], df['market_value_in_eur'])
plt.title('Player Transfer Value over Time')
plt.xlabel('Date')
plt.ylabel('Transfer Value')
plt.show()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['market_value_in_eur']])


sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape X to (samples, time_steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot predictions vs actual
plt.plot(actual, color='blue', label='Actual Transfer Value')
plt.plot(predicted, color='red', label='Predicted Transfer Value')
plt.title('Transfer Value Prediction')
plt.xlabel('Time')
plt.ylabel('Transfer Value')
plt.legend()
plt.show()
