import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from unidecode import unidecode

# --- 1. Load Your Clean Data ---
data_folder = 'data'
file_path = os.path.join(data_folder, 'final_top_10_player_data.csv')
df = pd.read_csv(file_path)

print("--- Data Loaded ---")

# --- 2. Clean and Engineer Features ---

# 2a. Clean the 'total_days_injured' column as requested
print("\n--- Cleaning 'total_days_injured' column ---")
df['total_days_injured'] = df['total_days_injured'].astype(str)
df['total_days_injured'] = df['total_days_injured'].str.extract('(\d+)', expand=False)
df['total_days_injured'] = df['total_days_injured'].fillna(0)
df['total_days_injured'] = df['total_days_injured'].astype(int)
print(f"Cleaned 'total_days_injured' - new data type: {df['total_days_injured'].dtype}")

# 2b. Engineer the 'injury_risk_score' feature
print("\n--- Engineering 'injury_risk_score' feature ---")
# Scale both injury columns between 0 and 1 to combine them fairly
injury_scaler = MinMaxScaler()
df[['scaled_days', 'scaled_count']] = injury_scaler.fit_transform(df[['total_days_injured', 'injury_count']])
# The risk score is the sum of the scaled values
df['injury_risk_score'] = df['scaled_days'] + df['scaled_count']
# Drop the temporary scaled columns
df.drop(columns=['scaled_days', 'scaled_count'], inplace=True)
print("Created 'injury_risk_score' to represent injury proneness.")

# --- 3. Select and Prepare All Features ---
# First, handle the categorical 'position' column with one-hot encoding
df_encoded = pd.get_dummies(df, columns=['position'])

# Define the final list of all features you want to use
feature_cols = [
    'goals',
    'assists',
    'successful_passes',
    'tackles_won',
    'avg_sentiment_score',
    'injury_risk_score' # Using our new engineered feature
]
# Add the new one-hot encoded position columns to our feature list
feature_cols.extend([col for col in df_encoded.columns if 'position_' in col])

target_col = 'Market Value 2015 (in millions €)'

# Create a new DataFrame with all selected features and the target
data = df_encoded[feature_cols + [target_col]].copy()
print("\n--- All features prepared ---")

# --- 4. Scale the Data for the Model ---
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
print("--- All data scaled ---")

# --- 5. Prepare Data for Multivariate LSTM ---
X = data_scaled[:, :-1] # All columns except the last are features
y = data_scaled[:, -1]  # The last column is the target
# Reshape X to the required 3D format: [samples, timesteps, features]
X = X.reshape(X.shape[0], 1, X.shape[1])
print(f"--- Data Reshaped for LSTM (Shape: {X.shape}) ---")

# --- 6. Build and Train the Final LSTM Model ---
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)) # Added return_sequences for stacking
model.add(LSTM(50)) # Added a second LSTM layer for more complexity
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("--- Training the Final LSTM Model ---")
model.fit(X, y, epochs=150, batch_size=1, verbose=0) # Increased epochs for more training
print("Model training complete!")

# --- 7. Make and Evaluate Predictions ---
predictions_scaled = model.predict(X)

# Create a dummy array to revert the scaling
dummy_array = np.zeros((len(predictions_scaled), data_scaled.shape[1]))
dummy_array[:, -1] = predictions_scaled.ravel()

# Inverse transform to get predictions in the original scale (€)
predictions = scaler.inverse_transform(dummy_array)[:, -1]

# Create a final DataFrame to compare actual vs. predicted values
results = pd.DataFrame({
    'Player Name': df['player_name'],
    'Actual Market Value (€M)': df[target_col],
    'Predicted Market Value (€M)': predictions.round(2)
})

print("\n--- Final Model Predictions ---")
print(results)