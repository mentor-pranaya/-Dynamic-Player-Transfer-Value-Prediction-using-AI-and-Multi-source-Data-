import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- CONFIGURATION ---
N_STEPS = 5 # Lookback window (timesteps)
TARGET_COL = 'current_value'
NUMERICAL_COLS = ['age', 'appearance', 'goals', 'assists', 'yellow cards', 'red cards',
                  'minutes played', 'days_injured', 'games_injured',
                  'highest_value', TARGET_COL]
FEATURE_COLS_MULTI = [col for col in NUMERICAL_COLS if col != TARGET_COL]
FEATURE_COL_UNI = 'goals'
TRAIN_SPLIT_RATIO = 0.8

# --- 1. Data Loading and Preprocessing ---

# Load the dataset
df = pd.read_csv(r"D:\Pythonproject\datasets\pythonProject\final_data.csv")
df_model = df[NUMERICAL_COLS].dropna().copy()

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_model)
scaled_df = pd.DataFrame(scaled_data, columns=df_model.columns)

# Separate target scaler for inverse transformation
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(df_model[[TARGET_COL]])

# --- 2. Sequence Generation Function ---

def create_sequences(data, target_col, n_steps):
    X, y = [], []
    target_idx = data.columns.get_loc(target_col)

    for i in range(len(data) - n_steps):
        # Sequence X: n_steps rows
        seq_x = data.iloc[i:(i + n_steps)].values
        X.append(seq_x)
        # Target y: target_col value at the next step (i + n_steps)
        seq_y = data.iloc[i + n_steps, target_idx]
        y.append(seq_y)
    return np.array(X), np.array(y)

# --- 3. Prepare Univariate Data (goals -> current_value) ---
uni_data = scaled_df[[FEATURE_COL_UNI, TARGET_COL]].copy()
X_uni_all, y_uni_all = create_sequences(uni_data, TARGET_COL, N_STEPS)
X_uni = X_uni_all[:, :, 0].reshape(-1, N_STEPS, 1) # Reshape for LSTM (samples, timesteps, features)
y_uni = y_uni_all

# Split Univariate Data
train_size_uni = int(len(X_uni) * TRAIN_SPLIT_RATIO)
X_train_uni, X_test_uni = X_uni[:train_size_uni], X_uni[train_size_uni:]
y_train_uni, y_test_uni = y_uni[:train_size_uni], y_uni[train_size_uni:]


# --- 4. Prepare Multivariate Data (10 features -> current_value) ---
multi_data = scaled_df[FEATURE_COLS_MULTI + [TARGET_COL]].copy()
X_multi_all, y_multi_all = create_sequences(multi_data, TARGET_COL, N_STEPS)
X_multi = X_multi_all[:, :, :-1] # Input features (exclude target from input)
y_multi = y_multi_all

# Split Multivariate Data
train_size_multi = int(len(X_multi) * TRAIN_SPLIT_RATIO)
X_train_multi, X_test_multi = X_multi[:train_size_multi], X_multi[train_size_multi:]
y_train_multi, y_test_multi = y_multi[:train_size_multi], y_multi[train_size_multi:]
N_FEATURES_MULTI = X_multi.shape[2]


# --- 5. Model Definition and Training (Re-training models) ---

# Univariate Model
uni_model = Sequential([
    LSTM(50, activation='relu', input_shape=(N_STEPS, 1)),
    Dropout(0.2),
    Dense(1)
])
uni_model.compile(optimizer='adam', loss='mse')
uni_model.fit(X_train_uni, y_train_uni, epochs=10, batch_size=32, verbose=0)

# Multivariate Model
multi_model = Sequential([
    LSTM(100, activation='relu', input_shape=(N_STEPS, N_FEATURES_MULTI)),
    Dropout(0.2),
    Dense(1)
])
multi_model.compile(optimizer='adam', loss='mse')
multi_model.fit(X_train_multi, y_train_multi, epochs=10, batch_size=32, verbose=0)


# --- 6. Prediction and Inverse Transformation (DELIVERABLE 1) ---

# 6.1 Generate Scaled Predictions on the Test Set
y_pred_uni_scaled = uni_model.predict(X_test_uni)
y_pred_multi_scaled = multi_model.predict(X_test_multi)

# 6.2 Inverse Transform to get real currency values
y_test_unscaled = target_scaler.inverse_transform(y_test_uni.reshape(-1, 1))
y_pred_uni_unscaled = target_scaler.inverse_transform(y_pred_uni_scaled)
y_pred_multi_unscaled = target_scaler.inverse_transform(y_pred_multi_scaled)

# 6.3 Create the results DataFrame
results_df = pd.DataFrame({
    'Actual_Value': y_test_unscaled.flatten().round(0),
    'Uni_Predicted_Value': y_pred_uni_unscaled.flatten().round(0),
    'Multi_Predicted_Value': y_pred_multi_unscaled.flatten().round(0)
})

# Format the values for better readability (currency style)
def format_currency(value):
    return f'€{value:,.0f}'

results_df['Actual_Value'] = results_df['Actual_Value'].apply(format_currency)
results_df['Uni_Predicted_Value'] = results_df['Uni_Predicted_Value'].apply(format_currency)
results_df['Multi_Predicted_Value'] = results_df['Multi_Predicted_Value'].apply(format_currency)

print("\n--- Initial Prediction Results (First 10 Test Samples) ---")
print(results_df.head(10).to_markdown(index=False))

# Save the predictions to a CSV file (Deliverable part)
results_file_path = 'lstm_prediction_results.csv'
results_df.to_csv(results_file_path, index=False)

print(f"\nFull prediction results saved to: {results_file_path}")
