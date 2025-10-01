import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- CONFIGURATION ---
N_STEPS = 5
TARGET_COL = 'current_value'
NUMERICAL_COLS = ['age', 'appearance', 'goals', 'assists', 'yellow cards', 'red cards',
                  'minutes played', 'days_injured', 'games_injured',
                  'highest_value', TARGET_COL]
FEATURE_COLS_MULTI = [col for col in NUMERICAL_COLS if col != TARGET_COL]
FEATURE_COL_UNI = 'goals'
TRAIN_SPLIT_RATIO = 0.8
N_EPOCHS = 20  # Increased epochs for better learning

# --- 1. Data Loading and Preprocessing with LOG TRANSFORMATION ---

# Load the dataset
df = pd.read_csv(r"D:\Pythonproject\datasets\pythonProject\final_data(punaya_murthy).csv")
df_model = df[NUMERICAL_COLS].dropna().copy()

# *** CRITICAL FIX: LOG TRANSFORM THE TARGET VARIABLE ***
# Apply log1p: log(1 + x) is used to handle values that are 0
df_model[TARGET_COL + '_log'] = np.log1p(df_model[TARGET_COL])
LOG_TARGET_COL = TARGET_COL + '_log'

# Scale the data (features and the log-transformed target)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_model[FEATURE_COLS_MULTI + [LOG_TARGET_COL]])
scaled_df = pd.DataFrame(scaled_data, columns=FEATURE_COLS_MULTI + [LOG_TARGET_COL])

# Separate target scaler for inverse transformation
target_log_scaler = MinMaxScaler(feature_range=(0, 1))
target_log_scaler.fit(df_model[[LOG_TARGET_COL]])


# --- 2. Sequence Generation Function (using log-target) ---

def create_sequences(data, target_col, n_steps):
    X, y = [], []
    target_idx = data.columns.get_loc(target_col)

    for i in range(len(data) - n_steps):
        seq_x = data.iloc[i:(i + n_steps)].values
        X.append(seq_x)
        seq_y = data.iloc[i + n_steps, target_idx]
        y.append(seq_y)
    return np.array(X), np.array(y)


# --- 3. Prepare Univariate Data (goals -> log(current_value)) ---
uni_data = scaled_df[[FEATURE_COL_UNI, LOG_TARGET_COL]].copy()
X_uni_all, y_uni_all = create_sequences(uni_data, LOG_TARGET_COL, N_STEPS)
X_uni = X_uni_all[:, :, 0].reshape(-1, N_STEPS, 1)
y_uni = y_uni_all

train_size_uni = int(len(X_uni) * TRAIN_SPLIT_RATIO)
X_train_uni, X_test_uni = X_uni[:train_size_uni], X_uni[train_size_uni:]
y_train_uni, y_test_uni = y_uni[:train_size_uni], y_uni[train_size_uni:]

# --- 4. Prepare Multivariate Data (10 features -> log(current_value)) ---
multi_data = scaled_df[FEATURE_COLS_MULTI + [LOG_TARGET_COL]].copy()
X_multi_all, y_multi_all = create_sequences(multi_data, LOG_TARGET_COL, N_STEPS)
X_multi = X_multi_all[:, :, :-1]
y_multi = y_multi_all
N_FEATURES_MULTI = X_multi.shape[2]

train_size_multi = int(len(X_multi) * TRAIN_SPLIT_RATIO)
X_train_multi, X_test_multi = X_multi[:train_size_multi], X_multi[train_size_multi:]
y_train_multi, y_test_multi = y_multi[:train_size_multi], y_multi[train_size_multi:]

# --- 5. Model Definition and Training (Re-training with Log Target) ---

# Univariate Model
uni_model = Sequential([
    LSTM(50, activation='relu', input_shape=(N_STEPS, 1)),
    Dropout(0.2),
    Dense(1)
])
uni_model.compile(optimizer='adam', loss='mse')
uni_history = uni_model.fit(X_train_uni, y_train_uni, epochs=N_EPOCHS, batch_size=32, verbose=0)

# Multivariate Model
multi_model = Sequential([
    LSTM(100, activation='relu', input_shape=(N_STEPS, N_FEATURES_MULTI)),
    Dropout(0.2),
    Dense(1)
])
multi_model.compile(optimizer='adam', loss='mse')
multi_history = multi_model.fit(X_train_multi, y_train_multi, epochs=N_EPOCHS, batch_size=32, verbose=0)

# --- 6. Prediction and INVERSE LOG TRANSFORMATION ---

# 6.1 Generate Scaled Predictions on the Test Set
y_pred_uni_scaled = uni_model.predict(X_test_uni)
y_pred_multi_scaled = multi_model.predict(X_test_multi)

# 6.2 Inverse Transform SCALING to get back log-values
y_test_log = target_log_scaler.inverse_transform(y_test_uni.reshape(-1, 1))
y_pred_uni_log = target_log_scaler.inverse_transform(y_pred_uni_scaled)
y_pred_multi_log = target_log_scaler.inverse_transform(y_pred_multi_scaled)

# 6.3 *** CRITICAL: Inverse Transform LOG (expm1) to get real currency values ***
y_test_unscaled = np.expm1(y_test_log).flatten()
y_pred_uni_unscaled = np.expm1(y_pred_uni_log).flatten()
y_pred_multi_unscaled = np.expm1(y_pred_multi_log).flatten()

# --- 7. Evaluation and Results (DELIVERABLES 2 & 3) ---

# Create the results DataFrame for display and file saving
results_df_new = pd.DataFrame({
    'Actual_Value': y_test_unscaled.round(0),
    'Uni_Predicted_Value': y_pred_uni_unscaled.round(0),
    'Multi_Predicted_Value': y_pred_multi_unscaled.round(0)
})


# --- Metric Calculation ---
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {model_name} Performance on Test Set ---")
    print(f"RMSE (Root Mean Squared Error): €{rmse:,.0f}")
    print(f"MAE (Mean Absolute Error):      €{mae:,.0f}")
    print(f"R-squared (R2 Score):           {r2:.4f}")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


uni_metrics = evaluate_model(y_test_unscaled, y_pred_uni_unscaled, "Univariate LSTM (Log Transformed)")
multi_metrics = evaluate_model(y_test_unscaled, y_pred_multi_unscaled, "Multivariate LSTM (Log Transformed)")


# Format and Display Predictions
def format_currency(value):
    return f'€{value:,.0f}'


results_df_display = results_df_new.copy()
results_df_display['Actual_Value'] = results_df_display['Actual_Value'].apply(format_currency)
results_df_display['Uni_Predicted_Value'] = results_df_display['Uni_Predicted_Value'].apply(format_currency)
results_df_display['Multi_Predicted_Value'] = results_df_display['Multi_Predicted_Value'].apply(format_currency)

print("\n--- NEW Initial Prediction Results (First 10 Test Samples) ---")
print("Note: Predictions should be much better now!")
print(results_df_display.head(10).to_markdown(index=False))

# Final Summary Table
summary_data = {
    'Metric': ['RMSE', 'MAE', 'R2 Score'],
    'Univariate Model': [f"€{uni_metrics['RMSE']:,.0f}", f"€{uni_metrics['MAE']:,.0f}", f"{uni_metrics['R2']:.4f}"],
    'Multivariate Model': [f"€{multi_metrics['RMSE']:,.0f}", f"€{multi_metrics['MAE']:,.0f}",
                           f"{multi_metrics['R2']:.4f}"]
}
summary_df = pd.DataFrame(summary_data)
print("\n--- Final Performance Summary (After Log Transformation) ---")
print(summary_df.to_string(index=False))

# --- 8. Final Deliverable: Save Best Model ---
# We will save the multivariate model as it generally performs better
best_model_path = 'best_multivariate_lstm_model.h5'
multi_model.save(best_model_path)
print(f"\nDeliverable 1: Best trained model saved to: {best_model_path}")

# Deliverable 2: Prediction results saved to CSV
results_file_path_new = 'log_transformed_lstm_prediction_results.csv'
results_df_new.to_csv(results_file_path_new, index=False)
print(f"Deliverable 2: Full prediction results saved to: {results_file_path_new}")