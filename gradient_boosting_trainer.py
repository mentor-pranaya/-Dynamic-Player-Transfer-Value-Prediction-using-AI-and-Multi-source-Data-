import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
DATA_PATH = r'D:\Pythonproject\datasets\pythonProject\final_data.csv'  # Ensure this file is in the same directory
TARGET_COL = 'current_value'
FEATURE_COLS = ['age', 'appearance', 'goals', 'assists', 'yellow cards', 'red cards',
                'minutes played', 'days_injured', 'games_injured',
                'highest_value', 'position_encoded', 'winger']

# --- 1. DATA LOADING AND PREPARATION ---

print("Starting data loading and preprocessing...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()  # Clean up column names
df_model = df[FEATURE_COLS + [TARGET_COL]].dropna().copy()

# Apply Log Transformation to the target variable
df_model['target_log'] = np.log1p(df_model[TARGET_COL])
y = df_model['target_log']
X = df_model[FEATURE_COLS]

# Train-Test Split (important for unbiased evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. SCALING ---
# Fit the scaler ONLY on the training features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for better feature name handling (optional but good practice)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS, index=X_test.index)

# --- 3. MODEL TUNING SETUP ---

# Define a dictionary of models and their parameter grids
models_to_tune = {
    'XGBoost': {
        'model': XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
    },
    'LightGBM': {
        'model': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),  # verbose=-1 silences warnings/output
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1],
        }
    }
}

best_r2 = -np.inf
best_model = None
best_model_name = ""

# --- 4. TUNING AND EVALUATION LOOP ---

for name, config in models_to_tune.items():
    print(f"\n--- Tuning {name} ---")

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        scoring='neg_mean_squared_error',  # Using negative MSE for maximization
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    # Fit the Grid Search on the scaled training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    current_best_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred_log = current_best_model.predict(X_test_scaled)

    # Calculate R2 score for evaluation
    r2 = r2_score(y_test, y_pred_log)

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"R-squared (R2) score for {name} on Test Set: {r2:.4f}")

    # Check if this is the overall best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = current_best_model
        best_model_name = name

# --- 5. FINAL RESULTS AND SAVING ---

print("\n==============================================")
print(f"Overall Best Model Found: {best_model_name}")
print(f"Best R-squared Score: {best_r2:.4f}")
print("==============================================")

# Save the best model
MODEL_FILENAME = 'best_gradient_boosting_model.pkl'
joblib.dump(best_model, MODEL_FILENAME)
print(f"✅ Best Model saved to {MODEL_FILENAME}")

# Save the scaler (re-saving just to ensure consistency)
SCALER_FILENAME = 'min_max_scaler_new.pkl'
joblib.dump(scaler, SCALER_FILENAME)
print(f"✅ Scaler saved to {SCALER_FILENAME}")

print("\nModel tuning complete. You can now use the saved files for deployment.")

