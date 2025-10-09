import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint
import os
import time
import joblib # Needed for saving the model and preprocessor

# --- Configuration ---
TARGET_COLUMN = 'market_value_in_eur'
RESULTS_FILE = '/Users/veerababu/Downloads/xgboost_tuned_results.md'

# Files to be saved for API deployment
MODEL_FILE = 'xgb_model_final.joblib'
PREPROCESSOR_FILE = 'preprocessor_final.joblib'

# Columns to drop to prevent memory overflow and irrelevant features
COLUMNS_TO_DROP = [
    'player', 'player_code', 'url', 'image_url', 'slug', 'player_id', 
    'city_of_birth', 'all_injuries_details', 'agent_name', 'date_of_birth', 
    'contract_expiration_date', 'text', 'most_severe_injury'
]

# --- Functions ---

def load_data(filepath):
    """Loads data and cleans column names."""
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {data.shape}")
        data.columns = data.columns.str.strip()
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

def preprocess_data(df):
    """
    Handles missing values, encodes categorical features, and scales numerical features.
    CRITICAL CHANGE: Uses handle_unknown and max_categories for speed.
    """
    
    # --- 1. CRITICAL: Clean Target Column ---
    initial_shape = df.shape[0]
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    final_shape = df.shape[0]
    print(f"\nCleaned target column. Dropped {initial_shape - final_shape} rows where '{TARGET_COLUMN}' was non-numeric or missing.")

    # --- 2. Drop Irrelevant/High-Cardinality Columns ---
    columns_present_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns and col != TARGET_COLUMN]
    df = df.drop(columns=columns_present_to_drop, errors='ignore')
    print(f"Dropped {len(columns_present_to_drop)} high-cardinality/irrelevant columns to conserve memory.")

    # --- 3. Separate Target (Y) and Features (X) ---
    Y = df.loc[:, TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # --- 4. Define feature types ---
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- 5. Create Preprocessing Pipelines ---
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # SPEED-UP: Limit categories to the top 50
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(
            handle_unknown='infrequent_if_exist', 
            max_categories=50, 
            sparse_output=False
        ))
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 6. Apply Preprocessor to X ---
    print(f"Numerical features remaining: {len(numerical_features)}")
    print(f"Categorical features remaining: {len(categorical_features)}")
    print("Starting One-Hot Encoding and Scaling...")

    X_processed = preprocessor.fit_transform(X)
    
    print(f"\n[CRITICAL SPEEDUP ACHIEVED] Reduced features from 23,822 to:")
    print(f"Final feature matrix shape after encoding: {X_processed.shape}")
    print(f"Total features created: {X_processed.shape[1]}")

    return X_processed, Y, preprocessor, numerical_pipeline['scaler'] 

def save_results_to_markdown(best_params, metrics, output_file):
    """Saves the best parameters and evaluation metrics to a markdown file."""
    
    content = f"""# XGBoost Model Hyperparameter Tuning Results

## Overview
This report documents the results of the Randomized Search Cross-Validation used to optimize the XGBoost Regressor for predicting transfer market values.

- **Baseline R-squared (Before Tuning):** 0.6384 (from initial run)
- **Tuning Method:** Randomized Search Cross-Validation (30 iterations, 5 folds) - Speed Optimized
- **Time of Analysis:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Best Parameters Found
| Parameter | Optimal Value |
| :--- | :--- |
{'\n'.join([f"| `{k}` | `{v}` |" for k, v in best_params.items()])}

## Final Evaluation Metrics (on Test Set)
Using the best parameters found by the search:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **R-squared (R²)** | **{metrics['r2']:.4f}** | Percentage of the target variance explained by the model. |
| **Root Mean Squared Error (RMSE)** | {metrics['rmse']:.4f} EUR | Standard deviation of the prediction errors (should be minimized). |
| **Mean Absolute Error (MAE)** | {metrics['mae']:.4f} EUR | Average magnitude of errors (less sensitive to outliers than RMSE). |

## Next Steps
The next step is to perform **Feature Importance Analysis** to understand which player attributes are most critical to the final market value prediction.
"""
    try:
        with open(output_file, 'w') as f:
            f.write(content)
        print(f"\n[SUCCESS] Results saved to '{output_file}'.")
    except Exception as e:
        print(f"\n[ERROR] Could not save results to file: {e}")

def train_and_evaluate_xgboost(X, Y):
    """Splits data, trains/tunes XGBoost Regressor, and evaluates performance."""
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"\nData split complete. Train set size: {X_train.shape}, Test set size: {X_test.shape}")

    # --- 1. Define Hyperparameter Search Space and Constants ---
    param_dist = {
        'n_estimators': randint(100, 500),      
        'learning_rate': uniform(0.01, 0.3),    
        'max_depth': randint(3, 10),            
        'colsample_bytree': uniform(0.7, 0.3),  
        'subsample': uniform(0.7, 0.3),         
        'gamma': uniform(0, 0.5),               
    }
    
    N_ITERATIONS = 30 
    N_CV_FOLDS = 5
    TOTAL_FITS = N_ITERATIONS * N_CV_FOLDS

    # --- 2. Run a small test to estimate time per fit ---
    print(f"\n[STATUS] Running quick test (5 iterations, single core) to estimate time per fit...")
    
    test_search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=1), 
        param_distributions=param_dist,
        n_iter=5, scoring='r2', cv=N_CV_FOLDS, verbose=0, random_state=42, n_jobs=1
    )
    
    start_test_time = time.time()
    test_search.fit(X_train, Y_train)
    end_test_time = time.time()
    
    time_per_fit = (end_test_time - start_test_time) / (5 * N_CV_FOLDS)
    estimated_total_time_seconds = time_per_fit * TOTAL_FITS
    
    print(f"[ESTIMATION] Time per single fit (CV fold) estimated at: {time_per_fit:.2f} seconds.")
    print(f"[ESTIMATION] Total estimated time for {TOTAL_FITS} fits: {estimated_total_time_seconds / 60:.2f} minutes.")
    print("--------------------------------------------------")


    # --- 3. Initialize Model and Randomized Search for full run ---
    base_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=N_ITERATIONS, scoring='r2', cv=N_CV_FOLDS, verbose=1, random_state=42, n_jobs=-1
    )

    print(f"\nStarting FULL Hyperparameter Tuning (Randomized Search) for {N_ITERATIONS} iterations (Total {TOTAL_FITS} fits)...")
    
    # --- 4. Run the Tuning ---
    start_time = time.time()
    random_search.fit(X_train, Y_train)
    end_time = time.time()

    tuning_duration = end_time - start_time
    print(f"\n[STATUS] Tuning complete in {tuning_duration:.2f} seconds ({tuning_duration / 60:.2f} minutes).")
    
    best_xgb_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print("\n" + "="*50)
    print("              Best Parameters Found")
    print("="*50)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("="*50)

    # --- 5. Predict and Evaluate Tuned Model ---
    Y_pred = best_xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    metrics = {'r2': r2, 'rmse': rmse, 'mae': mae}
    
    print("\n" + "="*50)
    print("          Tuned XGBoost Evaluation Results")
    print("="*50)
    print(f"R-squared Score (R²):           {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} EUR")
    print(f"Mean Absolute Error (MAE):      {mae:.4f} EUR")
    print("="*50)

    # --- 6. Save Results ---
    save_results_to_markdown(best_params, metrics, RESULTS_FILE)

    return best_xgb_model, r2

def feature_importance_analysis(xgb_model, preprocessor):
    """
    Extracts, maps, and prints the top 20 feature importances from the XGBoost model.
    """
    print("\n" + "="*50)
    print("      Starting Feature Importance Analysis")
    print("="*50)
    
    # 1. Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()
    
    # 2. Get importance scores
    importances = xgb_model.feature_importances_
    
    # 3. Create a DataFrame for easy sorting and display
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Clean up feature names (removes 'num__' or 'cat__' prefix)
    feature_df['Feature'] = feature_df['Feature'].str.split('__').str[-1]
    
    # 4. Sort and display top 20 features
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    
    print("Top 20 Features Influencing Market Value Prediction:")
    print(feature_df.head(20).to_markdown(index=False))
    print("="*50)

# --- NEW FUNCTION TO SAVE ASSETS ---
def save_model_assets(xgb_model, preprocessor, model_file, preprocessor_file):
    """Saves the trained model and the preprocessor using joblib."""
    
    print("\n" + "="*50)
    print("           Saving Model Assets for API")
    print("="*50)
    
    # 1. Save the Model
    try:
        joblib.dump(xgb_model, model_file)
        print(f"[SUCCESS] XGBoost Model saved to: {model_file}")
    except Exception as e:
        print(f"[ERROR] Could not save model: {e}")
        return
        
    # 2. Save the Preprocessor
    try:
        joblib.dump(preprocessor, preprocessor_file)
        print(f"[SUCCESS] Preprocessor saved to: {preprocessor_file}")
    except Exception as e:
        print(f"[ERROR] Could not save preprocessor: {e}")
        return
        
    print("The files above are mandatory for your Flask/FastAPI backend.")
    print("="*50)


# --- Main Execution ---
if __name__ == "__main__":
    FILE_PATH = 'master_list_cleaned.csv'

    # 1. Load Data
    data = load_data(FILE_PATH)
    if data is None:
        exit()

    # 2. Preprocess Data and Get Features/Target
    try:
        X_final, Y_final, preprocessor, scaler = preprocess_data(data) 
    except Exception as e:
        print(f"\nExiting due to critical processing error: {e}")
        exit()

    # 3. Train and Evaluate XGBoost
    xgb_model, final_r2 = train_and_evaluate_xgboost(X_final, Y_final)
    
    # 4. Feature Importance Analysis
    feature_importance_analysis(xgb_model, preprocessor)
    
    # 5. FINAL STEP: Save Model Assets for Deployment
    save_model_assets(xgb_model, preprocessor, MODEL_FILE, PREPROCESSOR_FILE)

    print("\nAnalysis and Asset Saving Complete!")
    print(f"You can now use '{MODEL_FILE}' and '{PREPROCESSOR_FILE}' for your back-end API.")
