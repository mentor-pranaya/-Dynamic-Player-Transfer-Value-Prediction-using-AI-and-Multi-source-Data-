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
import joblib

# --- Configuration ---
TARGET_COLUMN = 'market value'  # ✅ your actual column
RESULTS_FILE = '/Users/veerababu/Downloads/xgboost_tuned_results.md'
MODEL_FILE = 'xgb_model_final.joblib'
PREPROCESSOR_FILE = 'preprocessor_final.joblib'

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

        # Standardize column names
        data.columns = data.columns.str.strip()

        # Diagnostic printout
        print("\n--- Diagnostic: Available Columns in DataFrame ---")
        print(list(data.columns))
        print("--------------------------------------------------")

        # ✅ Auto-rename variant column names
        if 'market_value_in_euros' in data.columns and 'market value' not in data.columns:
            data.rename(columns={'market_value_in_euros': 'market value'}, inplace=True)
        if 'Market Value' in data.columns and 'market value' not in data.columns:
            data.rename(columns={'Market Value': 'market value'}, inplace=True)

        if TARGET_COLUMN not in data.columns:
            raise KeyError(f"Target column '{TARGET_COLUMN}' not found in DataFrame. "
                           "Check the list of available columns above.")

        return data

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None


def preprocess_data(df):
    """Handles missing values, encoding, and scaling."""
    print("\n--- Preprocessing Started ---")

    # --- Clean Target Column ---
    initial_rows = df.shape[0]
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    retained_rows = df.shape[0]
    print(f"Cleaned target column. Dropped {initial_rows - retained_rows} invalid rows "
          f"({retained_rows}/{initial_rows} retained = {retained_rows/initial_rows:.2%}).")

    # --- Drop Irrelevant Columns ---
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns and col != TARGET_COLUMN]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Dropped {len(cols_to_drop)} irrelevant/high-cardinality columns.")

    # --- Separate Features and Target ---
    Y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # --- Identify Data Types ---
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numerical features: {len(numerical_features)}, Categorical features: {len(categorical_features)}")

    # --- Define Pipelines ---
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=50, sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, numerical_features),
        ('cat', cat_pipe, categorical_features)
    ], remainder='drop')

    print("Starting encoding and scaling...")
    X_processed = preprocessor.fit_transform(X)

    print(f"✅ Preprocessing complete. Final feature matrix shape: {X_processed.shape}")
    return X_processed, Y, preprocessor


def save_results_to_markdown(best_params, metrics, output_file):
    """Save model performance report."""
    content = f"""# XGBoost Model Hyperparameter Tuning Results

**Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Best Parameters
| Parameter | Value |
|:--|:--|
{'\n'.join([f'| {k} | {v} |' for k, v in best_params.items()])}

## Metrics
| Metric | Value |
|:--|:--|
| R² | {metrics['r2']:.4f} |
| RMSE | {metrics['rmse']:.4f} |
| MAE | {metrics['mae']:.4f} |
"""
    with open(output_file, 'w') as f:
        f.write(content)
    print(f"\n✅ Results saved to {output_file}")


def train_and_evaluate_xgboost(X, Y):
    """Train and tune XGBoost model."""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"\nData split done: Train={X_train.shape}, Test={X_test.shape}")

    param_dist = {
        'n_estimators': randint(100, 400),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'colsample_bytree': uniform(0.7, 0.3),
        'subsample': uniform(0.7, 0.3),
        'gamma': uniform(0, 0.5)
    }

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=25,
        scoring='r2',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    print("\n🔍 Starting XGBoost hyperparameter tuning...")
    start = time.time()
    search.fit(X_train, Y_train)
    duration = (time.time() - start) / 60
    print(f"Tuning completed in {duration:.2f} minutes.")

    best_model = search.best_estimator_
    Y_pred = best_model.predict(X_test)

    metrics = {
        'r2': r2_score(Y_test, Y_pred),
        'rmse': np.sqrt(mean_squared_error(Y_test, Y_pred)),
        'mae': mean_absolute_error(Y_test, Y_pred)
    }

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    save_results_to_markdown(search.best_params_, metrics, RESULTS_FILE)
    return best_model


def feature_importance_analysis(xgb_model, preprocessor):
    """Show top 20 most important features."""
    print("\n=== Feature Importance Analysis ===")
    feature_names = preprocessor.get_feature_names_out()
    importances = xgb_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df['Feature'] = feat_df['Feature'].str.split('__').str[-1]
    feat_df = feat_df.sort_values('Importance', ascending=False)
    print(feat_df.head(20).to_markdown(index=False))


def save_model_assets(model, preprocessor):
    """Save model + preprocessor for API deployment."""
    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print(f"\n✅ Model saved to {MODEL_FILE}")
    print(f"✅ Preprocessor saved to {PREPROCESSOR_FILE}")


# --- Main Execution ---
if __name__ == "__main__":
    FILE_PATH = 'master_list_cleaned.csv'

    df = load_data(FILE_PATH)
    if df is None:
        exit()

    try:
        X, Y, preprocessor = preprocess_data(df)
    except Exception as e:
        print(f"\n❌ Exiting due to error: {e}")
        exit()

    model = train_and_evaluate_xgboost(X, Y)
    feature_importance_analysis(model, preprocessor)
    save_model_assets(model, preprocessor)

    print("\n🎯 All steps completed successfully! Ready for API deployment.")
