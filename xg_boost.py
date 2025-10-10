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
import time
import joblib

# --- Configuration ---
TARGET_COLUMN = 'market_value_in_euros'

RESULTS_FILE = r'C:\Users\ghans\OneDrive\Desktop\ai_project_1\xgboost_tuned_results.md'
MODEL_FILE = 'xgb_model_final.joblib'
PREPROCESSOR_FILE = 'preprocessor_final.joblib'

# Columns to drop for memory or irrelevant
COLUMNS_TO_DROP = [
    'player', 'player_code', 'url', 'image_url', 'slug', 'player_id',
    'city_of_birth', 'all_injuries_details', 'agent_name', 'date_of_birth',
    'contract_expiration_date', 'text', 'most_severe_injury'
]

# --- Load Data ---
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        print(f"Data loaded successfully: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- Preprocess Data ---
def preprocess_data(df):
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    # Drop irrelevant columns
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Separate X and Y
    Y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Identify feature types
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=50, sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ], remainder='passthrough')

    X_processed = preprocessor.fit_transform(X)
    print(f"Processed feature matrix: {X_processed.shape}")

    return X_processed, Y, preprocessor

# --- Save results ---
def save_results_to_markdown(best_params, metrics, output_file):
    content = f"""# XGBoost Hyperparameter Tuning Results
## Best Parameters
{best_params}

## Evaluation Metrics
R2: {metrics['r2']:.4f}
RMSE: {metrics['rmse']:.4f}
MAE: {metrics['mae']:.4f}
"""
    with open(output_file, 'w') as f:
        f.write(content)
    print(f"Results saved to {output_file}")

# --- Train XGBoost ---
def train_xgboost(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'colsample_bytree': uniform(0.7, 0.3),
        'subsample': uniform(0.7, 0.3),
        'gamma': uniform(0, 0.5),
    }

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(model, param_distributions=param_dist,
                                n_iter=30, cv=5, scoring='r2', verbose=1, random_state=42, n_jobs=-1)
    search.fit(X_train, Y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    # Evaluate
    Y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"Evaluation Metrics: {metrics}")

    # Save results
    save_results_to_markdown(best_params, metrics, RESULTS_FILE)
    return best_model

# --- Save Model and Preprocessor ---
def save_assets(model, preprocessor):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print(f"Model saved to {MODEL_FILE}, preprocessor saved to {PREPROCESSOR_FILE}")

# --- Main ---
if __name__ == "__main__":
    FILE_PATH = 'master_list.csv'
    df = load_data(FILE_PATH)
    if df is not None:
        X, Y, preprocessor = preprocess_data(df)
        model = train_xgboost(X, Y)
        save_assets(model, preprocessor)
