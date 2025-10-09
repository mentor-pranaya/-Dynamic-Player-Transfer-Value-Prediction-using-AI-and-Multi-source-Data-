import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
TARGET_COLUMN = 'market_value_in_eur'

# Columns to drop to prevent memory overflow and irrelevant features
COLUMNS_TO_DROP = [
    'player',                 
    'player_code',            
    'url',                    
    'image_url',              
    'slug',                   
    'player_id',              
    'city_of_birth',          
    'all_injuries_details',   
    'agent_name',             
    'date_of_birth',          
    'contract_expiration_date',
    'text',                   
    'most_severe_injury'      
]

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
    Separates the features (X) from the target (Y).
    """

    # --- 1. CRITICAL: Clean Target Column ---
    # Convert non-numeric values (like 'Unknown') to NaN, then drop rows with missing target.
    initial_shape = df.shape[0]
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], 
        errors='coerce' # This converts 'Unknown' (and any other strings) to NaN
    )
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

    # Pipeline for numerical features: Impute with median, then scale
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features: Impute with constant 'missing', then one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
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
    print(f"Final feature matrix shape after encoding: {X_processed.shape}")

    total_features = X_processed.shape[1]
    print(f"Total features created: {total_features}")

    return X_processed, Y, preprocessor, numerical_pipeline['scaler'] 


def train_and_evaluate_xgboost(X, Y):
    """Splits data, trains XGBoost Regressor, and evaluates performance."""

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"\nData split complete. Train set size: {X_train.shape}, Test set size: {X_test.shape}")

    # --- 1. Initialize XGBoost Regressor ---
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    print("XGBoost model initialized. Starting training...")

    # --- 2. Train the Model ---
    xgb_model.fit(X_train, Y_train)
    print("Model training complete.")

    # --- 3. Predict and Evaluate ---
    Y_pred = xgb_model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print("\n" + "="*50)
    print("          XGBoost Model Evaluation Results")
    print("="*50)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} EUR")
    print(f"Mean Absolute Error (MAE):      {mae:.4f} EUR")
    print(f"R-squared Score (R²):           {r2:.4f}")
    print("="*50)

    return xgb_model, r2

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

    print("\nReady for Milestone 6: Hyperparameter Tuning!")
