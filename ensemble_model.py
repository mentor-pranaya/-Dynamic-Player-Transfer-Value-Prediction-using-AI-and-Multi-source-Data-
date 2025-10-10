import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import warnings
import re

# --- Set Random Seeds for Reproducibility ---
np.random.seed(42)
tf.random.set_seed(42)
# ---------------------------------------------

# --- 1. Configuration ---
# File paths for models and data
XGB_MODEL_PATH = 'player_value_model.joblib'
LSTM_MODEL_PATH = 'multivariate_lstm_model.h5' # The single-step multivariate LSTM model
DATA_21_22_PATH = '2021_22_features_with_prior_injuries_and_sentiment.csv'
DATA_22_23_PATH = '2022_23_features_with_prior_injuries_and_sentiment.csv'
DATA_23_24_PATH = '2023_24_features_with_prior_injuries_and_sentiment.csv'
PREDICT_LIST_PATH = 'market_values_24_25.csv'

# Define the features used by each model
XGB_FEATURES = [
    'age', 'mp', 'starts', 'min', 'pos', 'gls_90', 'ast_90', 'xg_90', 'xag_90',
    'prog_carries', 'prog_passes', 'prior_injury_count', 'total_days_missed_prior', 'compound'
]
LSTM_FEATURES = [
    'age', 'mp', 'starts', 'min', 'gls_90', 'ast_90', 'xg_90', 'xag_90',
    'prog_carries', 'prog_passes', 'prior_injury_count', 'total_days_missed_prior',
    'compound', 'market_value_eur'
]
N_STEPS = 2 # The LSTM model was trained on 2 historical steps

# --- 2. Data Cleaning and Loading ---

def _clean_injury_days(value):
    """Cleans the injury days column by summing all numbers found in the string."""
    if isinstance(value, str):
        numbers = re.findall(r'\d+', value)
        return sum(int(num) for num in numbers)
    return value

def _standardize_columns(df, season_name):
    """Standardizes column names and calculates per-90 stats if they don't exist."""
    df = df.copy()
    rename_map = {'pasprog': 'prog_passes', 'carprog': 'prog_carries'}
    df.rename(columns=rename_map, inplace=True)
    if 'total_days_missed_prior' in df.columns:
        df['total_days_missed_prior'] = df['total_days_missed_prior'].apply(_clean_injury_days).fillna(0)
    per_90_stats = {'gls_90': 'goals', 'ast_90': 'assists', 'xg_90': 'xg', 'xag_90': 'xag'}
    for per_90_col, base_col in per_90_stats.items():
        if per_90_col not in df.columns:
            if base_col in df.columns and '90s' in df.columns and df['90s'].dtype in ['int64', 'float64']:
                 df[per_90_col] = np.divide(df[base_col], df['90s'], out=np.zeros_like(df[base_col], dtype=float), where=(df['90s']!=0))
            else:
                df[per_90_col] = 0
    return df

def load_all_data():
    """Loads and standardizes all seasonal data files."""
    try:
        data = {
            '21_22': _standardize_columns(pd.read_csv(DATA_21_22_PATH), '21-22'),
            '22_23': _standardize_columns(pd.read_csv(DATA_22_23_PATH), '22-23'),
            '23_24': _standardize_columns(pd.read_csv(DATA_23_24_PATH), '23-24'),
            'predict_list': pd.read_csv(PREDICT_LIST_PATH)
        }
        print("All data files loaded and standardized successfully.")
        return data
    except FileNotFoundError as e:
        print(f"Error loading files: {e}.")
        return None

# --- 3. Prediction Functions ---

def predict_with_xgb(data_dict, model):
    """Generates predictions using the static XGBoost model."""
    print("\nGenerating predictions with XGBoost model...")
    df_23 = data_dict['23_24'].copy()
    
    # Ensure all required features are present
    for feature in XGB_FEATURES:
        if feature not in df_23.columns:
            df_23[feature] = 0
            
    X_predict = df_23[XGB_FEATURES].fillna(0)
    
    predictions_log = model.predict(X_predict)
    predictions_eur = np.expm1(predictions_log)
    
    results = df_23[['player_id', 'player_name']].copy()
    results['xgb_prediction'] = predictions_eur
    return results

def predict_with_lstm(data_dict, model):
    """Generates predictions using the sequential LSTM model."""
    print("Generating predictions with LSTM model...")
    # Combine historical data to find players with complete history
    df_21 = data_dict['21_22']
    df_22 = data_dict['22_23']
    df_23 = data_dict['23_24']

    for df in [df_21, df_22, df_23]:
        for col in LSTM_FEATURES:
            if col not in df.columns:
                df[col] = 0

    df_21['season'] = 0
    df_22['season'] = 1
    df_23['season'] = 2

    combined_df = pd.concat([
        df_21[['player_id', 'player_name', 'season'] + LSTM_FEATURES],
        df_22[['player_id', 'player_name', 'season'] + LSTM_FEATURES],
        df_23[['player_id', 'player_name', 'season'] + LSTM_FEATURES]
    ]).dropna(subset=LSTM_FEATURES)
    
    player_counts = combined_df['player_id'].value_counts()
    players_with_history = player_counts[player_counts >= N_STEPS].index
    
    df_predict_sequences = combined_df[combined_df['player_id'].isin(players_with_history)]
    df_predict_sequences = df_predict_sequences.sort_values(['player_id', 'season'])

    player_groups = df_predict_sequences.groupby('player_id')
    
    # We need a scaler. It's best practice to save and load it, but here we refit it
    # on the entire available historical dataset for prediction purposes.
    scaler = StandardScaler()
    scaler.fit(df_predict_sequences[LSTM_FEATURES])
    
    predictions = {}
    for player_id, group in player_groups:
        if len(group) >= N_STEPS:
            # Get the last N_STEPS of data for prediction
            input_data = group[LSTM_FEATURES].iloc[-N_STEPS:].values
            input_data_scaled = scaler.transform(input_data)
            
            # FIX: Define N_FEATURES within the function's scope
            N_FEATURES = len(LSTM_FEATURES)
            input_data_reshaped = input_data_scaled.reshape((1, N_STEPS, N_FEATURES))
            
            # Predict the log-scaled value
            pred_log = model.predict(input_data_reshaped, verbose=0)[0][0]
            
            # Inverse transform to get the value in Euros
            pred_eur = np.expm1(pred_log)
            predictions[player_id] = pred_eur
            
    results = pd.DataFrame(list(predictions.items()), columns=['player_id', 'lstm_prediction'])
    return results

# --- 4. Main Execution ---
def main():
    """Main function to run the entire ensemble pipeline."""
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Load all data
    data_dict = load_all_data()
    if data_dict is None:
        return

    # Load trained models
    try:
        xgb_model = joblib.load(XGB_MODEL_PATH)
        lstm_model = load_model(LSTM_MODEL_PATH)
        print("\nAll models loaded successfully.")
    except Exception as e:
        print(f"\nError loading models: {e}. Please ensure model files are in the directory.")
        return

    # Generate predictions from both models
    xgb_predictions = predict_with_xgb(data_dict, xgb_model)
    lstm_predictions = predict_with_lstm(data_dict, lstm_model)

    # --- Ensemble the Predictions ---
    print("\nEnsembling predictions...")
    # Merge results, keeping only players predicted by both models
    ensemble_df = pd.merge(xgb_predictions, lstm_predictions, on='player_id', how='inner')
    
    # A simple average is a robust ensemble method.
    # You could experiment with weighted averages, e.g., 0.6 * xgb + 0.4 * lstm
    ensemble_df['ensemble_prediction'] = (ensemble_df['xgb_prediction'] + ensemble_df['lstm_prediction']) / 2
    
    # Merge with the list of players for the 24/25 season for comparison
    predict_list = data_dict['predict_list'][['player_id', 'market_value_eur']].rename(columns={'market_value_eur': 'actual_market_value_24_25'})
    ensemble_df = pd.merge(ensemble_df, predict_list, on='player_id', how='left')

    # Sort by the final ensemble prediction
    ensemble_df = ensemble_df.sort_values('ensemble_prediction', ascending=False)
    
    # --- Display Final Results ---
    print("\n--- Top 20 Player Market Value Forecasts (Ensemble Model) ---")
    
    display_cols = ['player_name', 'xgb_prediction', 'lstm_prediction', 'ensemble_prediction', 'actual_market_value_24_25']
    
    # Formatting for display
    formatters = {
        'xgb_prediction': '€{:,.0f}'.format,
        'lstm_prediction': '€{:,.0f}'.format,
        'ensemble_prediction': '€{:,.0f}'.format,
        'actual_market_value_24_25': '€{:,.0f}'.format
    }
    
    print(ensemble_df[display_cols].head(20).to_string(formatters=formatters))
    print("----------------------------------------------------------------------")


if __name__ == '__main__':
    main()

