import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Player Market Value Predictor",
    page_icon="⚽",
    layout="wide"
)

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings("ignore", category=FutureWarning)


# --- 1. Model and Data Loading (with Streamlit Caching) ---

@st.cache_resource
def load_models():
    """Loads all trained models into memory."""
    try:
        xgb_model = joblib.load('player_value_model.joblib')
        uni_lstm_model = load_model('univariate_lstm_model.h5')
        multi_lstm_model = load_model('multivariate_lstm_model.h5')
        return {
            "XGBoost": xgb_model,
            "Univariate LSTM": uni_lstm_model,
            "Multivariate LSTM": multi_lstm_model
        }
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure all model files are in the directory.")
        return None

@st.cache_data
def load_all_data():
    """Loads and standardizes all seasonal data files."""
    try:
        data = {
            '21_22': pd.read_csv('2021_22_features_with_prior_injuries_and_sentiment.csv'),
            '22_23': pd.read_csv('2022_23_features_with_prior_injuries_and_sentiment.csv'),
            '23_24': pd.read_csv('2023_24_features_with_prior_injuries_and_sentiment.csv'),
            'predict_list': pd.read_csv('market_values_24_25.csv')
        }
        for season in ['21_22', '22_23', '23_24']:
            data[season] = _standardize_columns(data[season])
        return data
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}. Please ensure all CSVs are in the directory.")
        return None

# --- 2. Data Cleaning and Standardization Helpers ---

def _clean_injury_days(value):
    if isinstance(value, str):
        numbers = re.findall(r'\d+', value)
        return sum(int(num) for num in numbers)
    return value

def _standardize_columns(df):
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

# --- 3. Prediction Functions ---

def predict_xgb(player_id, data_dict, model):
    df_23 = data_dict['23_24']
    player_data = df_23[df_23['player_id'] == player_id].head(1) # Ensure single row
    if player_data.empty:
        return None
    
    features = ['age', 'mp', 'starts', 'min', 'pos', 'gls_90', 'ast_90', 'xg_90', 'xag_90',
                'prog_carries', 'prog_passes', 'prior_injury_count', 'total_days_missed_prior', 'compound']
    
    for feature in features:
        if feature not in player_data.columns:
            player_data[feature] = 0

    X_predict = player_data[features].fillna(0)
    pred_log = model.predict(X_predict)
    pred_log = np.clip(pred_log, a_min=None, a_max=22.0) # Clip to prevent overflow
    return np.expm1(pred_log)[0]

def predict_univariate_lstm(player_id, data_dict, model):
    df_22 = data_dict['22_23']
    df_23 = data_dict['23_24']
    
    val_22_series = df_22[df_22['player_id'] == player_id]['market_value_eur']
    val_23_series = df_23[df_23['player_id'] == player_id]['market_value_eur']

    if val_22_series.empty or val_23_series.empty:
        return None
        
    input_data = np.array([val_22_series.iloc[0], val_23_series.iloc[0]], dtype=np.float32).reshape(-1, 1)
    input_log = np.log1p(input_data)

    scaler = MinMaxScaler()
    input_scaled = scaler.fit_transform(input_log)
    
    input_reshaped = input_scaled.reshape((1, 2, 1))

    pred_scaled = model.predict(input_reshaped, verbose=0)
    
    pred_log = scaler.inverse_transform(pred_scaled)
    pred_log = np.clip(pred_log, a_min=None, a_max=22.0) # Clip to prevent overflow
    
    return np.expm1(pred_log)[0][0]

def predict_multivariate_lstm(player_id, data_dict, model):
    features = ['age', 'mp', 'starts', 'min', 'gls_90', 'ast_90', 'xg_90', 'xag_90',
                'prog_carries', 'prog_passes', 'prior_injury_count', 'total_days_missed_prior',
                'compound', 'market_value_eur']
    
    df_22 = data_dict['22_23']
    df_23 = data_dict['23_24']

    # FIX: Use .head(1) to prevent errors from duplicate player entries in a season
    player_22 = df_22[df_22['player_id'] == player_id].head(1)
    player_23 = df_23[df_23['player_id'] == player_id].head(1)

    if player_22.empty or player_23.empty:
        return None
        
    for df in [player_22, player_23]:
        for col in features:
            if col not in df.columns:
                df[col] = 0

    input_data = pd.concat([player_22[features], player_23[features]], ignore_index=True).fillna(0).values
    
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)
    # The input_scaled array will now correctly have shape (2, 14)
    input_reshaped = input_scaled.reshape((1, 2, len(features)))

    pred_log_scaled = model.predict(input_reshaped, verbose=0)

    dummy_array = np.zeros((1, len(features)))
    dummy_array[0, -1] = pred_log_scaled[0, 0]
    inversed_dummy = scaler.inverse_transform(dummy_array)
    pred_log = inversed_dummy[0, -1]

    pred_log = np.clip(pred_log, a_min=None, a_max=22.0)

    return np.expm1(pred_log)

# --- 4. Streamlit App UI ---

st.title("⚽ Player Market Value Predictor")
st.markdown("Select players to forecast their market value for the 2024-25 season using multiple predictive models.")

# Load all necessary assets
models = load_models()
data = load_all_data()

if models and data:
    player_list_df = data['predict_list'].dropna(subset=['player_name', 'player_id'])
    player_list_df['display_name'] = player_list_df['player_name'] + " (" + player_list_df['player_id'].astype(str) + ")"
    player_options = player_list_df.set_index('player_id')['display_name'].to_dict()
    
    selected_ids = st.multiselect(
        "Choose players:",
        options=player_list_df['player_id'].unique(),
        format_func=lambda x: player_options.get(x, "Unknown Player")
    )

    if st.button("📈 Predict Market Values", type="primary"):
        if not selected_ids:
            st.warning("Please select at least one player.")
        else:
            with st.spinner("Running predictions..."):
                for player_id in selected_ids:
                    player_name = player_list_df[player_list_df['player_id'] == player_id]['player_name'].iloc[0]
                    
                    st.markdown(f"---")
                    st.header(f"Results for: {player_name}")

                    # Generate predictions
                    predictions = {
                        "XGBoost (Static)": predict_xgb(player_id, data, models["XGBoost"]),
                        "Univariate LSTM (Value Trend)": predict_univariate_lstm(player_id, data, models["Univariate LSTM"]),
                        "Multivariate LSTM (Performance Trend)": predict_multivariate_lstm(player_id, data, models["Multivariate LSTM"])
                    }
                    
                    # Filter out failed predictions and calculate ensemble
                    valid_preds = {k: v for k, v in predictions.items() if v is not None and np.isfinite(v)}
                    if valid_preds:
                        ensemble_value = np.mean(list(valid_preds.values()))
                        valid_preds["Ensemble (Average)"] = ensemble_value
                    else:
                        st.error("Could not generate any predictions for this player due to insufficient data.")
                        continue
                    
                    # Display results in columns
                    cols = st.columns(len(valid_preds))
                    
                    # Display Predictions
                    i = 0
                    for name, pred in valid_preds.items():
                        with cols[i]:
                            st.metric(
                                label=name,
                                value=f"€{pred:,.0f}"
                            )
                        i += 1
else:
    st.error("App could not start. Please check that all required data and model files are in the same directory.")
