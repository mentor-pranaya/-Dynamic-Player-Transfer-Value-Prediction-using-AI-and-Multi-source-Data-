import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import re

# --- 1. Configuration ---
# File paths
DATA_21_22_PATH = '2021_22_features_with_prior_injuries_and_sentiment.csv'
DATA_22_23_PATH = '2022_23_features_with_prior_injuries_and_sentiment.csv'
DATA_23_24_PATH = '2023_24_features_with_prior_injuries_and_sentiment.csv'
MODEL_SAVE_PATH = 'multivariate_lstm_model.h5'

# Model parameters
N_STEPS = 2  # Use 2 past seasons to predict the next one
# Features to be used by the model
FEATURES = [
    'age', 'mp', 'starts', 'min', 'gls_90', 'ast_90', 'xg_90', 'xag_90',
    'prog_carries', 'prog_passes', 'prior_injury_count', 'total_days_missed_prior',
    'compound', 'market_value_eur'
]
N_FEATURES = len(FEATURES) # This is a multivariate model

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

def load_and_prepare_data():
    """Loads, cleans, and prepares sequential data for all players."""
    try:
        df_21 = pd.read_csv(DATA_21_22_PATH)
        df_22 = pd.read_csv(DATA_22_23_PATH)
        df_23 = pd.read_csv(DATA_23_24_PATH)
        print("All seasonal data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}.")
        return None

    # Standardize columns for all dataframes
    df_21 = _standardize_columns(df_21, '21-22')
    df_22 = _standardize_columns(df_22, '22-23')
    df_23 = _standardize_columns(df_23, '23-24')

    # Ensure all required feature columns exist in all dataframes
    for df in [df_21, df_22, df_23]:
        for col in FEATURES + ['player_id', 'player_name']:
            if col not in df.columns:
                df[col] = 0 # Add missing columns and fill with 0

    df_21['season'] = 0
    df_22['season'] = 1
    df_23['season'] = 2

    combined_df = pd.concat([
        df_21[['player_id', 'player_name', 'season'] + FEATURES],
        df_22[['player_id', 'player_name', 'season'] + FEATURES],
        df_23[['player_id', 'player_name', 'season'] + FEATURES]
    ]).dropna(subset=FEATURES)

    player_counts = combined_df['player_id'].value_counts()
    players_with_3_seasons = player_counts[player_counts >= 3].index
    
    df_sequences = combined_df[combined_df['player_id'].isin(players_with_3_seasons)]
    df_sequences = df_sequences.sort_values(['player_id', 'season'])
    
    print(f"Found {len(players_with_3_seasons)} players with complete 3-year history.")
    return df_sequences

# --- 3. Sequence Creation and Scaling ---
def create_sequences_and_scale(df):
    """Creates input sequences (X) and target values (y) and scales them."""
    X, y = [], []
    player_groups = df.groupby('player_id')
    
    for _, group in player_groups:
        # Ensure we have at least N_STEPS + 1 records to create a sequence
        if len(group) >= N_STEPS + 1:
            # The first N_STEPS are the input features
            sequence = group[FEATURES].iloc[:N_STEPS].values
            # The target is the market value of the following season
            target = group['market_value_eur'].iloc[N_STEPS]
            X.append(sequence)
            y.append(target)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    # Log transform the target variable for better model performance
    y_log = np.log1p(y)

    # Split data before scaling to prevent data leakage
    X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    # Scale features based on the training data ONLY
    scaler = StandardScaler()
    # Reshape X_train to 2D for scaler, fit and transform, then reshape back to 3D
    X_train_reshaped = X_train.reshape(-1, N_FEATURES)
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    
    # Transform validation data using the SAME scaler
    X_val_reshaped = X_val.reshape(-1, N_FEATURES)
    X_val_scaled_reshaped = scaler.transform(X_val_reshaped)
    X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape)

    print(f"Data prepared for LSTM. Training shape: {X_train_scaled.shape}, Validation shape: {X_val_scaled.shape}")
    return X_train_scaled, y_train, X_val_scaled, y_val, scaler

# --- 4. Build and Train LSTM Model ---
def build_and_train_model(X_train, y_train, X_val, y_val):
    """Builds, compiles, and trains the multivariate LSTM model."""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(N_STEPS, N_FEATURES), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("\n--- Model Summary ---")
    model.summary()
    print("---------------------\n")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Starting model training...")
    model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    print("Model training complete.")
    return model

# --- 5. Main Execution ---
def main():
    """Main function to run the entire pipeline."""
    df = load_and_prepare_data()
    if df is None or df.empty:
        return
        
    X_train, y_train, X_val, y_val, scaler = create_sequences_and_scale(df)
    model = build_and_train_model(X_train, y_train, X_val, y_val)
    
    # --- Save the Trained Model ---
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")
    
    # --- Evaluation and Visualization ---
    print("\nEvaluating model on validation data...")
    y_pred_log = model.predict(X_val)
    
    # Inverse transform to get values in Euros
    y_pred_eur = np.expm1(y_pred_log)
    y_val_eur = np.expm1(y_val)
    
    # Calculate and print performance metrics
    r2 = r2_score(y_val_eur, y_pred_eur)
    mae = mean_absolute_error(y_val_eur, y_pred_eur)
    rmse = np.sqrt(mean_squared_error(y_val_eur, y_pred_eur))

    print("\n--- Multivariate Model Performance Metrics ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): €{mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): €{rmse:,.2f}")
    print("--------------------------------------------\n")
    
    # --- Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=y_val_eur.flatten(), y=y_pred_eur.flatten(), alpha=0.6)
    ax.plot([y_val_eur.min(), y_val_eur.max()], [y_val_eur.min(), y_val_eur.max()], '--', color='red', lw=2)
    ax.set_title('Actual vs. Predicted Market Value (Multivariate LSTM)', fontsize=16, weight='bold')
    ax.set_xlabel('Actual Market Value (€)', fontsize=12)
    ax.set_ylabel('Predicted Market Value (€)', fontsize=12)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"€{int(x/1e6)}M"))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"€{int(x/1e6)}M"))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()