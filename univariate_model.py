import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Configuration ---
# File paths for the data
DATA_21_22_PATH = '2021_22_features_with_prior_injuries_and_sentiment.csv'
DATA_22_23_PATH = '2022_23_features_with_prior_injuries_and_sentiment.csv'
DATA_23_24_PATH = '2023_24_features_with_prior_injuries_and_sentiment.csv'
PREDICT_LIST_PATH = 'market_values_24_25.csv'
MODEL_SAVE_PATH = 'univariate_lstm_model.h5' # Define the path to save the model

# Model parameters
N_STEPS = 2  # Number of past seasons to use for prediction (e.g., use 2 seasons to predict the 3rd)
N_FEATURES = 1 # This is a univariate model

# --- 2. Data Loading and Consolidation ---
def load_and_merge_values():
    """Loads market values from all seasons and merges them into a single DataFrame."""
    try:
        df_21 = pd.read_csv(DATA_21_22_PATH, usecols=['player_id', 'player_name', 'market_value_eur'])
        df_22 = pd.read_csv(DATA_22_23_PATH, usecols=['player_id', 'player_name', 'market_value_eur'])
        df_23 = pd.read_csv(DATA_23_24_PATH, usecols=['player_id', 'player_name', 'market_value_eur'])
        print("Seasonal data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}.")
        return None
    except ValueError:
         # Fallback if a file doesn't have the required columns
        print("Warning: A file was missing required columns. Trying to proceed.")
        # Create empty dataframes if any file is problematic
        df_21 = pd.DataFrame(columns=['player_id', 'player_name', 'market_value_eur'])
        df_22 = pd.DataFrame(columns=['player_id', 'player_name', 'market_value_eur'])
        df_23 = pd.DataFrame(columns=['player_id', 'player_name', 'market_value_eur'])


    # Rename columns for merging
    df_21.rename(columns={'market_value_eur': 'mv_21_22'}, inplace=True)
    df_22.rename(columns={'market_value_eur': 'mv_22_23'}, inplace=True)
    df_23.rename(columns={'market_value_eur': 'mv_23_24'}, inplace=True)
    
    # Merge the dataframes
    merged_df = pd.merge(df_21, df_22, on=['player_id', 'player_name'], how='outer')
    merged_df = pd.merge(merged_df, df_23, on=['player_id', 'player_name'], how='outer')
    
    # Drop rows where we don't have enough data to form a sequence
    merged_df.dropna(subset=['mv_21_22', 'mv_22_23', 'mv_23_24'], inplace=True)
    print(f"Merged data successfully. Found {len(merged_df)} players with complete 3-year market value history.")
    
    return merged_df

# --- 3. Data Preparation for LSTM ---
def prepare_sequences(df):
    """Creates input sequences and target values for the LSTM model."""
    # Using 21-22 and 22-23 values to predict 23-24 value
    X_raw = df[['mv_21_22', 'mv_22_23']].values
    y_raw = df['mv_23_24'].values.reshape(-1, 1)

    # Log transform to handle skewed data and improve model performance
    X_log = np.log1p(X_raw)
    y_log = np.log1p(y_raw)

    # Scale data to the range [0, 1]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X_log)
    y_scaled = scaler_y.fit_transform(y_log)
    
    # Reshape X into the 3D format required by LSTMs: [samples, timesteps, features]
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], N_STEPS, N_FEATURES))
    
    print(f"Data prepared for LSTM. Input shape: {X_reshaped.shape}, Target shape: {y_scaled.shape}")
    return X_reshaped, y_scaled, scaler_y

# --- 4. Build and Train LSTM Model ---
def build_and_train_model(X_train, y_train, X_val, y_val):
    """Builds, compiles, and trains the LSTM model."""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(N_STEPS, N_FEATURES), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("\n--- Model Summary ---")
    model.summary()
    print("---------------------\n")
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Starting model training...")
    history = model.fit(
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
    df = load_and_merge_values()
    if df is None or df.empty:
        print("Could not proceed due to data loading issues.")
        return
        
    X, y, scaler_y = prepare_sequences(df)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_and_train_model(X_train, y_train, X_val, y_val)
    
    # --- Save the Trained Model ---
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")
    # -----------------------------
    
    # --- Evaluation and Visualization ---
    print("\nEvaluating model on validation data...")
    y_pred_scaled = model.predict(X_val)
    
    # Inverse transform predictions and true values to get them back to Euros
    y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
    y_val_log = scaler_y.inverse_transform(y_val)
    
    y_pred_eur = np.expm1(y_pred_log)
    y_val_eur = np.expm1(y_val_log)
    
    # Calculate and print performance metrics
    r2 = r2_score(y_val_eur, y_pred_eur)
    mae = mean_absolute_error(y_val_eur, y_pred_eur)
    rmse = np.sqrt(mean_squared_error(y_val_eur, y_pred_eur))

    print("\n--- Univariate Model Performance Metrics ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): €{mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): €{rmse:,.2f}")
    print("------------------------------------------\n")
    
    # --- Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=y_val_eur.flatten(), y=y_pred_eur.flatten(), alpha=0.6, ax=ax)
    ax.plot([y_val_eur.min(), y_val_eur.max()], [y_val_eur.min(), y_val_eur.max()], '--', color='red', lw=2)
    ax.set_title('Actual vs. Predicted Market Value (Univariate LSTM)', fontsize=16, weight='bold')
    ax.set_xlabel('Actual Market Value (€)', fontsize=12)
    ax.set_ylabel('Predicted Market Value (€)', fontsize=12)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"€{int(x/1e6)}M"))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"€{int(x/1e6)}M"))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()