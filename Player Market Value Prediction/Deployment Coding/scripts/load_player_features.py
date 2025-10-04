import pandas as pd
import numpy as np
import joblib
import os

# --- File Paths ---
model_path = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\Trained Models\best_lstm_model.h5"
scaler_path = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\Trained Models\week6_lstm_model.h5"
path_name_source = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\data\feature_engineered\player_features_model_all_imputed.csv"
path_features = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\data\cleaned_data\Player Features.csv"
path_sentiment = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\data\cleaned_data\Sentiment Analysis Dataset.csv"
output_path = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\Deployment\Player_Market_Value_Prediction_Dataset.csv"

# --- Load Model and Scaler ---
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# --- Load Player Names ---
df_names = pd.read_csv(path_name_source)[['player']].rename(columns={'player': 'Player Name'})
df_features = pd.read_csv(path_features)[['Name', 'Age', 'Position', 'Injury Status']].rename(columns={'Name': 'Player Name'})
df_sentiment = pd.read_csv(path_sentiment)[['Player Name', 'Sentiment Label', 'Sentiment Score (0–1)']]

# --- Normalize Player Names ---
for df in [df_names, df_features, df_sentiment]:
    df['Player Name'] = df['Player Name'].str.lower().str.replace(' ', '', regex=False)

players = df_names['Player Name'].dropna().unique()

# --- Prediction Loop ---
records = []
for player in players:
    try:
        # --- Merge Features for Player ---
        f_row = df_features[df_features['Player Name'] == player].reset_index(drop=True)
        s_row = df_sentiment[df_sentiment['Player Name'] == player].reset_index(drop=True)
        if f_row.empty and s_row.empty:
            print(f"⚠️ Skipping {player}: No feature or sentiment data found")
            continue

        merged = pd.concat([f_row, s_row], axis=1)

        expected_features = [
            'Age', 'Position', 'Injury Status', 'Sentiment Label', 'Sentiment Score (0–1)',
            'shots', 'goals', 'passes_total', 'passes_completed', 'pass_accuracy',
            'assists', 'matches_played', 'injury_count', 'avg_days_missed',
            'max_days_missed', 'injuries_last_180d', 'avg_sentiment',
            'fee_million', 'is_free_transfer', 'high_value_flag',
            'expected_goals', 'total_days_out', 'days_per_injury',
            'avg_market_value', 'market_missing_flag',
            'xG', 'passes_attempted'
        ]

        feature_dict = {col: merged[col].iloc[0] if col in merged.columns and not merged[col].isna().all() else 0.0 for col in expected_features}
        features_df = pd.DataFrame([feature_dict])

        # --- Scale and Predict ---
        scaled = scaler.transform(features_df)
        preds = model.predict(scaled)

        lstm_preds = preds[0]
        ensemble_preds = preds[0]  # Replace with actual ensemble logic if needed

        lstm_market_value = scaler.inverse_transform([[lstm_preds]])[0][0]
        ensemble_market_value = scaler.inverse_transform([[ensemble_preds]])[0][0]
        market_value = round((lstm_market_value + ensemble_market_value) / 2, 2)

        records.append({
            'Player Name': player,
            'y_test': market_value,
            'lstm_preds': lstm_preds,
            'ensemble_preds': ensemble_preds,
            'lstm_market_value': lstm_market_value,
            'ensemble_market_value': ensemble_market_value,
            'Market Value (M)': market_value
        })

    except Exception as e:
        print(f"⚠️ Skipping {player}: {e}")

# --- Save Final Dataset ---
df_final = pd.DataFrame(records)
df_final.to_csv(output_path, index=False)
print("✅ Final prediction dataset saved to:", output_path)