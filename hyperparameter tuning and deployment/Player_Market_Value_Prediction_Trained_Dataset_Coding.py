import pandas as pd
import numpy as n

# --- File Paths ---
path_name_source = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\data\feature_engineered\player_features_model_all_imputed.csv"
path_features = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\data\cleaned_data\Player Features.csv"
path_sentiment = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\data\cleaned_data\Sentiment Analysis Dataset.csv"
path_predictions = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\CSV Files\week6_predictions.csv"
output_path = r"C:\Users\M.ANTONY ROJES\Downloads\Infosys\Deployment\Player_Market_Value_Prediction_Dataset.csv"

# --- Load Datasets ---
df_name = pd.read_csv(path_name_source)[['player']].rename(columns={'player': 'Player Name'})
df_features = pd.read_csv(path_features)[['Name', 'Age', 'Position', 'Injury Status']].rename(columns={'Name': 'Player Name'})
df_sentiment = pd.read_csv(path_sentiment)[['Player Name', 'Sentiment']].rename(columns={'Sentiment': 'Sentiment Label'})
df_preds = pd.read_csv(path_predictions)[[
    'Market Value (M)', 'y_test', 'lstm_preds', 'ensemble_preds',
    'lstm_market_value', 'ensemble_market_value'
]]

# --- Normalize Player Names ---
for df in [df_name, df_features, df_sentiment]:
    df['Player Name'] = df['Player Name'].str.lower().str.replace(' ', '', regex=False)

# --- Merge All Sources ---
merged = df_name.copy()
merged = merged.merge(df_features, on='Player Name', how='left')
merged = merged.merge(df_sentiment, on='Player Name', how='left')
merged = pd.concat([merged.reset_index(drop=True), df_preds.reset_index(drop=True)], axis=1)

# --- Final Column Selection ---
final_columns = [
    'Player Name', 'Age', 'Injury Status', 'Sentiment Label',
    'Market Value (M)', 'y_test', 'lstm_preds', 'ensemble_preds',
    'lstm_market_value', 'ensemble_market_value'
]
final_df = merged[final_columns]


# --- Save to CSV ---
final_df.to_csv(output_path, index=False)

# --- Display Confirmation ---
print("Final dataset created and saved to:", output_path)
print(final_df.head().to_markdown(index=False))