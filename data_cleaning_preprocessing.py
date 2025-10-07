"""
Data Cleaning and Preprocessing Script
--------------------------------------
Handles missing values, duplicate entries, scaling, and encoding.
Outputs: cleaned_dataset.csv
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load raw datasets (dummy merge for example)
performance = pd.read_csv("data/raw/premier_league_2024_player_stats.csv")
market = pd.read_csv("data/raw/transfermarkt_values.csv")
sentiment = pd.read_csv("data/raw/twitter_sentiment_scores.csv")

# Merge
df = performance.merge(market, on="player", how="left").merge(sentiment, on="player", how="left")

# Handle missing values
df.fillna({"goals": 0, "assists": 0, "shots": 0, "minutes": 0, "market_value": df["market_value"].median()}, inplace=True)
df.drop_duplicates(subset=["player"], inplace=True)

# Encode categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[["team"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["team"]))

# Scale numerical data
scaler = StandardScaler()
num_cols = ["minutes", "goals", "shots", "assists", "market_value", "sentiment_score"]
df[num_cols] = scaler.fit_transform(df[num_cols])

# Combine
clean_df = pd.concat([df.drop(columns=["team"]), encoded_df], axis=1)
clean_df.to_csv("data/cleaned/cleaned_dataset.csv", index=False)
print("✅ Cleaned dataset saved to data/cleaned/cleaned_dataset.csv")
