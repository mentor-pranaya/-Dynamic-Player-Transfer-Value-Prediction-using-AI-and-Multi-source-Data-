"""
data_cleaning_preprocessing.py
------------------------------
Cleans and merges FIFA player data with sentiment and market value.
Outputs: data/cleaned/cleaned_dataset.csv
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

fifa = pd.read_csv("data/raw/fifa_players_data_no_duplicates.csv")
sent = pd.read_csv("data/raw/sentiment_data.csv")
market = pd.read_csv("data/raw/transfermarkt_values.csv")

# Merge datasets
df = fifa.merge(sent, left_on="player_name", right_on="Player", how="left").merge(
    market, left_on="player_name", right_on="player", how="left"
)

# Fill missing values
df.fillna({"Sentiment": "Neutral", "Score": 0, "market_value": 0}, inplace=True)
df.drop_duplicates(subset=["player_name"], inplace=True)

# Scale numeric columns
num_cols = ["overall_rating", "potential", "value_euro", "wage_euro"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.to_csv("data/cleaned/cleaned_dataset.csv", index=False)
print("✅ Cleaned dataset saved to data/cleaned/cleaned_dataset.csv")
