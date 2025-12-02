# feature_engineering.py

import pandas as pd
import numpy as np
import os
from unidecode import unidecode
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the clean dataset you created in the last phase
data_folder = 'data'
clean_data_path = os.path.join(data_folder, 'final_top_10_player_data.csv')
df = pd.read_csv(clean_data_path)

# Re-use the name cleaning function for consistent merging
def clean_player_name(name):
    clean_name = unidecode(str(name).lower())
    parts = clean_name.split()
    return ' '.join(parts[:2])

print("--- Clean data loaded successfully ---")
print(df.head())

# The Fix: Add the encoding='latin-1' parameter to the read_csv function
tweets_df = pd.read_csv(
    os.path.join(data_folder, 'top_players_tweets.csv'),
    encoding='latin-1'
)

print("Successfully loaded tweets_df using 'latin-1' encoding!")

# ==> ACTION: Find the correct player name column and update this variable <==
player_name_col_in_tweets = 'player_name' # <-- REPLACE 'Player Name' WITH THE CORRECT NAME

# --- Calculate Sentiment for Each Tweet ---
analyzer = SentimentIntensityAnalyzer()
tweets_df['sentiment'] = tweets_df['text'].apply(lambda t: analyzer.polarity_scores(t)['compound'])

# --- Calculate Average Score per Player ---
# The groupby() now uses your corrected column name variable
player_sentiment = tweets_df.groupby(player_name_col_in_tweets)['sentiment'].mean().reset_index()
player_sentiment.rename(columns={'sentiment': 'avg_sentiment_score'}, inplace=True)

# --- Merge into Main DataFrame ---
df['merge_key'] = df['player_name'].apply(clean_player_name)
# The apply() now uses your corrected column name variable
player_sentiment['merge_key'] = player_sentiment[player_name_col_in_tweets].apply(clean_player_name)

df = pd.merge(df, player_sentiment[['merge_key', 'avg_sentiment_score']], on='merge_key', how='left')
df.drop('merge_key', axis=1, inplace=True)

print("\n--- Sentiment scores added ---")
print(df[['player_name', 'avg_sentiment_score']].head())

# --- Save DataFrame with Sentiment Scores ---

# Define the path to your main CSV file
output_path = os.path.join(data_folder, 'final_top_10_player_data.csv')

# Save the updated DataFrame, overwriting the old file
# index=False prevents adding an extra, unnamed column to your CSV
df.to_csv(output_path, index=False)

print(f"\nSuccessfully updated '{output_path}' with the new sentiment score column.")

