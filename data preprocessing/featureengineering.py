import os
import pandas as pd
import numpy as np
from unidecode import unidecode
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = r"C:\Users\Abhinav\Desktop\Project\data"
MAIN_FILE = os.path.join(BASE_DIR, "final_top_10_player_data.csv")
TWEETS_FILE = os.path.join(BASE_DIR, "top_players_tweets.csv")
INJURY_FILE = os.path.join(BASE_DIR, "injury_history.csv")
X_OUT = os.path.join(BASE_DIR, "X_scaled_features.csv")
Y_OUT = os.path.join(BASE_DIR, "y_target.csv")

def normalize_name(name: str) -> str:
    """Normalize player names (lowercase, remove accents, keep first two words)."""
    clean = unidecode(str(name).lower())
    parts = clean.split()
    return " ".join(parts[:2])

print("[1] Loading main dataset...")
df = pd.read_csv(MAIN_FILE)
print(f"✔ Loaded {len(df):,} rows from {MAIN_FILE}")

print("\n[2] Adding sentiment features from tweets...")

tweets_df = pd.read_csv(TWEETS_FILE, encoding="latin-1")
if "player_name" not in tweets_df.columns:
    raise ValueError("Expected a 'player_name' column in tweets CSV")

analyzer = SentimentIntensityAnalyzer()
tweets_df["sentiment_score"] = tweets_df["text"].apply(lambda txt: analyzer.polarity_scores(txt)["compound"])

sentiment_agg = tweets_df.groupby("player_name")["sentiment_score"].mean().reset_index()
sentiment_agg.rename(columns={"sentiment_score": "avg_sentiment"}, inplace=True)

df["merge_key"] = df["player_name"].map(normalize_name)
sentiment_agg["merge_key"] = sentiment_agg["player_name"].map(normalize_name)

df = df.merge(sentiment_agg[["merge_key", "avg_sentiment"]], on="merge_key", how="left")
df.drop(columns=["merge_key"], inplace=True)

print("✔ Sentiment merged successfully")

print("\n[3] Adding injury metrics...")

inj_df = pd.read_csv(INJURY_FILE, encoding="latin-1")
if "player_name" not in inj_df.columns or "days_missed" not in inj_df.columns:
    raise ValueError("Injury file must contain 'player_name' and 'days_missed' columns")

injury_stats = inj_df.groupby("player_name").agg(
    total_days_injured=("days_missed", "sum"),
    injuries=("player_name", "count")
).reset_index()

df["merge_key"] = df["player_name"].map(normalize_name)
injury_stats["merge_key"] = injury_stats["player_name"].map(normalize_name)

df = df.merge(injury_stats[["merge_key", "total_days_injured", "injuries"]], on="merge_key", how="left")
df.drop(columns=["merge_key"], inplace=True)

df["total_days_injured"] = df["total_days_injured"].fillna(0)
df["injuries"] = df["injuries"].fillna(0)

print("✔ Injury data merged successfully")

print("\n[4] Cleaning injury days column...")

df["total_days_injured"] = df["total_days_injured"].astype(str).str.extract(r"(\d+)", expand=False)
df["total_days_injured"] = df["total_days_injured"].fillna(0).astype(int)

print("✔ Injury data cleaned")

print("\n[5] One-hot encoding categorical columns...")

categorical_cols = ["position", "Nationality"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"✔ Encoded {len(categorical_cols)} categorical columns")

print("\n[6] Separating features and target...")

df_encoded.rename(columns={"Market Value 2015 (in millions €)": "market_value"}, inplace=True)

y = df_encoded["market_value"]
X = df_encoded.drop(columns=["market_value", "player_name"])

print(f"✔ Feature shape: {X.shape}, Target shape: {y.shape}")

print("\n[7] Scaling features with StandardScaler...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("✔ Features scaled")

print("\n[8] Saving processed outputs...")

X_scaled_df.to_csv(X_OUT, index=False)
y.to_csv(Y_OUT, index=False, header=True)

print(f"✔ Saved features → {X_OUT}")
print(f"✔ Saved target → {Y_OUT}")

print("\nPipeline finished successfully 🚀")