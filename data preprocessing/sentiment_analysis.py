
import os
import pandas as pd
from unidecode import unidecode
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = r"C:\Users\Abhinav\Desktop\Project\data"
TWEETS_FILE = os.path.join(BASE_DIR, "top_players_tweets.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "player_sentiment.csv")

def normalize_name(name: str) -> str:
    """Normalize player names (lowercase, remove accents, keep first two words)."""
    clean = unidecode(str(name).lower())
    parts = clean.split()
    return " ".join(parts[:2])

print("[1] Loading tweets dataset...")
tweets_df = pd.read_csv(TWEETS_FILE, encoding="latin-1")

if "player_name" not in tweets_df.columns or "text" not in tweets_df.columns:
    raise ValueError("Tweets file must contain 'player_name' and 'text' columns")

print(f"✔ Loaded {len(tweets_df):,} tweets")

print("\n[2] Running VADER Sentiment Analysis...")
analyzer = SentimentIntensityAnalyzer()

tweets_df["sentiment_compound"] = tweets_df["text"].apply(lambda txt: analyzer.polarity_scores(str(txt))["compound"])

print("\n[3] Aggregating sentiment scores per player...")

tweets_df["merge_key"] = tweets_df["player_name"].map(normalize_name)

player_sentiment = tweets_df.groupby("merge_key")["sentiment_compound"].mean().reset_index()
player_sentiment.rename(columns={"sentiment_compound": "avg_sentiment_score"}, inplace=True)

print("✔ Sentiment scores calculated for each player")

print("\n[4] Saving sentiment results...")
player_sentiment.to_csv(OUTPUT_FILE, index=False)

print(f"✔ Sentiment results saved to {OUTPUT_FILE}")
print("\n=== Sample Output ===")
print(player_sentiment.head())
