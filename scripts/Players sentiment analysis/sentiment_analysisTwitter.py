import tweepy
import pandas as pd
import sys
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- Twitter API ----------
bearer_token = "AAAAAAAAAAAAAAAAAAAAAKKq3gEAAAAA9%2BsVoiVEOj2pk2at8wZ7M%2Bcxfto%3D8ykU8O64YvtsGzHHQHSVrpCN9p5h5jylxBT8knPTuHTMiBrgvD"
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# ---------- Get player name from command line ----------
if len(sys.argv) < 2:
    print("⚠️ Please provide a player name. Example:")
    print("   python3 sentiment_analysis.py 'Lionel Messi'")
    sys.exit(1)

player = sys.argv[1]

# ---------- Sentiment Analyzer ----------
analyzer = SentimentIntensityAnalyzer()
results = []

print(f"Fetching tweets for: {player}")
query = f'"{player}" -is:retweet lang:en'

try:
    tweets = client.search_recent_tweets(query=query, max_results=10, tweet_fields=["text"])
except Exception as e:
    print(f"❌ Error fetching {player}: {e}")
    sys.exit(1)

if tweets.data:
    for tweet in tweets.data:
        vs = analyzer.polarity_scores(tweet.text)
        sentiment = "Positive" if vs["compound"] > 0 else "Negative" if vs["compound"] < 0 else "Neutral"
        results.append({
            "Player": player,
            "Tweet": tweet.text,
            "Sentiment": sentiment,
            "Score": vs["compound"]
        })
    time.sleep(2)

# ---------- Save results ----------
df = pd.DataFrame(results)

if not df.empty:
    summary = df.groupby("Sentiment").size()
    avg_score = df["Score"].mean()

    print("\n📊 Sentiment Summary:")
    print(summary)
    print(f"\n⭐ Average Sentiment Score: {avg_score:.2f}")

    df.to_csv(f"/Users/veerababu/Downloads/{player.replace(' ', '_')}_sentiment.csv",
              index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved detailed report for {player} in Downloads folder.")
else:
    print("⚠️ No tweets found.")
