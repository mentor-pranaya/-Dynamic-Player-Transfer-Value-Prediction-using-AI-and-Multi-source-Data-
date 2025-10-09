import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- Reddit API ----------
reddit = praw.Reddit(
    client_id="fHYPw2UASTGlYhbh1lT-9Q",
    client_secret="ZIBSqiYhep8uoZMlEIbwFa6xCXytMg",
    user_agent="football_sentiment by u/Stock_Confection5505"
)

# ---------- Load Players from CSV ----------
players_df = pd.read_csv("/Users/veerababu/Desktop/Infosys/CompleteList.csv")
players_list = players_df.iloc[:,0].dropna().tolist()   # takes first column

print(f"✅ Loaded {len(players_list)} players from CSV")

# ---------- Sentiment Analyzer ----------
analyzer = SentimentIntensityAnalyzer()

results = []
for player in players_list:
    print(f"🔎 Fetching posts for: {player}")
    try:
        for submission in reddit.subreddit("soccer").search(player, limit=10):  # limit reduced to avoid rate limits
            vs = analyzer.polarity_scores(submission.title)
            sentiment = "Positive" if vs["compound"] > 0 else "Negative" if vs["compound"] < 0 else "Neutral"
            results.append({
                "Player": player,
                "Text": submission.title,
                "Sentiment": sentiment,
                "Score": vs["compound"]
            })
    except Exception as e:
        print(f"❌ Error fetching {player}: {e}")

# ---------- Save results ----------
df = pd.DataFrame(results)
df.to_csv("/Users/veerababu/Downloads/sentiment_report.csv", index=False, encoding="utf-8-sig")
print("✅ Sentiment analysis complete. File saved to /Users/veerababu/Downloads/reddit_players_sentiment_report.csv")
