import praw  
import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- Initialize Reddit API ----------
reddit = praw.Reddit(
    client_id="fHYPw2UASTGlYhbh1lT-9Q",  
    client_secret="ZIBSqiYhep8uoZMlEIbwFa6xCXytMg",  
    user_agent="football_sentiment by u/Stock_Confection5505"  
)

# ---------- Load Player Names from CSV File ----------
# Read the CSV file containing player names
players_df = pd.read_csv("/Users/veerababu/Desktop/Infosys/CompleteList.csv")
# Extract the first column and convert it to a list, removing any empty entries
players_list = players_df.iloc[:, 0].dropna().tolist()

print(f"✅ Loaded {len(players_list)} players from CSV")

# ---------- Initialize Sentiment Analyzer ----------
analyzer = SentimentIntensityAnalyzer()

# List to store results
results = []

# ---------- Fetch and Analyze Sentiment for Each Player ----------
for player in players_list:
    print(f"🔎 Fetching posts for: {player}")
    try:
        # Search for posts related to the player in the 'soccer' subreddit
        for submission in reddit.subreddit("soccer").search(player, limit=10):  # Limit to 10 posts to avoid rate limits
            # Analyze the sentiment of the submission title
            sentiment_scores = analyzer.polarity_scores(submission.title)
            # Determine sentiment based on the compound score
            sentiment = "Positive" if sentiment_scores["compound"] > 0 else "Negative" if sentiment_scores["compound"] < 0 else "Neutral"
            # Append the results to the list
            results.append({
                "Player": player,
                "Text": submission.title,
                "Sentiment": sentiment,
                "Score": sentiment_scores["compound"]
            })
    except Exception as e:
        print(f"❌ Error fetching posts for {player}: {e}")

# ---------- Save Results to CSV File ----------
# Create a DataFrame from the results list
df = pd.DataFrame(results)
# Save the DataFrame to a CSV file
df.to_csv("/Users/veerababu/Desktop/sentiment_report.csv", index=False, encoding="utf-8-sig")  
print("✅ Sentiment analysis complete. File saved to path/to/your/sentiment_report.csv")
