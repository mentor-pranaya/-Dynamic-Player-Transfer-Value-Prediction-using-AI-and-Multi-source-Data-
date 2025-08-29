import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


sentiment_data = pd.read_csv('Sentiment analysis.csv')
sentiment_data = sentiment_data.loc[:, ~sentiment_data.columns.str.contains('^Unnamed')]

sentiment_data.fillna({"Text": "", "Sentiment": "Neutral"}, inplace=True)


analyzer = SentimentIntensityAnalyzer()
sentiment_data["Sentiment_Score"] = sentiment_data["Text"].astype(str).apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
sentiment_data["Sentiment_Label"] = sentiment_data["Sentiment"].map(sentiment_map)

player_sentiment = sentiment_data.groupby("User").agg(
    Avg_Sentiment_Score=("Sentiment_Score", "mean"),
    Sentiment_Volume=("Text", "count")
).reset_index()

print("Sentiment Data cleaned & engineered:")
print(player_sentiment.head())
