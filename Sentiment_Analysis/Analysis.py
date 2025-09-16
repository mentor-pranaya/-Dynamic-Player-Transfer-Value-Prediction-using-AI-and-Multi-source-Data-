import pandas as pd
from textblob import TextBlob

df = pd.read_csv(r'Data\Sentiment analysis.csv', encoding='utf-8')

df['sentiment_score'] = df['Likes'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

player_sentiment = df.groupby('Platform')['Likes'].mean().reset_index()
player_sentiment.rename(columns={'sentiment_score': 'avg_sentiment_score'}, inplace=True)

print(player_sentiment.head())