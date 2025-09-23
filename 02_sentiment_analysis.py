from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Let's use some of the tweets you collected as examples
tweets = [
    "Federer stands as one of just seven athletes to have crossed $1 billion in career pretax income while active in their sport, along with Los Angeles Lakers forward LeBron James, golfers Tiger Woods and Phil Mickelson, soccer players Cristiano Ronaldo and Lionel Messi, and boxer",
    "@ammunitiion @Cristiano @AlNassrFC_EN @AlNassrFC As of now, Cristiano Ronaldo's post has 266,007 likes. Since my last update (about 1 minute ago), it gained 213 likes.",
    "Cristiano Ronaldo all 7 World Cup Knockout goals [A thread]",
    "@thee_ovie Graveyard of talent Even the goat Cristiano Ronaldo"
]

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

results = []
for tweet in tweets:
    # The polarity_scores() method returns a dictionary with sentiment scores
    sentiment_scores = analyzer.polarity_scores(tweet)
    
    # The 'compound' score is a single value that summarizes the sentiment
    # Positive: compound > 0.05
    # Neutral: -0.05 <= compound <= 0.05
    # Negative: compound < -0.05
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    results.append({
        'Tweet': tweet,
        'Compound Score': compound_score,
        'Sentiment': sentiment,
        'All Scores': sentiment_scores
    })

# Convert the results to a DataFrame for easy viewing
df_sentiment = pd.DataFrame(results)

print("Sentiment Analysis Results:")
print(df_sentiment[['Tweet', 'Compound Score', 'Sentiment']])