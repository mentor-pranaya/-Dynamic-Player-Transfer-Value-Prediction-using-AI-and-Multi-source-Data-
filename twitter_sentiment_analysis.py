import tweepy
import pandas as pd
import snscrape.modules.twitter as sntwitter
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import time

# ========== STEP 1: Twitter API Setup ==========
# Replace these with your actual Twitter API credentials
#Client ID: THdUamtBTEpGU1NIRzVwOWJReFc6MTpjaQ
#Client Secret: Ch0HOE_WsSaKhMMLxWEd9sL-ZVr4edRVt3oDhV9nH7pAFwaMde
API_KEY = "######removed######"
API_SECRET = "######removed######"
ACCESS_TOKEN = "######removed######"
ACCESS_SECRET = "######removed######"

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

# ========== STEP 2: Helper Functions ==========

def clean_tweet(tweet):
    """
    Cleans a tweet by removing URLs, mentions, hashtags, emojis, and extra spaces.
    """
    tweet = re.sub(r"http\S+", "", tweet)       # Remove URLs
    tweet = re.sub(r"@\S+", "", tweet)          # Remove mentions
    tweet = re.sub(r"#\S+", "", tweet)          # Remove hashtags
    tweet = re.sub(r"[^A-Za-z0-9\s]", "", tweet) # Remove special chars/emojis
    return tweet.strip()

def get_player_tweets_api(player_name, count=50):
    """
    Fetches recent tweets mentioning the player.
    Returns a list of cleaned tweets.
    """
    try:
        query = f'"{player_name}" -filter:retweets'  # Search player name, exclude retweets
        tweets = api.search_tweets(q=query, lang="en", count=count, tweet_mode="extended")
        return [clean_tweet(tweet.full_text) for tweet in tweets]
    except Exception as e:
        print(f"Error fetching tweets for {player_name}: {e}")
        return []
# Twitter basic API did not allow search_tweet, using scraping.
def get_player_tweets(player_name, count=50):
    """
    Fetch recent tweets using snscrape (No API Key Required)
    """
    query = f'"{player_name}" lang:en -filter:retweets'
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= count:
            break
        tweets.append(tweet.content)
    return tweets

def analyze_sentiment(text):
    """
    Uses VADER to return the sentiment score of text (-1=Negative, +1=Positive).
    """
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)["compound"]

# ========== STEP 3: Load Player List ==========
players_df = pd.read_csv("market_values_all_competitions.csv")

# OPTIONAL: Limit for testing (remove this later)
players_df = players_df.head(20)  # Test with 20 players first

# ========== STEP 4: Sentiment Analysis ==========
sentiment_data = []

for _, row in players_df.iterrows():
    player_name = row["Player"]
    club = row["Club"]
    competition = row["Competition"]
    market_value = row["Market Value"]

    print(f"Fetching tweets for {player_name} ({club}, {competition})...")

    tweets = get_player_tweets(player_name, count=30)  # 30 tweets per player

    for tweet in tweets:
        score = analyze_sentiment(tweet)
        sentiment_data.append((player_name, club, competition, market_value, tweet, score))

    time.sleep(2)  # Sleep to avoid hitting rate limits

# ========== STEP 5: Save Sentiment Data ==========
sentiment_df = pd.DataFrame(sentiment_data, 
                             columns=["Player", "Club", "Competition", "Market Value", "Tweet", "Sentiment_Score"])

os.makedirs("sentiment_data", exist_ok=True)
sentiment_df.to_csv("sentiment_data/player_sentiment_all.csv", index=False)

print("Sentiment analysis completed for all players. Results saved to sentiment_data/player_sentiment_all.csv")

