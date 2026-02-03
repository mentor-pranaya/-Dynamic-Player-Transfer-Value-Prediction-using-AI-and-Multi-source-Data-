"""
Sentiment Analysis for Player Valuation
======================================

This module implements sentiment analysis using VADER and TextBlob for player-related tweets.
It processes social media data to extract sentiment features that can impact player valuations.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    print("Warning: vaderSentiment not available. Install with: pip install vaderSentiment")
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    print("Warning: textblob not available. Install with: pip install textblob")
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: nltk not available. Install with: pip install nltk")
    NLTK_AVAILABLE = False

class SentimentAnalyzer:
    """
    Sentiment analysis class for processing player-related social media data.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.vader_analyzer = None
        self.stop_words = set()
        
        # Initialize VADER
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize NLTK stopwords
        if NLTK_AVAILABLE:
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                self.stop_words = set(stopwords.words('english'))
            except:
                print("Warning: Could not download NLTK data")
                self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with stopwords removed
        """
        if not text or not NLTK_AVAILABLE:
            return text
        
        try:
            words = word_tokenize(text)
            filtered_words = [word for word in words if word not in self.stop_words]
            return ' '.join(filtered_words)
        except:
            return text
    
    def get_vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get VADER sentiment scores.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with VADER sentiment scores
        """
        if not VADER_AVAILABLE or not self.vader_analyzer:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        if not text or text.strip() == '':
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        scores = self.vader_analyzer.polarity_scores(text)
        return scores
    
    def get_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get TextBlob sentiment scores.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with TextBlob sentiment scores
        """
        if not TEXTBLOB_AVAILABLE:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        if not text or text.strip() == '':
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using both VADER and TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with all sentiment scores
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        cleaned_text = self.remove_stopwords(cleaned_text)
        
        # Get sentiment scores
        vader_scores = self.get_vader_sentiment(cleaned_text)
        textblob_scores = self.get_textblob_sentiment(cleaned_text)
        
        # Combine scores
        all_scores = {**vader_scores, **textblob_scores}
        
        return all_scores
    
    def process_twitter_data(self, twitter_file: str) -> pd.DataFrame:
        """
        Process Twitter data for sentiment analysis.
        
        Args:
            twitter_file: Path to Twitter CSV file
            
        Returns:
            DataFrame with sentiment scores
        """
        print(f"Processing Twitter data from {twitter_file}...")
        
        try:
            # Load Twitter data
            df = pd.read_csv(twitter_file, header=None)
            print(f"Loaded {len(df)} tweets")
            
            # Check if we have the expected columns and rename them
            if len(df.columns) >= 4:
                df.columns = ['id', 'game', 'sentiment', 'text']
                print("Renamed columns to: id, game, sentiment, text")
            else:
                print("Warning: Unexpected number of columns. Available columns:", df.columns.tolist())
                return pd.DataFrame()
            
            # Process each tweet
            sentiment_scores = []
            for idx, row in df.iterrows():
                if idx % 1000 == 0:
                    print(f"Processing tweet {idx}/{len(df)}")
                
                text = row['text'] if 'text' in row else ''
                scores = self.analyze_sentiment(text)
                sentiment_scores.append(scores)
            
            # Convert to DataFrame
            sentiment_df = pd.DataFrame(sentiment_scores)
            
            # Add original data
            result_df = df.copy()
            result_df = pd.concat([result_df, sentiment_df], axis=1)
            
            print("Twitter sentiment analysis completed")
            return result_df
            
        except Exception as e:
            print(f"Error processing Twitter data: {e}")
            return pd.DataFrame()
    
    def create_player_sentiment_features(self, sentiment_df: pd.DataFrame, 
                                       player_mapping: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create player-level sentiment features.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            player_mapping: Optional mapping from tweet to player
            
        Returns:
            DataFrame with player-level sentiment features
        """
        print("Creating player-level sentiment features...")
        
        if sentiment_df.empty:
            print("No sentiment data available")
            return pd.DataFrame()
        
        # If no player mapping provided, create a simple mapping
        if player_mapping is None:
            # Create a simple mapping based on available data
            # This is a placeholder - in practice, you'd need proper name matching
            sentiment_df['player_name'] = 'unknown'
        else:
            sentiment_df['player_name'] = sentiment_df.index.map(player_mapping)
        
        # If all players are 'unknown', create some sample players for demonstration
        if sentiment_df['player_name'].nunique() == 1 and sentiment_df['player_name'].iloc[0] == 'unknown':
            # Create sample players for demonstration
            unique_players = ['player_1', 'player_2', 'player_3', 'player_4', 'player_5']
            sentiment_df['player_name'] = np.random.choice(unique_players, len(sentiment_df))
        
        # Group by player and calculate aggregated features
        player_features = []
        
        for player in sentiment_df['player_name'].unique():
            if player == 'unknown':
                continue
                
            player_tweets = sentiment_df[sentiment_df['player_name'] == player]
            
            if len(player_tweets) == 0:
                continue
            
            # Basic sentiment statistics
            features = {
                'player_name': player,
                'tweet_count': len(player_tweets),
                'mean_compound_score': player_tweets['compound'].mean() if 'compound' in player_tweets.columns else 0,
                'std_compound_score': player_tweets['compound'].std() if 'compound' in player_tweets.columns else 0,
                'mean_polarity': player_tweets['polarity'].mean() if 'polarity' in player_tweets.columns else 0,
                'std_polarity': player_tweets['polarity'].std() if 'polarity' in player_tweets.columns else 0,
                'mean_subjectivity': player_tweets['subjectivity'].mean() if 'subjectivity' in player_tweets.columns else 0,
                'std_subjectivity': player_tweets['subjectivity'].std() if 'subjectivity' in player_tweets.columns else 0,
            }
            
            # Sentiment distribution
            if 'compound' in player_tweets.columns:
                positive_tweets = (player_tweets['compound'] > 0.05).sum()
                negative_tweets = (player_tweets['compound'] < -0.05).sum()
                neutral_tweets = len(player_tweets) - positive_tweets - negative_tweets
                
                features.update({
                    'positive_tweet_ratio': positive_tweets / len(player_tweets) if len(player_tweets) > 0 else 0,
                    'negative_tweet_ratio': negative_tweets / len(player_tweets) if len(player_tweets) > 0 else 0,
                    'neutral_tweet_ratio': neutral_tweets / len(player_tweets) if len(player_tweets) > 0 else 0,
                })
            
            # Sentiment volatility (standard deviation)
            features['sentiment_volatility'] = features['std_compound_score']
            
            # Sentiment trend (if we have temporal data)
            if 'created_at' in player_tweets.columns:
                try:
                    player_tweets_sorted = player_tweets.sort_values('created_at')
                    if len(player_tweets_sorted) > 1 and 'compound' in player_tweets_sorted.columns:
                        # Simple linear trend
                        x = np.arange(len(player_tweets_sorted))
                        y = player_tweets_sorted['compound'].values
                        if len(y) > 1 and not np.isnan(y).all():
                            trend = np.polyfit(x, y, 1)[0]
                            features['sentiment_trend'] = trend
                        else:
                            features['sentiment_trend'] = 0
                    else:
                        features['sentiment_trend'] = 0
                except:
                    features['sentiment_trend'] = 0
            else:
                features['sentiment_trend'] = 0
            
            player_features.append(features)
        
        result_df = pd.DataFrame(player_features)
        print(f"Created sentiment features for {len(result_df)} players")
        
        return result_df
    
    def create_synthetic_player_sentiment(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic sentiment data for players (for demonstration purposes).
        
        Args:
            players_df: DataFrame with player information
            
        Returns:
            DataFrame with synthetic sentiment features
        """
        print("Creating synthetic player sentiment data...")
        
        np.random.seed(42)  # For reproducibility
        
        player_sentiment = []
        
        for idx, player in players_df.iterrows():
            # Generate synthetic sentiment based on player performance
            # Better performing players tend to have more positive sentiment
            
            # Base sentiment influenced by FIFA rating
            if 'fifa_rating' in players_df.columns and pd.notna(player['fifa_rating']):
                base_sentiment = (player['fifa_rating'] - 50) / 50  # Normalize to -1 to 1
            else:
                base_sentiment = np.random.normal(0, 0.3)
            
            # Add some noise
            noise = np.random.normal(0, 0.2)
            mean_compound = np.clip(base_sentiment + noise, -1, 1)
            
            # Generate features
            features = {
                'player_name': player.get('p_id2', f'player_{idx}'),
                'tweet_count': np.random.poisson(50),  # Average 50 tweets per player
                'mean_compound_score': mean_compound,
                'std_compound_score': np.random.uniform(0.1, 0.5),
                'mean_polarity': mean_compound * 0.8,  # Slightly different from compound
                'std_polarity': np.random.uniform(0.1, 0.4),
                'mean_subjectivity': np.random.uniform(0.3, 0.8),
                'std_subjectivity': np.random.uniform(0.1, 0.3),
                'positive_tweet_ratio': max(0, min(1, (mean_compound + 1) / 2 + np.random.normal(0, 0.1))),
                'negative_tweet_ratio': max(0, min(1, (1 - mean_compound) / 2 + np.random.normal(0, 0.1))),
                'neutral_tweet_ratio': 1 - max(0, min(1, (mean_compound + 1) / 2)) - max(0, min(1, (1 - mean_compound) / 2)),
                'sentiment_volatility': np.random.uniform(0.1, 0.5),
                'sentiment_trend': np.random.normal(0, 0.1),
            }
            
            # Ensure ratios sum to 1
            total_ratio = features['positive_tweet_ratio'] + features['negative_tweet_ratio'] + features['neutral_tweet_ratio']
            if total_ratio > 0:
                features['positive_tweet_ratio'] /= total_ratio
                features['negative_tweet_ratio'] /= total_ratio
                features['neutral_tweet_ratio'] /= total_ratio
            
            player_sentiment.append(features)
        
        result_df = pd.DataFrame(player_sentiment)
        print(f"Created synthetic sentiment data for {len(result_df)} players")
        
        return result_df

def main():
    """Main function to run sentiment analysis."""
    print("=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Check if Twitter data is available
    twitter_files = ["../twitter_training.csv", "../twitter_validation.csv"]
    sentiment_data = None
    
    for twitter_file in twitter_files:
        try:
            print(f"\nTrying to process {twitter_file}...")
            sentiment_data = analyzer.process_twitter_data(twitter_file)
            if not sentiment_data.empty:
                print(f"Successfully processed {twitter_file}")
                break
        except FileNotFoundError:
            print(f"File {twitter_file} not found")
            continue
    
    # If no Twitter data available, create synthetic data
    if sentiment_data is None or sentiment_data.empty:
        print("\nNo Twitter data available. Creating synthetic sentiment data...")
        
        # Load player data to create synthetic sentiment
        try:
            players_df = pd.read_csv("../processed/dataset_processed.csv")
            sentiment_data = analyzer.create_synthetic_player_sentiment(players_df)
        except FileNotFoundError:
            print("Could not load player data for synthetic sentiment generation")
            return None
    
    # Create player-level sentiment features
    if not sentiment_data.empty:
        player_sentiment = analyzer.create_player_sentiment_features(sentiment_data)
        
        # Save results
        output_path = "../processed/player_sentiment_features.csv"
        player_sentiment.to_csv(output_path, index=False)
        print(f"\nPlayer sentiment features saved to {output_path}")
        
        # Print summary
        print(f"\nSentiment Analysis Summary:")
        print(f"Players analyzed: {len(player_sentiment)}")
        print(f"Average compound score: {player_sentiment['mean_compound_score'].mean():.3f}")
        print(f"Average positive ratio: {player_sentiment['positive_tweet_ratio'].mean():.3f}")
        print(f"Average negative ratio: {player_sentiment['negative_tweet_ratio'].mean():.3f}")
        
        return player_sentiment
    else:
        print("No sentiment data generated")
        return None

if __name__ == "__main__":
    main()
