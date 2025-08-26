import praw
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import unicodedata

nltk.download("vader_lexicon")

#I put my api key while running the program
reddit = praw.Reddit(
    client_id="reddit_api_key",
    client_secret="reddit_api_key_secret",
    user_agent="player-sentiment-analysis"
)

def fix_encoding(text):
    if isinstance(text, str):
        return text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    return text

def strip_accents(text):
    if isinstance(text, str):
        return "".join(
            c for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )
    return text


# Load player data
players_df = pd.read_csv("bundesliga_players_basic_info_23_24.csv")  # filename changes with the league
players_df["player_name"] = players_df["player_name"].apply(fix_encoding)
players_df["player_name_clean"] = players_df["player_name"].apply(strip_accents)


# Sentiment Analyzer

sia = SentimentIntensityAnalyzer()

def analyze_text(player_name, text):
    """
    Run VADER sentiment on a text and return row dict.
    """
    scores = sia.polarity_scores(text)
    return {
        "player": player_name,
        "comment": text,
        "compound": scores["compound"],
        "positive": scores["pos"],
        "neutral": scores["neu"],
        "negative": scores["neg"]
    }

def get_player_comments(player_name, limit=20):
    # Get comments + titles + selftext for player from Reddit.
    
    comments_data = []
    try:
        for submission in reddit.subreddit("soccer").search(player_name, limit=limit):
            comments_data.append(analyze_text(player_name, submission.title))
            if submission.selftext:
                comments_data.append(analyze_text(player_name, submission.selftext))

            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:limit]:
                comments_data.append(analyze_text(player_name, comment.body))

    except Exception as e:
        print(f"Error fetching {player_name}: {e}")

    return comments_data


# Main loop
all_data = []
for i, row in players_df.iterrows():
    pname = row["player_name_clean"]
    comments = get_player_comments(pname)

    if comments:
        df_temp = pd.DataFrame(comments)
        # Calculate averages per player
        avg_compound = df_temp["compound"].mean()
        avg_positive = df_temp["positive"].mean()
        df_temp["avg_compound"] = avg_compound
        df_temp["avg_positive"] = avg_positive
        all_data.append(df_temp)

    print(f"[{i+1}/{len(players_df)}] Processed {row['player_name']}")


# Save detailed dataset
# all the filenames included changes with the league
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("Bundesliga_2024_sentiment_detailed.csv", index=False, encoding="utf-8")
    print("Saved Bundesliga_2024_sentiment_detailed.csv")
