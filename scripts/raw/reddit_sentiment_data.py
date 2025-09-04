import os

os.environ["REDDIT_CLIENT_ID"] = "your REDDIT_CLIENT_ID"
os.environ["REDDIT_CLIENT_SECRET"] = "you REDDIT_CLIENT_SECRET"
os.environ["REDDIT_USER_AGENT"] = "your REDDIT_USER_AGENT"

import praw, os

# set credentials once
os.environ["REDDIT_CLIENT_ID"] = "your REDDIT_CLIENT_ID"
os.environ["REDDIT_CLIENT_SECRET"] = "you REDDIT_CLIENT_SECRET"
os.environ["REDDIT_USER_AGENT"] = "your REDDIT_USER_AGENT"

# use them here
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

print("Read-only mode:", reddit.read_only)
print("Subreddit test:", reddit.subreddit("soccer").display_name)



import praw, os

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

for submission in reddit.subreddit("soccer").hot(limit=5):
    print(submission.title)

import os
import re
import time
import logging
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import praw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor, as_completed

# =======================
# CONFIG
# =======================
INPUT_FILE = "/content/data/processed/final_top_players.csv"       # Input: player_name, season
OUTPUT_FILE = "final_reddit_sentiment.csv"    # Output file
SUBREDDITS = ["soccer", "premierleague", "LaLiga", "seriea", "bundesliga", "ligue1"]

MAX_POSTS_PER_PLAYER = 40
MAX_COMMENTS_PER_POST = 100
MIN_COMMENTS_THRESHOLD = 20
WORKERS = 4
SAVE_EVERY = 50   # Save progress every N players

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =======================
# NLTK Sentiment
# =======================
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# =======================
# Reddit Auth (from env vars)
# =======================
reddit = praw.Reddit(
    client_id=os.environ.get("REDDIT_CLIENT_ID"),
    client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
    user_agent=os.environ.get("REDDIT_USER_AGENT")
)

# =======================
# Helpers
# =======================
def season_window(season_year: int):
    start = datetime(season_year, 7, 1, tzinfo=timezone.utc)
    end = datetime(season_year + 1, 6, 30, 23, 59, 59, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())

def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.split())

def compile_name_pattern(name: str):
    return re.compile(r"\b" + re.escape(normalize_name(name)) + r"\b")

def sentiment_of(text: str):
    s = sia.polarity_scores(text)
    label = max(("pos", s["pos"]), ("neu", s["neu"]), ("neg", s["neg"]), key=lambda x: x[1])[0]
    return label, s["compound"]

# =======================
# Core
# =======================
def collect_for_player_season(player_name: str, season: int, subreddit_multi) -> dict:
    start_ts, end_ts = season_window(season)
    pattern = compile_name_pattern(player_name)
    query = f"\"{player_name}\""

    posts_seen = 0
    pos = neu = neg = 0
    compounds = []
    subreddits_hit = set()

    for submission in subreddit_multi.search(query=query, sort="new", limit=MAX_POSTS_PER_PLAYER):
        created = int(getattr(submission, "created_utc", 0))
       # if not (start_ts <= created <= end_ts):
       #     continue

        posts_seen += 1
        subreddits_hit.add(str(submission.subreddit).lower())

        # --- Post text ---
        text_norm = normalize_name(f"{submission.title or ''}\n{submission.selftext or ''}")
        if pattern.search(text_norm):
            label, comp = sentiment_of(text_norm)
            if label == "pos": pos += 1
            elif label == "neu": neu += 1
            else: neg += 1
            compounds.append(comp)

        # --- Comments ---
        try:
            submission.comments.replace_more(limit=0)
            for c in submission.comments[:MAX_COMMENTS_PER_POST]:
                c_created = int(getattr(c, "created_utc", 0))
               # if not (start_ts <= c_created <= end_ts):
                #    continue
                body_norm = normalize_name(getattr(c, "body", ""))
                if pattern.search(body_norm):
                    label, comp = sentiment_of(body_norm)
                    if label == "pos": pos += 1
                    elif label == "neu": neu += 1
                    else: neg += 1
                    compounds.append(comp)
        except Exception as ce:
            logging.debug(f"Comments error for {player_name} {season}: {ce}")

    total_used = pos + neu + neg
    if total_used < MIN_COMMENTS_THRESHOLD:
        return {
            "player_name": player_name,
            "season": season,
            "num_posts": posts_seen,
            "num_comments_used": total_used,
            "pos_ratio": np.nan,
            "neu_ratio": np.nan,
            "neg_ratio": np.nan,
            "mean_compound": np.nan,
            "fallback_used": True,
            "subreddits_covered": ",".join(sorted(subreddits_hit))
        }

    return {
        "player_name": player_name,
        "season": season,
        "num_posts": posts_seen,
        "num_comments_used": total_used,
        "pos_ratio": pos / total_used,
        "neu_ratio": neu / total_used,
        "neg_ratio": neg / total_used,
        "mean_compound": float(np.mean(compounds)) if compounds else 0.0,
        "fallback_used": False,
        "subreddits_covered": ",".join(sorted(subreddits_hit))
    }

# =======================
# Main
# =======================
def main():
    pairs = pd.read_csv(INPUT_FILE)[["player_name", "season"]]
    logging.info(f"Loaded {len(pairs)} player-season pairs")

    multi = reddit.subreddit("+".join(SUBREDDITS))
    rows = []
    processed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(collect_for_player_season, r.player_name, int(r.season), multi)
                for r in pairs.itertuples(index=False)]

        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception as e:
                logging.warning(f"Worker error: {e}")
            processed += 1

            if processed % SAVE_EVERY == 0:
                pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, mode="a", header=not os.path.exists(OUTPUT_FILE))
                rows.clear()
                logging.info(f"Progress saved at {processed} pairs")

    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, mode="a", header=not os.path.exists(OUTPUT_FILE))
        logging.info(f"Final results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
