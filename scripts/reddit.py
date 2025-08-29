import os
import re
import time
import logging
import unicodedata
from datetime import datetime, timezone
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from tqdm import tqdm
import praw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------- CONFIG -------------
PROJECT_PATH   = r"C:/Users/Abhinav/Desktop/Project"
PROCESSED_DIR  = os.path.join(PROJECT_PATH, "processed_data")
FINAL_DATA     = os.path.join(PROCESSED_DIR, "final_data.csv")
OUT_SENTIMENT  = os.path.join(PROCESSED_DIR, "reddit_sentiment.csv")

SUBREDDITS = ["soccer", "premierleague", "LaLiga", "seriea", "bundesliga", "ligue1"]

MAX_POSTS_PER_PLAYER   = 40
MAX_COMMENTS_PER_POST  = 100
MIN_COMMENTS_THRESHOLD = 20
WORKERS                = 4
SAVE_EVERY             = 200
BASE_SEARCH_SLEEP      = 0.2
# ----------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---- VADER ----
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
logging.info("VADER SentimentIntensityAnalyzer ready.")

# ---- Reddit ----
load_dotenv()
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", 
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36")

if not CLIENT_ID or not CLIENT_SECRET or not USER_AGENT:
    raise RuntimeError("Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT env vars in .env")

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# ---- Helpers ----
def season_window(season_year: int):
    start = datetime(season_year, 7, 1, tzinfo=timezone.utc)
    end   = datetime(season_year + 1, 6, 30, 23, 59, 59, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())

def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.split())

def compile_name_pattern(name: str):
    norm = normalize_name(name)
    return re.compile(r"\b" + re.escape(norm) + r"\b")

def sentiment_of(text: str):
    s = sia.polarity_scores(text)
    label = max(("pos", s["pos"]), ("neu", s["neu"]), ("neg", s["neg"]), key=lambda x: x[1])[0]
    return label, s["compound"]

def backoff_sleep(attempt: int, base: float = 1.5, cap: float = 60.0):
    t = min(cap, base * (2 ** attempt)) * (1 + 0.1 * np.random.rand())
    time.sleep(t)

# ---- Core ----
def collect_for_player_season(player_name: str, season: int, subreddit_multi) -> dict:
    start_ts, end_ts = season_window(season)
    pattern = compile_name_pattern(player_name)
    query   = f"\"{player_name}\""

    posts_seen = 0
    pos = neu = neg = 0
    compounds = []
    subreddits_hit = set()

    for attempt in range(5):
        try:
            time.sleep(BASE_SEARCH_SLEEP)
            for submission in subreddit_multi.search(query=query, sort="new", limit=MAX_POSTS_PER_PLAYER):
                created = int(getattr(submission, "created_utc", 0))
                if not (start_ts <= created <= end_ts):
                    continue

                posts_seen += 1
                subreddits_hit.add(str(submission.subreddit).lower())

                # post text
                text_norm = normalize_name(f"{submission.title or ''}\n{submission.selftext or ''}")
                if pattern.search(text_norm):
                    label, comp = sentiment_of(text_norm)
                    if label == "pos": pos += 1
                    elif label == "neu": neu += 1
                    else: neg += 1
                    compounds.append(comp)

                # comments
                try:
                    submission.comments.replace_more(limit=0)
                    for c in submission.comments[:MAX_COMMENTS_PER_POST]:
                        c_created = int(getattr(c, "created_utc", 0))
                        if not (start_ts <= c_created <= end_ts):
                            continue
                        body = getattr(c, "body", "")
                        if not body.strip():
                            continue
                        body_norm = normalize_name(body)
                        if pattern.search(body_norm):
                            label, comp = sentiment_of(body_norm)
                            if label == "pos": pos += 1
                            elif label == "neu": neu += 1
                            else: neg += 1
                            compounds.append(comp)
                except Exception as ce:
                    logging.debug(f"Comments error for {player_name} {season}: {ce}")

            break  # success
        except Exception as e:
            msg = str(e)
            logging.warning(f"Error {player_name} {season} attempt {attempt+1}: {msg}")
            backoff_sleep(attempt)

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

def load_pairs_from_final(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "player_name" not in df.columns or "season" not in df.columns:
        raise ValueError("final_data.csv must have 'player_name' and 'season'.")
    pairs = (
        df[["player_name", "season"]]
        .dropna()
        .assign(season=lambda d: pd.to_numeric(d["season"], errors="coerce").astype("Int64"))
        .dropna(subset=["season"])
        .drop_duplicates()
        .sort_values(["player_name", "season"])
        .reset_index(drop=True)
    )
    return pairs

# ---- Main ----
def main():
    pairs = load_pairs_from_final(FINAL_DATA)
    logging.info(f"Target player-season pairs from final_data: {len(pairs):,}")

    # resume support
    if os.path.exists(OUT_SENTIMENT):
        done = pd.read_csv(OUT_SENTIMENT)[["player_name", "season"]].drop_duplicates()
        before = len(pairs)
        pairs = (pairs.merge(done.assign(done=1), on=["player_name","season"], how="left")
                      .query("done != 1")
                      .drop(columns=["done"]))
        logging.info(f"Resuming: {before:,} → {len(pairs):,} remaining")

    if pairs.empty:
        logging.info("Nothing to do — all pairs already processed.")
        return

    multi = reddit.subreddit("+".join(SUBREDDITS))
    rows = []
    processed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(collect_for_player_season, r.player_name, int(r.season), multi)
                for r in pairs.itertuples(index=False)]

        for f in tqdm(as_completed(futs), total=len(futs), desc="Collecting sentiment", leave=False):
            try:
                rows.append(f.result())
            except Exception as e:
                logging.warning(f"Worker error: {e}")
            processed += 1

            if processed % SAVE_EVERY == 0:
                out_partial = pd.DataFrame(rows)
                if os.path.exists(OUT_SENTIMENT):
                    old = pd.read_csv(OUT_SENTIMENT)
                    out_partial = pd.concat([old, out_partial], ignore_index=True).drop_duplicates(
                        subset=["player_name","season"], keep="last"
                    )
                out_partial.to_csv(OUT_SENTIMENT, index=False)
                rows.clear()
                logging.info(f"Progress saved at {processed} pairs.")

    # final flush
    if rows:
        out = pd.DataFrame(rows)
        if os.path.exists(OUT_SENTIMENT):
            old = pd.read_csv(OUT_SENTIMENT)
            out = pd.concat([old, out], ignore_index=True).drop_duplicates(
                subset=["player_name","season"], keep="last"
            )
        out.to_csv(OUT_SENTIMENT, index=False)
        logging.info(f"Saved Reddit sentiment to: {OUT_SENTIMENT}")
        logging.info(out.head().to_string())

if __name__ == "__main__":
    main()
