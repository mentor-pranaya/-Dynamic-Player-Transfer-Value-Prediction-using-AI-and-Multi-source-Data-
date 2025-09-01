import os
import time
import math
import logging
import traceback
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

import tweepy
import praw
import mysql.connector
from mysql.connector import Error as MySQLError

from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG
# =========================

MYSQL_CFG = dict(
    host="localhost",
    user="himanshu",
    password="yahoonet",  
    database="AIProject",
    autocommit=False,
)

# Twitter API v2 
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAACsd3wEAAAAANCYjgfQmC015%2Bh5NWv75x7HxUWs%3D4zi50QL8WPjjh8UFp6BcB60UPvJcFVReCBD5PV3BKC4w4KRQXZ"
# Per-player tweet target (adjust to your access level). 7-day recent search without Academic.
TW_MAX_PER_PLAYER = 1200

# Reddit API
REDDIT_CFG = dict(
    client_id="O-8pbql513rmZav78DBzeA",
    client_secret="igz2dgdaBnYVg8xDMuE07-J9K52uew",
    user_agent="AIProject by u/phoenixeuphoric"
)
REDDIT_PER_PLAYER = 500

# Medium scrape pages per player (be polite; Medium blocks aggressive scraping)
MEDIUM_PAGES_PER_PLAYER = 25
HTTP_TIMEOUT = 15
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36"}

# Threading
MAX_WORKERS = 24  # network I/O bound ⇒ threads work well; tune 16–32
BATCH_INSERT_SIZE = 200  # bulk insert to DB per table

# Resume file
RESUME_FILE = "processed_players.txt"

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("sentiment_pipeline_v2.log"), logging.StreamHandler()],
)

# =========================
# DB HELPERS
# =========================
def get_db():
    return mysql.connector.connect(**MYSQL_CFG)

def ensure_tables():
    db = get_db()
    cur = db.cursor()
    # Twitter
    cur.execute("""
        CREATE TABLE IF NOT EXISTS twitter_sentiments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            player_name VARCHAR(100),
            tweet_id VARCHAR(64) UNIQUE,
            created_at DATETIME,
            content TEXT,
            sentiment VARCHAR(16),
            polarity FLOAT
        ) ENGINE=InnoDB;
    """)
    cur.execute("CREATE INDEX idx_tw_player ON twitter_sentiments(player_name);")

    # Reddit
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reddit_sentiments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            player_name VARCHAR(100),
            post_id VARCHAR(64) UNIQUE,
            subreddit VARCHAR(100),
            created_at DATETIME,
            title TEXT,
            selftext MEDIUMTEXT,
            sentiment VARCHAR(16),
            polarity FLOAT
        ) ENGINE=InnoDB;
    """)
    cur.execute("CREATE INDEX idx_rd_player ON reddit_sentiments(player_name);")

    # Medium
    cur.execute("""
        CREATE TABLE IF NOT EXISTS medium_sentiments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            player_name VARCHAR(100),
            url TEXT,
            url_hash VARCHAR(191) UNIQUE,
            title TEXT,
            created_at DATETIME,
            sentiment VARCHAR(16),
            polarity FLOAT
        ) ENGINE=InnoDB;
    """)
    cur.execute("CREATE INDEX idx_md_player ON medium_sentiments(player_name);")

    # Simple helper table to log runs (optional)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_run_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            run_started DATETIME,
            run_finished DATETIME
        ) ENGINE=InnoDB;
    """)
    db.commit()
    cur.close()
    db.close()

def upsert_bulk(table: str, rows: List[Tuple], on_dup: str, columns: List[str]):
    """Bulk upsert (INSERT ... ON DUPLICATE KEY UPDATE ...)"""
    if not rows:
        return
    db = get_db()
    cur = db.cursor()
    col_list = ",".join(columns)
    placeholders = ",".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) {on_dup}"
    for i in range(0, len(rows), BATCH_INSERT_SIZE):
        batch = rows[i:i+BATCH_INSERT_SIZE]
        cur.executemany(sql, batch)
        db.commit()
    cur.close()
    db.close()

# =========================
# SENTIMENT
# =========================
def analyze_sentiment(text: str) -> Tuple[str, float]:
    text = (text or "").strip()
    blob = TextBlob(text)
    p = blob.sentiment.polarity
    if p > 0.05:
        label = "positive"
    elif p < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, float(p)

# =========================
# TWITTER (tweepy v2)
# =========================
def twitter_client() -> tweepy.Client:
    return tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)

def fetch_twitter(player: str, max_items: int = TW_MAX_PER_PLAYER) -> List[Tuple]:
    """
    Returns list of tuples for bulk insert:
    (player_name, tweet_id, created_at, content, sentiment, polarity)
    """
    client = twitter_client()
    query = f'"{player}" -is:retweet lang:en'
    got = 0
    results = []

    # recent search (last 7 days) unless you have Academic full-archive set via 'search_all_tweets'
    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["created_at", "lang", "text", "id"],
        max_results=100,
    )

    for page in paginator:
        if page.data is None:
            break
        for tw in page.data:
            if tw.lang != "en":
                continue
            sent, pol = analyze_sentiment(tw.text)
            results.append((player, str(tw.id), tw.created_at.replace(tzinfo=None), tw.text, sent, pol))
            got += 1
            if got >= max_items:
                break
        if got >= max_items:
            break

    return results

# =========================
# REDDIT (PRAW)
# =========================
def reddit_client():
    return praw.Reddit(**REDDIT_CFG)

def fetch_reddit(player: str, max_items: int = REDDIT_PER_PLAYER) -> List[Tuple]:
    """
    (player_name, post_id, subreddit, created_at, title, selftext, sentiment, polarity)
    """
    reddit = reddit_client()
    results = []
    count = 0
    # PRAW search caps ~1000; we fetch in smaller chunks by sorting 'new' and paginating by 'limit'
    # This won’t get historical beyond a few months — for deep history, consider Pushshift (not always reliable).
    for submission in reddit.subreddit("all").search(player, sort="new", limit=max_items):
        title = submission.title or ""
        body = submission.selftext or ""
        text = f"{title}\n{body}".strip()
        sent, pol = analyze_sentiment(text)
        created = datetime.utcfromtimestamp(submission.created_utc)
        results.append((
            player,
            submission.id,
            submission.subreddit.display_name,
            created,
            title,
            body,
            sent,
            pol
        ))
        count += 1
        if count >= max_items:
            break
    return results

# =========================
# MEDIUM (scrape)
# =========================
def hash_for_url(url: str) -> str:
    # simple MySQL-safe hash (191 char unique column); we can store a truncated SHA256 hex
    import hashlib
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:191]

def fetch_medium(player: str, pages: int = MEDIUM_PAGES_PER_PLAYER) -> List[Tuple]:
    """
    (player_name, url, url_hash, title, created_at, sentiment, polarity)
    Note: Medium search page doesn’t expose article publish date reliably; we stamp 'now'.
    """
    results = []
    now = datetime.utcnow()
    for page in range(1, pages + 1):
        url = f"https://medium.com/search?q={quote_plus(player)}&page={page}"
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                logging.warning(f"Medium HTTP {r.status_code} for {player} page {page}")
                break
            soup = BeautifulSoup(r.text, "html.parser")

            # Medium markup changes often; this selector targets card titles broadly:
            for h in soup.find_all(["h2", "h3"]):
                title = (h.get_text() or "").strip()
                if not title:
                    continue
                # find nearest parent anchor
                a = h.find_parent("a")
                href = a["href"].split("?")[0] if a and a.has_attr("href") else None
                if not href:
                    continue
                sent, pol = analyze_sentiment(title)
                results.append((player, href, hash_for_url(href), title, now, sent, pol))
            time.sleep(0.6)  # be polite
        except Exception as e:
            logging.error(f"Medium error for {player} p{page}: {e}")
            break
    return results

# =========================
# PER-PLAYER WORK
# =========================
def process_player(player: str) -> Tuple[str, int, int, int]:
    tw_rows = rd_rows = md_rows = 0
    try:
        # --- Twitter ---
        try:
            twitter_rows = fetch_twitter(player, max_items=TW_MAX_PER_PLAYER)
            upsert_bulk(
                table="twitter_sentiments",
                rows=twitter_rows,
                columns=["player_name", "tweet_id", "created_at", "content", "sentiment", "polarity"],
                on_dup="ON DUPLICATE KEY UPDATE content=VALUES(content), sentiment=VALUES(sentiment), polarity=VALUES(polarity), created_at=VALUES(created_at)"
            )
            tw_rows = len(twitter_rows)
        except tweepy.TooManyRequests as e:
            # Hard backoff if we hit rate limit mid-batch
            logging.warning(f"Twitter rate limit for {player}: {e}. Backing off 900s.")
            time.sleep(900)
        except Exception as e:
            logging.error(f"Twitter error for {player}: {e}")

        # --- Reddit ---
        try:
            reddit_rows = fetch_reddit(player, max_items=REDDIT_PER_PLAYER)
            upsert_bulk(
                table="reddit_sentiments",
                rows=reddit_rows,
                columns=["player_name", "post_id", "subreddit", "created_at", "title", "selftext", "sentiment", "polarity"],
                on_dup="ON DUPLICATE KEY UPDATE title=VALUES(title), selftext=VALUES(selftext), sentiment=VALUES(sentiment), polarity=VALUES(polarity), created_at=VALUES(created_at)"
            )
            rd_rows = len(reddit_rows)
        except Exception as e:
            logging.error(f"Reddit error for {player}: {e}")

        # --- Medium ---
        try:
            medium_rows = fetch_medium(player, pages=MEDIUM_PAGES_PER_PLAYER)
            upsert_bulk(
                table="medium_sentiments",
                rows=medium_rows,
                columns=["player_name", "url", "url_hash", "title", "created_at", "sentiment", "polarity"],
                on_dup="ON DUPLICATE KEY UPDATE title=VALUES(title), sentiment=VALUES(sentiment), polarity=VALUES(polarity), created_at=VALUES(created_at)"
            )
            md_rows = len(medium_rows)
        except Exception as e:
            logging.error(f"Medium error for {player}: {e}")

        # Mark processed
        with open(RESUME_FILE, "a", encoding="utf-8") as f:
            f.write(player + "\n")

    except Exception as e:
        logging.error(f"General error for {player}: {e}\n{traceback.format_exc()}")

    return player, tw_rows, rd_rows, md_rows

# =========================
# MAIN
# =========================
def load_players() -> List[str]:
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT DISTINCT player FROM transfermrkt")
    players = [r[0] for r in cur.fetchall()]
    cur.close()
    db.close()
    return players

def already_processed() -> set:
    if not os.path.exists(RESUME_FILE):
        return set()
    with open(RESUME_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def main():
    ensure_tables()

    all_players = load_players()
    done = already_processed()
    pending = [p for p in all_players if p not in done]

    logging.info(f"Players total: {len(all_players)}; already done: {len(done)}; to do: {len(pending)}")

    run_start = datetime.utcnow()

    # Thread pool (network I/O bound)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_player, p): p for p in pending}
        for fut in as_completed(futures):
            player = futures[fut]
            try:
                p, tw_n, rd_n, md_n = fut.result()
                logging.info(f"✔ {p}: twitter={tw_n}, reddit={rd_n}, medium={md_n}")
            except Exception as e:
                logging.error(f"Worker crash for {player}: {e}")

    run_end = datetime.utcnow()
    # Log run
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO sentiment_run_log (run_started, run_finished) VALUES (%s, %s)", (run_start, run_end))
    db.commit()
    cur.close()
    db.close()

    logging.info("🎉 Sentiment pipeline finished.")

if __name__ == "__main__":
    main()

