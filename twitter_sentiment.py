import os, time, logging
from datetime import datetime
from typing import List, Tuple
import tweepy
from textblob import TextBlob
import mysql.connector

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

TWITTER_BEARER_TOKEN = "YOUR_BEARER_TOKEN"
TW_MAX_PER_PLAYER = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =========================
# DB HELPERS
# =========================
def get_db():
    return mysql.connector.connect(**MYSQL_CFG)

def ensure_twitter_table():
    db = get_db()
    cur = db.cursor()
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
    db.commit()
    cur.close(); db.close()

def upsert_bulk(rows: List[Tuple]):
    if not rows: return
    db = get_db()
    cur = db.cursor()
    sql = """INSERT INTO twitter_sentiments 
        (player_name, tweet_id, created_at, content, sentiment, polarity) 
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
        content=VALUES(content), sentiment=VALUES(sentiment), polarity=VALUES(polarity);"""
    cur.executemany(sql, rows)
    db.commit()
    cur.close(); db.close()

# =========================
# SENTIMENT
# =========================
def analyze_sentiment(text: str):
    p = TextBlob(text).sentiment.polarity
    if p > 0.05: return "positive", p
    if p < -0.05: return "negative", p
    return "neutral", p

# =========================
# TWITTER
# =========================
def twitter_client():
    return tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)

def fetch_twitter(player: str, max_items: int = TW_MAX_PER_PLAYER):
    client = twitter_client()
    query = f'"{player}" -is:retweet lang:en'
    rows = []
    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["created_at", "lang", "text", "id"],
        max_results=100,
    )
    got = 0
    for page in paginator:
        if page.data is None: break
        for tw in page.data:
            if tw.lang != "en": continue
            sent, pol = analyze_sentiment(tw.text)
            rows.append((player, str(tw.id), tw.created_at.replace(tzinfo=None), tw.text, sent, pol))
            got += 1
            if got >= max_items: break
        if got >= max_items: break
    return rows

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_twitter_table()
    players = ["Erling Haaland", "Bukayo Saka"]  # test set

    for p in players:
        rows = fetch_twitter(p)
        upsert_bulk(rows)
        logging.info(f"✅ {p}: {len(rows)} tweets saved")

