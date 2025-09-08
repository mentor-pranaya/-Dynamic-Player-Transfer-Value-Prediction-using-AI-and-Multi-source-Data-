import logging
from datetime import datetime
from typing import List, Tuple
import praw
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

REDDIT_CFG = dict(
    client_id="O-8pbql513rmZav78DBzeA",
    client_secret="igz2dgdaBnYVg8xDMuE07-J9K52uew",
    user_agent="AIProject by u/phoenixeuphoric"
)
REDDIT_PER_PLAYER = 300

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =========================
# DB HELPERS
# =========================
def get_db():
    return mysql.connector.connect(**MYSQL_CFG)

def ensure_reddit_table():
    db = get_db()
    cur = db.cursor()
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
    db.commit(); cur.close(); db.close()

def upsert_bulk(rows: List[Tuple]):
    if not rows: return
    db = get_db(); cur = db.cursor()
    sql = """INSERT INTO reddit_sentiments 
        (player_name, post_id, subreddit, created_at, title, selftext, sentiment, polarity)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
        title=VALUES(title), selftext=VALUES(selftext), sentiment=VALUES(sentiment), polarity=VALUES(polarity);"""
    cur.executemany(sql, rows); db.commit()
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
# REDDIT
# =========================
def reddit_client():
    return praw.Reddit(**REDDIT_CFG)

def fetch_reddit(player: str, max_items: int = REDDIT_PER_PLAYER):
    reddit = reddit_client()
    rows = []
    for submission in reddit.subreddit("all").search(player, sort="new", limit=max_items):
        text = f"{submission.title}\n{submission.selftext or ''}"
        sent, pol = analyze_sentiment(text)
        created = datetime.utcfromtimestamp(submission.created_utc)
        rows.append((
            player,
            submission.id,
            submission.subreddit.display_name,
            created,
            submission.title,
            submission.selftext or "",
            sent,
            pol
        ))
    return rows

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_reddit_table()
    #players = ["Erling Haaland", "Bukayo Saka"]  # test set
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT DISTINCT player FROM transfermrkt")
    players = [r[0] for r in cur.fetchall()]
    cur.close()
    db.close()

    for p in players:
        rows = fetch_reddit(p)
        upsert_bulk(rows)
        logging.info(f"✅ {p}: {len(rows)} reddit posts saved")

