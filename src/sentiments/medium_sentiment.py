import logging, time, hashlib
from datetime import datetime
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
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

MEDIUM_PAGES_PER_PLAYER = 3
HTTP_TIMEOUT = 15
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =========================
# DB HELPERS
# =========================
def get_db():
    return mysql.connector.connect(**MYSQL_CFG)

def ensure_medium_table():
    db = get_db(); cur = db.cursor()
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
    db.commit(); cur.close(); db.close()

def upsert_bulk(rows: List[Tuple]):
    if not rows: return
    db = get_db(); cur = db.cursor()
    sql = """INSERT INTO medium_sentiments
        (player_name, url, url_hash, title, created_at, sentiment, polarity)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE title=VALUES(title), sentiment=VALUES(sentiment), polarity=VALUES(polarity);"""
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
# MEDIUM
# =========================
def hash_for_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:191]

def fetch_medium(player: str, pages: int = MEDIUM_PAGES_PER_PLAYER):
    rows = []
    now = datetime.utcnow()
    for page in range(1, pages+1):
        url = f"https://medium.com/search?q={quote_plus(player)}&page={page}"
        r = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            logging.warning(f"Medium {r.status_code} for {player} p{page}")
            break
        soup = BeautifulSoup(r.text, "html.parser")
        for h in soup.find_all(["h2", "h3"]):
            title = h.get_text(strip=True)
            if not title: continue
            a = h.find_parent("a")
            href = a["href"].split("?")[0] if a and a.has_attr("href") else None
            if not href: continue
            sent, pol = analyze_sentiment(title)
            rows.append((player, href, hash_for_url(href), title, now, sent, pol))
        time.sleep(0.5)
    return rows

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_medium_table()
    players = ["Erling Haaland", "Bukayo Saka"]  # test set

    for p in players:
        rows = fetch_medium(p)
        upsert_bulk(rows)
        logging.info(f"✅ {p}: {len(rows)} medium articles saved")

