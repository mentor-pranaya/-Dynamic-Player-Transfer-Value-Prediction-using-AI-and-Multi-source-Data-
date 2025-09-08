# db_helpers.py
import mysql.connector
from typing import List, Tuple

MYSQL_CFG = dict(
    host="localhost",
    user="himanshu",
    password="yahoonet",
    database="AIProject",
    autocommit=False,
)

BATCH_INSERT_SIZE = 200

def get_db():
    return mysql.connector.connect(**MYSQL_CFG)

def ensure_tables():
    """Creates all sentiment tables if they don't exist"""
    db = get_db(); cur = db.cursor()

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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tw_player ON twitter_sentiments(player_name);")

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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rd_player ON reddit_sentiments(player_name);")

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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_md_player ON medium_sentiments(player_name);")

    db.commit(); cur.close(); db.close()

def upsert_bulk(table: str, rows: List[Tuple], columns: List[str], on_dup: str):
    """Bulk UPSERT rows into MySQL"""
    if not rows: return
    db = get_db(); cur = db.cursor()

    col_list = ",".join(columns)
    placeholders = ",".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) {on_dup}"

    for i in range(0, len(rows), BATCH_INSERT_SIZE):
        batch = rows[i:i+BATCH_INSERT_SIZE]
        cur.executemany(sql, batch)
        db.commit()

    cur.close(); db.close()

