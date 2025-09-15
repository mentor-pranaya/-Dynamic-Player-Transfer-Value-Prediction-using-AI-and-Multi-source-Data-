import mysql.connector
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# -----------------------
# DB Connection
# -----------------------
db = mysql.connector.connect(
    host="localhost",
    user="himanshu",
    password="yahoonet",
    database="AIProject"
)
cursor = db.cursor()
"""
    #2
    sentiment_trend
    Uses linear regression slope of sentiment polarity over time.
    Captures whether a player’s reputation is improving or declining.
    positions_played
    Collects all unique positions across matches.
    Example: "Right Wing, Left Wing, Center Forward".
    current_club_id
    Added column and foreign key for linking to clubs_trfrmrkt.
    You can populate it later from the latest market_values_trfrmrkt.
"""
# -----------------------
# Extend player_features table
# -----------------------
cursor.execute("""
ALTER TABLE player_features 
ADD COLUMN IF NOT EXISTS sentiment_trend FLOAT,
ADD COLUMN IF NOT EXISTS positions_played VARCHAR(500),
ADD COLUMN IF NOT EXISTS current_club_id INT,
ADD FOREIGN KEY (current_club_id) REFERENCES clubs_trfrmrkt(id);
""")
db.commit()

# -----------------------
# Helper: Sentiment Trend
# -----------------------
def calc_sentiment_trend(player_name):
    cursor.execute("""
        SELECT created_at, polarity
        FROM (
            SELECT player_name, created_at, polarity FROM twitter_sentiments
            UNION ALL
            SELECT player_name, created_at, polarity FROM reddit_sentiments
        ) s
        WHERE player_name=%s
        ORDER BY created_at
    """, (player_name,))
    rows = cursor.fetchall()
    if not rows or len(rows) < 5:
        return None  # not enough data

    dates = np.array([(r[0] - rows[0][0]).days for r in rows]).reshape(-1, 1)
    polarities = np.array([r[1] for r in rows])

    model = LinearRegression()
    model.fit(dates, polarities)
    return float(model.coef_[0])  # slope = sentiment trend

# -----------------------
# Helper: Positions Played
# -----------------------
def get_positions(player_id):
    cursor.execute("""
        SELECT DISTINCT position_name
        FROM (
            SELECT position_name FROM lineups WHERE player_id=%s
            UNION
            SELECT position_name FROM lineup_positions WHERE player_id=%s
        ) p
    """, (player_id, player_id))
    rows = cursor.fetchall()
    if not rows:
        return None
    return ", ".join(sorted({r[0] for r in rows if r[0]}))

# -----------------------
# Player Mapping
# -----------------------
cursor.execute("SELECT player_id_trfrmrkt, player_name_trfrmrkt FROM player_mapping WHERE is_confirmed=1")
mapping = dict(cursor.fetchall())

# -----------------------
# Update Player Features
# -----------------------
for pid, pname in mapping.items():
    try:
        sentiment_trend = calc_sentiment_trend(pname)
        positions = get_positions(pid)

        cursor.execute("""
            UPDATE player_features
            SET sentiment_trend=%s, positions_played=%s
            WHERE player_id=%s
        """, (sentiment_trend, positions, pid))
        db.commit()

        print(f"✅ Updated {pname} → sentiment_trend={sentiment_trend}, positions={positions}")
    except Exception as e:
        print(f"⚠️ Error updating {pname}: {e}")

cursor.close()
db.close()
print("🎉 Player features extended with sentiment trend & positions!")
