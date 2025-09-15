import mysql.connector
import pandas as pd
from datetime import datetime
from collections import Counter
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
    #3
    primary_position = Most frequent position across all matches.
    current_club_id = From the latest market value record.
    positions_played = All distinct positions.
"""

# -----------------------
# Extend player_features table
# -----------------------
cursor.execute("""
ALTER TABLE player_features 
ADD COLUMN IF NOT EXISTS primary_position VARCHAR(100),
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
        return None

    dates = np.array([(r[0] - rows[0][0]).days for r in rows]).reshape(-1, 1)
    polarities = np.array([r[1] for r in rows])

    model = LinearRegression()
    model.fit(dates, polarities)
    return float(model.coef_[0])

# -----------------------
# Helper: Positions
# -----------------------
def get_positions(player_id):
    cursor.execute("""
        SELECT position_name
        FROM (
            SELECT position_name FROM lineups WHERE player_id=%s
            UNION ALL
            SELECT position_name FROM lineup_positions WHERE player_id=%s
        ) p
    """, (player_id, player_id))
    rows = cursor.fetchall()
    if not rows:
        return None, None

    pos_list = [r[0] for r in rows if r[0]]
    unique_positions = ", ".join(sorted(set(pos_list)))

    # Find most frequent position
    primary = Counter(pos_list).most_common(1)[0][0] if pos_list else None
    return unique_positions, primary

# -----------------------
# Helper: Current Club
# -----------------------
def get_current_club(player_id):
    cursor.execute("""
        SELECT club_id
        FROM market_values_trfrmrkt
        WHERE player_id=%s
        ORDER BY snapshot_date DESC
        LIMIT 1
    """, (player_id,))
    row = cursor.fetchone()
    return row[0] if row else None

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
        positions, primary_position = get_positions(pid)
        current_club_id = get_current_club(pid)

        cursor.execute("""
            UPDATE player_features
            SET sentiment_trend=%s, positions_played=%s, primary_position=%s, current_club_id=%s
            WHERE player_id=%s
        """, (sentiment_trend, positions, primary_position, current_club_id, pid))
        db.commit()

        print(f"✅ {pname}: trend={sentiment_trend}, club_id={current_club_id}, primary={primary_position}, all={positions}")
    except Exception as e:
        print(f"⚠️ Error updating {pname}: {e}")

cursor.close()
db.close()
print("🎉 Player features updated with club + positions!")
