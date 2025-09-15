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
    #4
    total_injuries → Count of injuries.
    avg_days_out → Average days missed per injury.
    recent_injury → Flag (1 if player had injury in last 12 months, else 0).
    days_since_last_injury → Days since last injury (NULL if never injured).

    total_injuries: How many injuries a player has had.
    avg_days_out: Average recovery time.
    recent_injury: Boolean flag for injury in last 12 months.
    days_since_last_injury: Gap since last injury.
"""
# -----------------------
# Extend player_features table for injuries
# -----------------------
cursor.execute("""
ALTER TABLE player_features 
ADD COLUMN IF NOT EXISTS total_injuries INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS avg_days_out FLOAT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS recent_injury TINYINT(1) DEFAULT 0,
ADD COLUMN IF NOT EXISTS days_since_last_injury INT DEFAULT NULL;
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
# Helper: Injury Features
# -----------------------
def get_injury_features(player_id, transfermarkt_id):
    cursor.execute("""
        SELECT start_date, end_date, days_out
        FROM player_injuries_trfrmrkt
        WHERE player_id=%s AND transfermarkt_id=%s
    """, (player_id, transfermarkt_id))
    rows = cursor.fetchall()
    if not rows:
        return 0, None, 0, None

    total = len(rows)
    valid_days = [r[2] for r in rows if r[2] is not None]
    avg_days = float(sum(valid_days) / len(valid_days)) if valid_days else None

    today = datetime.today()
    last_injury_end = max([r[1] for r in rows if r[1] is not None], default=None)

    recent = 0
    days_since = None
    if last_injury_end:
        days_since = (today - last_injury_end).days
        if days_since <= 365:
            recent = 1

    return total, avg_days, recent, days_since

# -----------------------
# Player Mapping
# -----------------------
cursor.execute("SELECT player_id_trfrmrkt, player_name_trfrmrkt, transfermarkt_id FROM player_mapping WHERE is_confirmed=1")
mapping = cursor.fetchall()

# -----------------------
# Update Player Features
# -----------------------
for pid, pname, tm_id in mapping:
    try:
        sentiment_trend = calc_sentiment_trend(pname)
        positions, primary_position = get_positions(pid)
        current_club_id = get_current_club(pid)
        total_inj, avg_days, recent, days_since = get_injury_features(pid, tm_id)

        cursor.execute("""
            UPDATE player_features
            SET sentiment_trend=%s, positions_played=%s, primary_position=%s, 
                current_club_id=%s, total_injuries=%s, avg_days_out=%s, 
                recent_injury=%s, days_since_last_injury=%s
            WHERE player_id=%s
        """, (sentiment_trend, positions, primary_position, current_club_id,
              total_inj, avg_days, recent, days_since, pid))
        db.commit()

        print(f"✅ {pname}: injuries={total_inj}, avg_days={avg_days}, recent={recent}, days_since={days_since}")
    except Exception as e:
        print(f"⚠️ Error updating {pname}: {e}")

cursor.close()
db.close()
print("🎉 Player features updated with injury metrics!")
