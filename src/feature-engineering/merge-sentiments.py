import mysql.connector
import pandas as pd
from datetime import date, datetime
from collections import Counter
from sklearn.linear_model import LinearRegression
import numpy as np

# -----------------------
# DB Connection
# -----------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
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
#cursor.execute("""
#ALTER TABLE player_features 
#ADD COLUMN sentiment_trend FLOAT,
#ADD COLUMN positions_played VARCHAR(500),
#ADD COLUMN current_club_id INT,
#ADD FOREIGN KEY (current_club_id) REFERENCES clubs_trfrmrkt(id);
#""")
#db.commit()

# -----------------------
# Helper: Sentiment Trend
# -----------------------
def calc_sentiment_trend(player_name):
    cursor.execute("""
        SELECT created_at, polarity
        FROM (
            SELECT transfermarkt_id, created_at, polarity FROM reddit_sentiments
        ) s
        WHERE transfermarkt_id=%s
        ORDER BY created_at
    """, (player_name,))
    rows = cursor.fetchall()
    if not rows or len(rows) < 5:
        return None  # not enough data

    dates = np.array([(r[0] - rows[0][0]).days for r in rows]).reshape(-1, 1)
    polarities = np.array([r[1] for r in rows])

    model = LinearRegression()
    model.fit(dates, polarities)
    print(f"Sentiment trend for {player_name}: {model.coef_[0]}")
    return float(model.coef_[0])  # slope = sentiment trend

# -----------------------
# Helper: Positions Played
# -----------------------

def get_positions_old(player_id):
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
        SELECT club_join_id
        FROM player_transfer_history
        WHERE transfermarkt_id=%s
        ORDER BY transfer_date DESC
        LIMIT 1
    """, (player_id,))
    row = cursor.fetchone()
    return row[0] if row else None

# -----------------------
# Helper: Injury Features
# -----------------------
def get_injury_features(player_id):
    cursor.execute("""
        SELECT start_date, if(end_date<>'1990-01-01',end_date,curdate()) as end_date, days_out
        FROM player_injuries_trfrmrkt
        WHERE transfermarkt_id=%s
    """, [player_id])
    rows = cursor.fetchall()
    if not rows:
        return 0, None, 0, None

    total = len(rows)
    valid_days = [r[2] for r in rows if r[2] is not None]
    avg_days = float(sum(valid_days) / len(valid_days)) if valid_days else None

    today = date.today()
    last_injury_end = max([r[1] for r in rows if r[1] is not None], default=None)

    recent = 0
    days_since = None
    if last_injury_end:
        print(f"Last injury end date for {player_id}: {last_injury_end}, today: {today}")
        days_since = (today - last_injury_end).days
        if days_since <= 365:
            recent = 1

    return total, avg_days, recent, days_since
# -----------------------
# Player Mapping
# -----------------------
cursor.execute("SELECT transfermarkt_id, statsbomb_player_id FROM player_mapping")
mapping = dict(cursor.fetchall())

# -----------------------
# Update Player Features
# -----------------------
playercnt=1
for pid, spid in mapping.items():
    try:
        sentiment_trend = calc_sentiment_trend(pid)
        positions, primary_position = get_positions(spid)
        current_club_id = get_current_club(pid)
        total_inj, avg_days, recent, days_since = get_injury_features(pid)
        
        cursor.execute("""
            UPDATE player_features
            SET sentiment_trend=%s, positions_played=%s, primary_position=%s, current_club_id=%s, total_injuries=%s, avg_days_out=%s, 
                recent_injury=%s, days_since_last_injury=%s
            WHERE player_id=%s
        """, (sentiment_trend, positions, primary_position, current_club_id, total_inj, avg_days, recent, days_since, pid))
        db.commit()

        print(f"✅ Updated {pid} → sentiment_trend={sentiment_trend}, club_id={current_club_id}, primary={primary_position}, all={positions}, total_inj {total_inj}, avg_days {avg_days}, recent{recent}, days_since {days_since}. Progress: {playercnt}/{len(mapping)}. Remaining: {len(mapping)-playercnt}")
    except Exception as e:
        print(f"⚠️ Error updating {pid}: {e}")
    playercnt+=1

cursor.close()
db.close()
print("🎉 Player features extended with sentiment trend & positions!")
