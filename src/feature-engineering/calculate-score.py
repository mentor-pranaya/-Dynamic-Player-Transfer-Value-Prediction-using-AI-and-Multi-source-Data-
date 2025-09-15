import mysql.connector
import pandas as pd
from datetime import datetime

# ------------------------
# DB Connection
# ------------------------
db = mysql.connector.connect(
    host="localhost",
    user="himanshu",
    password="yahoonet",
    database="AIProject"
)
cursor = db.cursor(dictionary=True)

"""
    #5
    1. Pulls required inputs (injuries, sentiments, market value).
    2. Computes **Injury Score, Sentiment Score, Availability Score, Value Score, Composite Score**.
    3. Updates the `player_features` table.
    
CREATE TABLE IF NOT EXISTS player_features (
       mapping_id INT PRIMARY KEY,
       composite_score FLOAT,
       injury_score FLOAT,
       sentiment_score FLOAT,
       availability_score FLOAT,
       value_score FLOAT,
       FOREIGN KEY (mapping_id) REFERENCES player_mapping(id) ON DELETE CASCADE
   );

it will **fill/update composite scores for all players**.
"""

# ------------------------
# Helper: Compute Scores
# ------------------------
def compute_composite_score(inj_total, avg_days, days_since, sentiment_slope, latest_val, last_year_val):
    # --- Injury Score ---
    injury_penalty = (inj_total * 10 + (avg_days or 0)) / 5
    injury_score = max(0, 100 - min(100, injury_penalty))

    # --- Sentiment Score ---
    sentiment_score = max(0, min(100, 50 + (sentiment_slope or 0) * 1000))

    # --- Availability Score ---
    if days_since is None or days_since > 365:
        availability_score = 100
    else:
        availability_score = max(0, (days_since / 365) * 100)

    # --- Value Score ---
    if latest_val and last_year_val and last_year_val > 0:
        growth = (latest_val - last_year_val) / last_year_val
        value_score = max(0, min(100, 50 + growth * 50))
    else:
        value_score = 50  # neutral if missing

    # --- Weighted Final Score ---
    composite = (
        0.4 * injury_score +
        0.3 * sentiment_score +
        0.2 * availability_score +
        0.1 * value_score
    )

    return round(composite, 2), round(injury_score, 2), round(sentiment_score, 2), round(availability_score, 2), round(value_score, 2)

# ------------------------
# Fetch Player Data
# ------------------------
cursor.execute("""
    SELECT pm.id AS mapping_id, pt.id AS trfrmrkt_id, ps.player_id AS statsbomb_id, pi.id AS injury_id
    FROM player_mapping pm
    LEFT JOIN players_trfrmrkt pt ON pm.trfrmrkt_id = pt.id
    LEFT JOIN players ps ON pm.statsbomb_id = ps.player_id
    LEFT JOIN player_injuries pi ON pm.trfrmrkt_id = pi.id
""")
mappings = cursor.fetchall()

for m in mappings:
    player_id = m["mapping_id"]

    # --- Injuries ---
    cursor.execute("""
        SELECT COUNT(*) AS total_inj, AVG(DATEDIFF(date_of_return, date_of_injury)) AS avg_days,
               MAX(DATEDIFF(CURDATE(), date_of_injury)) AS days_since
        FROM player_injuries
        WHERE name = (SELECT name FROM players_trfrmrkt WHERE id = %s)
    """, (m["trfrmrkt_id"],))
    inj = cursor.fetchone() or {"total_inj": 0, "avg_days": 0, "days_since": None}

    # --- Sentiment slope ---
    cursor.execute("""
        SELECT AVG(polarity) AS avg_pol
        FROM reddit_sentiments
        WHERE player_name = (SELECT name FROM players_trfrmrkt WHERE id = %s)
    """, (m["trfrmrkt_id"],))
    sent = cursor.fetchone() or {"avg_pol": 0}
    sentiment_slope = sent["avg_pol"]

    # --- Market values ---
    cursor.execute("""
        SELECT market_value, snapshot_date
        FROM market_values_trfrmrkt
        WHERE player_id = %s
        ORDER BY snapshot_date DESC
        LIMIT 2
    """, (m["trfrmrkt_id"],))
    mv = cursor.fetchall()
    latest_val = mv[0]["market_value"] if mv else None
    last_year_val = mv[1]["market_value"] if len(mv) > 1 else latest_val

    # --- Compute ---
    comp, inj_s, sent_s, avail_s, val_s = compute_composite_score(
        inj["total_inj"], inj["avg_days"], inj["days_since"], sentiment_slope, latest_val, last_year_val
    )

    # --- Save to player_features ---
    cursor.execute("""
        INSERT INTO player_features (mapping_id, composite_score, injury_score, sentiment_score, availability_score, value_score)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            composite_score=VALUES(composite_score),
            injury_score=VALUES(injury_score),
            sentiment_score=VALUES(sentiment_score),
            availability_score=VALUES(availability_score),
            value_score=VALUES(value_score)
    """, (player_id, comp, inj_s, sent_s, avail_s, val_s))
    db.commit()

cursor.close()
db.close()
print("✅ Composite scores updated in player_features")
