import mysql.connector
import pandas as pd
from datetime import datetime

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
    #1
	Creates a player_features table if not exists.
	Aggregates from:
	market_values_trfrmrkt → market value features
	player_injuries_trfrmrkt → injury stats
	player_transfers_trfrmrkt → transfer stats
	twitter_sentiments + reddit_sentiments → sentiment scores
	player_cards + matches → cards per match
	Maps names via player_mapping to join sentiment data.
	Inserts/updates features into player_features.
"""
# -----------------------
# Create Player Features Table
# -----------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS player_features (
    player_id INT PRIMARY KEY,
    latest_market_value BIGINT,
    market_value_growth FLOAT,
    total_injuries INT,
    avg_days_out FLOAT,
    last_injury_date DATE,
    total_transfers INT,
    total_transfer_fees BIGINT,
    free_transfers INT,
    sentiment_mean FLOAT,
    sentiment_positive_ratio FLOAT,
    sentiment_trend FLOAT,
    avg_cards_per_match FLOAT,
    positions_played VARCHAR(200),
    current_club_id INT
) ENGINE=InnoDB;
""")

db.commit()

# -----------------------
# Helper Queries
# -----------------------

# Market Value Features
cursor.execute("""
SELECT mv.transfermarkt_id,
       MAX(mv.market_value) AS latest_value,
       (MAX(mv.market_value) - MIN(mv.market_value)) / if(ifnull(MIN(mv.market_value),0)=0,1,ifnull(MIN(mv.market_value),0)) AS growth
FROM player_transfer_history mv
GROUP BY mv.transfermarkt_id
""")
market_df = pd.DataFrame(cursor.fetchall(), columns=["player_id", "latest_value", "growth"])

# Injury Features
cursor.execute("""
SELECT pi.transfermarkt_id,
       COUNT(*) AS total_injuries,
       AVG(days_out) AS avg_days_out,
       MAX(start_date) AS last_injury
FROM player_injuries_trfrmrkt pi
GROUP BY pi.transfermarkt_id
""")
injury_df = pd.DataFrame(cursor.fetchall(), columns=["player_id", "total_injuries", "avg_days_out", "last_injury"])

# Transfer History
cursor.execute("""
SELECT transfermarkt_id,
       COUNT(*) AS total_transfers,
       SUM(fee) AS total_fees,
       SUM(CASE WHEN reason='Free Transfer' THEN 1 ELSE 0 END) AS free_transfers
FROM player_transfer_history
GROUP BY transfermarkt_id
""")
transfer_df = pd.DataFrame(cursor.fetchall(), columns=["player_id", "total_transfers", "total_fees", "free_transfers"])

# Sentiment (Reddit)
cursor.execute("""
SELECT transfermarkt_id, AVG(polarity) AS avg_polarity,
       SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) / COUNT(*) AS positive_ratio
FROM (
    SELECT transfermarkt_id, polarity, sentiment FROM reddit_sentiments
) s
GROUP BY transfermarkt_id
""")
sentiment_df = pd.DataFrame(cursor.fetchall(), columns=["player_id", "avg_polarity", "positive_ratio"])

# Cards per match
cursor.execute("""
SELECT p.transfermarkt_id,
       COUNT(*) / NULLIF(COUNT(DISTINCT m.match_id),0) AS avg_cards
FROM player_cards pc
JOIN matches m ON pc.team_id IN (m.competition_id) join player_mapping p on p.statsbomb_player_id=pc.player_id
GROUP BY p.transfermarkt_id;
""")
cards_df = pd.DataFrame(cursor.fetchall(), columns=["player_id", "avg_cards"])

# -----------------------
# Merge into one DF
# -----------------------
features = pd.merge(market_df, injury_df, on="player_id", how="left")
features = pd.merge(features, transfer_df, on="player_id", how="left")
features = pd.merge(features, sentiment_df, on="player_id", how="left")
features = pd.merge(features, cards_df, on="player_id", how="left")

"""
# Player Cards needs mapping via player_mapping
cursor.execute("SELECT transfermarkt_id, canonical_name FROM player_mapping")
mapping = dict(cursor.fetchall())
cards_df["player_id"] = cards_df["player_name"].map({v:k for k,v in mapping.items()})
"""
features = pd.merge(features, cards_df, on="player_id", how="left")

# -----------------------
# Save into DB
# -----------------------
for _, row in features.iterrows():
    cursor.execute("""
        INSERT INTO player_features
        (player_id, latest_market_value, market_value_growth, total_injuries, avg_days_out,
         last_injury_date, total_transfers, total_transfer_fees, free_transfers,
         sentiment_mean, sentiment_positive_ratio, avg_cards_per_match)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          latest_market_value=VALUES(latest_market_value),
          market_value_growth=VALUES(market_value_growth),
          total_injuries=VALUES(total_injuries),
          avg_days_out=VALUES(avg_days_out),
          last_injury_date=VALUES(last_injury_date),
          total_transfers=VALUES(total_transfers),
          total_transfer_fees=VALUES(total_transfer_fees),
          free_transfers=VALUES(free_transfers),
          sentiment_mean=VALUES(sentiment_mean),
          sentiment_positive_ratio=VALUES(sentiment_positive_ratio),
          avg_cards_per_match=VALUES(avg_cards_per_match)
    """, (
        row.get("player_id"),
        row.get("latest_value"),
        row.get("growth"),
        row.get("total_injuries"),
        row.get("avg_days_out"),
        row.get("last_injury"),
        row.get("total_transfers"),
        row.get("total_fees"),
        row.get("free_transfers"),
        row.get("avg_polarity"),
        row.get("positive_ratio"),
        row.get("avg_cards")
    ))

db.commit()
cursor.close()
db.close()

print("✅ Player features table updated successfully!")
