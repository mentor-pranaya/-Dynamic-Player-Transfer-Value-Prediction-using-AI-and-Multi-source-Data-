import mysql.connector
import pandas as pd
from decimal import Decimal

# -----------------------------
# DB connection
# -----------------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yahoonet",
    database="AIProject"
)
cursor = db.cursor()


#  Mapping StatsBomb → Transfermarkt
cursor.execute("SELECT statsbomb_player_id, transfermarkt_id FROM player_mapping WHERE transfermarkt_id IS NOT NULL")
mapping_df = pd.DataFrame(cursor.fetchall(), columns=["statsbomb_player_id","transfermarkt_id"])
#print("Mapping size:", mapping_df.shape)
print(mapping_df.head())
print(mapping_df.info)
#print(mapping_df.describe())
#  Minutes played per player per season
#    (each lineup row = player appears in match)
cursor.execute("""
SELECT l.player_id, m.season_id,
       COUNT(*)*90 AS minutes_played
FROM lineups l
JOIN matches m ON l.match_id = m.match_id where l.player_id in (select statsbomb_player_id from player_mapping)
GROUP BY l.player_id, m.season_id
""")
minutes_df = pd.DataFrame(cursor.fetchall(), columns=["statsbomb_player_id","season_id","minutes_played"])
print(minutes_df.head())
print(minutes_df.info)

#  Shots & Pressures from events
#    (player_id from StatsBomb events)
cursor.execute("""
SELECT e.player, m.season_id,
 SUM(CASE WHEN e.type='Shot' THEN 1 ELSE 0 END) AS shots,
 SUM(CASE WHEN e.type='Pressure' THEN 1 ELSE 0 END) AS pressures
FROM eventsnew1 e
JOIN matches m ON e.match_id = m.match_id  where e.player in (select p.player_name from player_mapping m, players p where p.player_id=m.statsbomb_player_id)
GROUP BY e.player, m.season_id
""")
events_df = pd.DataFrame(cursor.fetchall(), columns=["player_name","season_id","shots","pressures"])
print(events_df.head())
print(events_df.info)

# Map player_name → player_id
cursor.execute("SELECT player_id, player_name FROM players")
players_df = pd.DataFrame(cursor.fetchall(), columns=["statsbomb_player_id","player_name"])
print(players_df.head())
print(players_df.info)

events_df = events_df.merge(players_df, on="player_name", how="left")
print(events_df.head())
print(events_df.info)

# now events_df has statsbomb_player_id
events_df = events_df.drop(columns=["player_name"])
print(events_df.head())
print(events_df.info)

# Merge minutes and events
stats_df = minutes_df.merge(events_df, on=["statsbomb_player_id","season_id"], how="left")
print(stats_df.head())
print(stats_df.info)

stats_df["shots"] = stats_df["shots"].fillna(0)
stats_df["pressures"] = stats_df["pressures"].fillna(0)
print(stats_df.head())
print(stats_df.info)

# per 90
stats_df["shots_per90"] = stats_df.apply(lambda r: Decimal(r["shots"])/(Decimal(r["minutes_played"])/90) if r["minutes_played"]>0 else 0, axis=1)
stats_df["pressures_per90"] = stats_df.apply(lambda r: Decimal(r["pressures"])/(Decimal(r["minutes_played"])/90) if r["minutes_played"]>0 else 0, axis=1)
print(stats_df.head())
print(stats_df.info)

# Map transfermarkt_id
stats_df = stats_df.merge(mapping_df, on="statsbomb_player_id", how="left")
print(stats_df.head())
print(stats_df.info)

# Insert/Update into player_features
for _, row in stats_df.iterrows():
    if pd.isna(row.transfermarkt_id):
        continue  # no mapping
    cursor.execute("""
        INSERT INTO player_features
        (player_id, season_id, minutes_played, shots_per90, pressures_per90)
        VALUES (%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          minutes_played=VALUES(minutes_played),
          shots_per90=VALUES(shots_per90),
          pressures_per90=VALUES(pressures_per90)
    """, (int(row.transfermarkt_id),
          int(row.season_id),
          int(row.minutes_played),
          float(row.shots_per90),
          float(row.pressures_per90)))
db.commit()

cursor.close()
db.close()

print("✅ StatsBomb features merged into player_features!")
