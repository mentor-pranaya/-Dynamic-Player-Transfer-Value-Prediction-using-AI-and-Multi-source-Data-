import os
import json
import mysql.connector

# ------------------------------
# Database Connection
# ------------------------------
db = mysql.connector.connect(
    host="localhost",
    user="himanshu",
    password="yahoonet",  # Replace with your MySQL password
    database="AIProject"
)
cursor = db.cursor()

# ------------------------------
# Create Tables if Not Exists
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS eventsNew1 (
    id VARCHAR(50) PRIMARY KEY,
    file_name VARCHAR(100),
    match_id VARCHAR(100),
    index_no INT,
    period INT,
    timestamp VARCHAR(20),
    type VARCHAR(100),
    player VARCHAR(100),
    team VARCHAR(100),
    location_x FLOAT,
    location_y FLOAT,
    related_events VARCHAR(50)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS lineups (
    file_name VARCHAR(100),
    comp_id VARCHAR(50),
    id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT,
    team_id INT,
    team_name VARCHAR(100),
    player_id INT,
    player_name VARCHAR(100),
    position_id INT,
    position_name VARCHAR(100),
    jersey_number INT
);
""")

# ------------------------------
# Utility Insert Function
# ------------------------------
def insert_many(query, data):
    """Insert many records with executemany."""
    cursor.executemany(query, data)

# ------------------------------
# Paths
# ------------------------------
events_path = "/home/gubsend/Infosys Springboard/open-data-master/data/events"

# ------------------------------
# Process Events + Lineups (Batched)
# ------------------------------
BATCH_SIZE = 100  # files per commit
event_batch, lineup_batch = [], []
filecnt = 1

for file in os.listdir(events_path):
    if file.endswith(".json"):
        with open(os.path.join(events_path, file), "r", encoding="utf-8") as f:
            events = json.load(f)

        for i, event in enumerate(events):
            # Event details
            event_id = event.get("id")
            match_id = event.get("match_id")
            period = event.get("period")
            timestamp = event.get("timestamp")
            event_type = event.get("type", {}).get("name")
            player = event.get("player", {}).get("name")
            team = event.get("team", {}).get("name")
            location = event.get("location") if event.get("location") else [0, 0]
            related_events_list = event.get("related_events") or [""]
            related_events = related_events_list[0] if related_events_list else ""

            event_batch.append((
                file, event_id, match_id, i, period, timestamp, event_type,
                player, team, location[0], location[1], related_events
            ))

            # Lineups (from "Starting XI")
            if event_type == "Starting XI":
                team_id = event["team"]["id"]
                team_name = event["team"]["name"]

                for lineup in event.get("tactics", {}).get("lineup", []):
                    lineup_batch.append((
                        file, event_id, match_id, team_id, team_name,
                        lineup["player"]["id"], lineup["player"]["name"],
                        lineup["position"]["id"], lineup["position"]["name"],
                        lineup["jersey_number"]
                    ))

        # Commit every BATCH_SIZE files
        if filecnt % BATCH_SIZE == 0 or filecnt>=3464:
            if event_batch:
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS eventsNew{filecnt} (
                    id VARCHAR(50) PRIMARY KEY,
                    file_name VARCHAR(100),
                    match_id VARCHAR(100),
                    index_no INT,
                    period INT,
                    timestamp VARCHAR(20),
                    type VARCHAR(100),
                    player VARCHAR(100),
                    team VARCHAR(100),
                    location_x FLOAT,
                    location_y FLOAT,
                    related_events VARCHAR(50)
                );
                """)
                insert_many(f"""
                    INSERT INTO eventsNew{filecnt} (file_name, id, match_id, index_no, period, timestamp, type, player, team, location_x, location_y, related_events)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ;
                """, event_batch)
                event_batch.clear()

            if lineup_batch:
                insert_many("""
                    INSERT INTO lineups (file_name, comp_id, match_id, team_id, team_name, player_id, player_name, position_id, position_name, jersey_number)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
                """, lineup_batch)
                lineup_batch.clear()

            db.commit()
            print(f"✅ Batch committed at file {filecnt}")
            
        print(f"✅ File# read into memory {filecnt}")
        filecnt += 1

# ------------------------------
# Final commit for leftovers
# ------------------------------
if event_batch:
    insert_many("""
        INSERT INTO eventsNew (file_name, id, match_id, index_no, period, timestamp, type, player, team, location_x, location_y, related_events)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE type=VALUES(type);
    """, event_batch)

if lineup_batch:
    insert_many("""
        INSERT INTO lineups (file_name, comp_id, match_id, team_id, team_name, player_id, player_name, position_id, position_name, jersey_number)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
    """, lineup_batch)

db.commit()

# ------------------------------
# Close Connection
# ------------------------------
cursor.close()
db.close()
print("🎉 Events and lineups imported successfully (Batched)!")

