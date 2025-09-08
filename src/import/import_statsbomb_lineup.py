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
cursor.execute("""
CREATE TABLE IF NOT EXISTS eventsNew (
    id VARCHAR(50) PRIMARY KEY,
    file_name varchar(100),
    match_id varchar(100),
    index_no INT,
    period INT,
    timestamp VARCHAR(20),
    type VARCHAR(100),
    player VARCHAR(100),
    team VARCHAR(100),
    location_x FLOAT,
    location_y FLOAT,
    related_events varchar(50)
);
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS lineups (
    file_name varchar(100),
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
);""")
filecnt=1
def insert_many(query, data):
    cursor.executemany(query, data)
    db.commit()

# ------------------------------
# Paths
# ------------------------------
events_path = "/home/gubsend/Infosys Springboard/open-data-master/data/events"

# ------------------------------
# Process Events + Lineups
# ------------------------------
for file in os.listdir(events_path):
    if file.endswith(".json"):
        with open(os.path.join(events_path, file), "r", encoding="utf-8") as f:
            events = json.load(f)

        # Collect for batch insert
        event_records, lineup_records = [], []

        for i, event in enumerate(events):
            # Insert into events table
            event_id = event.get("id")
            match_id = event.get("match_id")
            period = event.get("period")
            timestamp = event.get("timestamp")
            event_type = event.get("type", {}).get("name")
            player = event.get("player", {}).get("name")
            team = event.get("team", {}).get("name")
            location = event.get("location") if event.get("location") else [0,0]
            related_events_list=event.get("related_events") if event.get("related_events") else [0]
            related_events = ''
            if related_events_list[0]:
                related_events = related_events_list[0]
            event_records.append((
                f"{file}",event_id, match_id, i, period, timestamp, event_type,
                player, team, location[0], location[1],related_events
            ))

            # Extract lineup info if this is a "Starting XI" event
            if event_type == "Starting XI":
                team_id = event["team"]["id"]
                team_name = event["team"]["name"]

                for lineup in event.get("tactics", {}).get("lineup", []):
                    player_id = lineup["player"]["id"]
                    player_name = lineup["player"]["name"]
                    pos_id = lineup["position"]["id"]
                    pos_name = lineup["position"]["name"]
                    jersey = lineup["jersey_number"]

                    lineup_records.append((
                        f"{file}", event_id, match_id, team_id, team_name,
                        player_id, player_name,
                        pos_id, pos_name, jersey
                    ))
        # print(event_records)
        # Insert into events table
        if event_records:
            insert_many("""
                INSERT INTO eventsNew (file_name, id, match_id, index_no, period, timestamp, type, player, team, location_x, location_y, related_events)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE type=VALUES(type);
            """, event_records)

        # Insert into lineups table
        if lineup_records:
            insert_many("""
                INSERT INTO lineups (file_name, comp_id, match_id, team_id, team_name, player_id, player_name, position_id, position_name, jersey_number)   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """, lineup_records)

        print(f"✅ Processed {file}: {len(event_records)} events, {len(lineup_records)} lineup records. File# {filecnt}")
        filecnt=filecnt+1

# ------------------------------
# Close Connection
# ------------------------------
cursor.close()
db.close()
print("🎉 Events and lineups imported successfully!")

