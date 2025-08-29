import os
import json
import pandas as pd
import mysql.connector

# ------------------------------
# Database Connection Setup
# ------------------------------
db = mysql.connector.connect(
    host="localhost",
    user="himanshu",     
    password="yahoonet",  
    database="AIProject"
)

cursor = db.cursor()

# ------------------------------
# Create a sample table (for Events data)
# ------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT,
    index_no INT,
    period INT,
    timestamp VARCHAR(20),
    type VARCHAR(100),
    player VARCHAR(100),
    team VARCHAR(100),
    location_x FLOAT,
    location_y FLOAT
);
""")

# ------------------------------
# Path to StatsBomb open data
# ------------------------------
base_path = "/home/gubsend/Infosys Springboard/open-data-master/data/events"

# Loop through JSON files in events folder
for file in os.listdir(base_path):
    if file.endswith(".json"):
        file_path = os.path.join(base_path, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            #print(data)
        # Convert JSON into pandas DataFrame (flatten only required fields)
        records = []
        for i, event in enumerate(data):
            match_id = event.get("match_id")
            period = event.get("period")
            timestamp = event.get("timestamp")
            event_type = event.get("type", {}).get("name")
            player = event.get("player", {}).get("name")
            team = event.get("team", {}).get("name")
            location = event.get("location") if event.get("location") else [0, 0]

            records.append([
                match_id, i, period, timestamp, event_type,
                player, team, location[0], location[1]
            ])
            #print(match_id, i, period, timestamp, event_type,player, team, location[0], location[1])
        df = pd.DataFrame(records, columns=[
            "match_id", "index_no", "period", "timestamp", "type",
            "player", "team", "location_x", "location_y"
        ])
        print(df)
        # Insert into MySQL row by row
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO events
                (match_id, index_no, period, timestamp, type, player, team, location_x, location_y)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, tuple(row))

        db.commit()
        print(f"Inserted {len(df)} rows from {file}")

# ------------------------------
# Close connection
# ------------------------------
cursor.close()
db.close()
print("✅ Import completed successfully!")
