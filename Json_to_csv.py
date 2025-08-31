# Convert JSON files to CSV files
import pandas as pd
import os
import json

# Paths
json_directory = "/User/performance data/data/lineups"   # folder containing 1000 JSON files
csv_directory = "player/lineups"     # output folder for CSV files

# Create output folder if not exists
os.makedirs(csv_directory, exist_ok=True)

# Loop through each JSON file
for filename in os.listdir(json_directory):
    if filename.endswith(".json"):
        json_path = os.path.join(json_directory, filename)
        csv_filename = os.path.splitext(filename)[0] + ".csv"
        csv_path = os.path.join(csv_directory, csv_filename)

        try:
            # Read JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Flatten JSON if nested
            df = pd.json_normalize(data)

            # Save as CSV
            df.to_csv(csv_path, index=False)
            print(f" Converted {filename} → {csv_filename}")
        
        except Exception as e:
            print(f" Error processing {filename}: {e}")

print("🎉 Conversion completed! All CSVs saved in:", csv_directory)
