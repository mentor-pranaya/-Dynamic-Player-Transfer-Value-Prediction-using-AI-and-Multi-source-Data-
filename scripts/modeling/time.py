import pandas as pd

# Load your master list
file_path = "/Users/veerababu/Desktop/Infosys/master_list_cleaned.csv"
df = pd.read_csv(file_path)

# ---- Option 1: If you have a date column already ----
if "date" in df.columns:
    # Convert to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Extract year, month, and season
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["date"].dt.year.astype(str) + "/" + (df["date"].dt.year + 1).astype(str)
    
    # Add a match counter for each player (time series index)
    if "player_id" in df.columns:
        df["time_step"] = df.groupby("player_id").cumcount() + 1

# ---- Option 2: If you don’t have a date column ----
else:
    if "player_id" in df.columns:
        df["time_step"] = df.groupby("player_id").cumcount() + 1
    else:
        # If only one row per player, just number rows as timeline
        df["time_step"] = range(1, len(df) + 1)

# ✅ Save the updated file to Downloads folder
output_path = "/Users/veerababu/Downloads/master_list_with_timeseries.csv"
df.to_csv(output_path, index=False)

print(f"✅ Time series column added! File saved to: {output_path}")
print(df.head())
