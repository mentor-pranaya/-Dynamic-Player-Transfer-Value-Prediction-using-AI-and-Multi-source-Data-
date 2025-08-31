import pandas as pd
from pathlib import Path

# === CONFIG ===
files = {
    "sentiment_report": "/Users/veerababu/Downloads/sentiment_report.csv",
    "injuries": "/Users/veerababu/Downloads/all_players_injuries_fast.csv",
    "completelist": "/Users/veerababu/Downloads/CompleteList.csv",
}
save_dir = Path("/Users/veerababu/Downloads/cleaned/")  # Save all cleaned files here
save_dir.mkdir(parents=True, exist_ok=True)

# === CLEANING FUNCTION WITH SUMMARY ===
def clean_file(name, path):
    print(f"\n📂 Processing: {name}")
    try:
        df = pd.read_csv(path)

        # Store initial stats
        initial_shape = df.shape
        initial_missing = df.isna().sum().sum()
        initial_dupes = df.duplicated().sum()

        # Cleaning
        df = df.drop_duplicates()
        df = df.fillna("Unknown")

        # Store final stats
        final_shape = df.shape
        final_missing = df.isna().sum().sum()
        final_dupes = df.duplicated().sum()

        # Save cleaned file
        save_path = save_dir / f"{name}_clean.csv"
        df.to_csv(save_path, index=False)

        # === PRINT SUMMARY REPORT ===
        print(f"   ✅ Original rows: {initial_shape[0]}, After cleaning: {final_shape[0]}")
        print(f"   🗑️ Duplicates removed: {initial_dupes}")
        print(f"   ❓ Missing values filled: {initial_missing - final_missing}")
        print(f"   💾 Saved cleaned file → {save_path}")

        return df

    except Exception as e:
        print(f"   ❌ Error cleaning {name}: {e}")
        return None


# === RUN CLEANING FOR ALL FILES ===
cleaned_data = {}
for name, path in files.items():
    cleaned_data[name] = clean_file(name, path)

# === OPTIONAL: MERGE IF POSSIBLE ===
if all(df is not None and "player" in df.columns for df in cleaned_data.values()):
    print("\n🔗 All files have 'player' column → merging...")
    merged = pd.merge(cleaned_data["sentiment_report"], cleaned_data["injuries"], on="player", how="outer")
    merged = pd.merge(merged, cleaned_data["completelist"], on="player", how="outer")
    merged.to_csv(save_dir / "merged_clean.csv", index=False)
    print(f"   ✅ Merged dataset saved → {save_dir / 'merged_clean.csv'}")
else:
    print("\n⚠️ Skipped merging: not all files have 'player' column.")
