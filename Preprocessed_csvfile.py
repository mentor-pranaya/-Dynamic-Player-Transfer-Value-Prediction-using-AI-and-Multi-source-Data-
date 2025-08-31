# Preprocess multiple CSV files in a directory
import pandas as pd
import glob
import os

# Path where all your CSV files are stored
path = "/User/Desktop/Transfermarkt "  # change to your folder
output_path = "/User/Desktop/preprocessing_dataset/Preprocessed_injury_tweeter_data"

# Create output folder if not exists
os.makedirs(output_path, exist_ok=True)

# Get list of all CSV files
all_files = glob.glob(os.path.join(path, "*.csv"))

print(f"Found {len(all_files)} CSV files.")

for file in all_files:
    try:
        # Read CSV
        df = pd.read_csv(file)

        # -----------------
        # Preprocessing Steps
        # -----------------

        # 1. Remove duplicates
        df.drop_duplicates(inplace=True)

        # 2. Handle missing values (example: fill NaN with 0)
        df.fillna(0, inplace=True)

        # 3. Standardize column names
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        # 4. (Optional) Convert data types if needed
        # df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Save cleaned file
        filename = os.path.basename(file)
        df.to_csv(os.path.join(output_path, filename), index=False)

        print(f"✅ Processed: {filename}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

print("🎉 Preprocessing completed for all CSV files!")
