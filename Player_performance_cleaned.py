import os
import pandas as pd

# Define paths
input_folder = '/content/StateBomb Player Performance dataset/LineUp_csv'   
output_folder = '/content/Player Performance dataset/Cleaned dataset/Lineup' 

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# List all files in the input folder (assuming CSV format)
all_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Function to clean and preprocess a dataframe - customize as per your dataset
def clean_and_preprocess(df):
    # Example cleaning steps:
    # 1. Drop duplicates
    df = df.drop_duplicates()

    # 2. Handle missing values - example: fill numeric NaNs with median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    # 3. Handle missing categorical data - fill with mode
    for col in df.select_dtypes(include=['object']).columns:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col].fillna(mode_val, inplace=True)

    # 4. Convert date columns if any (example)
    # df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')

    # 5. Rename columns to lower case and strip spaces (example)
    df.columns = [col.strip().lower() for col in df.columns]

    # Add more cleaning steps as necessary based on your dataset columns

    return df

# Process all files
for file in all_files:
    file_path = os.path.join(input_folder, file)

    # Read the dataset
    df = pd.read_csv(file_path)

    # Clean and preprocess
    cleaned_df = clean_and_preprocess(df)

    # Save cleaned file to output folder
    output_path = os.path.join(output_folder, file)
    cleaned_df.to_csv(output_path, index=False)

print(f"Processed {len(all_files)} files. Cleaned files are saved in {output_folder}")
