import pandas as pd

# Define file locations
file_complete = "/Users/veerababu/Downloads/cleaned/completelist_clean2.csv"
file_injuries = "/Users/veerababu/Downloads/cleaned/injuries_clean2.csv"
file_sentiment = "/Users/veerababu/Downloads/cleaned/sentiment_report_clean2.csv"

# Read CSV files into dataframes
df_complete = pd.read_csv(file_complete)
df_injuries = pd.read_csv(file_injuries)
df_sentiment = pd.read_csv(file_sentiment)

# Convert all column headers to lowercase and strip whitespace
df_complete.columns = [col.strip().lower() for col in df_complete.columns]
df_injuries.columns = [col.strip().lower() for col in df_injuries.columns]
df_sentiment.columns = [col.strip().lower() for col in df_sentiment.columns]

# Make sure 'player' column exists in each dataframe, rename first column if needed
def ensure_player_column(df):
    if 'player' not in df.columns:
        df.rename(columns={df.columns[0]: 'player'}, inplace=True)
    return df

df_complete = ensure_player_column(df_complete)
df_injuries = ensure_player_column(df_injuries)
df_sentiment = ensure_player_column(df_sentiment)

# Merge dataframes on 'player' column
merged_df = pd.merge(df_complete, df_injuries, on='player', how='left')
merged_df = pd.merge(merged_df, df_sentiment, on='player', how='left')

# Save the combined dataframe to a CSV file
output_file = "/Users/veerababu/Downloads/cleaned/master_player_data.csv"
merged_df.to_csv(output_file, index=False)

print("Master dataset successfully saved!")
print("Dataset dimensions:", merged_df.shape)
print("Columns included:", merged_df.columns.tolist())
