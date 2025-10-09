import pandas as pd
import unicodedata

# Load CSV
file_sentiment = "/Users/veerababu/Downloads/cleaned/sentiment_report_clean2.csv"
df = pd.read_csv(file_sentiment)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Remove accents from player names
def remove_accents(text):
    if isinstance(text, str):
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if ord(c) < 128)
    return text

df['player'] = df['player'].apply(remove_accents)

# Aggregate data by player to keep all details
agg_dict = {}
for col in df.columns:
    if col != 'player':
        # Combine all unique values separated by " | " for text columns
        agg_dict[col] = lambda x: " | ".join([str(v) for v in x.unique() if pd.notna(v)])

df = df.groupby('player').agg(agg_dict).reset_index()

# Sort alphabetically
df = df.sort_values(by='player').reset_index(drop=True)

# Save cleaned file
output_file = "/Users/veerababu/Downloads/cleaned/sentiment_master_cleaned_sorted.csv"
df.to_csv(output_file, index=False)

print(f"✅ Sentiment data cleaned and aggregated. Saved to: {output_file}")
