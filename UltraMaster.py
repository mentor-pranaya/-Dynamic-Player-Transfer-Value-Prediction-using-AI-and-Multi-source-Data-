import pandas as pd
import unicodedata

# ---------- Input files ----------
file_complete = "/Users/veerababu/Downloads/cleaned/completelist_cleaned_sorted.csv"
file_injuries = "/Users/veerababu/Downloads/cleaned/injuries_master_cleaned_final_sorted.csv"
file_sentiment = "/Users/veerababu/Downloads/cleaned/sentiment_master_cleaned_sorted.csv"

# ---------- Load CSVs ----------
df_complete = pd.read_csv(file_complete)
df_injuries = pd.read_csv(file_injuries)
df_sentiment = pd.read_csv(file_sentiment)

# ---------- Normalize column names ----------
for df in [df_complete, df_injuries, df_sentiment]:
    df.columns = df.columns.str.strip().str.lower()

# ---------- Remove accents from player names ----------
def remove_accents(text):
    if isinstance(text, str):
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if ord(c) < 128)
    return text

for df in [df_complete, df_injuries, df_sentiment]:
    df['player'] = df['player'].apply(lambda x: remove_accents(x).strip() if pd.notna(x) else x)

# ---------- Merge datasets ----------
df_master = df_complete.merge(df_injuries, on='player', how='left')
df_master = df_master.merge(df_sentiment, on='player', how='left')

# ---------- Optional: remove columns if entirely empty ----------
for col in ['days missed', 'games missed']:
    if col in df_master.columns and df_master[col].isna().all():
        df_master = df_master.drop(columns=[col])

# ---------- Remove duplicate rows ----------
df_master = df_master.drop_duplicates(subset=['player']).reset_index(drop=True)

# ---------- Save final master list ----------
output_file = "/Users/veerababu/Downloads/cleaned/master_list_final.csv"
df_master.to_csv(output_file, index=False)

print(f"✅ Master list created and saved: {output_file}")
