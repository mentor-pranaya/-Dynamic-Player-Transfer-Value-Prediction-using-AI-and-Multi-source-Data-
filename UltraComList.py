import pandas as pd
import unicodedata

# ---------- Input file ----------
file_complete = "/Users/veerababu/Downloads/cleaned/completelist_clean2.csv"

# ---------- Read CSV ----------
df = pd.read_csv(file_complete)

# ---------- Normalize column names ----------
df.columns = df.columns.str.strip().str.lower()

# ---------- Function to remove accents ----------
def remove_accents(text):
    if isinstance(text, str):
        # Remove accents and special chars
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if ord(c) < 128)
    return text

# ---------- Clean player names ----------
df['player'] = df['player'].apply(lambda x: remove_accents(x).strip() if pd.notna(x) else x)

# ---------- Remove duplicates ----------
df = df.drop_duplicates(subset=['player'])

# ---------- Sort alphabetically ----------
df = df.sort_values(by='player').reset_index(drop=True)

# ---------- Save cleaned file ----------
output_file = "/Users/veerababu/Downloads/cleaned/completelist_cleaned_sorted.csv"
df.to_csv(output_file, index=False)

print(f"✅ Completed: Cleaned and sorted players. Saved to {output_file}")
