import os
import pandas as pd
import ast

# Define input and output directories
input_folder = '/content/Player Performance dataset/Cleaned dataset/Events'     # Folder containing raw datasets
output_folder = '/content/Data2/Feature Engineering/Events'   # Folder to save processed datasets
os.makedirs(output_folder, exist_ok=True)

def process_location(col):
    """Convert location columns from string '[x, y]' to two separate numerical columns."""
    def parse_location(val):
        try:
            coords = ast.literal_eval(val)
            if isinstance(coords, list) and len(coords) == 2:
                return pd.Series({'loc_x': coords[0], 'loc_y': coords[1]})
            else:
                return pd.Series({'loc_x': None, 'loc_y': None})
        except:
            return pd.Series({'loc_x': None, 'loc_y': None})
    return col.apply(parse_location)

def feature_engineering(df):
    # Handle missing values
    df = df.fillna(0)

    # Convert boolean columns to integers
    bool_cols = [c for c in df.columns if df[c].dtype == 'bool']
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Detect and process location columns dynamically
    location_cols = [c for c in df.columns if 'location' in c or 'freeze_frame' in c]
    for col in location_cols:
        loc_df = process_location(df[col])
        df[f'{col}_x'] = loc_df['loc_x']
        df[f'{col}_y'] = loc_df['loc_y']
        df.drop(columns=[col], inplace=True)

    # Drop identifier columns dynamically (columns ending with '.id' or named 'id')
    id_cols = [c for c in df.columns if c.endswith('.id') or c == 'id']
    df.drop(columns=id_cols, errors='ignore', inplace=True)

    return df

# Process all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        processed_df = feature_engineering(df)

        output_file_path = os.path.join(output_folder, filename)
        processed_df.to_csv(output_file_path, index=False)

print("Feature engineering completed for all datasets.")
