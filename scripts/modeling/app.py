import joblib
import pandas as pd
import numpy as np
import os # Import os for path handling
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration ---
MODEL_FILE = 'xgb_model_final.joblib'
PREPROCESSOR_FILE = 'preprocessor_final.joblib'

# CRITICAL: List of ALL expected one-hot encoded (OHE) columns the model needs.
REQUIRED_OHE_COLUMNS = [
    'position_defender', 'position_goalkeeper', 'position_midfield', 'position_missing', 
    'foot_right', 'foot_left', 'foot_unknown'
]

# Define the fields the front-end wants to display on lookup.
# These keys MUST match the HTML IDs (e.g., id="detail-age" expects key "age").
DISPLAY_FIELDS = [
    'age', 
    'country_of_citizenship', 
    'sentiment', 
    'current_club_name'
]


# --- LOAD YOUR REAL CSV DATA HERE ---
LOADED_PLAYER_DB = {}
# FIX: Using the ABSOLUTE PATH provided by the user
CSV_FILE_NAME = '/Users/veerababu/Downloads/master_list_cleaned.csv'

try:
    # 1. Load your CSV into a DataFrame using the absolute path
    df_all_players = pd.read_csv(CSV_FILE_NAME) 
    
    # CRITICAL FIX 1: Convert all column names to lowercase for consistent access
    df_all_players.columns = df_all_players.columns.str.lower()
    
    # 2. Set the player name column as the index for quick lookup
    #    NOTE: This assumes your player name column is named 'player' (now lowercased)
    if 'player' not in df_all_players.columns:
        raise KeyError('The CSV must contain a column named "player" (case-insensitive).')
        
    df_all_players = df_all_players.set_index('player')
    
    # 3. Convert the DataFrame to a dictionary for fast lookups in the Flask route
    LOADED_PLAYER_DB = df_all_players.to_dict('index')
    
    print(f"Successfully loaded {len(LOADED_PLAYER_DB)} players from {CSV_FILE_NAME}.")

except FileNotFoundError:
    print(f"CRITICAL ERROR: File '{CSV_FILE_NAME}' not found. Double check the file path.")
except KeyError as e:
    print(f"CRITICAL ERROR: Data processing failed: {e}")
except Exception as e:
    print(f"CRITICAL ERROR: An unexpected error occurred during data load: {e}")
# --- END CSV DATA LOAD ---


# --- Initialize Flask App and Load Assets ---
app = Flask(__name__)
CORS(app) 
model = None
preprocessor = None

def load_assets():
    """Load the saved XGBoost model and preprocessor."""
    global model, preprocessor
    try:
        # Load files relative to the current script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, MODEL_FILE)
        preprocessor_path = os.path.join(script_dir, PREPROCESSOR_FILE)

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        print("Model and Preprocessor assets loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Asset file not found: {e}. Ensure '{MODEL_FILE}' and '{PREPROCESSOR_FILE}' are in the same directory.")
    except Exception as e:
         print(f"ERROR: Failed to load model or preprocessor: {e}")

load_assets()

# FIX: Shared function to manually add OHE columns to input_df before preprocessing.
def preprocess_input(input_df):
    """
    Manually one-hot encode 'position' and 'foot' columns, add all required OHE columns (fill missing with 0),
    and drop raw categorical columns. Assumes input_df has raw 'position' and 'foot'.
    """
    if input_df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    # CRITICAL FIX 2: Ensure input keys are lowercased to match expected structure
    # This happens when processing the lookup data before sending it to the model.
    input_df.columns = input_df.columns.str.lower()

    row = input_df.iloc[0]  # Single row
    
    # Initialize all OHE columns to 0
    for col in REQUIRED_OHE_COLUMNS:
        input_df[col] = 0.0
    
    # Map and set position OHE (handle known mappings; default to missing)
    position = str(row.get('position', 'Unknown')).title()
    position_map = {
        'Goalkeeper': 'position_goalkeeper',
        'Defence': 'position_defender',  # Mapping 'Defence' to defender
        'Midfield': 'position_midfield',
        'Attack': 'position_missing',  # No 'Attack' column, maps to missing/forward
        'Defender': 'position_defender', # Handle common capitalization
    }
    position_col = position_map.get(position, 'position_missing')
    if position_col in REQUIRED_OHE_COLUMNS:
        input_df[position_col] = 1.0
    else:
        input_df['position_missing'] = 1.0 
    
    # Map and set foot OHE
    foot = str(row.get('foot', 'unknown')).lower()
    foot_map = {
        'right': 'foot_right',
        'left': 'foot_left',
        'unknown': 'foot_unknown'
    }
    foot_col = foot_map.get(foot, 'foot_unknown')
    if foot_col in REQUIRED_OHE_COLUMNS:
        input_df[foot_col] = 1.0
    else:
        input_df['foot_unknown'] = 1.0 
    
    # Drop raw categorical columns (preprocessor shouldn't see them)
    if 'position' in input_df.columns:
        input_df = input_df.drop('position', axis=1)
    if 'foot' in input_df.columns:
        input_df = input_df.drop('foot', axis=1)
    
    # Ensure numeric columns are float
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns
    input_df[numeric_cols] = input_df[numeric_cols].astype(float)
    
    return input_df


def process_and_predict(raw_data):
    """Core function to transform raw data, predict the market value (normalized), and denormalize it."""
    if not model or not preprocessor:
        raise Exception("Model assets not loaded for prediction. Check server startup logs.")
        
    # Create a single-row DataFrame and apply manual OHE preprocessing
    input_df = pd.DataFrame([raw_data])
    input_df = preprocess_input(input_df)   # This adds/fixes OHE columns
    
    # Apply the preprocessor
    transformed_data = preprocessor.transform(input_df)
    
    # Handle sparse matrix
    if hasattr(transformed_data, 'todense'):
        transformed_data = transformed_data.todense()
    
    # Ensure transformed_data is 2D array (single sample)
    if transformed_data.ndim == 1:
        transformed_data = transformed_data.reshape(1, -1)
    
    normalized_prediction = model.predict(transformed_data)[0]
    denormalized_prediction = np.exp(normalized_prediction) 
    
    return denormalized_prediction


# --- API Endpoint: Player Lookup and Prediction ---
@app.route('/predict/lookup', methods=['POST'])
def lookup_predict():
    """Accepts player name (JSON payload) and returns a prediction using loaded data."""
    try:
        request_data = request.get_json()
        player_name = request_data.get('player_name')
        
        if not player_name:
            return jsonify({"error": "Missing 'player_name' in request."}), 400
        
        player_features = None
        
        # 1. Retrieve player features from the LOADED_PLAYER_DB (case-insensitive lookup)
        for key, features in LOADED_PLAYER_DB.items():
            if key.lower() == player_name.lower():
                player_features = features
                player_name = key # Use the exact casing from the DB for display
                break
        
        if not player_features:
            return jsonify({"error": f"Player '{player_name}' not found in the loaded database."}), 404
            
        # 2. Run prediction
        prediction = process_and_predict(player_features)

        # 3. Build the response, including required display fields (CRITICAL FIX)
        response = {
            "player_name": player_name,
            "predicted_value": float(prediction),
            "R2_score": 0.6414
        }
        
        # Add the specific player details for the HTML to display
        for field in DISPLAY_FIELDS:
            # Use .get() method on the dictionary to safely access data
            value = player_features.get(field)
            
            # Handle numerical values (like Age) and potential NaN/None
            if isinstance(value, (float, int, np.number)) and not pd.isna(value):
                # Format floats to int if they are whole numbers (like age)
                response[field] = int(value) if value == int(value) else float(value)
            elif pd.isna(value) or value is None:
                response[field] = 'N/A'
            else:
                response[field] = str(value)

        return jsonify(response)

    except Exception as e:
        print(f"Lookup Prediction Error: {e}")
        error_message = f"Internal lookup prediction error: {e}. Check the console."
        return jsonify({"error": error_message}), 500


# --- API Endpoint: Manual Prediction ---
@app.route('/predict/manual', methods=['POST'])
def manual_predict():
    """Accepts raw player features (JSON payload) and returns a prediction."""
    try:
        raw_data = request.get_json()
        if not raw_data:
            return jsonify({"error": "No data provided in request."}), 400
        
        # Apply the same OHE preprocessing as lookup
        prediction = process_and_predict(raw_data)
        
        return jsonify({
            "predicted_value": float(prediction),
            "R2_score": 0.6414
        })

    except Exception as e:
        print(f"Manual Prediction Error: {e}")
        return jsonify({"error": f"Internal prediction error: {e}. Ensure input includes 'position' and 'foot' as raw strings."}), 500


# --- API Endpoint: Home/Status Check ---
@app.route('/', methods=['GET'])
def home():
    # Provide a list of available players for the front end's hint
    player_names = list(LOADED_PLAYER_DB.keys())
    return jsonify({
        "status": "Transfer IQ Prediction API is running.",
        "lookup_players": player_names
    })


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
