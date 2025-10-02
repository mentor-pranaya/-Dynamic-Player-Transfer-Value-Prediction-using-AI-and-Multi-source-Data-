from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# --- CONFIGURATION ---
# Define the order of features expected by the model.
# This order MUST match the 'FEATURE_COLS' array used during training.
FEATURE_ORDER = ['age', 'appearance', 'goals', 'assists', 'yellow cards', 'red cards',
                 'minutes played', 'days_injured', 'games_injured',
                 'highest_value', 'position_encoded', 'winger']

# --- 1. APPLICATION SETUP ---
app = Flask(__name__)
# Enable CORS to allow your index.html (running from a browser file path)
# to make requests to this server (running on localhost:5000).
CORS(app)

# Initialize variables to hold the loaded assets
model = None
scaler = None

# --- 2. MODEL AND SCALER LOADING (Executed once when server starts) ---
try:
    # Load the trained Random Forest model
    model = joblib.load('final_tuned_random_forest_model.pkl')

    # Load the fitted MinMaxScaler object (CRITICAL for data pre-processing)
    scaler = joblib.load('min_max_scaler.pkl')

    print("✅ Assets Loaded: Random Forest Model and MinMaxScaler are ready.")
except FileNotFoundError as e:
    print(
        f"❌ Asset Error: Could not find file {e.filename}. Ensure 'final_tuned_random_forest_model.pkl' and 'min_max_scaler.pkl' are in the same directory.")
except Exception as e:
    print(f"❌ Loading Error: An unexpected error occurred: {e}")
    model = None
    scaler = None


# --- 3. API ENDPOINT ---

@app.route('/predict', methods=['POST'])
def predict():
    # Health check for model availability
    if not model or not scaler:
        return jsonify({'error': 'Server assets (Model or Scaler) failed to load.'}), 500

    try:
        # Get the JSON data sent from the front-end
        data = request.json
        raw_features = data['features']  # This is an ordered list of 12 raw numbers

        # Validation: Check if the correct number of features was received
        if len(raw_features) != len(FEATURE_ORDER):
            return jsonify({'error': f'Expected {len(FEATURE_ORDER)} features, but received {len(raw_features)}.'}), 400

        # --- PRE-PROCESSING PIPELINE ---

        # 1. Convert the list of features into a 2D NumPy array (required by model.predict)
        input_array = np.array(raw_features).reshape(1, -1)

        # 2. Scale the raw input features (Crucial Step!)
        # This transforms the raw inputs (like age=25, goals=10) into scaled values (0-1)
        input_scaled = scaler.transform(input_array)

        # 3. Predict the LOG value
        log_prediction = model.predict(input_scaled)[0]

        # 4. Inverse Transform (expm1) to get the final currency value in Euros
        predicted_value = np.expm1(log_prediction)

        # Return the final result
        return jsonify({
            'success': True,
            'predicted_value': float(predicted_value),
            'model_source': 'Final Tuned Random Forest Regressor'
        })

    except Exception as e:
        # Catch any errors during prediction (e.g., incorrect data types)
        print(f"Prediction Runtime Error: {e}")
        return jsonify({'error': f'Prediction processing failed. Detail: {str(e)}'}), 400


# --- 4. SERVER RUNNING COMMAND ---
if __name__ == '__main__':
    # Run the server on localhost port 5000 in debug mode
    # Access via http://127.0.0.1:5000
    app.run(debug=True, port=5000)
