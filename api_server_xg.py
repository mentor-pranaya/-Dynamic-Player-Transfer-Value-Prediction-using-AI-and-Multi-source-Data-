from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# --- CONFIGURATION ---
# Define the order of features expected by the model.
# This order MUST match the 'FEATURE_COLS' array used during training and the front-end.
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

# --- 2. MODEL AND SCALER LOADING (Updated to use new files) ---
try:
    # Load the best gradient boosting model (XGBoost or LightGBM)
    # NOTE: This API uses specific filenames different from the previous version.
    MODEL_FILE = 'best_gradient_boosting_model.pkl'
    SCALER_FILE = 'min_max_scaler_new.pkl'

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    # We now check which specific model was loaded to inform the user
    model_name = type(model).__name__

    print(f"✅ Assets Loaded: Model ({model_name}) and Scaler are ready.")

except FileNotFoundError as e:
    print(
        f"❌ Asset Error: Could not find file {e.filename}. Ensure '{MODEL_FILE}' and '{SCALER_FILE}' are in the same directory.")
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

        # 1. Convert the list of features into a 2D NumPy array
        input_array = np.array(raw_features).reshape(1, -1)

        # 2. Scale the raw input features (Crucial Step!)
        input_scaled = scaler.transform(input_array)

        # 3. Predict the LOG value
        log_prediction = model.predict(input_scaled)[0]

        # 4. Inverse Transform (expm1) to get the final currency value in Euros
        predicted_value = np.expm1(log_prediction)

        # Determine the model name for the output message
        model_name = type(model).__name__

        # Return the final result
        return jsonify({
            'success': True,
            'predicted_value': float(predicted_value),
            'model_source': model_name
        })

    except Exception as e:
        # Catch any errors during prediction (e.g., incorrect data types)
        print(f"Prediction Runtime Error: {e}")
        return jsonify({'error': f'Prediction processing failed. Detail: {str(e)}'}), 400


# --- 4. SERVER RUNNING COMMAND ---
if __name__ == '__main__':
    # Run the server on localhost port 5000 in debug mode
    app.run(debug=True, port=5000)
