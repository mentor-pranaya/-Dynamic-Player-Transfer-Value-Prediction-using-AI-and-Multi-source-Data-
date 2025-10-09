import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

# --- 1. Initialize the Flask Application ---
app = Flask(__name__)

# --- 2. Load the Saved Model and Columns ---
try:
    model = joblib.load('final_player_value_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
    print("--- Model and columns loaded successfully ---")
except Exception as e:
    print(f"Error loading model or columns: {e}")
    model = None
    model_columns = None

# --- 3. Create the Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or model_columns is None:
        return jsonify({'error': 'Model is not loaded properly. Check server logs.'}), 500

    try:
        # Get the JSON data sent to the API
        json_data = request.get_json()
        input_df = pd.DataFrame(json_data, index=[0])
        
        print("\n--- Received input data ---")
        print(input_df)
        
        # --- 4. THE FIX: Prepare Input Data Robustly ---
        # Reindex the input DataFrame to match the model's required columns.
        # This ensures all 19 columns are present and in the correct order.
        # 'fill_value=0' automatically sets any missing columns (like 'lstm_prediction') to 0.
        final_input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # --- 5. Make the Prediction ---
        prediction = model.predict(final_input_df)
        output = float(prediction[0])
        
        print(f"--- Model prediction: {output:.2f} million € ---")
        
        # --- 6. Return the Prediction as a JSON Response ---
        return jsonify({'predicted_market_value_in_millions_eur': round(output, 2)})
    
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# --- 7. Run the Flask Application ---
if __name__ == '__main__':
    app.run(debug=True)