import numpy as np
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- SIMULATION PARAMETERS FROM train_model.py ---
W_LSTM = 0.50
W_LGBM = 0.30
W_XGB = 0.20
LOG_TRANSFORM_MULTIPLIER = 0.99


def get_simulated_prediction(data):
    """
    Simulates the ensemble prediction based on the logic in train_model.py.
    Now incorporates Minutes Played into the simulation.
    """
    # 1. Read input data
    market_score = data.get('Market_Value_Score', 1.0)
    nationality = data.get('Nationality', 'Other').lower()
    player_name = data.get('Player_Name', 'Unknown Player')
    minutes_played = data.get('Minutes_Played', 1000)  # New field

    # 2. Assign multipliers based on Nationality & Minutes Played for simulation
    # Nationality Multiplier (Higher market premium for certain nations)
    nationality_multipliers = {
        'brazil': 1.10, 'france': 1.08, 'england': 1.05,
        'argentina': 1.07, 'portugal': 1.05
    }
    nat_multiplier = nationality_multipliers.get(nationality, 1.0)

    # Minutes Played Multiplier (Higher multiplier for being a key player with high minutes)
    # Assuming max minutes in a season is around 3,420 (38 games * 90 min)
    # Scale minutes_played from 0 to 1, then apply a factor
    minutes_factor = (minutes_played / 3420.0)
    minutes_multiplier = 1.0 + (0.15 * min(minutes_factor, 1.0))  # Max 15% boost for max minutes

    # 3. Simulate the base log value
    # Base value of '10,000,000' is assumed, log(1 + 10M) ~= 16.11
    log_pred_base = 16.11 * market_score * 0.95 * nat_multiplier * minutes_multiplier

    # 4. Simulating the ensemble components
    noise_lstm = np.random.normal(0, 0.04)
    noise_xgb = np.random.normal(0, 0.05)
    noise_lgbm = np.random.normal(0, 0.06)

    lstm_pred_log = log_pred_base * LOG_TRANSFORM_MULTIPLIER + noise_lstm
    xgb_pred_log = log_pred_base * 0.98 + noise_xgb
    lgbm_pred_log = log_pred_base * 0.97 + noise_lgbm

    # Final Weighted Ensemble Prediction
    ensemble_pred_log = (W_LSTM * lstm_pred_log) + \
                        (W_LGBM * lgbm_pred_log) + \
                        (W_XGB * xgb_pred_log)

    # Inverse Transformation to original currency value
    ensemble_pred = np.expm1(ensemble_pred_log)

    return ensemble_pred, player_name


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        predicted_value, player_name = get_simulated_prediction(data)

        # Format the output value nicely
        formatted_value = f"${predicted_value:,.2f}"

        return jsonify({
            "status": "success",
            "player_name": player_name,
            "predicted_player_value": formatted_value,
            "raw_value": predicted_value,
            "message": f"Prediction generated for {player_name} using the simulated Ensemble."
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


if __name__ == '__main__':
    # Run the API server
    print("Starting Flask server on http://127.0.0.1:5000/predict")
    app.run(debug=True)