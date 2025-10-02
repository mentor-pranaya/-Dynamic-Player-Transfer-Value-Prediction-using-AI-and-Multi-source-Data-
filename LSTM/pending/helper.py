import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def forecast_future_multivariate(model, recent_seq, scaler, n_future=3, target_index=0):
    """
    model: trained multivariate LSTM
    recent_seq: shape (1, n_steps, n_features) scaled
    scaler: fitted MinMaxScaler for all features
    n_future: steps ahead
    target_index: column index of the target in feature vector (market value)
    """
    preds_scaled = []
    seq = recent_seq.copy()
    for _ in range(n_future):
        # predict next target
        yhat_scaled = model.predict(seq, verbose=0)[0][0]
        preds_scaled.append(yhat_scaled)
        # create new step with same features as last step
        new_step = seq[:,-1,:].copy()  # copy last step features
        new_step[target_index] = yhat_scaled  # overwrite target with predicted
        # shift and append
        seq = np.append(seq[:,1:,:], new_step.reshape(1,1,-1), axis=1)

    # inverse transform: build dummy array for scaler
    dummy = np.zeros((len(preds_scaled), scaler.n_features_in_))
    for i, p in enumerate(preds_scaled):
        dummy[i, target_index] = p
    preds_inv = scaler.inverse_transform(dummy)[:,target_index]
    return preds_inv

def forecast_future(model, recent_sequence, scaler, n_future=3):
    """
    model: trained LSTM
    recent_sequence: shape (1, n_steps, n_features) already scaled
    scaler: same scaler used during training
    n_future: how many steps to predict ahead
    """
    preds = []
    seq = recent_sequence.copy()
    for _ in range(n_future):
        yhat = model.predict(seq, verbose=0)[0][0]
        preds.append(yhat)
        # shift window and append prediction
        new_step = np.array([[yhat]])  # shape (1,1)
        seq = np.append(seq[:,1:,:], new_step.reshape(1,1,1), axis=1)
    # inverse scale
    dummy = np.zeros((len(preds), scaler.n_features_in_))
    dummy[:,0] = preds
    preds_inv = scaler.inverse_transform(dummy)[:,0]
    return preds_inv


def evaluate_model(model, X, y, player_index, scaler):
    """
    Returns overall RMSE/MAE and per-player metrics.
    """
    y_pred = model.predict(X)
    y_pred = y_pred.reshape(-1)
    y = y.reshape(-1)

    # Inverse transform only the market_value dimension
    dummy_pred = np.zeros((len(y_pred), scaler.n_features_in_))
    dummy_true = np.zeros((len(y), scaler.n_features_in_))
    dummy_pred[:,0] = y_pred
    dummy_true[:,0] = y

    y_pred_inv = scaler.inverse_transform(dummy_pred)[:,0]
    y_true_inv = scaler.inverse_transform(dummy_true)[:,0]

    overall_rmse = mean_squared_error(y_true_inv, y_pred_inv, squared=False)
    overall_mae = mean_absolute_error(y_true_inv, y_pred_inv)

    # Per player metrics
    df_eval = pd.DataFrame({
        "player_id": player_index,
        "y_true": y_true_inv,
        "y_pred": y_pred_inv
    })

    per_player = (
        df_eval.groupby("player_id")
        .apply(lambda g: pd.Series({
            "rmse": mean_squared_error(g.y_true, g.y_pred, squared=False),
            "mae": mean_absolute_error(g.y_true, g.y_pred)
        }))
        .reset_index()
    )

    return overall_rmse, overall_mae, per_player

def get_player_predictions(model, X, y, player_index, scaler, player_id):
    """Return inverse-scaled actual & predicted series for a given player_id."""
    y_pred = model.predict(X).reshape(-1)
    y = y.reshape(-1)

    dummy_pred = np.zeros((len(y_pred), scaler.n_features_in_))
    dummy_true = np.zeros((len(y), scaler.n_features_in_))
    dummy_pred[:,0] = y_pred
    dummy_true[:,0] = y

    y_pred_inv = scaler.inverse_transform(dummy_pred)[:,0]
    y_true_inv = scaler.inverse_transform(dummy_true)[:,0]

    df = pd.DataFrame({
        "player_id": player_index,
        "actual": y_true_inv,
        "predicted": y_pred_inv
    })
    return df[df.player_id==player_id].reset_index(drop=True)


def prepare_multivariate_sequences(df, features_cols, n_steps=3, n_future=1):
    """
    df: DataFrame containing transfermarkt_id and features_cols.
    features_cols: list of column names to use as features
    n_steps: look-back window
    n_future: how many steps ahead to predict (usually 1)
    """
    scaler = MinMaxScaler()
    scaler.fit(df[features_cols].fillna(0))
    
    X_list, y_list, player_index = [], [], []
    
    for pid, group in df.groupby("transfermarkt_id"):
        arr = group[features_cols].fillna(0).values
        arr_scaled = scaler.transform(arr)
        # create sequences
        for i in range(len(arr_scaled) - n_steps - n_future + 1):
            X_list.append(arr_scaled[i:i+n_steps])
            y_list.append(arr_scaled[i+n_steps+n_future-1, 0])  # predict market_value
            player_index.append(pid)
    
    if not X_list:
        return None, None, None, scaler
    
    X = np.array(X_list)   # (samples, n_steps, features)
    y = np.array(y_list)   # (samples,)
    return X, y, player_index, scaler
