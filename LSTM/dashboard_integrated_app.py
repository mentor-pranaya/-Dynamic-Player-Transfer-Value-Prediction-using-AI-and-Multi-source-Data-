import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback
import xgboost as xgb
import matplotlib.pyplot as plt
import mysql.connector
from tensorflow.keras.optimizers import Adam
import joblib, tempfile
import streamlit_ext as ste
import json

#To track script re-run
st.write("Script re-run at:", pd.Timestamp.now())

xgb_models = {}

if "xgb_models" not in st.session_state:
    st.session_state["xgb_models"] = {}
if "trained" not in st.session_state:
    st.session_state.trained = False
if "xgb_models_final" not in st.session_state:
    st.session_state.xgb_models_final = None
if "final_seq_model" not in st.session_state:
    st.session_state.final_seq_model = None
if "final_screen" not in st.session_state:
    st.session_state.final_screen = None


# ---------- Helper Functions ----------

def make_multivariate_multistep(array, n_steps, n_future=3):
    X, y = [], []
    for i in range(len(array)-n_steps-n_future+1):
        X.append(array[i:i+n_steps, :])
        y.append(array[i+n_steps:i+n_steps+n_future, 0])  # predict market_value
    return np.array(X), np.array(y)

def inverse_transform_col_series(col_scaled, scaler, n_features_total):
    col_scaled = np.array(col_scaled).reshape(-1, 1)
    expanded = np.zeros((len(col_scaled), n_features_total))
    expanded[:, 0] = col_scaled.flatten()
    return scaler.inverse_transform(expanded)[:, 0]

# ---------- Build Seq2Seq Training Model ----------

def build_seq2seq_train(n_steps, n_features, n_future, latent_dim=64):
    # Encoder
    encoder_inputs = Input(shape=(n_steps, n_features), name="encoder_inputs")
    encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(encoder_inputs)

    # Decoder
    decoder_inputs = Input(shape=(n_future, 1), name="decoder_inputs")
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    decoder_dense = Dense(1, name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], [decoder_outputs, state_h, state_c])
    #model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------- Build Seq2Seq Inference Models ----------
def build_seq2seq_inference_models(train_model, n_steps, n_features, latent_dim=64):
    # Encoder Model
    encoder_inputs = Input(shape=(n_steps, n_features), name="encoder_inputs_inference")
    encoder_lstm = train_model.get_layer("encoder_lstm")
    _, state_h_enc, state_c_enc = encoder_lstm(encoder_inputs)
    encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

    # Decoder Model
    decoder_inputs = Input(shape=(1, 1), name="decoder_inputs_inference")
    decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = train_model.get_layer("decoder_lstm")
    decoder_dense = train_model.get_layer("decoder_dense")

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs, state_h, state_c]
    )

    return encoder_model, decoder_model

# ---------- Recursive Seq2Seq Prediction ----------
def predict_seq2seq_recursive(input_seq_scaled, encoder_model, decoder_model, n_future):
    state_h, state_c = encoder_model.predict(input_seq_scaled)
    start_val = input_seq_scaled[0, -1, 0]
    decoder_input = np.array([[[start_val]]])
    preds = []
    for _ in range(n_future):
        out, state_h, state_c = decoder_model.predict([decoder_input, state_h, state_c])
        pred_val = out[0, 0, 0]
        preds.append(pred_val)
        decoder_input = np.array([[[pred_val]]])
    return np.array(preds)


# ---------- Initialize session state ----------
st.session_state.lstm_results = []
st.session_state.xgb_results = []

# ---------- Streamlit UI ----------
st.title("Training Encoder-Decoder LSTM")
st.write("Forecast multi-step player transfer market values using an Encoder-Decoder LSTM.")

n_steps = st.sidebar.slider("Past windows (n_steps)", 2, 10, 3)
n_future = st.sidebar.slider("Future horizons (n_future)", 1, 5, 3)
epochs = st.sidebar.slider("Epochs", 10, 100, 50)
if st.session_state.trained:
    if hasattr(st.session_state, 'n_steps') and hasattr(st.session_state, 'n_future'):
        if st.session_state.n_steps != n_steps or st.session_state.n_future < n_future:
            st.session_state.n_future = n_future
            st.session_state.trained = False  # reset if params changed
            @st.dialog("Retraining Initiated", width="small", on_dismiss="ignore")
            def show_pop(): 
                st.write("n_steps changed or n_future increased, retraining required.\n Retraining Initiated with the new parameters.")

            show_pop()
    else:
        st.session_state.n_steps = n_steps
        st.session_state.n_future = n_future            

st.session_state.n_steps = n_steps
if hasattr(st.session_state, 'n_future'):
    if st.session_state.n_future < n_future:
        st.session_state.n_future = n_future
else:
    st.session_state.n_future = n_future
# ---------- Hyperparameter Tuning Controls ----------
st.sidebar.subheader("Hyperparameter Tuning Options")
tune_lstm = st.sidebar.checkbox("Tune LSTM Hyperparameters")
tune_xgb = st.sidebar.checkbox("Tune XGBoost Hyperparameters")

lstm_search_size = st.sidebar.slider("LSTM Random Search Iterations", 1, 21, 3)
lstm_search_epoches = st.sidebar.slider("LSTM Random Search Epoch Count", 5, 50, 10)
xgb_search_size = st.sidebar.slider("XGBoost Random Search Iterations", 1, 21, 3)


# DB Connection
db = mysql.connector.connect(
    host="localhost", user="root", password="yahoonet", database="AIProject"
)

df_t = pd.read_sql(
    "SELECT transfermarkt_id, transfer_date, market_value FROM player_transfer_history ORDER BY transfermarkt_id, transfer_date", db
)
df_f = pd.read_sql(
    "SELECT distinct p.*, t.name as player_name FROM player_features p, players_trfrmrkt t where p.player_id=t.transfermarkt_id", db
)

df_params = pd.read_sql("select run_date, XGBoost_RSME, Val_Loss as `LSTM_Loss`, Best_XGBoost_params, Best_LSTM_params from hyper_parameter_results order by id;", db)
st.subheader("Saved Hyperparameter Tuning Results in DB")
st.write(df_params)
#st.session_state.xgb_results= df_params[['LSTM_Loss', 'Best_XGBoost_params']].sort_values(by='LSTM_Loss').head(1)
rs_best_lstm = df_params[['LSTM_Loss', 'Best_LSTM_params']].sort_values(by='LSTM_Loss').head(1) #['Best_LSTM_params','LSTM_Loss'].head(1).values)

rs_best_lstm_str='{"iteration": 1,' + str(rs_best_lstm['Best_LSTM_params'].values[0].replace("'", '"').replace("}", "").replace("{", "")) + ',"rmse":' + str(rs_best_lstm['LSTM_Loss'].values[0]) + '}'
#st.write(rs_best_lstm_str)
#rs_best_lstm = str(df_params.sort_values(by='LSTM_Loss')['Best_LSTM_params'].head(1).values[0]).replace("'", '"')
#, []
rs_best_xgb = df_params[['XGBoost_RSME','Best_XGBoost_params']].sort_values(by='XGBoost_RSME').head(1)
rs_best_xgb_str='{"iteration": 1,' + str(rs_best_xgb['Best_XGBoost_params'].values[0].replace("'", '"').replace("}", "").replace("{", "")) + ',"rmse":' + str(rs_best_xgb['XGBoost_RSME'].values[0]) + '}'
#st.write(rs_best_xgb_str)
st.write("Best LSTM Params:", rs_best_lstm, "  \nBest XGB Params:", rs_best_xgb)
rs_best_lstm_json=json.loads(rs_best_lstm_str)
rs_best_xgb_json=json.loads(rs_best_xgb_str)
#st.json(rs_best_lstm_json)
#st.json(rs_best_xgb_json)
st.session_state.lstm_results=rs_best_lstm_json
st.session_state.xgb_results=rs_best_xgb_json
df = df_t.merge(
    df_f[
        [
            "player_id", "player_name", "total_injuries", "sentiment_mean",
            "avg_cards_per_match", "avg_days_out", "recent_injury",
            "days_since_last_injury", "minutes_played", "shots_per90", "pressures_per90"
        ]
    ],
    left_on="transfermarkt_id", right_on="player_id", how="left"
)

df = df.sort_values(["transfermarkt_id", "transfer_date"])

player_options = df.sort_values("player_name").apply(
    lambda row: f"{row['player_name']} ({row['player_id']})", axis=1
).unique().tolist()
player_choice = st.sidebar.selectbox("Select Player", player_options)
pid = int(player_choice.split("(")[-1].replace(")", ""))

# ---------- TRAINING AND DOWNLOAD HANDLING ----------

if st.sidebar.button("Click here to start training and evaluation"):
    st.session_state.trained = False  # reset before training

#st.write(st.session_state)

# Prepare sequences
X_list, y_list, player_index = [], [], []
scaler = MinMaxScaler()
feature_cols = [
    "market_value", "total_injuries", "sentiment_mean", "avg_cards_per_match",
    "avg_days_out", "recent_injury", "days_since_last_injury",
    "minutes_played", "shots_per90", "pressures_per90"
]
scaler.fit(df[feature_cols].fillna(0).values)

for p, group in df.groupby("transfermarkt_id"):    
    features = group[feature_cols].fillna(0).values
    #scaled = scaler.fit_transform(features)
    scaled = scaler.transform(features)
    Xp, yp = make_multivariate_multistep(scaled, n_steps, n_future)
    if len(Xp) == 0:
        continue
    X_list.append(Xp)
    y_list.append(yp)
    player_index.extend([p] * len(yp))

X = np.vstack(X_list)
y = np.vstack(y_list)
player_index = np.array(player_index)

# Train/Validation split
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X, y, player_index, test_size=0.2, shuffle=True, random_state=42
)

# Decoder inputs for teacher forcing
decoder_input_train = np.zeros((y_train.shape[0], n_future, 1))
decoder_input_train[:, 1:, 0] = y_train[:, :-1]
decoder_input_val = np.zeros((y_val.shape[0], n_future, 1))
decoder_input_val[:, 1:, 0] = y_val[:, :-1]

# Build and train Seq2Seq
latent_dim = 64
seq2seq = build_seq2seq_train(n_steps, X.shape[2], n_future, latent_dim=latent_dim)

#st.subheader("Training Seq2Seq Encoder-Decoder LSTM (with teacher forcing)")

class StreamlitLogger(Callback):
    def __init__(self, total_epochs, modname=""):
        super().__init__()
        self.total_epochs = total_epochs
        self.history = {"loss": [], "val_loss": []}
        self.modname = modname

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["loss"].append(logs.get("loss"))
        self.history["val_loss"].append(logs.get("val_loss"))
        epoch_log.text(
            f"Epoch {epoch+1}/{self.total_epochs} - "
            f"Loss: {logs.get('loss'):.4f}, "
            f"Val Loss: {logs.get('val_loss'):.4f}"
        )
        progress_bar.progress((epoch+1)/self.total_epochs)
        new_data = {
            "Train Loss": self.history["loss"][-1:],
            "Val Loss": self.history["val_loss"][-1:]
        }
        loss_chart.add_rows(new_data)
        #if self.modname=="":
        #    loss_chart.add_rows(new_data)
        #else:
        #    loss_chart1.add_rows(new_data)

#es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

## Dummy outputs for state placeholders
#dummy_train = np.zeros((len(y_train), latent_dim))
#dummy_val = np.zeros((len(y_val), latent_dim))
#
#history = seq2seq.fit(
#    [X_train, decoder_input_train],
#    [y_train[..., np.newaxis], dummy_train, dummy_train],
#    #y_train[..., np.newaxis],
#    epochs=epochs,
#    batch_size=32,
#    validation_data=([X_val, decoder_input_val], [y_val[..., np.newaxis], dummy_val, dummy_val]),
#    #validation_data=([X_val, decoder_input_val], y_val[..., np.newaxis]),
#    verbose=0,
#    callbacks=[es, StreamlitLogger(epochs)]
#)

## Plot loss curve
#fig, ax = plt.subplots()
#ax.plot(history.history["loss"], label="Train Loss")
#ax.plot(history.history["val_loss"], label="Val Loss")
#ax.legend()
#ax.set_title("Loss Curves")
#st.pyplot(fig)
#st.write("Training and validation loss curves.")
## ---------- Build inference models ----------
#encoder_model, decoder_model = build_seq2seq_inference_models(
#    seq2seq, n_steps=X.shape[1], n_features=X.shape[2], latent_dim=latent_dim
#)

#st.success("Training completed!")

# -------------------- Ensemble + Retrain best models + Comparison Dashboard --------------------

# Helper: safe get best params from session_state lists
def get_best_from_results(results_list, score_col):
    if not results_list:
        return None, None, None
    df = pd.DataFrame(results_list)
    idx = df[score_col].idxmin()
    return df.loc[idx].to_dict(), df, idx

# Retrain final LSTM (seq2seq) on training set using best hyperparams (or defaults)
#chart_placeholder1 = st.empty()
#loss_chart1 = chart_placeholder1.line_chart({"Train Loss": [], "Val Loss": []})
def retrain_final_seq2seq(X_train, decoder_input_train, y_train, X_val, decoder_input_val, y_val,
                        best_params=None, epochs_final=50):
    # defaults
    latent_dim = int(best_params.get("latent_dim", 64)) if best_params else 64
    batch_size = int(best_params.get("batch_size", 32)) if best_params else 32
    lr = float(best_params.get("learning_rate", 0.001)) if best_params else 0.001

    # build model (note: this train model returns decoder_outputs only)
    encoder_inputs = Input(shape=(X_train.shape[1], X_train.shape[2]), name="encoder_inputs")
    encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(encoder_inputs)

    decoder_inputs = Input(shape=(y_train.shape[1], 1), name="decoder_inputs")
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    decoder_dense = Dense(1, name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    opt = Adam(learning_rate=lr)
    train_model.compile(optimizer=opt, loss="mse")
    es_local = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = train_model.fit(
        [X_train, decoder_input_train],
        y_train[..., np.newaxis],
        validation_data=([X_val, decoder_input_val], y_val[..., np.newaxis]),
        epochs=epochs_final,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es_local, StreamlitLogger(epochs_final,"1")]
    )

    # build inference models (encoder_model, decoder_model) using trained layers
    # Encoder inference
    enc_in = Input(shape=(X_train.shape[1], X_train.shape[2]), name="enc_infer_inputs")
    enc_lstm_layer = train_model.get_layer("encoder_lstm")
    _, st_h, st_c = enc_lstm_layer(enc_in)
    encoder_model_inf = Model(enc_in, [st_h, st_c])

    # Decoder inference
    dec_in = Input(shape=(1, 1), name="dec_infer_inputs")
    dec_state_h = Input(shape=(latent_dim,), name="dec_state_h")
    dec_state_c = Input(shape=(latent_dim,), name="dec_state_c")
    dec_lstm_layer = train_model.get_layer("decoder_lstm")
    dec_dense_layer = train_model.get_layer("decoder_dense")

    dec_out, dec_h, dec_c = dec_lstm_layer(dec_in, initial_state=[dec_state_h, dec_state_c])
    dec_out = dec_dense_layer(dec_out)
    decoder_model_inf = Model([dec_in, dec_state_h, dec_state_c], [dec_out, dec_h, dec_c])

    return train_model, encoder_model_inf, decoder_model_inf, history

# Retrain final XGBoost per step using best params if present, else defaults
def retrain_final_xgboost_per_step(X_train_flat, y_train, X_val_flat, y_val, best_params_per_step=None):
    xgb_models_final = []
    n_future_local = y_train.shape[1]
    y_val_pred_xgb_all = np.zeros((X_val_flat.shape[0], n_future_local))
    for step in range(n_future_local):
        meta_train = np.hstack([X_train_flat, y_train_pred_seq_train[:, step].reshape(-1,1)])
        meta_val = np.hstack([X_val_flat, y_val_pred_seq_val[:, step].reshape(-1,1)])

        # if best params were provided per-step, use them; otherwise defaults / first best global
        bp = None
        if best_params_per_step is not None and step < len(best_params_per_step):
            bp = best_params_per_step[step]

        n_estimators = int(bp.get("n_estimators", 300)) if bp else 300
        max_depth = int(bp.get("max_depth", 6)) if bp else 6
        learning_rate = float(bp.get("learning_rate", 0.05)) if bp else 0.05
        subsample = float(bp.get("subsample", 0.8)) if bp else 0.8
        colsample = float(bp.get("colsample_bytree", 0.8)) if bp else 0.8

        model_x = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample, random_state=42, verbosity=0
        )
        model_x.fit(meta_train, y_train[:, step])
        xgb_models_final.append(model_x)
        y_val_pred_xgb_all[:, step] = model_x.predict(meta_val)
    return xgb_models_final, y_val_pred_xgb_all

# --- Prepare meta-outputs we need for retraining ---
# if seq2seq is already trained in-memory as seq2seq (older structure returning [decoder, h, c])
# we prefer to use predictions from the "train_model" used previously. To avoid confusion, we'll derive LSTM preds below.

# 1) Get best hyperparams from tuning results (if available).
best_lstm_row, df_lstm_results, best_lstm_idx = get_best_from_results((st.session_state.get("lstm_results"),), "rmse")
best_xgb_row, df_xgb_results, best_xgb_idx = get_best_from_results((st.session_state.get("xgb_results"),), "rmse")
st.write(best_lstm_row, df_lstm_results, best_lstm_idx)
st.write(best_xgb_row, df_xgb_results, best_xgb_idx)
st.subheader("Final model training & comparison")
progress_bar = st.progress(0)
epoch_log = st.empty()
chart_placeholder = st.empty()
loss_chart = chart_placeholder.line_chart({"Train Loss": [], "Val Loss": []})

# Use a little more epochs for final training
epochs_final = max(epochs, 50)

if st.session_state.trained == False:
    # 2) Retrain final seq2seq on training set using best LSTM params (or defaults)
    st.write("Retraining final Seq2Seq LSTM on training set with best hyperparameters (or defaults)...")
    best_lstm_params = None
    if best_lstm_row:
        best_lstm_params = {
            "latent_dim": int(best_lstm_row.get("latent_dim", 64)),
            "batch_size": int(best_lstm_row.get("batch_size", 32)),
            "learning_rate": float(best_lstm_row.get("learning_rate", 0.001))
        }

    
    final_seq_model, encoder_model_inf, decoder_model_inf, history_final = retrain_final_seq2seq(
        X_train, decoder_input_train, y_train, X_val, decoder_input_val, y_val,
        best_params=best_lstm_params, epochs_final=epochs_final
    )
    # save to session
    st.session_state.final_seq_model = final_seq_model
    st.session_state.encoder_model_inf = encoder_model_inf
    st.session_state.decoder_model_inf = decoder_model_inf
    st.session_state.history_final = history_final.history

# plot final training loss
fig_final, axf = plt.subplots()
axf.plot(st.session_state.history_final["loss"], label="Train Loss")
axf.plot(st.session_state.history_final["val_loss"], label="Val Loss")
axf.set_title("Final Seq2Seq Loss Curves")
axf.legend()
st.pyplot(fig_final)

if st.session_state.trained:
    history_final = st.session_state.history_final
    for x in range(len(history_final["loss"])):
        new_data = {
                "Train Loss": [st.session_state.history_final["loss"][x]],
                "Val Loss": [st.session_state.history_final["val_loss"][x]]
            }
        loss_chart.add_rows(new_data)
        epoch_log.text(
            f"Epoch {len(history_final["loss"])}/{epochs_final} - "
            f"Loss: {st.session_state.history_final["loss"][x]:.4f}, "
            f"Val Loss: {st.session_state.history_final["val_loss"][x]:.4f}"
        )
        progress_bar.progress((len(history_final["loss"]))/epochs_final)

# 3) Get Seq2Seq predictions on train & val to build meta-features for XGBoost retraining
y_train_pred_seq_train = st.session_state.final_seq_model.predict([X_train, decoder_input_train]).squeeze(-1)  # shape (n_samples, n_future)
y_val_pred_seq_val = st.session_state.final_seq_model.predict([X_val, decoder_input_val]).squeeze(-1)

# 4) Retrain XGBoost per step using best params from tuning set if any
st.write("Retraining final XGBoost models (one per horizon) using seq2seq predictions as meta-feature...")
# Optionally you can use best params per-step extracted from df_xgb_results; here we just use the single best set found earlier.
best_xgb_params_global = None
if best_xgb_row:
    # df_xgb_results row contains a single best set; use it across steps
    best_xgb_params_global = {
        "n_estimators": int(best_xgb_row.get("n_estimators", 300)),
        "max_depth": int(best_xgb_row.get("max_depth", 6)),
        "learning_rate": float(best_xgb_row.get("learning_rate", 0.05)),
        "subsample": float(best_xgb_row.get("subsample", 0.8)),
        "colsample_bytree": float(best_xgb_row.get("colsample_bytree", 0.8))
    }

y_val_pred_xgb_all_final = np.zeros_like(y_val_pred_seq_val)

if st.session_state.trained == False:
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    # retrain xgb models
    xgb_models_final = []
    for step in range(n_future):
        X_meta_train = np.hstack([X_train_flat, y_train_pred_seq_train[:, step].reshape(-1,1)])
        X_meta_val = np.hstack([X_val_flat, y_val_pred_seq_val[:, step].reshape(-1,1)])

        params = best_xgb_params_global or {"n_estimators":300,"max_depth":6,"learning_rate":0.05,"subsample":0.8,"colsample_bytree":0.8}
        model_x = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        model_x.fit(X_meta_train, y_train[:, step])
        ## ✅ Assign human-readable feature names
        #model_x.get_booster().feature_names = feature_cols + ["seq2seq_pred"]

        xgb_models_final.append(model_x)
        y_val_pred_xgb_all_final[:, step] = model_x.predict(X_meta_val)

    st.session_state.xgb_models_final = xgb_models_final
    st.session_state.trained = True

if st.session_state.trained:
    # 5) Ensemble predictions (simple average)
    y_val_seq_lstm_final_inv_list = []
    y_val_xgb_final_inv_list = []
    y_val_ensemble_inv_list = []

    n_features_total = X.shape[2]
    # inverse transform each step separately and compute error metrics
    rmse_per_step = []
    mae_per_step = []
    for step in range(n_future):
        # inverse transform scaled data -> actual euros
        y_true_step_inv = inverse_transform_col_series(y_val[:, step], scaler, n_features_total)
        y_seq_step_inv = inverse_transform_col_series(y_val_pred_seq_val[:, step], scaler, n_features_total)
        y_xgb_step_inv = inverse_transform_col_series(y_val_pred_xgb_all_final[:, step], scaler, n_features_total)
        y_ens_step_inv = (y_seq_step_inv + y_xgb_step_inv) / 2.0

        y_val_seq_lstm_final_inv_list.append(y_seq_step_inv)
        y_val_xgb_final_inv_list.append(y_xgb_step_inv)
        y_val_ensemble_inv_list.append(y_ens_step_inv)

        # metrics
        rmse = np.sqrt(((y_true_step_inv - y_seq_step_inv)**2).mean())
        rmse_x = np.sqrt(((y_true_step_inv - y_xgb_step_inv)**2).mean())
        rmse_ens = np.sqrt(((y_true_step_inv - y_ens_step_inv)**2).mean())

        mae = np.mean(np.abs(y_true_step_inv - y_seq_step_inv))
        mae_x = np.mean(np.abs(y_true_step_inv - y_xgb_step_inv))
        mae_ens = np.mean(np.abs(y_true_step_inv - y_ens_step_inv))

        rmse_per_step.append({"step": step+1, "rmse_lstm": rmse, "rmse_xgb": rmse_x, "rmse_ensemble": rmse_ens,
                            "mae_lstm": mae, "mae_xgb": mae_x, "mae_ensemble": mae_ens})

    rmse_df = pd.DataFrame(rmse_per_step).set_index("step")

    st.write("### Evaluation Metrics (Validation Set) — per horizon (in €)")
    st.dataframe(rmse_df.style.format("{:.3f}"))

    # 6) Plot Actual vs Predictions for Step+1..N (show first N samples)
    n_show = min(80, len(y_val))
    for step in range(n_future):
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_val[:, step].shape[0] and inverse_transform_col_series(y_val[:, step], scaler, n_features_total)[:n_show], label="Actual", color="black")
        ax.plot(y_val_seq_lstm_final_inv_list[step][:n_show], label="LSTM Pred", linestyle="--")
        ax.plot(y_val_xgb_final_inv_list[step][:n_show], label="XGBoost Pred", linestyle="--")
        ax.plot(y_val_ensemble_inv_list[step][:n_show], label="Ensemble Avg", linestyle=":")
        ax.set_title(f"Actual vs Predicted — Step+{step+1} (first {n_show} val samples)")
        ax.set_xlabel("Validation sample index")
        ax.set_ylabel("Transfer Value (€)")
        ax.legend()
        st.pyplot(fig)

    # 7) Residual histogram for Step+1
    step_to_plot = 0
    res_lstm = inverse_transform_col_series(y_val[:, step_to_plot], scaler, n_features_total) - y_val_seq_lstm_final_inv_list[step_to_plot]
    res_xgb = inverse_transform_col_series(y_val[:, step_to_plot], scaler, n_features_total) - y_val_xgb_final_inv_list[step_to_plot]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(res_lstm, bins=30, alpha=0.6, label="LSTM residuals")
    ax.hist(res_xgb, bins=30, alpha=0.6, label="XGBoost residuals")
    ax.set_title(f"Residuals distribution (Step+{step_to_plot+1})")
    ax.legend()
    st.pyplot(fig)

    # 8) XGBoost feature importance for the first horizon's final model (meta features)
    if st.session_state.xgb_models_final:
        fig, ax = plt.subplots(figsize=(8,4))
        m0 = st.session_state.xgb_models_final[0]
        # get importance and convert to nicer labels
        fmap = m0.get_booster().get_score(importance_type='gain')  # dict {feature:score}
        # features are 'f0','f1',... where the last one is the lstm-pred meta feature
        items = sorted(fmap.items(), key=lambda t: t[1], reverse=True)[:20]
        names = [k for k, v in items]
        scores = [v for k, v in items]
        ax.barh(range(len(names))[::-1], scores, align='center')
        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(names[::-1])
        ax.set_title("XGBoost feature importance (gain) — Step+1 meta-model")
        st.pyplot(fig)

    st.success("Final models trained and comparison dashboard ready. Models are stored in session_state for reuse.")

    encoder_model = st.session_state.encoder_model_inf
    decoder_model = st.session_state.decoder_model_inf
    xgb_models = st.session_state.xgb_models_final

    st.divider()
    # -------------------- Download Buttons --------------------
    st.subheader("💾 Download Trained Models")
    col1, col2 = st.columns(2)

    with col1:
        ste.download_button(
            label="Download LSTM Seq2Seq Model",
            data=open("lstm_seq2seq_model.h5", "rb").read() if st.session_state.trained else None,
            file_name="lstm_seq2seq_model.h5"
        )

    with col2:
        # Save XGBoost ensemble to temp file
        temp_xgb_file = tempfile.NamedTemporaryFile(delete=False)
        joblib.dump(xgb_models, temp_xgb_file.name)
        temp_xgb_file.close()

        ste.download_button(
            label="Download XGBoost Models",
            data=open(temp_xgb_file.name, "rb").read(),
            file_name="xgb_models.pkl"
        )


    # ---------- Player-specific recursive forecast ----------
    st.write(f"### {n_future}-Step Forecast for Player: {player_choice}")
    player_data = df[df["player_id"] == pid].sort_values("transfer_date")

    if len(player_data) >= n_steps:
        last_seq = player_data.iloc[-n_steps:][feature_cols].fillna(0).values
        scaled_last_seq = scaler.transform(last_seq)[np.newaxis, :, :]
        preds_seq_scaled = predict_seq2seq_recursive(scaled_last_seq, encoder_model, decoder_model, n_future)
        preds_seq_inv = inverse_transform_col_series(preds_seq_scaled, scaler, X.shape[2])

        X_seq_flat = scaled_last_seq.reshape(1, -1)
        preds_xgb = []
        for step in range(n_future):
            meta_input = np.hstack([X_seq_flat, np.array(preds_seq_scaled[step]).reshape(1,1)])
            #preds_xgb.append(xgb_models[step].predict(meta_input)[0])
            preds_xgb.append(st.session_state.xgb_models_final[step].predict(meta_input)[0])
        preds_xgb_inv = inverse_transform_col_series(np.array(preds_xgb), scaler, X.shape[2])
        preds_ensemble_inv = (preds_seq_inv + preds_xgb_inv)/2

        # Actual last values for comparison
        actual_last = player_data["market_value"].values[-n_future:] if len(player_data) >= n_future else player_data["market_value"].values

        # Plot
        fig, ax = plt.subplots(figsize=(10,5))
        steps = np.arange(1, n_future+1)
        ax.plot(steps[:len(actual_last)], actual_last, marker='o', label="Actual", color="blue")
        ax.plot(steps, preds_seq_inv, marker='o', label="Seq2Seq LSTM", color="orange")
        ax.plot(steps, preds_xgb_inv, marker='o', label="XGBoost refined", color="green")
        ax.plot(steps, preds_ensemble_inv, marker='o', label="Ensemble (avg)", color="red", linestyle=":")
        ax.set_xlabel("Steps Ahead")
        ax.set_ylabel("Transfer Value (€ millions)")
        ax.set_title(f"{n_future}-Step Forecast for Player {player_choice}")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Not enough data for recursive forecasting for this player.")


# Close DB connection
db.close()
