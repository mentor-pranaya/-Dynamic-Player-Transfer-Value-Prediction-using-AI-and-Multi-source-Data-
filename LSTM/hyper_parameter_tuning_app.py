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
import random
import time
from tensorflow.keras.optimizers import Adam


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

if st.sidebar.button("Click here to start training and evaluation"):

    # Prepare sequences
    X_list, y_list, player_index = [], [], []
    scaler = MinMaxScaler()
    feature_cols = [
        "market_value", "total_injuries", "sentiment_mean", "avg_cards_per_match",
        "avg_days_out", "recent_injury", "days_since_last_injury",
        "minutes_played", "shots_per90", "pressures_per90"
    ]

    for p, group in df.groupby("transfermarkt_id"):
        features = group[feature_cols].fillna(0).values
        scaled = scaler.fit_transform(features)
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

    st.subheader("Training Seq2Seq Encoder-Decoder LSTM (with teacher forcing)")
    progress_bar = st.progress(0)
    epoch_log = st.empty()
    chart_placeholder = st.empty()
    loss_chart = chart_placeholder.line_chart({"Train Loss": [], "Val Loss": []})

    class StreamlitLogger(Callback):
        def __init__(self, total_epochs):
            super().__init__()
            self.total_epochs = total_epochs
            self.history = {"loss": [], "val_loss": []}

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

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # Dummy outputs for state placeholders
    dummy_train = np.zeros((len(y_train), latent_dim))
    dummy_val = np.zeros((len(y_val), latent_dim))

    history = seq2seq.fit(
        [X_train, decoder_input_train],
        [y_train[..., np.newaxis], dummy_train, dummy_train],
        #y_train[..., np.newaxis],
        epochs=epochs,
        batch_size=32,
        validation_data=([X_val, decoder_input_val], [y_val[..., np.newaxis], dummy_val, dummy_val]),
        #validation_data=([X_val, decoder_input_val], y_val[..., np.newaxis]),
        verbose=0,
        callbacks=[es, StreamlitLogger(epochs)]
    )

    # Plot loss curve
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.legend()
    ax.set_title("Loss Curves")
    st.pyplot(fig)
    st.write("Training and validation loss curves.")
    # ---------- Build inference models ----------
    encoder_model, decoder_model = build_seq2seq_inference_models(
        seq2seq, n_steps=X.shape[1], n_features=X.shape[2], latent_dim=latent_dim
    )

    st.success("Training completed!")


    # ---------- XGBoost ensemble ----------

    y_train_seq_pred = seq2seq.predict([X_train, decoder_input_train])[0].squeeze(-1)
    y_val_seq_pred = seq2seq.predict([X_val, decoder_input_val])[0].squeeze(-1)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    xgb_models = []
    y_val_pred_xgb_all = np.zeros_like(y_val_seq_pred)

    for step in range(n_future):
        X_meta_train = np.hstack([X_train_flat, y_train_seq_pred[:, step].reshape(-1,1)])
        X_meta_val = np.hstack([X_val_flat, y_val_seq_pred[:, step].reshape(-1,1)])
        model_xgb = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42)
        model_xgb.fit(X_meta_train, y_train[:, step])
        xgb_models.append(model_xgb)
        y_val_pred_xgb_all[:, step] = model_xgb.predict(X_meta_val)
        # Ensemble predictions
        y_val_ensemble = (y_val_seq_pred + y_val_pred_xgb_all) / 2
        st.subheader("Validation Results Comparison")
         # Validation plotting ----------
        def inverse_transform_col(col_scaled, scaler, n_features_total):
            col_scaled = np.array(col_scaled).reshape(-1, 1)
            expanded = np.zeros((len(col_scaled), n_features_total))
            expanded[:, 0] = col_scaled.flatten()
            return scaler.inverse_transform(expanded)[:, 0]

        y_val_true_inv = inverse_transform_col(y_val[:, 0], scaler, X.shape[2])
        y_val_seq_lstm_inv = inverse_transform_col(y_val_seq_pred[:, 0], scaler, X.shape[2])
        y_val_xgb_inv = inverse_transform_col(y_val_pred_xgb_all[:, 0], scaler, X.shape[2])
        y_val_ensemble_inv = inverse_transform_col(y_val_ensemble[:, 0], scaler, X.shape[2])

        fig, ax = plt.subplots(figsize=(12,6))
        n_show = min(50, len(y_val_true_inv))
        ax.plot(y_val_true_inv[:n_show], label="Actual", color="blue")
        ax.plot(y_val_seq_lstm_inv[:n_show], label="Seq2Seq LSTM", color="orange", linestyle="--")
        ax.plot(y_val_xgb_inv[:n_show], label="XGBoost", color="green", linestyle="--")
        ax.plot(y_val_ensemble_inv[:n_show], label="Ensemble (avg)", color="red", linestyle=":")
        ax.set_title(f"Validation Forecast (Step+{step+1}) - Seq2Seq + XGBoost Ensemble")
        ax.set_xlabel("Validation Samples")
        ax.set_ylabel("Transfer Value (€ millions)")
        ax.legend()
        st.pyplot(fig)
        st.write("Validation forecast (first step) comparison for Seq2Seq LSTM, XGBoost, and Ensemble.")

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
            preds_xgb.append(xgb_models[step].predict(meta_input)[0])
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
    
    # ---------- Hyperparameter Tuning ----------
    #if st.sidebar.button("Run Hyperparameter Tuning"):
    ## Init session state for LSTM
    #if "lstm_results" not in st.session_state:
    #    st.session_state.lstm_results = []
    #if "lstm_step" not in st.session_state:
    #    st.session_state.lstm_step = 0

    ## Init session state for XGBoost
    #if "xgb_results" not in st.session_state:
    #    st.session_state.xgb_results = []
    #if "xgb_step" not in st.session_state:
    #    st.session_state.xgb_step = 0

    # ---------- Hyperparameter Tuning XGB ----------
    if tune_xgb:
        st.subheader("XGBoost Hyperparameter Tuning")
        def random_search_xgb(X_meta_train, y_train_step, X_meta_val, y_val_step):
            progress_bar = st.progress(0)
            table_placeholder = st.empty()
            chart_placeholder = st.empty()
            rmse_list = []

            best_rmse = float("inf")
            best_params = {}

            n_estimators_options = [100, 300, 500]
            max_depth_options = [4, 6, 8]
            lr_options = [0.01, 0.05, 0.1]
            subsample_options = [0.7, 0.8, 1.0]
            colsample_options = [0.7, 0.8, 1.0]

            for i in range(xgb_search_size):
                n_estimators = random.choice(n_estimators_options)
                max_depth = random.choice(max_depth_options)
                learning_rate = random.choice(lr_options)
                subsample = random.choice(subsample_options)
                colsample_bytree = random.choice(colsample_options)

                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42
                )
                model.fit(X_meta_train, y_train_step)
                preds = model.predict(X_meta_val)
                rmse = np.sqrt(((preds - y_val_step)**2).mean())

                rmse_list.append(rmse)
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree
                    }
                
                # Update session results
                st.session_state.xgb_results.append({
                    "iteration": i+1,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "rmse": rmse
                })

                # Update table and chart
                df_table = pd.DataFrame(st.session_state.xgb_results)
                table_placeholder.dataframe(df_table)
                chart_placeholder.line_chart(df_table.set_index("iteration")["rmse"])

                # Progress bar
                progress_bar.progress((i+1)/xgb_search_size)
            return best_params, best_rmse
     
        st.write("Running random search for XGBoost hyperparameters...")
        # Prepare meta-train for first step as example
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        y_train_seq_pred = seq2seq.predict([X_train, decoder_input_train])[0].squeeze(-1)
        y_val_seq_pred = seq2seq.predict([X_val, decoder_input_val])[0].squeeze(-1)
        X_meta_train = np.hstack([X_train_flat, y_train_seq_pred[:, 0].reshape(-1,1)])
        X_meta_val = np.hstack([X_val_flat, y_val_seq_pred[:, 0].reshape(-1,1)])
        
        best_xgb_params, best_xgb_rmse = random_search_xgb(
            X_meta_train, y_train[:, 0], X_meta_val, y_val[:, 0]
        )
        st.success(f"Best XGBoost params: {best_xgb_params} | RMSE: {best_xgb_rmse:.4f}")

    # ---------- Hyperparameter Tuning LSTM ----------
    if tune_lstm:
        st.subheader("LSTM Hyperparameter Tuning")
        st.write("Running random search for LSTM hyperparameters...")
        progress_bar_epoch = st.progress(0)
        epoch_log_hyper = st.empty()
        class UpdateEpochProgress(Callback):
            def __init__(self, total_epochs):
                super().__init__()
                self.total_epochs = total_epochs
                #self.epoch_progress = st.progress(0)

            def on_epoch_end(self, epoch, logs=None):
                #self.epoch_progress.progress((epoch + 1) / self.total_epochs)
                progress_bar_epoch.progress((epoch + 1) / self.total_epochs)
                epoch_log_hyper.text(f"Epoch {epoch+1}/{self.total_epochs} - Loss: {logs.get('loss'):.4f}, Val Loss: {logs.get('val_loss'):.4f}")

        def random_search_lstm(X_train, decoder_input_train, y_train, X_val, decoder_input_val, y_val, dummy_train, dummy_val, n_steps, n_future, n_features):
            best_val_loss = float("inf")
            best_params = {}
            progress_bar = st.progress(0)
            table_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            latent_options = [32, 64, 128]
            batch_options = [16, 32, 64]
            lr_options = [0.001, 0.005, 0.01]

            losses = []

            for i in range(lstm_search_size):
                latent_dim = random.choice(latent_options)
                batch_size = random.choice(batch_options)
                lr = random.choice(lr_options)
                progress_bar_epoch.progress(0)
                model = build_seq2seq_train(n_steps, n_features, n_future, latent_dim=latent_dim)
                model.compile(optimizer="adam", loss="mse")  # optionally modify optimizer with lr
                
                dummy_train = np.zeros((X_train.shape[0], latent_dim))
                dummy_val = np.zeros((X_val.shape[0], latent_dim))
                
                history = model.fit(
                    [X_train, decoder_input_train],
                    [y_train[..., np.newaxis], dummy_train, dummy_train],
                    #y_train[..., np.newaxis],
                    epochs=lstm_search_epoches,  # keep short for tuning
                    batch_size=batch_size,
                    validation_data=([X_val, decoder_input_val], [y_val[..., np.newaxis], dummy_val, dummy_val]),
                    #validation_data=([X_val, decoder_input_val], y_val[..., np.newaxis]),
                    verbose=0,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True), 
                            UpdateEpochProgress(lstm_search_epoches)]
                )
                val_loss = min(history.history["val_loss"])

                losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {"latent_dim": latent_dim, "batch_size": batch_size, "learning_rate": lr}
                
                # Update session results
                st.session_state.lstm_results.append({
                    "iteration": i+1,
                    "latent_dim": latent_dim,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "rmse": val_loss
                })

                # Update table
                df_table = pd.DataFrame(st.session_state.lstm_results)
                table_placeholder.dataframe(df_table)
                # Update chart
                chart_placeholder.line_chart(df_table.set_index("iteration")["rmse"])
                # Update progress bar
                progress_bar.progress((i+1)/lstm_search_size)
                epoch_log_hyper.text(f"Completed {i+1}/{lstm_search_size} iterations")

            return best_params, best_val_loss
    
        best_lstm_params, best_lstm_loss = random_search_lstm(
            X_train, decoder_input_train, y_train,
            X_val, decoder_input_val, y_val,
            dummy_train, dummy_val,
            n_steps, n_future, X.shape[2]
        )
        st.success(f"Best LSTM params: {best_lstm_params} | Val Loss: {best_lstm_loss:.4f}")

        
        #tab1, tab2 = st.tabs(["🔵 LSTM Random Search", "🟠 XGBoost Random Search"])
        #tab1, tab2 = st.columns(2)

        ##with tab1:
        #st.markdown("### 🔵 LSTM Random Search")
        #run_random_search(
        #    "lstm",
        #    param_grid={
        #        "units": [32, 64, 128],
        #        "dropout": [0.1, 0.2, 0.3],
        #        "batch_size": [16, 32, 64]
        #    },
        #    n_iter=10
        #)

        ##with tab2:
        #st.markdown("### 🟠 XGBoost Random Search")
        #run_random_search(
        #    "xgb",
        #    param_grid={
        #        "max_depth": [3, 5, 7],
        #        "learning_rate": [0.01, 0.05, 0.1],
        #        "n_estimators": [100, 200, 300]
        #    },
        #    n_iter=10
        #)

    

    # --- Progress visualisation for Hyper Parameter Tuning ---
    def run_random_search(model_name, param_grid, n_iter=10):
        results_key = f"{model_name}_results"
        step_key = f"{model_name}_step"

        progress_bar = st.progress(0)
        status_text = st.empty()
        table_placeholder = st.empty()
        chart_placeholder = st.empty()

        st.session_state[results_key] = []

        for i in range(n_iter):
            # pick random params
            params = {k: np.random.choice(v) for k, v in param_grid.items()}

            # --- Train model & get RMSE ---
            # Replace this dummy line with actual model training + validation
            rmse = np.random.uniform(0.5, 2.0)

            # store result
            st.session_state[results_key].append({"iteration": i+1, "params": params, "rmse": rmse})
            st.session_state[step_key] = i+1

            # update progress
            progress_bar.progress((i+1)/n_iter)
            status_text.text(f"{model_name.upper()} Iteration {i+1}/{n_iter} | RMSE: {rmse:.4f}")

            # update results table
            df = pd.DataFrame(st.session_state[results_key])
            table_placeholder.dataframe(df)

            # update graph
            fig, ax = plt.subplots()
            ax.plot(df["iteration"], df["rmse"], marker="o", color="blue" if model_name=="lstm" else "orange")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("RMSE")
            ax.set_title(f"{model_name.upper()} Random Search Progress")
            chart_placeholder.pyplot(fig)

            #time.sleep(1)  # simulate long training
    # Check if we have results for both models
    if st.session_state.lstm_results and st.session_state.xgb_results:

        st.subheader("📊 LSTM vs XGBoost Random Search Comparison")

        # Convert to DataFrames
        df_lstm = pd.DataFrame(st.session_state.lstm_results)
        df_xgb = pd.DataFrame(st.session_state.xgb_results)

        # Fill missing iterations if different
        max_iter = max(len(df_lstm), len(df_xgb))
        df_lstm = df_lstm.reindex(range(max_iter), fill_value=np.nan)
        df_xgb = df_xgb.reindex(range(max_iter), fill_value=np.nan)

        # Side-by-side columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔵 LSTM RMSE")
            st.line_chart(df_lstm["rmse"])

        with col2:
            st.markdown("### 🟠 XGBoost RMSE")
            st.line_chart(df_xgb["rmse"])

        # Combined comparison plot
        st.markdown("### Combined Comparison")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df_lstm["iteration"], df_lstm["rmse"], marker="o", color="blue", label="LSTM")
        ax.plot(df_xgb["iteration"], df_xgb["rmse"], marker="o", color="orange", label="XGBoost")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE / Loss")
        ax.set_title("Random Search Progress Comparison")
        ax.legend()
        st.pyplot(fig)

        best_lstm_idx = df_lstm["rmse"].idxmin()
        best_xgb_idx = df_xgb["rmse"].idxmin()
        st.write(f"✅ Best LSTM RMSE: {df_lstm.loc[best_lstm_idx, 'rmse']:.4f} at iteration {best_lstm_idx+1}")
        st.write(f"✅ Best XGBoost RMSE: {df_xgb.loc[best_xgb_idx, 'rmse']:.4f} at iteration {best_xgb_idx+1}")


# Close DB connection
db.close()
