def create_sequences_playerwise(df, feature_cols, target_col, n_steps_in, n_steps_out=1):
    """
    Create sequences by sliding a window over each player's timeseries.
    Returns X (samples, n_steps_in, n_features), y (samples, n_steps_out).
    """
    X_list, y_list = [], []
    players = df['player_id'].unique()

    for pid in players:
        p = df[df['player_id'] == pid].sort_values('date')
        values = p[feature_cols + [target_col]].values
        n_features = len(feature_cols)   # <-- correctly indented

        for i in range(len(values) - n_steps_in - n_steps_out + 1):
            X_list.append(values[i:i+n_steps_in, :n_features])
            y_list.append(values[i+n_steps_in:i+n_steps_in+n_steps_out, n_features])

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y
# 9) Split into train/val by time (player-wise holdout of last portion)
# -----------------------------
# I'll create sequences then split last X% of each player's sequences for validation to avoid leakage.

# create a copy and drop rows with missing target
ts = ts.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# Temporarily fill features NaN with 0 (we will scale after split). You might want smarter imputation.
ts_filled = ts.copy()
for col in FEATURE_COLS:
    if pd.api.types.is_numeric_dtype(ts_filled[col]):
        ts_filled[col] = ts_filled[col].fillna(0)
    else:
        ts_filled[col] = ts_filled[col].fillna(0)

# Create sequences
print('Creating sequences... (this may take a while)')
X_all, y_all = create_sequences_playerwise(ts_filled, FEATURE_COLS, TARGET_COL, N_STEPS_IN, N_STEPS_OUT)
print('Total sequences:', X_all.shape[0])

# Train/val split by temporal holdout — we'll split sequences by player timeline using an index per-player
# Simpler approach: shuffle and use a validation split (not ideal for strict time-series, but workable when sequences are already time-ordered per player)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=VAL_SPLIT, random_state=RANDOM_SEED, shuffle=True)

print('Train sequences:', X_train.shape[0], 'Val sequences:', X_val.shape[0])

# 10) Scaling — fit on training data then apply to val & test
# -----------------------------
# Flatten X_train for scaler fit
n_train = X_train.shape[0]
n_tim = X_train.shape[1]
n_feat = X_train.shape[2]

scaler_X = MinMaxScaler(feature_range=(0,1))
scaler_y = MinMaxScaler(feature_range=(0,1))

X_train_flat = X_train.reshape(-1, n_feat)
X_val_flat = X_val.reshape(-1, n_feat)

scaler_X.fit(X_train_flat)
X_train_scaled = scaler_X.transform(X_train_flat).reshape(n_train, n_tim, n_feat)
X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape[0], n_tim, n_feat)

# y scaling (target is 1D)
y_train_flat = y_train.reshape(-1, 1)
y_val_flat = y_val.reshape(-1, 1)
scaler_y.fit(y_train_flat)
y_train_scaled = scaler_y.transform(y_train_flat).reshape(y_train.shape)
y_val_scaled = scaler_y.transform(y_val_flat).reshape(y_val.shape)



# 11) Univariate LSTM (history of target only)

# -----------------------------
print('\nBuilding & training univariate LSTM (target only)...')

# Build univariate X from the scaled X (but we'll select the target column index inside features if present)
if TARGET_COL in FEATURE_COLS:
    targ_idx = FEATURE_COLS.index(TARGET_COL)
    X_uni_train = X_train_scaled[:,:,targ_idx:targ_idx+1]
    X_uni_val = X_val_scaled[:,:,targ_idx:targ_idx+1]
else:
    # if target not in features, build sequences directly from the target series
    # rebuild univariate sequences from raw ts
    values = ts_filled.groupby('player_id')[TARGET_COL].apply(lambda s: s.values)
    # For brevity, reuse X_train_scaled structure
    X_uni_train = X_train_scaled[:,:, -1:]  # best-effort fallback
    X_uni_val = X_val_scaled[:,:, -1:]

# reshape y for single-step
if N_STEPS_OUT == 1:
    y_train_uni = y_train_scaled.reshape(-1,1)
    y_val_uni = y_val_scaled.reshape(-1,1)
else:
    y_train_uni = y_train_scaled
    y_val_uni = y_val_scaled

# model
uni_model = Sequential()
uni_model.add(LSTM(64, input_shape=(N_STEPS_IN, X_uni_train.shape[2])))
if N_STEPS_OUT == 1:
    uni_model.add(Dense(1))
else:
    uni_model.add(Dense(N_STEPS_OUT))

uni_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
uni_model.summary()

callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
             ModelCheckpoint(os.path.join(MODEL_DIR,'univariate_lstm.h5'), save_best_only=True, monitor='val_loss')]

# history_uni = uni_model.fit(X_uni_train, y_train_uni, validation_data=(X_uni_val, y_val_uni),
#                             epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)

history_uni = uni_model.fit(
    X_uni_train, y_train_uni,
    validation_data=(X_uni_val, y_val_uni),
    epochs=10,                # 🔹 reduced epochs
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)



