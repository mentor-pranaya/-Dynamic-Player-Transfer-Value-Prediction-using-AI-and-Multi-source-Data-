#  Encoder-Decoder LSTM for multi-step forecasting
# -----------------------------
if N_STEPS_OUT > 1:
    print('\nBuilding encoder-decoder LSTM for multi-step forecasting...')
    n_features = n_feat
    encoder_inputs = Input(shape=(N_STEPS_IN, n_features))
    encoder = LSTM(128, activation='tanh', return_sequences=False)(encoder_inputs)
    # Repeat & decode
    decoder = RepeatVector(N_STEPS_OUT)(encoder)
    decoder = LSTM(128, activation='tanh', return_sequences=True)(decoder)
    decoder_outputs = TimeDistributed(Dense(1))(decoder)

    seq2seq = Model(encoder_inputs, decoder_outputs)
    seq2seq.compile(optimizer='adam', loss='mse', metrics=['mae'])
    seq2seq.summary()

    callbacks_ed = [EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                    ModelCheckpoint(os.path.join(MODEL_DIR,'encoder_decoder_lstm.h5'), save_best_only=True, monitor='val_loss')]

    # reshape y to (samples, n_out, 1)
    y_train_ed = y_train_scaled.reshape(y_train_scaled.shape[0], N_STEPS_OUT, 1)
    y_val_ed = y_val_scaled.reshape(y_val_scaled.shape[0], N_STEPS_OUT, 1)

    # history_ed = seq2seq.fit(X_train_scaled, y_train_ed, validation_data=(X_val_scaled, y_val_ed),
    #                          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_ed, verbose=2)

    history_ed = seq2seq.fit(
    X_train_scaled, y_train_ed,
    validation_data=(X_val_scaled, y_val_ed),
    epochs=10,           
    batch_size=BATCH_SIZE,
    callbacks=callbacks_ed,
    verbose=2
)

