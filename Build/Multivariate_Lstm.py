#  Multivariate LSTM — uses all selected features
# -----------------------------
print('\nBuilding & training multivariate LSTM...')

multi_model = Sequential()
multi_model.add(LSTM(128, input_shape=(N_STEPS_IN, n_feat)))
if N_STEPS_OUT == 1:
    multi_model.add(Dense(1))
else:
    multi_model.add(Dense(N_STEPS_OUT))

multi_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
multi_model.summary()

callbacks_multi = [EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                   ModelCheckpoint(os.path.join(MODEL_DIR,'multivariate_lstm.h5'), save_best_only=True, monitor='val_loss')]

# history_multi = multi_model.fit(X_train_scaled, y_train_scaled if N_STEPS_OUT>1 else y_train_scaled.reshape(-1,1),
#                                 validation_data=(X_val_scaled, y_val_scaled if N_STEPS_OUT>1 else y_val_scaled.reshape(-1,1)),
#                                 epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_multi, verbose=2)

history_multi = multi_model.fit(
    X_train_scaled,
    y_train_scaled if N_STEPS_OUT > 1 else y_train_scaled.reshape(-1, 1),
    validation_data=(
        X_val_scaled,
        y_val_scaled if N_STEPS_OUT > 1 else y_val_scaled.reshape(-1, 1)
    ),
    epochs=10,               # 🔹 reduced epochs
    batch_size=BATCH_SIZE,
    callbacks=callbacks_multi,
    verbose=2
)
