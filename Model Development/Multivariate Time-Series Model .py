# ...existing code...

# Ensure these columns exist in your DataFrame
required_features = ['performance_stat1', 'performance_stat2', 'injury_history', 'sentiment_score', 'market_value']
assert all(col in df.columns for col in required_features), "Missing required columns!"

# Use only relevant features
df_multivariate = df[required_features]

X, y = create_sequences(df_multivariate, sequence_length, 'market_value')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Re-train LSTM with multivariate input
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)