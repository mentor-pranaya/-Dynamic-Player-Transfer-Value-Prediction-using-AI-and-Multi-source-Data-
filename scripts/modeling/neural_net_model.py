import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("/Users/veerababu/Downloads/master_list_cleaned.csv")
print("Dataset loaded:", df.shape)

# -------------------------------
# 2. Numeric and categorical features
# -------------------------------
numeric_features = ['age', 'experience_years', 'height_in_cm',
                    'total_injuries', 'total_days_missed',
                    'contract_years_remaining']

cat_features = ['current_club_name', 'sub_position']

# Ensure numeric columns are numeric
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values with median
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Label encode categorical columns
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# -------------------------------
# 3. Target and train-test split
# -------------------------------
y = pd.to_numeric(df['market_value_in_eur'], errors='coerce').fillna(0).values

X_num = df[numeric_features].values
X_cat = df[cat_features].values

X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

print("Train shape:", X_train_num.shape, "Test shape:", X_test_num.shape)

# -------------------------------
# 4. Build model with embeddings
# -------------------------------
inputs = []
embeddings = []

for i, col in enumerate(cat_features):
    vocab_size = df[col].nunique() + 1
    embed_dim = min(50, max(2, vocab_size // 2))  # safe embedding size
    input_i = Input(shape=(1,))
    embedding_i = Embedding(input_dim=vocab_size, output_dim=embed_dim)(input_i)
    embedding_i = Flatten()(embedding_i)
    inputs.append(input_i)
    embeddings.append(embedding_i)

# Numeric input
num_input = Input(shape=(X_train_num.shape[1],))
inputs.append(num_input)
embeddings.append(num_input)

# Concatenate embeddings + numeric features
x = Concatenate()(embeddings)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(1)(x)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -------------------------------
# 5. Callbacks
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Save model directly in Downloads
model_path = "/Users/veerababu/Downloads/market_value_model.keras"
checkpoint = ModelCheckpoint(
    model_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# -------------------------------
# 6. Train model
# -------------------------------
history = model.fit(
    [X_train_cat[:, i] for i in range(X_train_cat.shape[1])] + [X_train_num],
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# -------------------------------
# 7. Evaluate model
# -------------------------------
loss, mae = model.evaluate(
    [X_test_cat[:, i] for i in range(X_test_cat.shape[1])] + [X_test_num],
    y_test
)
print("Test MAE:", mae)

# -------------------------------
# 8. Save predictions
# -------------------------------
predictions = model.predict([X_test_cat[:, i] for i in range(X_test_cat.shape[1])] + [X_test_num])
pred_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": predictions.flatten()
})
pred_csv_path = "/Users/veerababu/Downloads/nn_predictions.csv"
pred_df.to_csv(pred_csv_path, index=False)
print(f"Predictions saved to CSV at {pred_csv_path}")
print(f"Model saved at {model_path}")
