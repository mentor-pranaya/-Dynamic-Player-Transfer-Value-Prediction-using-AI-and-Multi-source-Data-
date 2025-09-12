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
df = pd.read_csv("/Users/veerababu/Downloads/cleaned/master_list_final_features.csv")
print("Dataset loaded:", df.shape)

# -------------------------------
# 2. Numeric and categorical features
# -------------------------------
numeric_features = ['age', 'experience_years', 'height_in_cm',
                    'total_injuries', 'total_days_missed',
                    'contract_years_remaining']

cat_features = ['current_club_name', 'sub_position']

# Fill missing numeric values
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Label encode categorical columns
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Target
y = df['market_value_in_eur'].astype(float).values

# Train-test split
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
# 3. Build model with embeddings
# -------------------------------
inputs = []
embeddings = []

for i, col in enumerate(cat_features):
    vocab_size = df[col].nunique() + 1
    embed_dim = min(50, vocab_size // 2)  # reasonable embedding size
    input_i = Input(shape=(1,))
    embedding_i = Embedding(vocab_size, embed_dim)(input_i)
    embedding_i = Flatten()(embedding_i)
    inputs.append(input_i)
    embeddings.append(embedding_i)

# Numeric input
num_input = Input(shape=(X_train_num.shape[1],))
inputs.append(num_input)
embeddings.append(num_input)

# Concatenate all inputs
x = Concatenate()(embeddings)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(1)(x)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -------------------------------
# 4. Callbacks for safe training
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model_path = "/Users/veerababu/Downloads/cleaned/market_value_model.keras"
checkpoint = ModelCheckpoint(
    model_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# -------------------------------
# 5. Train model
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
# 6. Evaluate
# -------------------------------
loss, mae = model.evaluate(
    [X_test_cat[:, i] for i in range(X_test_cat.shape[1])] + [X_test_num],
    y_test
)
print("Test MAE:", mae)

# -------------------------------
# 7. Save predictions
# -------------------------------
predictions = model.predict([X_test_cat[:, i] for i in range(X_test_cat.shape[1])] + [X_test_num])
pred_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": predictions.flatten()
})
pred_df.to_csv("/Users/veerababu/Downloads/cleaned/nn_predictions.csv", index=False)
print("Predictions saved to CSV.")
