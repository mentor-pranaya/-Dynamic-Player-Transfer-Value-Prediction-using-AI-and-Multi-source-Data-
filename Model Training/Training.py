import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# Example dataset: [perf_mean, perf_max, perf_min, perf_std, sentiment, injury, contract]
X = np.array([
    [28, 30, 25, 2.16, 0.8, 1, 24],
    [26, 27, 25, 1.0, 0.6, 2, 18],
    [27, 29, 24, 2.05, 0.7, 0, 36]
])
y = [60, 40, 70]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
pickle.dump(model, open("transferiq_model.pkl", "wb"))
print("Model trained and saved as transferiq_model.pkl")
