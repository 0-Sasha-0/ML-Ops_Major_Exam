# src/quantize.py

import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

data = fetch_california_housing()
X = data.data
y = data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load("artifacts/linear_model.joblib")

weights = model.coef_
bias = model.intercept_

joblib.dump({"weights": weights, "bias": bias}, "artifacts/unquant_params.joblib")

scale = 0.01

quantized_weights = np.clip(np.round(weights / scale), 0, 255).astype(np.uint8)
quantized_bias = np.clip(np.round(bias / scale), 0, 255).astype(np.uint8)

joblib.dump({
    "quant_weights": quantized_weights,
    "quant_bias": quantized_bias,
    "scale": scale
}, "artifacts/quant_params.joblib")

dequantized_weights = quantized_weights.astype(np.float32) * scale
dequantized_bias = quantized_bias * scale

y_pred = np.dot(X_test, dequantized_weights) + dequantized_bias

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Sample dequantized predictions:", y_pred[:5])
print(f"R² Score after dequantization: {r2:.4f}")
print(f"Mean Squared Error after dequantization: {mse:.4f}")
