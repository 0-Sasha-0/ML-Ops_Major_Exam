# src/predict.py

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load dataset
data = fetch_california_housing()
X = data.data
_, X_test, _, _ = train_test_split(X, data.target, test_size=0.2, random_state=42)

# Load model
model = joblib.load("artifacts/linear_model.joblib")

# Predict
predictions = model.predict(X_test[:5])

# Show output
print("Sample predictions:", predictions)
