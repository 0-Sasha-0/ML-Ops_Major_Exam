import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Load the model
model = joblib.load("../artifacts/linear_model.joblib")

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions
predictions = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("Sample predictions:", predictions[:5])
print("R² Score:", round(r2, 4))
print("Mean Squared Error:", round(mse, 4))

