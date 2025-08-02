import numpy as np
import pandas as pd  # <-- ADD THIS LINE to load CSVs
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load the model
model = joblib.load("artifacts/linear_model.joblib")

# Load test data
X_test = pd.read_csv("artifacts/X_test.csv")
y_test = pd.read_csv("artifacts/y_test.csv")

# Make predictions
predictions = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("Sample predictions:", predictions[:5])
print("R² Score:", round(r2, 4))
print("Mean Squared Error:", round(mse, 4))

