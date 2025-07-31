# tests/test_train.py

import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def test_dataset_loading():
    data = fetch_california_housing()
    assert data.data.shape[0] > 0
    assert data.target.shape[0] > 0

def test_model_training_and_r2_score():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    assert hasattr(model, 'coef_')  # Check if model is trained

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    assert r2 > 0.5  # Arbitrary threshold to ensure model is learning
