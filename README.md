# MLOps Major Exam
## 📝 Project Overview

This project implements a complete MLOps workflow for predicting housing prices using the California Housing dataset. The goal was to build a clean and modular machine learning pipeline using Linear Regression and apply MLOps best practices.

## 📁 Project Structure
```
california-housing-mlops/
├── src/
│   ├── train.py          # Script to train and save the model
│   ├── predict.py        # Script to load model and make predictions
│   └── artifacts/        # Contains saved model and test data
│       ├── linear_model.joblib
│       ├── X_test.csv
│       └── y_test.csv
├── requirements.txt
└── README.md
```


## 🔧 Tools and Libraries

- Python
- scikit-learn
- pandas
- numpy
- joblib

## 📈 Model Performance

- **R² Score**: 0.5758  
- **Mean Squared Error (MSE)**: 0.5559

These metrics are based on predictions made on the test set using the trained Linear Regression model.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/0-Sasha-0/ML-Ops_Major_Exam.git
   cd ML-Ops_Major_Exam

2. Install the dependencies:
pip install -r requirements.txt

3. Train the model:
python src/train.py

4. Make predictions:
python src/predict.py
