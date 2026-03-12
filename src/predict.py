import joblib
import numpy as np
import pandas as pd

model = joblib.load("model/medipredict_model.pkl")
scaler = joblib.load("model/scaler.pkl")
imputer = joblib.load("model/imputer.pkl")

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

def prepare_input(data_dict):
    df = pd.DataFrame([data_dict])[FEATURES]
    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)
    return df, X_scaled

def predict_risk(data_dict):
    df, X_scaled = prepare_input(data_dict)
    proba = model.predict_proba(X_scaled)[0, 1]

    if proba < 0.33:
        level = "Faible"
    elif proba < 0.66:
        level = "Modéré"
    else:
        level = "Élevé"

    return proba, level, df, X_scaled