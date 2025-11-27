# src/api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from enum import Enum
import shap

from src.Pipelines.predict_pipeline import ChurnPredictor

app = FastAPI(title="Churn Prediction API", version="1.0")

predictor = ChurnPredictor()


# ----------- Request Schema -----------

class GeographyEnum(str, Enum):
    france = "France"
    germany = "Germany"
    spain = "Spain"

class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"

class Customer(BaseModel):
    CreditScore: int
    Geography: GeographyEnum
    Gender: GenderEnum
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    


# ----------- API Endpoints -----------

@app.get("/")
def root():
    return {"message": "Churn Prediction App"}



@app.post("/predict_batch")
def predict_batch(customers: list[Customer]):
    # convert payload into dataframe
    df = pd.DataFrame([c.model_dump() for c in customers])

    # predict
    probas = predictor.predict_proba(df)
  

    return {
        "churn_probabilities": probas.tolist(),
        }

@app.post("/shap_values")
def shap_values(customers: list[Customer]):
    df = pd.DataFrame([c.model_dump() for c in customers])

    # Preprocess input using saved preprocessor
    X_processed = predictor.preprocess(df)
    # Compute SHAP Values
    if predictor.model_name() in ["RandomForestClassifier", "XGBClassifier"]:
        explainer = shap.TreeExplainer(predictor.model)
    
    
    if predictor.model_name() in ["LogisticRegression"]:
        explainer = shap.LinearExplainer(
            predictor.model,
            X_processed,
            feature_perturbation = "interventional"
        )

    shap_vals = explainer.shap_values(X_processed)   # Convert numpy arrays → lists for JSON
    
    shap_values_list = shap_vals.tolist()

    expected_value = explainer.expected_value.tolist() if hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value)

    return {
        "shap_values": shap_values_list,
        "expected_value": expected_value,
        "feature_names": list(X_processed.columns)
    }



@app.get("/health")
def health_check():
    return {"status": "ok"}
