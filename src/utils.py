import os
import sys
import pandas as pd
import numpy as np
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report



from src.logger import logging
from src.exception import CustomException


import os

def get_artifacts_path() -> str:
    """
    Returns the artifacts folder path.
    - Training: uses local ./artifacts
    - Inference: uses ARTIFACTS_PATH env variable inside Docker
    """
    return os.getenv("ARTIFACTS_PATH", "artifacts")



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, X_test, y_train, y_test, models, params):
    try:
        report={}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(params.keys())[i]]
            
            GridSearch = GridSearchCV(estimator=model, param_grid=param, cv=2, n_jobs=-1, verbose=2, scoring='average_precision')
            GridSearch.fit(X_train, y_train) 

            logging.info(f"Best GridSearch params: {model}: {GridSearch.best_params_} ")
            
            model.set_params(**GridSearch.best_params_)
            model.fit(X_train, y_train) #Train model with best_params from GridSearchCV

            y_pred_test = model.predict(X_test)
            
            # Evaluate best model
            
            print("Accuracy:", accuracy_score(y_test, y_pred_test))
            #print("Classification Report:\n", classification_report(y_test, y_pred_test))
            test_model_score = accuracy_score(y_test, y_pred_test)
            
            report[list(models.keys())[i]] = test_model_score                 
        
        return report

    except Exception as e:
        raise CustomException(e, sys)
    

