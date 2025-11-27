import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, get_artifacts_path

import os

ARTIFACTS_DIR = get_artifacts_path()
MODEL_PATH = os.path.join(ARTIFACTS_DIR,"model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR,'preprocessor.pkl')


class ChurnPredictor:
    def __init__(self):
        self.model = load_object(file_path=MODEL_PATH)
        self.preprocessor = load_object(file_path=PREPROCESSOR_PATH)

    def model_name(self):
        return type(self.model).__name__

    def preprocess(self, features):
        try:         
            X = self.preprocessor.transform(features)
            return X
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_proba(self, features):
        try:
            X = self.preprocess(features)
            proba = self.model.predict_proba(X)[:, 1]
            return proba
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_label(self, features, threshold=0.5):
        try:
            proba = self.predict_proba(features)
            return (proba >= threshold).astype(int)
        except Exception as e:
            raise CustomException(e, sys)
   
        








