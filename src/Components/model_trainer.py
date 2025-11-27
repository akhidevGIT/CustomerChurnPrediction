from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from dataclasses import dataclass
import os
import sys
from src.logger import logging
from src.utils import evaluate_model, save_object, get_artifacts_path
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    artifacts_path = get_artifacts_path()
    trained_model_path = os.path.join(artifacts_path, "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train, test):
        try:
            logging.info("Splitting input and target features from train and test arrays")
            X_train, X_test, y_train, y_test = (
                train.drop(columns = 'Exited'),
                test.drop(columns = 'Exited'),
                train['Exited'],
                test['Exited']
                )
            models = {
                "Logistic Regression":LogisticRegression(),
                "Random Forest":RandomForestClassifier(),
                #"Support Vector Machines":SVC(),
                "XGBoost Classifier": XGBClassifier(eval_metric="logloss")
                }
            params={
                "logistic_regression":{
                    #  "C": [0.1, 1, 10],
                    # "class_weight": [None, "balanced"],
                    # "solver": ["lbfgs"],
                    # "max_iter": [1000]
                },
                "Random Forest":{
                    "n_estimators": [100, 300],
                    # "max_depth": [None, 5, 10, 20],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 4],
                    # "max_features": ["sqrt", "log2"],
                    # "class_weight": [None, "balanced", "balanced_subsample"]
                    },
                # "svc":{
                #     'C': [0.1, 1, 10],
                #     'kernel': ['poly', 'rbf', 'sigmoid'],
                #     'gamma': ['scale', 'auto']

                #     },
                "xgboost":{
                    "n_estimators": [100, 300],
                    "learning_rate": [0.01, 0.05]
                    # "n_estimators": [100, 300, 500],
                    # "learning_rate": [0.01, 0.05, 0.1],
                    # "max_depth": [3, 5, 7],
                    # "subsample": [0.8, 1.0],
                    # "colsample_bytree": [0.8, 1.0],
                    # "gamma": [0, 1],
                    # "min_child_weight": [1, 3, 5],
                    # "scale_pos_weight": [1, 2, 5]   # for handling imbalance
                    }
                }
            logging.info("Model Evaluation Started")
            model_report:dict = evaluate_model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models = models,
                params = params
                )
            logging.info("Model Evaluation Completed")
            logging.info("Best Model Selected")

             #To get best model score from dict
            best_model_score = max(model_report.values())
        
            #To get best name from dict
            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found", sys)
            
            logging.info("Best model found")

            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj = best_model
            )

            print(f"best_model_score:{best_model_score}")
            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)

