import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as scis
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, get_artifacts_path

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline, FunctionTransformer


from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifacts_path = get_artifacts_path()
    preprocessor_obj_path = os.path.join(artifacts_path, "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
            self.preprocessor_path = DataTransformationConfig()
    
    def TransformerObject(self):
            try:
                
                def initial_process(X):
                    import pandas as pd
                    X = X.copy()
                    # Remove unimportant columns 
                    X = X.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore')

                    # add AgeGroup    
                    X['AgeGroup'] = pd.cut(X['Age'], bins = [0, 30, 50, 100], labels= ['young', 'adults', 'senior'], right = False)   
                    
                    # add HasBalance 
                    X['HasBalance'] = (X['Balance']>0).astype(int)
                    # binary gender 
                    X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})

                    return X
                
                initial_process_tf = FunctionTransformer(initial_process, validate=False)
                
                
                balance_pipe = Pipeline([
                        ("power", PowerTransformer(method="yeo-johnson")),
                        ("scale", StandardScaler())
                        ])            
                num_cols = ['CreditScore', 'Age', 'Tenure', 'EstimatedSalary']
                cat_cols = ['Geography', 'AgeGroup']

                # Combine transformations using ColumnTransformer
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("balance", balance_pipe, ["Balance"]),
                        ("num", StandardScaler(), num_cols),     
                        ("cat", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False,
                                                drop="first"), cat_cols)
                    ], verbose_feature_names_out=False, remainder= "passthrough")
                preprocessor.set_output(transform='pandas')

                

                # Create the full pipeline including initial preprocess
                full_pipeline = Pipeline(steps=[
                                    ('initial_process', initial_process_tf), # Apply initial cleaning 
                                    ('preprocessing', preprocessor), # Apply transformations
                                    
                                ])
                
                return full_pipeline
            except Exception as e:
                  raise CustomException(e, sys)
    
    def InitiateDataTransformation(self, train_path, val_path):
            try:
                  train_df = pd.read_csv(train_path)
                  val_df = pd.read_csv(val_path)
                  
                  target_column = 'Exited'

                  logging.info("Read train and val data completed")
                  logging.info("instantiating preprocessing object")

                  PreprocessorObj = self.TransformerObject()

                  input_feature_train = train_df.drop(columns=target_column, axis=1)
                  target_feature_train = train_df[target_column]

                  input_feature_test = val_df.drop(columns=target_column, axis=1)
                  target_feature_test = val_df[target_column]
                  
                  logging.info("Applying Preprocessing object on train and test data frames Start")
                  # 1. Apply preprocessing object to the train and test data sets
                  input_feature_preprocess_train = PreprocessorObj.fit_transform(input_feature_train)
                  input_feature_preprocess_test = PreprocessorObj.transform(input_feature_test)
                  logging.info("Applying Preprocessing object on train and test data frames End")
                  
                  logging.info("SMOTE on train data initiation")
                  # 2. Applying SMOTE on training data
                  smote = SMOTE(random_state=42)
                  X_train_smote, y_train_smote = tuple(smote.fit_resample(input_feature_preprocess_train, target_feature_train))
                  logging.info("SMOTE on train data End")
                  
                  # 3. Combine the preprocessed train and test data sets with the target column
                  train_preprocess = pd.concat([pd.DataFrame(X_train_smote), pd.DataFrame(y_train_smote)], axis=1)
                  test_preprocess = pd.concat([pd.DataFrame(input_feature_preprocess_test), pd.DataFrame(target_feature_test)], axis=1)
                  
                  
                  logging.info("Saving preprocessing object")
                  save_object(
                       file_path=self.preprocessor_path.preprocessor_obj_path,
                       obj= PreprocessorObj
                  )
                  logging.info("Saved preprocessing object")
                  return (
                    train_preprocess,
                    test_preprocess,
                    self.preprocessor_path.preprocessor_obj_path
                  )

                  
            except Exception as e:
                raise CustomException(e, sys)
            

