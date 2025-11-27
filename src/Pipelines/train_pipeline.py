from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation 
from src.Components.model_trainer import ModelTrainer   
from src.logger import logging
from src.exception import CustomException

import sys

def run_train_pipeline():
    try: 
        logging.info("Starting train pipeline ...")
        print("Data Ingestion")   
        data_ingest = DataIngestion()
        train_data_path, val_data_path  = data_ingest.initiate_data_ingestion()

        print("Data Transformation")   
        data_trans =  DataTransformation()
        train_preprocess, test_preprocess, _ = data_trans.InitiateDataTransformation(train_data_path, val_data_path)
        
        print("Model training")
        model = ModelTrainer()
        best_score = model.initiate_model_training(train_preprocess, test_preprocess)
        logging.info("Training complete...")
    except Exception as e:
        raise CustomException(e, sys)

    return best_score


if __name__ == '__main__':
    training = run_train_pipeline()
    



    

