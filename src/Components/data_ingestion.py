import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.utils import get_artifacts_path


@dataclass
class DataIngestionConfig:
    artifacts_path = get_artifacts_path()
    train_data_path:str = os.path.join(artifacts_path, 'train.csv')
    test_data_path:str = os.path.join(artifacts_path, 'test.csv')
    raw_data_path:str = os.path.join(artifacts_path, 'data.csv')
    val_data_path:str = os.path.join(artifacts_path, 'validation.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion...")
            # Read raw data from the source
            df = pd.read_csv(os.path.join('data', 'Churn_Modelling.csv'))
            logging.info("Read dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # Split the dataset into training and testing sets
            logging.info("train test split initiated")
            train_val_data, test_data = train_test_split(df, test_size=0.2,random_state=42, stratify=df['Exited'])
            train_data, val_data = train_test_split(train_val_data, test_size=0.2,random_state=42, stratify=train_val_data['Exited'])

            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            val_data.to_csv(self.ingestion_config.val_data_path, index=False, header = True)

            test_data = test_data.drop(columns = 'Exited')
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header = True)



            logging.info("Train and test data saved successfully")

            logging.info('data ingestion is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path
            )

            
        except Exception as e:
            raise CustomException(e, sys)
            

