import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## Intialize the data ingestion configuratiom
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')

## Create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        
        try:
            df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info("Train Test Split started")
            
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)
            
            logging.info("Train Test Split Done")
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info(f"Train data Saved to the path {self.ingestion_config.train_data_path}")
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info(f"Test data Saved to the path {self.ingestion_config.test_data_path}")
            
            logging.info("Ingestion of data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Error occured in Data Ingestion config')
        

