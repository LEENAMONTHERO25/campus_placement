import pandas as pd
import numpy as np
import os
import sys
from src.Campus_Placement.logger import logging
from src.Campus_Placement.exception import customexception
from dataclasses import dataclass
from src.Campus_Placement.utils.utils import save_object
from src.Campus_Placement.utils.utils import evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
        
        
            models={
             "Logistic Regression": LogisticRegression(),
             "RandomForestClassifier":RandomForestClassifier(n_estimators =14,
                                                             criterion = 'entropy', random_state = 42)

        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            best_model_score= round(best_model_score*100,2)
            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
        
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)