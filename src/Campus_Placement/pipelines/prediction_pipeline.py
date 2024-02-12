import os
import sys
import pandas as pd
from src.Campus_Placement.exception import customexception
from src.Campus_Placement.logger import logging
from src.Campus_Placement.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise customexception(e,sys)
    
    
    
class CustomData:
    def __init__(self,
                 gender:str,
                 ssc_p:float,
                 ssc_b:str,
                 hsc_p:float,
                 hsc_b:str,
                 hsc_s:str,
                 degree_p:float,
                 degree_t:str,
                 workex:str,
                 etest_p:float,
                 specialisation:str,
                 mba_p:float):
        
        self.gender=gender
        self.ssc_p=ssc_p
        self.ssc_b=ssc_b
        self.hsc_p=hsc_p
        self.hsc_b=hsc_b
        self.hsc_s=hsc_s
        self.degree_p = degree_p
        self.degree_t= degree_t
        self.workex = workex
        self.etest_p=etest_p
        self.specialisation=specialisation
        self.mba_p=mba_p
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'gender':[self.gender],
                    'ssc_p':[self.ssc_p],
                    'ssc_b':[self.ssc_b],
                    'hsc_p':[self.hsc_p],
                    'hsc_b':[self.hsc_b],
                    'hsc_s':[self.hsc_s],
                    'degree_p':[self.degree_p],
                    'degree_t':[self.degree_t],
                    'workex':[self.workex],
                    'etest_p':[self.etest_p],
                    'specialisation':[self.specialisation],
                    'mba_p':[self.mba_p],
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)