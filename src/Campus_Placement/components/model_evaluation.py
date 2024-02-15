import os
import sys
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.Campus_Placement.utils.utils import load_object



class ModelEvaluation:
    def __init__(self):
        pass

    
    def eval_metrics(self,actual, pred):
         acc = accuracy_score(actual, pred) # Calculate Accuracy
         f1 = f1_score(actual, pred) # Calculate F1-score
         precision = precision_score(actual, pred) # Calculate Precision
         recall = recall_score(actual, pred)  # Calculate Recall
         roc_auc = roc_auc_score(actual, pred) #Calculate Roc
         return acc, f1 , precision, recall, roc_auc

    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])

            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)

        

          
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            print(tracking_url_type_store)



            with mlflow.start_run():

                predicted_qualities = model.predict(X_test)

                (acc, f1 , precision, recall, roc_auc) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("roc_auc", roc_auc)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")


                

            
        except Exception as e:
            raise e