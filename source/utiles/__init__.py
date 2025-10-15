import pandas as pd
import numpy as np
import os
import sys
from source.exceptions import CustomException
import joblib
import json


def save_csv_data(data: pd.DataFrame, folder_name, file_name, path) -> None:
    try:
        dir_path = os.path.join(path,folder_name)
        os.makedirs(dir_path, exist_ok=True)
        data.to_csv (os.path.join(dir_path,file_name), index=False)

    except Exception as e:
        raise CustomException (e,sys)
    

def save_numpyArr_data(data: pd.DataFrame, folder_name, file_name, path) -> None:
    try:
        dir_path = os.path.join(path,folder_name)
        os.makedirs(dir_path, exist_ok=True)
        np.save(os.path.join(dir_path,file_name), data)

    except Exception as e:
        raise CustomException (e,sys)
    


def save_model(model,file_name, path):
    try:
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path,file_name)
        joblib.dump(model,model_path)
    
    except Exception as e:
        raise CustomException (e,sys)
    

def save_json(data, folder_name, file_name, path ):
    try:
        dir_path = os.path.join(path,folder_name)
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path,file_name), 'w') as f:
            json.dump(data, f, indent=4)

    except Exception as e:
        raise CustomException (e,sys)