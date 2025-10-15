import pandas as pd
import numpy as np

from source.exceptions import CustomException
from source.logger import logging
import sys

from sklearn.linear_model import LogisticRegression
from source.utiles import save_model

def train_model(X_train, y_train):
    try:
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train,y_train)

        return lr_model

    except Exception as e:
        raise CustomException(e, sys)
    

def main():
    try:

        logging.info("Stage Model Training Started")
        train_data = np.load("./data/transformed/transformed_train.npy")

        X_train = train_data[:,:-1]
        y_train = train_data[:,-1]

        model = train_model(X_train, y_train)

        save_model(model,"lr_model.pkl","./models")

        logging.info("Stage Model Training Completed")

    except Exception as e:
        raise CustomException(e, sys)
    


if __name__ == "__main__":
    main()