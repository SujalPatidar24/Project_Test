from source.exceptions import CustomException
from source.logger import logging
import sys

import pandas as pd
import numpy as np

from source.utiles import save_csv_data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame :
    try:
        data.drop(['return_id','return_date','pickup_delay_days','order_id','product_id','customer_id'], axis = 1, inplace = True)

        date_col = ['order_date','delivery_date','join_date']
        for col in date_col:
            data[col] = pd.to_datetime(data[col])

        obj_col = ['is_returned','order_month','order_year','is_expensive']
        for col in obj_col:
            data[col] = data[col].astype('object')

        return data

    except Exception as e:
        raise CustomException (e,sys)

def main():
    try:
        logging.info("Stage : Data Preprocessing Started")

        train_df = pd.read_csv("./data/raw/train_data.csv")
        test_df = pd.read_csv("./data/raw/test_data.csv")

        preprocess_train_df = preprocess_data(train_df)
        preprocess_test_df = preprocess_data(test_df)

        save_csv_data(preprocess_train_df, "preprocessed", "preprocessed_train.csv", "./data")
        save_csv_data(preprocess_test_df, "preprocessed", "preprocessed_test.csv", "./data")


        logging.info("Stage : Data Preprocessing Completed")

    except Exception as e:
        raise CustomException (e,sys)
    


if __name__ =="__main__":
    main()