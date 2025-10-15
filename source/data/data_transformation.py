import pandas as pd
import numpy as np
import sys

from source.exceptions import CustomException
from source.logger import logging

from source.utiles import save_numpyArr_data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def transform_data_obj(data: pd.DataFrame):
    try:
        date_col = ['order_date','delivery_date','join_date']
        for col in date_col:
            data[col] = pd.to_datetime(data[col])

        obj_col = ['order_month','order_year','is_expensive']
        for col in obj_col:
            data[col] = data[col].astype('object')

        numerical_cols = data.select_dtypes(include=['int32', 'float32','float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        logging.info(numerical_cols)
        logging.info(categorical_cols)

        num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("labelencoder", OneHotEncoder(handle_unknown='ignore'))
            ])
        preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ],
            )
        
        return preprocessor

    except Exception as e:
        raise CustomException (e,sys)


def main():
    try:
        logging.info("Stage : Transformation Started")

        processed_train_df = pd.read_csv("./data/preprocessed/preprocessed_train.csv")
        processed_test_df = pd.read_csv("./data/preprocessed/preprocessed_test.csv")

        X_train = processed_train_df.drop("is_returned",axis=1)
        y_train = processed_train_df["is_returned"]

        X_test = processed_test_df.drop("is_returned",axis=1)
        y_test = processed_test_df["is_returned"]

        preprocessor = transform_data_obj(X_train)
        
        logging.info("Transforming data")
        

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        logging.info("Data transformed")

        # FIX: Convert the sparse matrix output to a dense NumPy array using .toarray()
        X_train_transformed = preprocessor.fit_transform(X_train).toarray()
        X_test_transformed = preprocessor.transform(X_test).toarray()
        

        y_train_array = np.array(y_train).reshape(-1, 1)
        y_test_array = np.array(y_test).reshape(-1, 1)


        transformed_train_df = np.c_[X_train_transformed,y_train_array]
        transformed_test_df = np.c_[X_test_transformed,y_test_array]

        save_numpyArr_data(transformed_train_df, "transformed", "transformed_train.npy", "./data")
        save_numpyArr_data(transformed_test_df, "transformed", "transformed_test.npy", "./data")

        logging.info("Stage : Transformation Completed")

    except Exception as e:
        raise CustomException (e,sys)
    

if __name__ == "__main__":
    main()