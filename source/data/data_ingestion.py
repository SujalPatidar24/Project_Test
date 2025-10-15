import pandas as pd
import sys
from source.exceptions import CustomException
from source.logger import logging


import os
from azure.storage.blob import BlobServiceClient
import pyarrow.parquet as pq
import io

from sklearn.model_selection import train_test_split
from source.utiles import save_csv_data


def ingest_data(CONTAINER_NAME: str,BLOB_NAME: str,CONNECTION_STRING: str) -> pd.DataFrame:
    try:
        # -----------------------------
        # Connect to Azure Blob Storage
        # -----------------------------
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)

        # -----------------------------
        # Download parquet blob
        # -----------------------------
        data = blob_client.download_blob().readall()

        LOCAL_FILE_PATH = os.path.join("data", BLOB_NAME.split("/")[-1])  # saves as data/orders.parquet
        os.makedirs(os.path.dirname(LOCAL_FILE_PATH), exist_ok=True)

        # Download blob
        with open(LOCAL_FILE_PATH, "wb") as file:
            file.write(blob_client.download_blob().readall())

        print(f"✅ Blob downloaded successfully and saved at: {LOCAL_FILE_PATH}")

        # Convert Parquet bytes → Pandas DataFrame
        table = pq.read_table(io.BytesIO(data))
        df = table.to_pandas()

        print("Shape:", df.shape)
        print(df.head(2))

        return df
    
    except Exception as e:
        raise CustomException (e,sys)



def split_data(X: pd.DataFrame, y: pd.DataFrame,test_size: int) -> pd.DataFrame:

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42, stratify=y)

        train_data = pd.concat([X_train.reset_index(drop=True), pd.Series(y_train, name='is_returned').reset_index(drop=True)], axis=1)
        test_data = pd.concat([X_test.reset_index(drop=True), pd.Series(y_test, name='is_returned').reset_index(drop=True)], axis=1)
       
        return train_data, test_data
    
    except Exception as e:
        raise CustomException (e, sys)


 


def main():
    try:
        logging.info("Stage : Data Ingestion Started")
        CONNECTION_STRING = os.getenv('CONNECTION_STRING')
        CONTAINER_NAME = "gold"
        BLOB_NAME = "final_df/part-00000-1ac71545-82f8-44e5-a0d8-6a0299dccc3f.c000.snappy.parquet"   # example parquet file

        df = ingest_data(CONTAINER_NAME,BLOB_NAME,CONNECTION_STRING)
        test_size = 0.2

        save_csv_data(df,"raw","raw_data.csv","./data")

        X = df.drop("is_returned",axis=1)
        y = df["is_returned"]
        
        train_data ,test_data = split_data(X, y ,test_size)
        
        

        save_csv_data(train_data,"raw","train_data.csv","./data")
        save_csv_data(test_data,"raw","test_data.csv","./data")

        logging.info("Stage : Data Ingestion Completed")

    except Exception as e:
        raise CustomException(e, sys)
    


if __name__ == "__main__":
    main()