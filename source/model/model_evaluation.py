from source.logger import logging
from source.exceptions import CustomException
import sys

import numpy as np
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from source.utiles import save_json

def evaluate_model(X_test, y_test, model):
    try:
        y_pred = model.predict(X_test)

        metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
        
        return metrics

    except Exception as e:
        raise CustomException(e,sys)
    


def main():
    try:
        logging.info("Stage Model Evaluation Started")

        test_data = np.load("./data/transformed/transformed_test.npy")
        model = joblib.load("./models/lr_model.pkl")

        X_test = test_data[:,:-1]
        y_test = test_data[:,-1]

        results = evaluate_model(X_test, y_test, model)
        save_json(results,"evaluation_metrics","scores.json","./results")

        logging.info("Stage Model Evaluation Completed")


    except Exception as e:
        raise CustomException(e,sys)
    


if __name__ == "__main__":
    main()