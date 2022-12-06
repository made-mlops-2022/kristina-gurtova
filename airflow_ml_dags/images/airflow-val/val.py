import json
import os
import pickle

import click
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--transformer-dir")
@click.option("--model-dir")
@click.option("--metrics-dir")
def val(input_dir: str, transformer_dir: str, model_dir: str, metrics_dir: str):
    X_test = pd.read_csv(os.path.join(input_dir, "data_test.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "target_test.csv"))
    with open(transformer_dir + "/transformer.pkl", "rb") as transform_file:
        transformer = pickle.load(transform_file)
    X_test = transformer.transform(X_test)
    with open(model_dir + "/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    y_pred = model.predict(X_test)
    metrics_dict = {
        "accuracy_score": accuracy_score(y_pred, y_test),
        "precision_score": precision_score(y_pred, y_test),
        "recall_score": recall_score(y_pred, y_test),
        "roc_auc_score": roc_auc_score(y_pred, y_test)
    }
    os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_dir + "/metrics.json", "w+") as model_file:
        json.dump(metrics_dict, model_file)


if __name__ == "__main__":
    val()
