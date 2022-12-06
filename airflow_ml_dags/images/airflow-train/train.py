import os
import pickle

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--transformer-dir")
@click.option("--model-output-dir")
def train(input_dir: str, transformer_dir: str, model_output_dir: str):
    X_train = pd.read_csv(os.path.join(input_dir, "data_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "target_train.csv"))
    with open(transformer_dir + "/transformer.pkl", "rb") as transform_file:
        transformer = pickle.load(transform_file)
    X_train = transformer.transform(X_train)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    os.makedirs(model_output_dir, exist_ok=True)
    with open(model_output_dir + "/model.pkl", "wb+") as model_file:
        pickle.dump(model, model_file)


if __name__ == "__main__":
    train()
