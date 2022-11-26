import os
import pickle

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    X_train = pd.read_csv(os.path.join(input_dir, "data_train.csv"))
    scaler = StandardScaler()
    scaler.fit(X_train)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + "/transformer.pkl", "wb+") as output:
        pickle.dump(scaler, output)


if __name__ == "__main__":
    preprocess()
