import os.path

import click
import pandas as pd

from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "data_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "data_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "target_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "target_test.csv"), index=False)


if __name__ == "__main__":
    split()