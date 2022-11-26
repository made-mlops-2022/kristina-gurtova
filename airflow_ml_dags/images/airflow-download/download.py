import os

import click
from sklearn.datasets import load_breast_cancer


@click.command("download")
@click.option("--output-dir")
def download(output_dir: str):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    download()
