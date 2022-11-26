import os
import pickle

import click
import pandas as pd


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--transformer-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, transformer_dir: str, model_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    with open(transformer_dir + "/transformer.pkl", "rb") as transform_file:
        transformer = pickle.load(transform_file)
    data = transformer.transform(data)
    with open(model_dir + "/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    predictions = model.predict(data)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(predictions).to_csv(os.path.join(output_dir, 'predicts.csv'), index=False, header=None)


if __name__ == "__main__":
    predict()
