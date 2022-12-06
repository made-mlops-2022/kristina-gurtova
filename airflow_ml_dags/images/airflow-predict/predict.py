import os
import pickle

import click
import pandas as pd


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--transformer-path")
@click.option("--model-path")
@click.option("--output-dir")
def predict(input_dir: str, transformer_path: str, model_path: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    print(transformer_path)
    print(model_path)
    with open("/data/" + transformer_path, "rb") as transform_file:
        transformer = pickle.load(transform_file)
    data = transformer.transform(data)
    with open("/data/" + model_path, "rb") as model_file:
        model = pickle.load(model_file)
    predictions = model.predict(data)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(predictions).to_csv(os.path.join(output_dir, 'predicts.csv'), index=False, header=None)


if __name__ == "__main__":
    predict()
