import pickle

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(model_dir: str, model_name: str = None) -> Pipeline:
    if model_name:
        model_path = model_dir + f"/{model_name}_model.pkl"
    else:
        model_path = model_dir
    with open(model_path, "rb") as model_input:
        model = pickle.load(model_input)
    return model


def predict_labels(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(data)
    return predicts


def write_predictions(predicts: np.ndarray, test_output: str) -> None:
    df = pd.DataFrame(data=predicts, columns=['target'])
    df.to_csv(test_output, index_label=False)
