import json
import os
import sys

import pathlib

cur_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(f"{cur_path}/../")
sys.path.append(f"{cur_path}/../model_usage")

from model_usage import train_pipeline, predict_pipeline

from fixtures import config_file

config_path = "configs/logreg_config.yaml"


def test_end2end_training(config_file):
    train_pipeline(config_path)
    assert os.path.exists(config_file.paths.model_path + f"/{config_file.train_params.model_type}_model.pkl")
    assert os.path.exists(config_file.paths.metric_data_path + f"/{config_file.train_params.model_type}_metrics.json")
    with open(config_file.paths.metric_data_path + f"/{config_file.train_params.model_type}_metrics.json", "r") as f:
        metrics = json.load(f)
    for metric in config_file.metrics:
        assert metric in metrics


def test_end2end_predicting():
    predict_pipeline(config_path)
