import json
import os
import pickle
import sys

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier

import pathlib

cur_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(f"{cur_path}/../")
sys.path.append(f"{cur_path}/../model_usage")

import pandas as pd
import pytest
from pathlib import Path

from model_usage.schemes import get_params_from_config
from model_usage.data_manipulation import read_data, train_val_split, get_target, build_transformer, make_features
from model_usage.model_training import train_model, create_fitted_pipeline, serialize_model
from model_usage import train_pipeline, predict
from test_utils import generate_synthetic_data

config_path = "configs/logreg_config.yaml"


@pytest.fixture()
def config_file():
    model_config = get_params_from_config(config_path)
    return model_config


@pytest.fixture
def synthetic_data(config_file):
    gen_data_file = Path(config_file.paths.train_data_path.replace(".csv", "_synthetic.csv"))
    if not gen_data_file.exists():
        generate_synthetic_data(config_file.paths.train_data_path)
    data = pd.read_csv(config_file.paths.train_data_path.replace(".csv", "_synthetic.csv"))
    lab = preprocessing.LabelEncoder()
    data_transformed = lab.fit_transform(data[config_file.feature_params.target_col])
    data[config_file.feature_params.target_col] = data_transformed
    return data


def test_read_data(config_file, synthetic_data):
    data = read_data(config_file.paths.train_data_path.replace(".csv", "_synthetic.csv"))
    assert isinstance(data, pd.DataFrame)
    assert len(data) == len(synthetic_data)


def test_train_val_split(config_file, synthetic_data):
    data_train, data_val = train_val_split(synthetic_data, config_file.splitting_params)
    assert config_file.splitting_params.val_size - 0.05 <= len(data_val) / len(
        synthetic_data) <= config_file.splitting_params.val_size + 0.05


def test_get_target(config_file, synthetic_data):
    data_target = get_target(synthetic_data, config_file.feature_params)
    assert len(data_target) == len(synthetic_data[config_file.feature_params.target_col])
    comp_data = data_target == synthetic_data[config_file.feature_params.target_col]
    assert all(comp_data)


def test_make_features(config_file, synthetic_data):
    data = synthetic_data.drop(columns=[config_file.feature_params.target_col], axis=1)
    transformer = build_transformer(config_file.feature_params)
    transformer.fit(data)
    features = make_features(data, transformer)
    assert len(features) == len(data)


def test_train_model(config_file, synthetic_data):
    features = synthetic_data.drop(columns=[config_file.feature_params.target_col], axis=1)
    target = synthetic_data[config_file.feature_params.target_col]
    model_cls = train_model(features, target, config_file.train_params)
    if config_file.train_params.model_type == "LogisticRegression":
        assert isinstance(model_cls, LogisticRegression)
    elif config_file.train_params.model_type == "RandomForestClassifier":
        assert isinstance(model_cls, RandomForestClassifier)
    check_is_fitted(model_cls)


def test_serialize_model(config_file, synthetic_data):
    transformer = build_transformer(config_file.feature_params)
    transformer.fit(synthetic_data)
    features = make_features(synthetic_data, transformer)
    target = synthetic_data[config_file.feature_params.target_col]
    cls = train_model(features, target, config_file.train_params)
    fitted_pipeline = create_fitted_pipeline(transformer, cls)
    serialize_model(fitted_pipeline, config_file.paths.model_path, config_file.train_params.model_type)
    assert os.path.exists(config_file.paths.model_path + f"/{config_file.train_params.model_type}_model.pkl")
    with open(config_file.paths.model_path + f"/{config_file.train_params.model_type}_model.pkl", "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, Pipeline)


def test_end2end_training(config_file, synthetic_data):
    train_pipeline(config_path)
    assert os.path.exists(config_file.paths.model_path + f"/{config_file.train_params.model_type}_model.pkl")
    assert os.path.exists(config_file.paths.metric_data_path + f"/{config_file.train_params.model_type}_metrics.json")
    with open(config_file.paths.metric_data_path + f"/{config_file.train_params.model_type}_metrics.json", "r") as f:
        metrics = json.load(f)
    for metric in config_file.metrics:
        assert metric in metrics
