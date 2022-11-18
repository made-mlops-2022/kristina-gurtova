import os
import pickle
import sys

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

from model_usage.data_manipulation import read_data, train_val_split, get_target, build_transformer, make_features
from model_usage.model_training import train_model, create_fitted_pipeline, serialize_model
from model_usage import train_pipeline, predict_pipeline
from model_usage.model_testing import load_model, predict_labels, write_predictions

from fixtures import config_file, synthetic_data

config_path = "configs/logreg_config.yaml"


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


def test_load_model(config_file):
    model = load_model(config_file.paths.model_path, config_file.train_params.model_type)
    assert isinstance(model, Pipeline)
    check_is_fitted(model)


def test_predict_labels(config_file, synthetic_data):
    model = load_model(config_file.paths.model_path, config_file.train_params.model_type)
    predicts = predict_labels(model, synthetic_data)
    assert len(predicts) == len(synthetic_data)
    assert all(predicts >= 0.0)
    assert all(predicts <= 1.1)


def test_write_predictions(config_file, synthetic_data):
    model = load_model(config_file.paths.model_path, config_file.train_params.model_type)
    predicts = predict_labels(model, synthetic_data)
    write_predictions(predicts, config_file.paths.predictions_path)
    assert os.path.exists(config_file.paths.predictions_path)
