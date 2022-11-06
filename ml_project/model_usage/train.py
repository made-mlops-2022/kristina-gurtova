import json
import logging
import sys
from typing import Dict, Tuple

import hydra
from omegaconf import DictConfig
import mlflow
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from data_manipulation import (
    read_data, train_val_split,
    get_target, make_features
)
from data_manipulation import build_transformer
from model_training import (
    train_model, create_fitted_pipeline,
    predict, cnt_metrics, serialize_model
)
from schemes import get_params_from_config


def train_pipeline(config_path: DictConfig) -> Tuple[Dict[str, float], Pipeline]:
    train_params = get_params_from_config(config_path)
    logging.info(f"Train pipeline started with parameters {train_params}")
    data = read_data(train_params.paths.train_data_path)
    logger.info(f"Succesfully read data from {train_params.paths.train_data_path} of shape {data.shape}")
    train_data, val_data = train_val_split(data, train_params.splitting_params)

    train_target = get_target(train_data, train_params.feature_params)
    val_target = get_target(val_data, train_params.feature_params)
    train_data = train_data.drop(columns=train_params.feature_params.target_col, axis=1)
    val_data = val_data.drop(columns=train_params.feature_params.target_col, axis=1)
    logger.info(f"Splitted data into train and val of shapes {train_data.shape} and {val_data.shape}")

    transformer = build_transformer(train_params.feature_params)
    transformer.fit(train_data)
    train_features = make_features(train_data, transformer)
    logger.info("Preprocessed features from train data")

    cls = train_model(train_features, train_target, train_params.train_params)
    logger.info(f"Trained classifier {cls}")

    fitted_pipeline = create_fitted_pipeline(transformer, cls)
    predicts = predict(fitted_pipeline, val_data)
    res_metrics = cnt_metrics(predicts, val_target, train_params.metrics)
    logger.info(f"Metrics for val data {res_metrics}")

    metric_path = train_params.paths.metric_data_path + f"/{train_params.train_params.model_type}_metrics.json"
    with open(metric_path, "w+") as output_metrics_file:
        json.dump(res_metrics, output_metrics_file)
    logger.info(f"Wrote metrics to {metric_path}")

    serialize_model(fitted_pipeline, train_params.paths.model_path, train_params.train_params.model_type)
    logger.info(f"Serialized model into {train_params.paths.model_path}")
    return res_metrics, fitted_pipeline


@hydra.main(version_base=None, config_path="../configs")
def train(config: DictConfig) -> None:
    if config.use_mlflow:
        with mlflow.start_run():
            metrics, model = train_pipeline(config)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(config.paths.model_path + f"/{config.train_params.model_type}_model.pkl")
            mlflow.sklearn.log_model(model, config.train_params.model_type)
    else:
        train_pipeline(config)


if __name__ == "__main__":
    sys.argv.append('hydra.job.chdir=False')
    train()
