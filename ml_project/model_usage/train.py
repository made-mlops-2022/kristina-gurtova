import json

import click

from data_manipulation import read_data, train_val_split, get_target, make_features
from data_manipulation import build_transformer
from model_training import train_model, create_fitted_pipeline, predict, cnt_metrics, serialize_model
from schemes import get_params_from_config


def train_pipeline(config_path: str) -> None:
    train_params = get_params_from_config(config_path)
    data = read_data(train_params.paths.train_data_path)
    train_data, val_data = train_val_split(data, train_params.splitting_params)

    train_target = get_target(train_data, train_params.feature_params)
    val_target = get_target(val_data, train_params.feature_params)
    train_data = train_data.drop(columns=train_params.feature_params.target_col, axis=1)
    val_data = val_data.drop(columns=train_params.feature_params.target_col, axis=1)

    transformer = build_transformer(train_params.feature_params)
    transformer.fit(train_data)
    train_features = make_features(train_data, transformer)

    cls = train_model(train_features, train_target, train_params.train_params)
    fitted_pipeline = create_fitted_pipeline(transformer, cls)
    predicts = predict(fitted_pipeline, val_data)
    res_metrics = cnt_metrics(predicts, val_target, train_params.metrics)

    metric_path = train_params.paths.metric_data_path + f"/{train_params.train_params.model_type}_metrics.json"

    with open(metric_path, "w+") as output_metrics_file:
        json.dump(res_metrics, output_metrics_file)

    serialize_model(fitted_pipeline, train_params.paths.model_path, train_params.train_params.model_type)


@click.command()
@click.argument("config_path")
def train(config_path: str) -> None:
    train_pipeline(config_path)


if __name__ == "__main__":
    train()
