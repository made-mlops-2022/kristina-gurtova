import click

from data_manipulation import read_data, train_val_split, get_target, make_features
from data_manipulation import build_transformer
from schemes import get_params_from_config, TrainingParams


def train_pipeline(config_path):
    train_params = get_params_from_config(config_path)
    data = read_data(train_params.paths.input_data_path)
    train_data, val_data = train_val_split(data, train_params.split_params)

    train_target = get_target(train_data, train_params.feature_params)
    val_target = get_target(val_data, train_params.feature_params)
    train_data = train_data.drop(train_params.feature_params.target_col, 1)
    val_data = val_data.drop(train_params.feature_params.target_col, 1)

    transformer = build_transformer(train_params.feature_params)
    transformer.fit(train_data)
    train_features = make_features(train_data, transformer)



@click.argument("config_path")
def train(config_path: str) -> None:
    train_pipeline(config_path)


if __name__ == "__main__":
    train()
