import click

from schemes import get_params_from_config
from data_manipulation import read_data
from model_testing import load_model, predict_labels, write_predictions


def predict_pipeline(config_path: str) -> None:
    test_params = get_params_from_config(config_path)
    data = read_data(test_params.paths.test_data_path)
    model = load_model(test_params.paths.model_path, test_params.train_params.model_type)
    predictions = predict_labels(model, data)
    write_predictions(predictions, test_params.paths.predictions_path)


@click.command()
@click.argument("config_path")
def predict(config_path: str) -> None:
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict()
