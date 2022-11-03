import click

from schemes import get_params_from_config


def predict_pipeline(config_path: str):
    test_params = get_params_from_config(config_path)



@click.command()
@click.argument("config_path")
def predict(config_path: str) -> None:
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict()
