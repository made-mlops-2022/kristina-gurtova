import pandas as pd
from fixtures import config_file
from model_usage.data_manipulation import RandomCustomTransformer


def test_custom_transformer(config_file):
    data = pd.read_csv(config_file.paths.train_data_path)
    data_age = data["age"]
    transformed_data = RandomCustomTransformer().transform(data_age)
    assert all(transformed_data <= data_age)
    assert len(round(transformed_data / data_age, 1).unique()) == 1
