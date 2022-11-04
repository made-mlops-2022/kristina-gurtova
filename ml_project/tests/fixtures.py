import sys
import pathlib
from pathlib import Path
import pandas as pd
import pytest

cur_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(f"{cur_path}/../")
sys.path.append(f"{cur_path}/../model_usage")

from model_usage.schemes import get_params_from_config
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
    data[data[config_file.feature_params.target_col] >= 0.5] = 1
    data[data[config_file.feature_params.target_col] < 0.5] = 0
    return data