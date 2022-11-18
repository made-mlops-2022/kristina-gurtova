from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from schemes import SplitParams


def read_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    return data


def train_val_split(data: pd.DataFrame, params: SplitParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(data, test_size=params.val_size, random_state=params.random_state)
    return train_data, val_data
