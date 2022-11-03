from dataclasses import dataclass


@dataclass()
class Paths:
    train_data_path: str
    test_data_path: str
    model_path: str
    metric_data_path: str
    predictions_path: str
