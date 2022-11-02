from dataclasses import dataclass


@dataclass()
class Paths:
    input_data_path: str
    output_model_path: str
    metric_data_path: str
