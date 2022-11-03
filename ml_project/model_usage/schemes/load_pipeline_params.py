from dataclasses import dataclass
from typing import List

from .model_split_params import SplitParams
from .model_paths import Paths
from .model_train_params import TrainParams
from .model_feature_params import FeatureParams

from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingParams:
    paths: Paths
    splitting_params: SplitParams
    train_params: TrainParams
    feature_params: FeatureParams
    metrics: List[str]


TrainingParamsSchema = class_schema(TrainingParams)


def get_params_from_config(config_path: str) -> TrainingParams:
    with open(config_path, "r") as config:
        schema = TrainingParamsSchema()
        return schema.load(yaml.safe_load(config))
