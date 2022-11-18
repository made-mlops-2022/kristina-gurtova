from dataclasses import dataclass, field
from typing import List, Optional

from .model_split_params import SplitParams
from .model_paths import Paths
from .model_train_params import TrainParams
from .model_feature_params import FeatureParams

from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class TrainingParams:
    paths: Paths
    splitting_params: SplitParams
    train_params: TrainParams
    feature_params: FeatureParams
    metrics: List[str]
    use_mlflow: Optional[bool] = field(default=False)


TrainingParamsSchema = class_schema(TrainingParams)


def get_params_from_config(config: DictConfig) -> TrainingParams:
    schema = TrainingParamsSchema()
    return schema.load(config)
