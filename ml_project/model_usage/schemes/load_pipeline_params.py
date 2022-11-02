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
        # print(yaml.safe_load(config))
        return schema.load(yaml.safe_load(config))

'''
{'paths': {'input_data_path': 'data_manipulation/raw/heart_cleveland_upload.csv', 'output_model_path': 'models/model.pkl', 'metric_data_path': 'metrics/metrics.json'}, 'splitting_params': {'val_size': 0.2, 'random_state': 42}, 'train_params': {'model_type': 'LogisticRegression', 'params': {'random_state': 42}}, 'feature_params': {'preprocess': {'numerical_preprocessor': 'StandardScaler', 'numerical_columns': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], 'categorical_preprocessor': 'OneHotEncoder', 'categorical_columns': ['cp', 'restecg', 'slope', 'ca', 'thal']}, 'target_col': 'condition'}, 'metrics': ['accuracy_score', 'precision_score', 'recall_score', 'roc_auc_score']}

'''