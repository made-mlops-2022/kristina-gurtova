import pickle
from typing import Union, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from schemes import TrainParams

SklearnClassificationModel = Union[LogisticRegression, RandomForestClassifier]


def train_model(features: pd.DataFrame, target: pd.Series, train_params: TrainParams) -> SklearnClassificationModel:
    if train_params.model_type == "LogisticRegression":
        cls = LogisticRegression(random_state=train_params.params.random_state)
    elif train_params.model_type == "RandomForestClassifier":
        cls = RandomForestClassifier(random_state=train_params.params.random_state)
    else:
        raise NotImplementedError()
    cls.fit(features, target)
    return cls


def create_fitted_pipeline(transformer: ColumnTransformer, model: SklearnClassificationModel) -> Pipeline:
    pipeline = Pipeline([("feature_transform", transformer), ("classifier", model)])
    return pipeline


def predict(predict_pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = predict_pipeline.predict(features)
    return predicts


def cnt_metrics(predictions: np.ndarray, real_values: pd.Series, metrics: List[str]) -> Dict[str, float]:
    metrics_dict = {}
    if "accuracy_score" in metrics:
        metrics_dict["accuracy"] = accuracy_score(predictions, real_values)
    if "precision_score" in metrics:
        metrics_dict["precision"] = precision_score(predictions, real_values)
    if "recall_score" in metrics:
        metrics_dict["recall"] = recall_score(predictions, real_values)
    if "roc_auc_score" in metrics:
        metrics_dict["roc_auc"] = roc_auc_score(predictions, real_values)
    return metrics_dict


def serialize_model(model: Pipeline, output_file: str) -> None:
    with open(output_file, "wb+") as output:
        pickle.dump(model, output)
