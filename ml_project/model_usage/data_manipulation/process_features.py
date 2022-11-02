import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder

from ..schemes import FeatureParams


def get_target(data: pd.DataFrame, feature_params: FeatureParams) -> pd.Series:
    target = data[feature_params.target_col]
    return target


def build_num_transform_pipeline(num_preprocessor: str) -> Pipeline:
    if num_preprocessor == "StandardScaler":
        pipeline = Pipeline([("standard_scaler", StandardScaler())])
    elif num_preprocessor == "MinMaxScaler":
        pipeline = Pipeline([("min_max_scaler", MinMaxScaler())])
    else:
        raise NotImplemented
    return pipeline


def build_categorical_transform_pipeline(categorical_preprocessor: str) -> Pipeline:
    if categorical_preprocessor == "OneHotEncoder":
        pipeline = Pipeline([("one_hot_enc", OneHotEncoder())])
    elif categorical_preprocessor == "OrdinalEncoder":
        pipeline = Pipeline([("ordinal_enc", OrdinalEncoder())])
    else:
        raise NotImplemented
    return pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                build_num_transform_pipeline(params.preprocess.numerical_preprocessor),
                params.preprocess.numerical_columns
            ),
            (
                "categorical_pipeline",
                build_categorical_transform_pipeline(params.preprocess.categorical_preprocessor),
                params.preprocess.categorical_columns
            )
        ]
    )
    return transformer


def make_features(data: pd.DataFrame, transformer: ColumnTransformer) -> pd.DataFrame:
    transformed_data = transformer.transform(data)
    return transformed_data
