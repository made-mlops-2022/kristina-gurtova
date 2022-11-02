from dataclasses import dataclass
from typing import List


@dataclass()
class PreprocessParams:
    numerical_preprocessor: str
    numerical_columns: List[str]
    categorical_preprocessor: str
    categorical_columns: List[str]


@dataclass()
class FeatureParams:
    target_col: str
    preprocess: PreprocessParams
