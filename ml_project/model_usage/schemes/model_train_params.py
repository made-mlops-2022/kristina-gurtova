from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class Params:
    random_state: int = field(default=42)
    solver: Optional[str] = field(default="lbfgs")
    n_estimators: Optional[int] = field(default=100)


@dataclass()
class TrainParams:
    model_type: str
    params: Params
