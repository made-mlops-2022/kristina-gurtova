from dataclasses import dataclass, field


@dataclass()
class Params:
    random_state: int = field(default=42)


@dataclass()
class TrainParams:
    model_type: str
    params: Params
