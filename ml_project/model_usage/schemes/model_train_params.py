from dataclasses import dataclass, field


@dataclass()
class TrainParams:
    model_type: str
    random_state: int = field(default=42)
