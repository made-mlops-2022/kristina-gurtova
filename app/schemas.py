from typing import List

from pydantic import BaseModel, Field


class InputFeatures(BaseModel):
    age: int = Field(ge=0, le=100)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int
    chol: int
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=2)
    thal: int = Field(ge=0, le=2)


class Input(BaseModel):
    data: List[InputFeatures]
