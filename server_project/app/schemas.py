from typing import List

from pydantic import BaseModel, Field, validator, root_validator


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

    @root_validator(pre=True)
    def check_if_has_condition(cls, vals):
        print(vals)
        if "condition" in vals:
            vals.pop("condition")
        return vals

    @root_validator(pre=True)
    def only_legit_fields(cls, vals):
        if len(vals) != 13:
            raise ValueError("Incorrect input parameters\n"
                             "Use fields ['age', 'sex', 'cp',"
                             "'trestbps', 'chol', 'fbs',"
                             "'restecg', 'thalach', 'exang',"
                             "'oldpeak', 'slope', 'ca', 'thal']")
        return vals

    @validator("chol")
    def check_chol_value(cls, v):
        if 100 < v < 400:
            return v
        else:
            raise ValueError('this cholesterol is impossible')


class Input(BaseModel):
    data: List[InputFeatures]

    @validator("data")
    def data_length(cls, v):
        if len(v) == 0:
            raise ValueError('data must contain more than zero elements')
        return v

    @root_validator(pre=True)
    def check_target_contains(cls, vals):
        if "target" in vals:
            raise ValueError('you should not put target')
        return vals

    @root_validator(pre=True)
    def check_same_data(cls, vals):
        check_set = set()
        for l in vals["data"]:
            sample = l.values()
            if tuple(sample) in check_set:
                raise ValueError("You gave the same data")
            check_set.add(tuple(sample))
        return vals