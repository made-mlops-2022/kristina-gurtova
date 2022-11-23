import os

import pandas as pd
import requests
from fastapi import FastAPI
import pickle
from fastapi import status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse

from app.schemas import Input

app = FastAPI()
model = None

MODEL_PATH = os.environ.get("MODEL_PATH")
MODEL_PATH = "https://storage.yandexcloud.net/model-storage-bucket/RandomForestClassifier_model.pkl"


@app.exception_handler(RequestValidationError)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

@app.on_event("startup")
async def download_model():
    global model
    url = MODEL_PATH
    r = requests.get(url, allow_redirects=True)
    model = pickle.loads(r.content)


@app.get("/health")
async def check_health(status_code=status.HTTP_200_OK):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Downloading model. Try later.',
        )
    else:
        return {'message': 'Service is ready to accept requests!'}


@app.post("/predict")
async def predict(request: Input):
    if len(request.data) == 0:
        return {"target": []}
    X = pd.DataFrame(columns=["age", "sex", "cp", "trestbps",
                              "chol", "fbs", "restecg", "thalach",
                              "exang", "oldpeak", "slope",
                              "ca", "thal"])
    for row in request.data:
        X.loc[len(X.index)] = [row.age, row.sex, row.cp,
                               row.trestbps, row.chol, row.fbs,
                               row.restecg, row.thalach, row.exang,
                               row.oldpeak, row.slope, row.ca,
                               row.thal]
    y_pred = model.predict(X)
    return {"target": y_pred.tolist()}


@app.get("/")
async def read_main():
    return {"message": "Hello!"}
