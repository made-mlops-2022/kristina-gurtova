import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
import pickle
import asyncio
from s3 import S3Client
from fastapi import status, HTTPException

from app.schemas import InputFeatures, Input

app = FastAPI()
model = None

ACCESS_KEY = "YCAJESwFRwRWLhILFnRuG5-yk"
SECRET_KEY = "YCNoDmKySTR_RZ0eHsSGwSPw76eX83hrsWP1xROC"


@app.on_event("startup")
async def download_model():
    global model
    s3 = S3Client(
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        region='ru-central1',
        s3_bucket='model-storage-bucket'
    )
    files = [f async for f in s3.list()]
    print(files)

    content = await s3.download('RandomForestClassifier_model.pkl')
    model = pickle.loads(content)


@app.get("/health")
def check_health(status_code=status.HTTP_200_OK):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Downloading model. Try later.',
        )
    else:
        return {'message': 'Service is ready to accept requests!'}


@app.post("/predict")
def predict(request: Input):
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
