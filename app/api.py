from fastapi import FastAPI, Response
import pickle
import asyncio
from s3 import S3Client
from fastapi import status, HTTPException

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
def check_health(response: Response, status_code=status.HTTP_200_OK):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Downloading model. Try later.',
        )
    else:
        return {'message': 'Service is ready to accept requests!'}
