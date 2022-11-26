from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello!"}


def test_correct_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data='{"data": [{"age": 69, "sex": 1, "cp": 0, "trestbps": 160, '
                 '"chol": 234, "fbs": 1, "restecg": 2, "thalach": 131,'
                 '"exang": 0, "oldpeak": 0.1, "slope": 1, "ca": 1,'
                 '"thal": 0},'
                 '{"age": 35, "sex": 1, "cp": 3, "trestbps": 126,'
                 '"chol": 282, "fbs": 0, "restecg": 2, "thalach": 156,'
                 '"exang": 1, "oldpeak": 0, "slope": 0, "ca": 0,'
                 '"thal": 2}]}'
        )
        print(response.json())
        assert response.status_code == 200
        assert "target" in response.json()
        assert len(response.json()["target"]) == 2


def test_empty_input_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data='{"data": []}'
        )
        assert response.status_code == 400


def test_incorrect_input_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data='{"data": [{"sex": 1, "cp": 0, "trestbps": 160, '
                 '"chol": 234, "fbs": 1, "restecg": 2, "thalach": 131,'
                 '"exang": 0, "oldpeak": 0.1, "slope": 1, "ca": 1,'
                 '"thal": 0},'
                 '{"age": 35, "sex": 1, "cp": 3, "trestbps": 126,'
                 '"chol": 282, "fbs": 0, "restecg": 2, "thalach": 156,'
                 '"exang": 1, "oldpeak": 0, "slope": 0, "ca": 0,'
                 '"thal": 2}]}'
        )
        assert response.status_code == 400
