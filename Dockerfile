FROM python:3.8-slim-buster
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app project/app
COPY ./server.py project/server.py
COPY ./test_server.py project/test_server.py

WORKDIR /project
CMD ["python3", "server.py"]