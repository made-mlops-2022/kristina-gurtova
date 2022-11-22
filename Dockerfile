FROM python:3.8-slim-buster
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./app project/app
COPY ./server.py project/server.py

WORKDIR /project
CMD ["python", "server.py"]