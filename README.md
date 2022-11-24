# kristina-gurtova
MADE 2022

Run app:
```bash
python3 server.py
```

Health example:
```bash
python3 client.py http://localhost:8080/health
```

Predict example:
```bash
python3 client.py http://localhost:8080/predict -t POST -d '{"data": [{"age":50, "sex":0, "cp":1, "trestbps":100, "chol":101, "fbs":0, "restecg":0, "thalach":100, "exang":0, "oldpeak":100.10, "slope":0, "ca":0, "thal":0}]}'
```

Test:
```bash
pytest test_server.py
```

Locally build and run docker:
```bash
docker build -t heart_cleveland_server .
docker run --env-file ./.env -p 8080:8080 heart_cleveland_server
```

Build and run docker from docker hub:
```bash
docker pull questina/heart_cleveland_server:0.0.2
docker run --env-file ./.env -p 8080:8080 questina/heart_cleveland_server:0.0.2
```

Оптимизация docker image:
1. Использую легковесный образ питона python:3.8-slim-buster (размер 567.47 MB)
2. Использую --no-cache-dir при загрузке пакетов (размер 477.71 MB)
3. Использую последние версии всех использующихся библиотек питона (размер 477.75 MB)
4. Использую более раннюю версию pandas (1.4.4) (размер 475.23 MB)

Основная часть:

1) Оберните inference вашей модели в rest сервис на FastAPI, должен быть endpoint /predict (3/3)

2) Напишите endpoint /health, который должен возращать 200, если ваша модель готова к работе (такой чек особенно актуален, если делаете доп задание про скачивание из хранилища) (1/1)

3) Напишите unit тест для /predict (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/) (3/3)

4) Напишите скрипт, который будет делать запросы к вашему сервису (2/2)

5) Напишите Dockerfile, соберите на его основе образ и запустите локально контейнер (docker build, docker run). Внутри контейнера должен запускаться сервис, написанный в предущем пункте. Закоммитьте его, напишите в README.md корректную команду сборки (4/4)

6) Опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (2/2)

7) Опишите в README.md корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель. Убедитесь, что вы можете протыкать его скриптом из пункта 3 (1/1)

8) Проведите самооценку - распишите в реквесте какие пункты выполнили и на сколько баллов, укажите общую сумму баллов (0/1)

9) Ваш сервис скачивает модель из S3 или любого другого хранилища при старте, путь для скачивания передается через переменные окружения (2/2)
10) Оптимизируйте размер docker image. Опишите в README.md, что вы предприняли для сокращения размера и каких результатов удалось добиться. Должно получиться мини исследование -- я сделал тото и получился такой-то результат (2/2)
11) Сделайте валидацию входных данных https://pydantic-docs.helpmanual.io/usage/validators/ . Например, порядок колонок не совпадает с трейном, типы, допустимые максимальные и минимальные значения. Проявите фантазию, это доп. баллы, проверка не должна быть тривиальной. Вы можете сохранить вместе с моделью доп информацию о структуре входных данных, если это нужно (2/2). возращайте 400, в случае, если валидация не пройдена