# kristina-gurtova
MADE 2022

Train model:
```bash
cd ml_project
python3 model_usage/train.py --config-name logreg_config.yaml
```
Predict: 
```
cd ml_project
python3 model_usage/predict.py --config-name rf_config.yaml
```
Структура проекта:
```
├── __init__.py
├── configs
│   ├── logreg_config.yaml
│   └── rf_config.yaml
├── data
│   ├── predictions
│   │   └── test_labels.csv
│   └── raw
│       ├── heart_cleveland_test_data.csv
│       ├── heart_cleveland_upload.csv
│       └── heart_cleveland_upload_synthetic.csv
├── metrics
│   ├── LogisticRegression_metrics.json
│   └── RandomForestClassifier_metrics.json
├── model_usage
│   ├── __init__.py
│   ├── data_manipulation
│   │   ├── __init__.py
│   │   ├── custom_transformer.py
│   │   ├── make_data.py
│   │   └── process_features.py
│   ├── model_testing
│   │   ├── __init__.py
│   │   └── model_predict.py
│   ├── model_training
│   │   ├── __init__.py
│   │   └── model_fit_predict.py
│   ├── predict.py
│   ├── schemes
│   │   ├── __init__.py
│   │   ├── load_pipeline_params.py
│   │   ├── model_feature_params.py
│   │   ├── model_paths.py
│   │   ├── model_split_params.py
│   │   └── model_train_params.py
│   └── train.py
├── models
│   ├── LogisticRegression_model.pkl
│   └── RandomForestClassifier_model.pkl
├── notebooks
│   └── EDA.ipynb
├── report
│   ├── figures
│   │   ├── data_heatmap.png
│   │   └── data_hist.png
│   ├── gen_report.py
│   └── report.md
└── tests
    ├── fixtures.py
    ├── test_modules.py
    ├── test_pipeline.py
    ├── test_transformer.py
    └── test_utils.py
```
Архитектурные решения:
* В папке configs лежат конфиги, которые подаются на вход скрипту обучения или предсказания через флаг --config-name
* В папке data лежат данные. В data/raw необработанные csv для теста и обучения, в data/predictions предсказания
* В папке metrics лежат json, содержащие метрики обучения на валидационных данных
* В папке models лежат обученные модели в формате pickle
* В папке notebooks лежат ноутбуки с EDA
* В папке report лежит скрипт для генирирования отчета по данным
* В папке tests лежат тесты на модули, трейн и предикт и трансформер, а также скрипт для генерации синтетических данных
* В model_usage/data_manipulation лежат скрипты для работы с данными (чтение, запись, обработка)
* В model_usage/schemes лежат основные схемы для работы с конфигом, а также скрипт для обработки конфига
* В model_usage/model_training лежат скрипты для получения обученного пайплайна
* В model_usage/model_testing лежат скрипты для получения предсказаний
* В .github/workflows лежит конфиг для ci

Синтетические данные генерируются с помощью библиотеки synthia 

Файл с метриками модели именуется как {имя_модели}_metrics.json

Файл с моделями аналогично как {имя_модели}_model.pkl

Самооценка:
--------
1) В описании к пулл реквесту описаны основные "архитектурные" и тактические решения (1/1)

2) В пулл-реквесте проведена самооценка (1/1)

3) Выполнено EDA, закоммитьте ноутбук в папку с ноутбуками (1/1)

4) Использован скрипт, который сгенерит отчет (1/1)

5) Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкцию по запуску (3/3)

6) Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3/3)

7) Проект имеет модульную структуру (2/2)

8) Использованы логгеры (2/2)

9) Написаны тесты на отдельные модули и на прогон обучения и predict (3/3)

10) Для тестов генерируются синтетические данные, приближенные к реальным (2/2)

11) Обучение модели конфигурируется с помощью конфигов в json или yaml, имеется как минимум 2 корректные конфигурации (3/3)

12) Используются датаклассы для сущностей из конфига, а не голые dict (2/2)

13) Написан кастомный трансформер и протестирован его (3/3)

14) В проекте зафиксированы все зависимости (1/1)

15) Настроен CI для прогона тестов, линтера на основе github actions (3/3).

16) Использована hydra для конфигурирования (3/3)

17) Развернут локально mlflow (1/1)

18) Залогированны метрики (1/1)

Остальное не делала :(
