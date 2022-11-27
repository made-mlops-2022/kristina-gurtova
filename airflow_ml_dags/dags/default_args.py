from datetime import timedelta

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

RAW_DATA_DIR = "/data/raw/{{ ds }}"
PROCESSED_DATA_DIR = "/data/processed/{{ ds }}"
MODEL_DIR = "/data/models/{{ ds }}"
METRICS_DIR = "/data/metrics/{{ ds }}"
TRANSFORMER_DIR = "/data/transformers/{{ ds }}"
PREDICTIONS_DIR = "/data/predictions/{{ ds }}"
