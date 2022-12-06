import os

import airflow

from airflow import DAG
from airflow.models import Variable
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

from datetime import timedelta

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

RAW_DATA_DIR = os.environ["RAW_DATA_DIR"]
PREDICTIONS_DIR = os.environ["PREDICTIONS_DIR"]

MODEL_PATH = Variable.get("model_path")
TRANSFORMER_PATH = Variable.get("transformer_path")


with DAG(
        dag_id="predict",
        start_date=airflow.utils.dates.days_ago(5),
        schedule_interval="@daily",
        default_args=default_args,
) as dag:
    wait_data = FileSensor(
        task_id="wait-data",
        filepath="raw/{{ ds }}/data.csv",
        fs_conn_id="ADMIN_CONN",
        mode="poke",
        timeout=6000,
        poke_interval=10,
        retries=100,
    )

    wait_model = FileSensor(
        task_id="wait-model",
        filepath=MODEL_PATH,
        fs_conn_id="ADMIN_CONN",
        mode="poke",
        timeout=6000,
        poke_interval=10,
        retries=100,
    )

    wait_transformer = FileSensor(
        task_id="wait-transformer",
        filepath=TRANSFORMER_PATH,
        fs_conn_id="ADMIN_CONN",
        mode="poke",
        timeout=6000,
        poke_interval=10,
        retries=100,
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir={RAW_DATA_DIR} "
                f"--transformer-path={TRANSFORMER_PATH} "
                f"--model-path={MODEL_PATH} "
                f"--output-dir={PREDICTIONS_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/questina/Desktop/kristina-gurtova/airflow_ml_dags/data/",
                      target="/data",
                      type='bind')]
    )
    [wait_data, wait_model, wait_transformer] >> predict
