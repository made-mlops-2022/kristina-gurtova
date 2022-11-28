import os

import airflow
from airflow import DAG
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


with DAG(
        dag_id="generate_data",
        start_date=airflow.utils.dates.days_ago(5),
        schedule_interval="@daily",
        default_args=default_args,
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command=f"--output-dir {RAW_DATA_DIR}",
        task_id="docker-airflow-download",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/questina/Desktop/kristina-gurtova/airflow_ml_dags/data/",
                      target="/data",
                      type='bind')]
    )
    download