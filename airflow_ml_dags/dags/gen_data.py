from datetime import timedelta

import airflow
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "admin",
    "email": ["admin@example.org"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

RAW_DATA_DIR = "/data/raw/{{ ds }}"

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