from datetime import timedelta

import airflow

from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

RAW_DATA_DIR = "/data/raw/{{ ds }}"
PROCESSED_DATA_DIR = "/data/processed/{{ ds }}"
MODEL_DIR = "/data/models/{{ ds }}"
METRICS_DIR = "/data/metrics/{{ ds }}"
TRANSFORMER_DIR = "/data/transformers/{{ ds }}"

with DAG(
        dag_id="train",
        start_date=airflow.utils.dates.days_ago(5),
        schedule_interval="@weekly",
        default_args=default_args,
) as dag:

    wait_target = FileSensor(
        task_id="wait-target",
        filepath="raw/{{ ds }}/target.csv",
        fs_conn_id="ADMIN_CONN",
        mode="poke",
        timeout=6000,
        poke_interval=10,
        retries=100
    )

    wait_data = FileSensor(
        task_id="wait-data",
        filepath="raw/{{ ds }}/data.csv",
        fs_conn_id="ADMIN_CONN",
        mode="poke",
        timeout=6000,
        poke_interval=10,
        retries=100
    )

    split_data = DockerOperator(
        image="airflow-split",
        command=f"--input-dir={RAW_DATA_DIR} --output-dir={PROCESSED_DATA_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/Users/questina/Desktop/kristina-gurtova/airflow_ml_dags/data/",
                      target="/data",
                      type='bind')]
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir={PROCESSED_DATA_DIR} --output-dir={TRANSFORMER_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-fit-scaler",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/Users/questina/Desktop/kristina-gurtova/airflow_ml_dags/data/",
                      target="/data",
                      type='bind')]
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--input-dir={PROCESSED_DATA_DIR} "
                f"--transformer-dir={TRANSFORMER_DIR} "
                f"--model-output-dir={MODEL_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-fit-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/Users/questina/Desktop/kristina-gurtova/airflow_ml_dags/data/",
                      target="/data",
                      type='bind')]
    )

    validate = DockerOperator(
        image="airflow-val",
        command=f"--input-dir={PROCESSED_DATA_DIR} "
                f"--transformer-dir={TRANSFORMER_DIR} "
                f"--model-dir={MODEL_DIR} "
                f"--metrics-dir={METRICS_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-val",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/Users/questina/Desktop/kristina-gurtova/airflow_ml_dags/data/",
                      target="/data",
                      type='bind')]
    )

    [wait_target, wait_data] >> split_data >> preprocess >> train >> validate