import os

import pytest

from airflow.models import DagBag

os.environ["AIRFLOW_VAR_MODEL_PATH"] = "models/2022-11-27/model.pkl"
os.environ["AIRFLOW_VAR_TRANSFORMER_PATH"] = "transformers/2022-11-27/transformer.pkl"
os.environ["RAW_DATA_DIR"] = '/data/raw/{{ ds }}'
os.environ["PROCESSED_DATA_DIR"] = '/data/processed/{{ ds }}'
os.environ["MODEL_DIR"] = '/data/models/{{ ds }}'
os.environ["METRICS_DIR"] = '/data/metrics/{{ ds }}'
os.environ["TRANSFORMER_DIR"] = '/data/transformers/{{ ds }}'
os.environ["PREDICTIONS_DIR"] = '/data/predictions/{{ ds }}'

import sys

sys.path.append('../')

from airflow_ml_dags.dags.gen_data import dag as dag_download
from airflow_ml_dags.dags.train import dag as dag_train
from airflow_ml_dags.dags.predict import dag as dag_predict


@pytest.fixture()
def dagbag():
    return DagBag(dag_folder=f"{os.path.abspath('./dags')}", include_examples=False)


def test_base_generate_dag(dagbag):
    dag = dagbag.get_dag(dag_id='generate_data')
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 1


def test_base_train_dag(dagbag):
    dag = dagbag.get_dag(dag_id='train')
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 6


def test_base_predict_dag(dagbag):
    dag = dagbag.get_dag(dag_id='predict')
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 4


def assert_dag_dict_equal(source, dag):
    assert dag.task_dict.keys() == source.keys()
    for task_id, downstream_list in source.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)


def test_dag_download_order():
    assert_dag_dict_equal(
        {
            "docker-airflow-download": [],
        },
        dag_download,
    )


def test_dag_train_order():
    assert_dag_dict_equal(
        {
            "wait-target": ["docker-airflow-split"],
            "wait-data": ["docker-airflow-split"],
            "docker-airflow-split": ["docker-airflow-fit-scaler"],
            "docker-airflow-fit-scaler": ["docker-airflow-fit-model"],
            "docker-airflow-fit-model": ["docker-airflow-val"],
            "docker-airflow-val": [],
        },
        dag_train,
    )


def test_dag_predict_order():
    assert_dag_dict_equal(
        {
            "wait-data": ["docker-airflow-predict"],
            "wait-model": ["docker-airflow-predict"],
            "wait-transformer": ["docker-airflow-predict"],
            "docker-airflow-predict": [],
        },
        dag_predict,
    )
