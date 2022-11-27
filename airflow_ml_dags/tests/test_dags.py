import datetime
import random

import pytest

from airflow.models import DagBag
from airflow.utils.state import DagRunState, TaskInstanceState
from airflow.utils.types import DagRunType
from sqlalchemy_utils.types.enriched_datetime.pendulum_date import pendulum

from airflow_ml_dags.dags.gen_data import dag as dag_download
from airflow_ml_dags.dags.train import dag as dag_train
from airflow_ml_dags.dags.predict import dag as dag_predict


DATA_INTERVAL_START = pendulum.datetime(2022, 11, 26, tz="UTC") + datetime.timedelta(seconds=random.randint(0, 60),
                                                                                     minutes=random.randint(0, 60),
                                                                                     hours=random.randint(0, 24))
DATA_INTERVAL_END = DATA_INTERVAL_START + datetime.timedelta(days=1)


@pytest.fixture()
def dagbag():
    return DagBag(dag_folder="../dags", include_examples=False)


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


def test_download_dag_success():
    dagrun = dag_download.create_dagrun(
        state=DagRunState.RUNNING,
        execution_date=DATA_INTERVAL_START,
        data_interval=(DATA_INTERVAL_START, DATA_INTERVAL_END),
        start_date=DATA_INTERVAL_END,
        run_type=DagRunType.MANUAL,
        run_id=str(random.randint(0, 100_000))
    )
    ti = dagrun.get_task_instance(task_id="docker-airflow-download")
    ti.task = dag_download.get_task(task_id="docker-airflow-download")
    ti.run(ignore_ti_state=True)
    assert ti.state == TaskInstanceState.SUCCESS