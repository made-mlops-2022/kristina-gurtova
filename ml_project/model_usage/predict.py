from schemes import get_params_from_config
from data_manipulation import read_data
from model_testing import load_model, predict_labels, write_predictions

import logging
import sys
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_pipeline(config_path: DictConfig) -> None:
    test_params = get_params_from_config(config_path)
    logging.info(f"Predict pipeline started with parameters {test_params}")
    data = read_data(test_params.paths.test_data_path)
    logger.info(f"Succesfully read data from {test_params.paths.train_data_path} of shape {data.shape}")
    model = load_model(test_params.paths.model_path,
                       test_params.train_params.model_type)
    logger.info(f"Succesfully read pipeline {model}")
    predictions = predict_labels(model, data)
    logger.info(f"Got predictions of shape {predictions.shape}")
    write_predictions(predictions, test_params.paths.predictions_path)
    logger.info(f"Wrote predictions to {test_params.paths.predictions_path}")


@hydra.main(version_base=None, config_path="../configs")
def predict(config_path: DictConfig) -> None:
    predict_pipeline(config_path)


if __name__ == "__main__":
    sys.argv.append('hydra.job.chdir=False')
    predict()
