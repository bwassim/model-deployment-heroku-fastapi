import logging
from sklearn.linear_model import LogisticRegression
import starter.starter.ml.model as model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_train(process_data_train_sample):
    """This function tests the capacity of the train_model function to generate a model
    """
    X, y, encoder, lb = process_data_train_sample

    lg_model_best = model.train_model(X_train=X, y_train=y)
    assert lg_model_best is not None
    assert isinstance(lg_model_best, LogisticRegression)


def test_model_metrics(process_data_train_sample):
    logger.info("Test the model and evaluate it against the test data")

    X, y, encoder, lb = process_data_train_sample
    model_lg = model.train_model(X_train=X, y_train=y)

    predictions = model.inference(model_lg, X)
    # Calculate the scores of the test data
    precision, recall, fbeta = model.compute_model_metrics(y, predictions)
    assert isinstance(precision, float), "check precision"
    assert isinstance(recall, float), "check recall"
    assert isinstance(fbeta, float), "check fbeta"
