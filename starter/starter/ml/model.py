import logging
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# Optional: implement hyper parameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 150],
                  'max_iter': [400, 1000]}
    logger.info("Train the Census model with the following hyperparameters: %s", param_grid)
    lr = LogisticRegression()
    grid_search = GridSearchCV(lr, param_grid, cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)
    logger.info("The best parameters: %s", grid_search.best_params_)
    logger.info("The best score: %s", grid_search.best_score_)

    return grid_search.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    return model.predict(X)
