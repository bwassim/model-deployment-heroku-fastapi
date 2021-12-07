# Script to train machine learning model.
import logging
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def go(args):
    logger.info("Load cleaned Census data")
    census_df = pd.read_csv(args.input_artifact)

    logger.info("create train and test dataset ")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(census_df, test_size=args.test_size)

    logger.info("Save the test to census_cleaned_test.csv")
    test.to_csv(args.save_test_data, index=False)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    logger.info("Process the train data with process_data function")

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # --------
    logger.info("Train a simple Logistic regression model")

    model_census = train_model(X_train, y_train)
    # --------
    logger.info("Test the model and evaluate it against the test data: %s", args.save_test_data)

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    predictions = inference(model_census, X_test)
    # Calculate the scores of the test data
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    # --------
    logger.info("Scores for test data: precision: %.2f, recall: %.2f, fbeta: %.2f", precision, recall, fbeta)

    # --------
    logger.info("Saving the trained model to %s", args.save_model)
    joblib.dump(model_census, args.save_model)

    # --------
    logger.info("Save the encoder to %s", args.save_encoder)
    joblib.dump(encoder, args.save_encoder)

    # -------
    logger.info("Save label binarizer to: %s", args.save_lb)
    joblib.dump(lb, args.save_lb)

    # -------
    logger.info("Successfully trained, tested and saved a classifier for the Census data")

    # Train and save a model.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and train a model")
    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Cleaned data for Census",
        default="./starter/data/census_cleaned.csv",
        required=False
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="Percentage of test data while the remaining is for training",
        default=0.2,
        required=False
    )
    parser.add_argument(
        "--save_test_data",
        type=str,
        help="Test data for model evaluation",
        default="./starter/data/census_cleaned_test.csv",
        required=False
    )
    parser.add_argument(
        "--target_label",
        type=str,
        help=" Target variable name",
        default="salary",
        required=False
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="./starter/model/census_model_classifier.joblib",
        required=False
    )
    parser.add_argument(
        "--save_encoder",
        type=str,
        default="./starter/model/encoder_census.joblib",
        required=False
    )
    parser.add_argument(
        "--save_lb",
        type=str,
        default="./starter/model/lb_census.joblib"
    )
    args = parser.parse_args()
    go(args)
