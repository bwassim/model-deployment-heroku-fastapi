import argparse
import json
import logging
import pandas as pd

from ml.model import compute_model_metrics, inference
from ml.data import process_data
from joblib import load
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def go(args):
    """Evaluate the trained model with slicing: consider taking a unique feature in a certain feature column
        and see how it affects the score results
    As a result of this code we will have two file scores
    - scores_all.json representing the full test dataset: census_cleaned.csv
    _ slice_scores.json representing the scores for each slice
    """
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

    logger.info("Load the the saved cleaned dataset ")
    census_test = pd.read_csv(args.input_artifact)
    lr_model = load(args.saved_model)
    encoder = load(args.saved_encoder)
    lb = load(args.saved_lb)
    # --------
    logger.info("Save scores for the full test data")

    X_test, y_test, _, _ = process_data(
        census_test,
        categorical_features=cat_features,
        label=args.target_value,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    predictions = inference(lr_model, X_test)
    # Calculate the scores of the test data
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    # --------
    logger.info(
        "Scores for test data: precision: %.2f, recall: %.2f, fbeta: %.2f",
        precision,
        recall,
        fbeta,
    )
    # --------
    with open(args.save_score, "w") as f:
        json.dump(obj={"precision": precision, "recall": recall, "fbeta": fbeta}, fp=f, indent=5)

    logger.info("Save scores for the sliced test dataset")
    scores = defaultdict(list)
    for feature in cat_features:
        for name in census_test[feature].unique():
            df_slice = census_test[census_test[feature] == name]
            X_test, y_test, _, _ = process_data(
                df_slice,
                categorical_features=cat_features,
                label=args.target_value,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = inference(lr_model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)
            scores[feature].append(
                {
                    "feature_name": name,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                }
            )

    with open(args.save_score_slices, "w") as f:
        json.dump(obj=scores, fp=f, indent=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate slices of the model")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="This is the cleaned test dataset",
        default="./starter/data/census_cleaned_test.csv",
        required=False,
    )
    parser.add_argument(
        "--saved_model",
        type=str,
        help="This is the saved model",
        default="./starter/model/census_model_classifier.joblib",
        required=False,
    )
    parser.add_argument(
        "--target_value",
        type=str,
        help="Target value for classification",
        default="salary",
        required=False,
    )
    parser.add_argument(
        "--saved_encoder",
        type=str,
        help="Saved encoder",
        default="./starter/model/encoder_census.joblib",
        required=False,
    )
    parser.add_argument(
        "--saved_lb",
        type=str,
        help="Load saved Label binarizer",
        default="./starter/model/lb_census.joblib",
        required=False,
    )
    parser.add_argument(
        "--save_score",
        type=str,
        help="Saved file for full test data scores",
        default="./starter/model/score_full.json",
        required=False,
    )
    parser.add_argument(
        "--save_score_slices",
        type=str,
        help="Save file scores of sliced test data",
        default="./starter/model/score_slices.json",
        required=False,
    )
    args = parser.parse_args()
    go(args)
