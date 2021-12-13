import pandas as pd
import pytest
from starter.starter.ml.data import process_data


@pytest.fixture(scope="session")
def data():
    """Load original cleaned test data"""
    df_test = pd.read_csv("./data/census_cleaned_test.csv", nrows=200)
    return df_test


@pytest.fixture()
def process_data_train_sample(data):

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
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=None,
        lb=None,
    )
    return X, y, encoder, lb
