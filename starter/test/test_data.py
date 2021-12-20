
from starter.starter.ml.data import process_data

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer


def test_class_names(data):
    """Check that only the known classes are present
    """
    required_columns = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    ]

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns))


def test_process_data(data):
"""Test the shape of the test data and make sure that the encoders are correctly retrieved.
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
    label = "salary"
    X, y, encoder, lb = process_data(data, cat_features, label, training=True)
    assert data.shape == (200, 15)
    assert y.shape == (200,)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
