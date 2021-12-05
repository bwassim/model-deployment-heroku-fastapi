import pandas as pd
import pytest

@pytest.fixture(scope="session")
def data():
    """Load original cleaned data """
    df = pd.read_csv("../data/census_cleaned.csv")
    return df

