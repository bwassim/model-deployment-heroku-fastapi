from main import CensusData
from fastapi.testclient import TestClient
from main import app


# Test fastapi root
def test_root():
    """Test root route"""
    client = TestClient(app)  # another way
    # with TestClient(app) as client:
    r = client.get("/")
    assert r.status_code == 200


def test_get():
    """Test reception from endpoint"""
    client = TestClient(app)
    response = client.get("/send")
    assert response.status_code == 200
    assert response.json()["send"] == "well received"


def test_prediction_below_50K():
    body = CensusData(
        **{
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        }
    )

    client = TestClient(app)
    response = client.post("/predict", data=body.json(by_alias=True))
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_prediction_above_50K():
    body = CensusData(
        **{
            "age": 31,
            "workclass": "Private",
            "fnlgt": 45781,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 14084,
            "capital-loss": 0,
            "hours-per-week": 50,
            "native-country": "United-States",
        }
    )

    client = TestClient(app)
    response = client.post("/predict", data=body.json(by_alias=True))
    assert response.status_code == 200
    assert response.json()["prediction"] == ">=50K"
