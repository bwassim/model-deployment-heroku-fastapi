import json

import requests
import argparse
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def prediction_below(args):
    """Test prediction results for the case where the result is >=50K"""
    body = {
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
    response = requests.post(f"{args.base_url}/predict", json=body)
    assert response.status_code == 200
    print(f"Response status code: {response.status_code}")
    # logger.info("Response status code for prediction below %s", response.status_code)
    # assert response.status_code == 200
    r = response.json()
    print(
        f"A person with {body['education']} aged {body['age']} will probably earn {r['prediction']}"
    )
    return response.json()


def prediction_above(args):
    """Test prediction results for the case where the result is <=50K"""
    body = {
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
    response = requests.post(f"{args.base_url}/predict", json=body)
    assert response.status_code == 200
    print(f"Response status code: {response.status_code}")
    # logger.info("Response status code for prediction below %s", response.status_code)
    # assert response.status_code == 200
    r = response.json()
    print(
        f"A person with {body['education']} aged {body['age']} will probably earn {r['prediction']}"
    )
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heroku app test")
    parser.add_argument(
        "--base_url",
        type=str,
        help="This is the base heroku url",
        required=False,
        default="https://census-bureau-predict-salary.herokuapp.com/",
    )
    args = parser.parse_args()
    prediction_below(args)
    prediction_above(args)
    # print(res)
