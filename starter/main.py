# Api to make prediction with the trained Census model


from typing import Union
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

app = FastAPI()

model = joblib.load("model/census_model_classifier.joblib")
encoder = joblib.load("model/encoder_census.joblib")
lb = joblib.load("model/lb_census.joblib")


class CensusData(BaseModel):
    """
    Census prediction
    """
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example="77516")
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


class CensusResponse(BaseModel):
    """Census response data """
    prediction: str


@app.get("/", summary="Root path API", description="Census prediction API")
async def root():
    return {"message": "Hello there!"}


@app.post("/predict", summary="Predict API endpoint",
          description="predict classification result for Census data",
          response_model=CensusResponse)
async def get_prediction(request: CensusData = Body(default=None, examples= {
    "below": {
        "summary": "<=50K",
        "description": "Sample data with prediction <=50K",
        "value": {
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
    }
})):
    # convert data into a dictionary, then a pandas dataframe
    census_df = pd.DataFrame.from_dict([request.dict(by_alias=True)])

    # process data
    cat_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
    X, _, _, _ = process_data(census_df, cat_features, None, training=False, encoder=encoder, lb=lb)
    preds = inference(model, X)
    logger.info("This is prediction %.2f", preds)
    return {"prediction": "<=50K" if preds <=0.5 else ">=50K"}





