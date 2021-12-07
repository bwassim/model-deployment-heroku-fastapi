# Api to make prediction with the trained Census model



from typing import Union
import logging
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

app = FastAPI()

model = joblib.load("./starter/ml/m")


# Declare the data object
class TaggedItem(BaseModel):
    age: int


@app.get("/")
async def greetings():
    return {"Starting test our api"}
