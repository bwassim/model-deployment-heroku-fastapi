"""
Clean Census data and return artifact
"""

import logging
import argparse
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    input_artifact = os.path.join(os.getcwd(), "starter/data", args.input_artifact)
    df_census = pd.read_csv(input_artifact)

    logger.info("Remove spaces in column names %s", input_artifact)
    df_census.columns = df_census.columns.str.replace(' ', '')

    logger.info("Remove spaces for each row of each columns")
    df_categorical = df_census.select_dtypes(['object'])
    df_census[df_categorical.columns] = df_categorical.apply(lambda x: x.str.strip())

    output_artifact = os.path.join(os.getcwd(), "starter/data", args.output_artifact)
    logger.info("Save the cleaned data %s", output_artifact)
    df_census.to_csv(output_artifact, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Census data from empty spaces and Interrogation mark")
    parser.add_argument(
        "--input_artifact",
        type=str,
        help="This is the original data census.csv",
        default="census.csv",
        required=False
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="This is the cleaned census data",
        default="census_cleaned.csv",
        required=False
    )
    args = parser.parse_args()
    go(args)
