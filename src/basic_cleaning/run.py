#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import os
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Downloading the artifact from previous step
    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    # Limiting the range of price
    logger.info("Limiting the range of price")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # convert the column 'last_review to datetime object
    logger.info("Covert 'last_review' to datetime object")
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Saving and log the artifact
    filename = args.output_artifact
    df.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Fully-qualified name for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price", type=int, help="Lower bound for the price", required=True
    )

    parser.add_argument(
        "--max_price", type=int, help="Upper bound for the price", required=True
    )

    args = parser.parse_args()

    go(args)
