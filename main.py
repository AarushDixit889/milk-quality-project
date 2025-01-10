from src.project.pipeline.training_pipeline import training_pipeline
from src.project.utils.common import read_yaml

import os

os.environ['MLFLOW_TRACKING_URL']="https://dagshub.com/aarushdixit73androi/milk-quality-project.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="aarushdixit73androi"
os.environ['MLFLOW_TRACKING_PASSWORD']="28b53480818ced457dfa2539ff696cb9d98be199"


if __name__ == "__main__":
    config = read_yaml("config/config.yaml")
    training_pipeline(config.data_ingestion.source)