from src.project.pipeline.training_pipeline import training_pipeline
from src.project.pipeline.deployment_pipeline import deployment_pipeline
from src.project.utils.common import read_yaml

if __name__ == "__main__":
    if input("Training Pipeline? (y/n)").lower() == "y":
        config = read_yaml("config/config.yaml")
        training_pipeline(config.data_ingestion.source)
    else:
        deployment_pipeline()