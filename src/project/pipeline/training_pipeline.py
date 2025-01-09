from src.project.components.data_ingestion import DataIngestor, ZipDataIngestor
from src.project.components.data_cleaning import MappingStrategy

import pandas as pd


class IngestorPipeline:
    def __init__(self):
        self.data_ingestion_strategy=ZipDataIngestor()

    def run(self, file_path:str)->pd.DataFrame:
        """
            Runs the data ingestion pipeline

            Parameters
            ----------
            file_path: str
                Path to the file

            Returns
            -------
            pd.DataFrame
                DataFrame containing the data

        """

        data_ingestor = DataIngestor(self.data_ingestion_strategy)
        return data_ingestor.read_data(file_path=file_path)

class DataCleaningPipeline:
    def encode(self, df: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
        """
            Encode data using label encoding

            Parameters
            ----------
            df: pd.DataFrame
                DataFrame to clean
            column: str
                Column to clean

            Returns
            -------
            pd.DataFrame
                Cleaned DataFrame
        """

        encoder_strategy = MappingStrategy()
        df = encoder_strategy.encode(df=df, column=column, mapping=mapping)
        return df



def training_pipeline(data_source:str)->pd.DataFrame:
    """
        Runs the training pipeline

        Parameters
        ----------
        data_source: str
            Path to the file

        Returns
        -------
        pd.DataFrame
            DataFrame containing the data
    """
    # Ingestion Pipeline
    ingestor_pipeline = IngestorPipeline()
    data = ingestor_pipeline.run(data_source)

    # Cleaning Pipeline
    cleaning_pipeline = DataCleaningPipeline()
    data = cleaning_pipeline.encode(df=data, column="Grade", mapping={"low":0, "medium":1, "high":2})

    print(data.head())