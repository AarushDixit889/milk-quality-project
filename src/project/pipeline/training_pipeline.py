from src.project.components.data_ingestion import DataIngestor, ZipDataIngestor
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
    print(data)