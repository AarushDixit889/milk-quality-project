from abc import ABC, abstractmethod
import pandas as pd
import zipfile
import os
class DataIngestionStrategy(ABC):
    @abstractmethod
    def read_data(self, file_path:str)->pd.DataFrame:
        """
            Abstract method to read data
        """   
        pass


class ZipDataIngestor(DataIngestionStrategy):
    def read_data(self, file_path:str)->pd.DataFrame:
        """
            Reads data from a zip file

            Parameters
            ----------
            file_path: str
                Path to the zip file
            
            Returns
            -------
            pd.DataFrame
                DataFrame containing the data
        """


        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall("extracted_data")

        csvs=os.listdir("extracted_data")

        if len(csvs)==0:
            raise Exception("No csv files found in the zip file")
        

        if len(csvs)>1:
            raise Exception("More than one csv file found in the zip file")
        

        return pd.read_csv(f"extracted_data/{csvs[0]}")

class DataIngestor:
    def __init__(self, data_ingestion_strategy:DataIngestionStrategy):
        self.data_ingestion_strategy=data_ingestion_strategy

    def read_data(self, file_path:str)->pd.DataFrame:
        """
            Reads data from a file

            Parameters
            ----------
            file_path: str
                Path to the file
            
            Returns
            -------
            pd.DataFrame
                DataFrame containing the data
        """
        
        return self.data_ingestion_strategy.read_data(file_path)