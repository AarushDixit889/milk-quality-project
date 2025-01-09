from abc import ABC, abstractmethod
import pandas as pd

class DataCleaningStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
        """
            Abstract method to clean data
        """
        pass

class MappingStrategy(DataCleaningStrategy):
    def encode(self, df: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
        """
            Encode data using label encoding

            Parameters
            ----------
            df: pd.DataFrame
                DataFrame to clean
            column: str
                Column to clean
            mapping: dict
                Mapping dictionary

            Returns
            -------
            pd.DataFrame
                Cleaned DataFrame
        """
        df[column]=df[column].map(mapping)
        return df