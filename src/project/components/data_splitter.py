from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitterStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, test_size: float, y_column:str) -> tuple:
        """
            Abstract method to split data
        """
        pass

class TrainTestSplitter(DataSplitterStrategy):
    def split_data(self, df: pd.DataFrame, test_size: float, y_column:str) -> tuple:
        """
            Split data into train and test

            Parameters
            ----------
            df: pd.DataFrame
                DataFrame to split
            test_size: float
                Test size

            Returns
            -------
            tuple
                Train and test data
        """
        X = df.drop(y_column, axis=1)
        y = df[y_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=56)
        return (X_train, X_test, y_train, y_test)