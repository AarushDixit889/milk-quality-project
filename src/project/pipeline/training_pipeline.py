from src.project.components.data_ingestion import DataIngestor, ZipDataIngestor

from src.project.components.data_cleaning import MappingStrategy

from src.project.components.data_splitter import TrainTestSplitter

from src.project.components.model_training import LogisticRegressionModelTrainingStrategy
from src.project.components.model_training import RandomForestModelTrainingStrategy
from src.project.components.model_training import DecisionTreeModelTrainingStrategy
from src.project.components.model_training import SVCModelTrainingStrategy

from src.project.components.model_evaluation import ClassificationEvaluationStrategy

from sklearn.base import ClassifierMixin

import pandas as pd

import mlflow


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


class DataSplitterPipeline:
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

        splitter_strategy = TrainTestSplitter()
        return splitter_strategy.split_data(df=df, test_size=test_size, y_column=y_column)


class ModelTrainingPipeline:
    def train(self, X_train, y_train, X_test, y_test)->ClassifierMixin:
        """
            Train model

            Parameters
            ----------
            X_train: pd.DataFrame
                Training data
            y_train: pd.DataFrame
                Target values
            X_test: pd.DataFrame
                Test data
            y_test: pd.DataFrame
                Test labels
                
            Returns
            -------
            ClassifierMixin
                Trained model
        """
        
        mlflow.set_experiment("Milk Quality Prediction")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # Logistic Regression
        if not mlflow.active_run():
            mlflow.start_run(run_name="Logistic Regression")
        mlflow.autolog(log_models=True)
        strategy = LogisticRegressionModelTrainingStrategy()
        strategy.train(X_train, y_train)
        evaluate_strategy = ClassificationEvaluationStrategy(strategy.model)
        mlflow.log_params(strategy.get_params())
        mlflow.log_metrics(evaluate_strategy.evaluate(X_test, y_test))
        mlflow.end_run()

        # Random Forest
        if not mlflow.active_run():
            mlflow.start_run(run_name="Random Forest")
        mlflow.autolog(log_models=True)
        strategy = RandomForestModelTrainingStrategy()
        strategy.train(X_train, y_train)
        evaluate_strategy = ClassificationEvaluationStrategy(strategy.model)
        mlflow.log_params(strategy.get_params())
        mlflow.log_metrics(evaluate_strategy.evaluate(X_test, y_test))
        mlflow.end_run()

        # Decision Tree
        if not mlflow.active_run():
            mlflow.start_run(run_name="Decision Tree")
        mlflow.autolog(log_models=True)
        strategy = DecisionTreeModelTrainingStrategy()
        strategy.train(X_train, y_train)
        evaluate_strategy = ClassificationEvaluationStrategy(strategy.model)
        mlflow.log_params(strategy.get_params())
        mlflow.log_metrics(evaluate_strategy.evaluate(X_test, y_test))
        mlflow.end_run()

        # SVC
        if not mlflow.active_run():
            mlflow.start_run(run_name="SVM")
        mlflow.autolog(log_models=True)
        strategy = SVCModelTrainingStrategy()
        strategy.train(X_train, y_train)
        evaluate_strategy = ClassificationEvaluationStrategy(strategy.model)
        mlflow.log_params(strategy.get_params())
        mlflow.log_metrics(evaluate_strategy.evaluate(X_test, y_test))
        mlflow.end_run()



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

    # Splitting Pipeline
    splitter_pipeline = DataSplitterPipeline()
    X_train, X_test, y_train, y_test = splitter_pipeline.split_data(df=data, test_size=0.2, y_column="Grade")

    # Training Pipeline
    training_pipeline = ModelTrainingPipeline()
    model = training_pipeline.train(X_train, y_train, X_test, y_test)

    return model