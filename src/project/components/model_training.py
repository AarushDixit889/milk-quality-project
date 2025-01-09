from abc import ABC, abstractmethod
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from src.project.utils.common import read_yaml

class ModelTrainingStrategy(ABC):
    @abstractmethod
    def train(self,X_train,y_train)->ClassifierMixin:
        """
            Abstract method to train model
        """
        pass
    
    @abstractmethod
    def get_params(self)->dict:
        """
            Abstract method to get params
        """
        pass

class LogisticRegressionModelTrainingStrategy(ModelTrainingStrategy):
    def __init__(self):
        params=read_yaml("D:\Aarush\data science projects\Milk Quality Prediction\params.yaml").logistic_regression
        self.model = LogisticRegression(C=params.C, penalty=params.penalty, solver=params.solver)

    def train(self,X_train,y_train)->ClassifierMixin:
        """
            Train model

            Parameters
            ----------
            X_train: pd.DataFrame
                Training data
            y_train: pd.DataFrame
                Target values

            Returns
            -------
            ClassifierMixin
                Trained model
        """
        self.model.fit(X_train,y_train)
        return self.model
    
    def get_params(self)->dict:
        return self.model.get_params()

class RandomForestModelTrainingStrategy(ModelTrainingStrategy):

    def __init__(self):
        params=read_yaml("D:\Aarush\data science projects\Milk Quality Prediction\params.yaml").random_forest
        self.model = RandomForestClassifier(n_estimators=params.n_estimators)

    def train(self,X_train,y_train)->ClassifierMixin:
        """
            Train model

            Parameters
            ----------
            X_train: pd.DataFrame
                Training data
            y_train: pd.DataFrame
                Target values

            Returns
            -------
            ClassifierMixin
                Trained model
        """
        self.model.fit(X_train,y_train)
        return self.model

    def get_params(self)->dict:
        return self.model.get_params()

class DecisionTreeModelTrainingStrategy(ModelTrainingStrategy):
    def __init__(self):
        params=read_yaml("D:\Aarush\data science projects\Milk Quality Prediction\params.yaml").decison_tree
        self.model = DecisionTreeClassifier(max_depth=params.max_depth)

    def train(self,X_train,y_train)->ClassifierMixin:
        """
            Train model

            Parameters
            ----------
            X_train: pd.DataFrame
                Training data
            y_train: pd.DataFrame
                Target values

            Returns
            -------
            ClassifierMixin
                Trained model
        """
        self.model.fit(X_train,y_train)
        return self.model
    
    def get_params(self)->dict:
        return self.model.get_params()
    
class SVCModelTrainingStrategy(ModelTrainingStrategy):
    def __init__(self):
        params=read_yaml("D:\Aarush\data science projects\Milk Quality Prediction\params.yaml").svc
        self.model = SVC(C=params.C, kernel=params.kernel)

    def train(self,X_train,y_train)->ClassifierMixin:
        """
            Train model

            Parameters
            ----------
            X_train: pd.DataFrame
                Training data
            y_train: pd.DataFrame
                Target values

            Returns
            -------
            ClassifierMixin
                Trained model
        """
        self.model.fit(X_train,y_train)
        return self.model
    
    def get_params(self)->dict:
        return self.model.get_params()