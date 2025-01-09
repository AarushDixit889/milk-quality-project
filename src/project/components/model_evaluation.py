from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, X_test, y_test)->dict:
        pass

class ClassificationEvaluationStrategy(ModelEvaluationStrategy):
    def __init__(self, model):
        """
            Initialize the model evaluation strategy

            Parameters
            ----------
            model: ClassifierMixin
                Model to evaluate
            
        """

        self.model = model
    def evaluate(self, X_test, y_test)->dict:
        """
            Evaluate the model

            Parameters
            ----------
            X_test: pd.DataFrame
                Test data
            y_test: pd.DataFrame
                Test labels

            Returns
            -------
            dict
                Dictionary of evaluation metrics
        """

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}