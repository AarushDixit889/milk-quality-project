from abc import ABC, abstractmethod
import mlflow

class ModelRegistryStrategy(ABC):
    @abstractmethod
    def register_model(self, model_name, run_id)->dict:
        pass

class ModelRegistry(ModelRegistryStrategy):
    def register_model(self, model_name, run_id)->dict:
        mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)