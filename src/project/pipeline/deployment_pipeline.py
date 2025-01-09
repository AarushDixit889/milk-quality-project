from src.project.components.model_registry import ModelRegistry

def deployment_pipeline():
    run_id = input("Run ID: ")
    model_name = input("Model Name: ")
    model_registry = ModelRegistry()
    model_registry.register_model(model_name=model_name, run_id=run_id)