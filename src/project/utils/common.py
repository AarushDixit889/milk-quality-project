import yaml
from pathlib import Path
import os
import json
import pickle

def read_yaml(path_to_yaml: Path) -> dict:
    """
        Reads a YAML file and returns the contents as a dictionary

        Parameters
        ----------
        path_to_yaml: str
            Path to the YAML file

        Returns
        -------
        dict
            Contents of the YAML file
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
        return content
    except Exception as e:
        raise e
    


def create_directories(path_to_directories: list, verbose=True):
    """
        Creates directories if they don't exist

        Parameters
        ----------
        path_to_directories: list
            List of directories to create
        
            
        Returns
        -------
        None
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"Created directory at: {path}")

def save_json(path: Path, data: dict):
    """
        Saves a dictionary to a JSON file

        Parameters
        ----------
        path: Path
            Path to the JSON file

        data: dict
            Dictionary to save
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved json report at: {path}")



def load_json(path: Path) -> dict:
    """
        Loads a JSON file and returns the contents as a dictionary

        Parameters
        ----------
        path: Path
            Path to the JSON file

        Returns
        -------
        dict
            Contents of the JSON file
    """
    with open(path, "r") as f:
        return json.load(f)
    
def save_object(file_path: Path, obj: object) -> None:
    """
        Saves an object to a pickle file

        Parameters
        ----------
        file_path: Path
            Path to the pickle file

        obj: object
            Object to save
    """
    try:
        dir_path = Path(file_path).parent
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise e



def load_object(file_path: Path) -> object:
    """
        Loads an object from a pickle file

        Parameters
        ----------
        file_path: Path
            Path to the pickle file

        Returns
        -------
        object 
            Object loaded from the pickle file
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise e

