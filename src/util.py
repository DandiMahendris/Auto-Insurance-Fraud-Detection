import yaml
import joblib
from datetime import datetime

config_dir = "config/config.yaml"

def time_stamp() -> datetime:
    #return current date and time
    return datetime.now()

def load_config() -> dict:
    # Try to load yaml file
    try:
        with open(config_dir, "r") as file:
            config = yaml.safe_load(file)
    except:
        raise RuntimeError("parameter file not found in path")

    # Return params in dictionary format
    return config

def pickle_load(file_path: str):
    # Load and return pickle file
    return joblib.load(file_path)

def pickle_dump(data, file_path: str) -> None:
    # Dump into file
    joblib.dump(data, file_path)

params = load_config()
PRINT_DEBUG = params["print_debug"]

def print_debug(message: str) -> None:
    # Check wether user want to use print or not
    if PRINT_DEBUG == True:
        print(time_stamp(), message)