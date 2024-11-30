import yaml
import os

def load_config(file_path='config.yaml'):
    """
    Loads configuration from a YAML file.
    
    :param file_path: Path to the YAML configuration file
    :return: Parsed configuration as a dictionary
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config
