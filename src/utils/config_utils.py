# config_utils.py

import yaml

def load_config(config_path):
    """
    Load the YAML configuration file.

    :param config_path: Path to the configuration file (YAML).
    :return: Configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config
