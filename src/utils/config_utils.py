# src/utils/config_utils.py

import yaml

# Function to load the configuration from config.yaml
def load_config(config_path="config/config.yaml"):
    """Load configuration from the provided YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Function to save updated configuration back to config.yaml
def save_config(config, config_path="config/config.yaml"):
    """Save updated configuration to the YAML file."""
    with open(config_path, "w") as file:
        yaml.safe_dump(config, file)

# Function to get a specific value from the config
def get_config_value(key, config=None, config_path="config/config.yaml"):
    """Get a specific configuration value by key."""
    if config is None:
        config = load_config(config_path)
    keys = key.split('.')
    value = config
    for k in keys:
        value = value.get(k, None)
        if value is None:
            raise KeyError(f"Key '{key}' not found in configuration.")
    return value

# Example usage
# if __name__ == "__main__":
#     config = load_config()
#     print(get_config_value("model.wav2vec2.sample_rate", config))
