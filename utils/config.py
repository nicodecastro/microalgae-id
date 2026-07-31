import yaml
from pathlib import Path

def load_config(config_path="utils/config.yaml"):
    """
    Load YAML configuration file and return as a Python dictionary.

    Parameters:
        config_path (str or Path): Path to YAML config

    Returns:
        dict: configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config