import yaml
from pathlib import Path

def load_config(config_path=None):
    """
    Load YAML configuration file and return as a Python dictionary.

    Parameters:
        config_path (str or Path or None): Path to YAML config. If omitted,
            load configs/training.yaml from the repository.

    Returns:
        dict: configuration dictionary
    """
    config_path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
