"""Compatibility entry point for configured research experiments."""

from training.config import load_config
from training.experiments.runner import run_configured_experiment


def run_experiment(config_path=None, *, output_dir):
    """Load settings and run into an explicitly selected fresh directory."""
    return run_configured_experiment(load_config(config_path), output_dir)
