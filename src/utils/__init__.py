"""Utility functions and helpers"""

import logging
from typing import Dict, List
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict, output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to {output_path}")


def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)


def create_directories(paths: List[str]) -> None:
    """
    Create multiple directories.
    
    Args:
        paths: List of directory paths
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created {len(paths)} directories")
