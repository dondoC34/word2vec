"""
MLflow tracking configuration management module.

This module handles:
- Default configuration settings for the local MLflow SQLite backend and artifact store.
- Parsing and validation of custom YAML configuration files (e.g., 'mlflow-config.yml').
- Transparent cascading fallback to project defaults when no configuration is provided.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("MLflowConfig")

# Default settings for the MLflow tracking environment
DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 5000,
        "backend_store_uri": "sqlite:///mlflow.db",
        "default_artifact_root": "./mlartifacts",
    },
    "tracking": {
        "experiment_prefix": "Word2Vec",
        "autolog_figures": True,
        "autolog_model": True,
        "system_metrics": True,
        "system_metrics_sampling_interval": 10,
    },
}


def load_mlflow_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Loads MLflow configuration from a YAML file or falls back to standard defaults.

    Resolution order:
    1. The explicit path supplied via `config_path` (if provided).
    2. 'mlflow-config.yml' in the current working directory.
    3. 'mlflow-config.yml' in the project repository root.
    4. Built-in `DEFAULT_CONFIG` fallback dictionary.

    Args:
        config_path: Optional file path or string pointing to a YAML configuration file.

    Returns:
        dict[str, Any]: Resolved configuration dictionary containing 'server' and 'tracking' blocks.
    """
    candidate_paths: list[Path] = []
    if config_path:
        candidate_paths.append(Path(config_path))
    else:
        candidate_paths.append(Path("mlflow-config.yml"))
        candidate_paths.append(Path(__file__).resolve().parent.parent.parent / "mlflow-config.yml")

    for p in candidate_paths:
        if p.exists() and p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    # Deep merge with DEFAULT_CONFIG to guarantee complete keys
                    cfg = {
                        "server": {**DEFAULT_CONFIG["server"], **data.get("server", {})},
                        "tracking": {**DEFAULT_CONFIG["tracking"], **data.get("tracking", {})},
                    }
                    logger.debug("Loaded MLflow config from '%s'.", p)
                    return cfg
            except Exception as e:
                logger.warning("Error reading config file '%s': %s. Using default config.", p, e)

    return DEFAULT_CONFIG.copy()
