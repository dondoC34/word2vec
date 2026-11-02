"""
MLflow tracking, experiment monitoring, and server lifecycle package for Word2Vec.

This package provides a clean, modular architecture for experiment tracking:
- `MLflowTracker`: High-level session manager for PyTorch Word2Vec model architectures.
- `config`: Configuration loading and default settings management.
- `plotting`: Diagnostic Matplotlib visualization generators (loss curves, PCA, heatmaps).
- `server`: Server lifecycle controls (process launching and port health checks).
- `manager`: Storage inspection, run query tables, best run discovery, and cleanup.
"""

from __future__ import annotations

from w2v.tracking.config import DEFAULT_CONFIG, load_mlflow_config
from w2v.tracking.manager import (
    clear_mlflow_data,
    export_run_artifacts,
    get_best_mlflow_run,
    get_mlflow_status,
    list_mlflow_runs,
    resolve_storage_paths,
)
from w2v.tracking.plotting import (
    create_heatmap_figure,
    create_loss_figure,
    create_pca_figure,
)
from w2v.tracking.server import (
    check_mlflow_server_online,
    start_mlflow_server,
)
from w2v.tracking.tracker import MLflowTracker

__all__ = [
    "DEFAULT_CONFIG",
    "MLflowTracker",
    "check_mlflow_server_online",
    "clear_mlflow_data",
    "create_heatmap_figure",
    "create_loss_figure",
    "create_pca_figure",
    "export_run_artifacts",
    "get_best_mlflow_run",
    "get_mlflow_status",
    "list_mlflow_runs",
    "load_mlflow_config",
    "resolve_storage_paths",
    "start_mlflow_server",
]
