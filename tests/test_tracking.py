import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np

from w2v.tracking import (
    DEFAULT_CONFIG,
    MLflowTracker,
    create_heatmap_figure,
    create_loss_figure,
    create_pca_figure,
    load_mlflow_config,
)


def test_load_mlflow_config_defaults(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.yml"
    cfg = load_mlflow_config(non_existent)
    assert "server" in cfg
    assert "tracking" in cfg
    assert cfg["server"]["host"] == DEFAULT_CONFIG["server"]["host"]
    assert cfg["server"]["port"] == DEFAULT_CONFIG["server"]["port"]


def test_load_mlflow_config_custom(tmp_path: Path) -> None:
    custom_yaml = tmp_path / "custom_mlflow.yml"
    custom_yaml.write_text(
        """
server:
  host: "0.0.0.0"
  port: 8080
  backend_store_uri: "sqlite:///custom.db"
tracking:
  experiment_prefix: "TestExp"
  autolog_figures: false
""",
        encoding="utf-8",
    )
    cfg = load_mlflow_config(custom_yaml)
    assert cfg["server"]["host"] == "0.0.0.0"
    assert cfg["server"]["port"] == 8080
    assert cfg["server"]["backend_store_uri"] == "sqlite:///custom.db"
    assert cfg["tracking"]["experiment_prefix"] == "TestExp"
    assert cfg["tracking"]["autolog_figures"] is False


def test_figure_creation() -> None:
    # 1. Loss figure
    fig_loss = create_loss_figure([10.5, 8.2, 6.1], model_name="SkipGram")
    assert isinstance(fig_loss, plt.Figure)
    plt.close(fig_loss)

    # 2. Mock model & dataset
    mock_dataset = MagicMock()
    mock_dataset.stoi = {"apple": 0, "banana": 1, "cherry": 2, "date": 3}
    mock_dataset.counts = {"apple": 10, "banana": 8, "cherry": 5, "date": 2}

    mock_model = MagicMock()
    rng = np.random.RandomState(42)
    mock_model.get_embedding.side_effect = lambda w: rng.randn(8) if w in mock_dataset.stoi else None

    # Heatmap figure
    fig_hm = create_heatmap_figure(mock_model, mock_dataset, candidate_words=["apple", "banana", "cherry", "date"])
    assert isinstance(fig_hm, plt.Figure)
    plt.close(fig_hm)

    # PCA figure
    fig_pca = create_pca_figure(mock_model, mock_dataset, candidate_words=["apple", "banana", "cherry", "date"])
    assert isinstance(fig_pca, plt.Figure)
    plt.close(fig_pca)


def test_mlflow_tracker_disabled() -> None:
    tracker = MLflowTracker(model_name="SkipGram", enabled=False)
    assert tracker.start_run(params={"lr": 0.01}) is None
    tracker.log_epoch_loss(1, 0.5)
    tracker.log_summary_metrics([0.5], 1.0)
    tracker.log_figures([0.5], None, None)
    tracker.end_run()
    assert tracker.run_id is None


def test_mlflow_tracker_sqlite_lifecycle(tmp_path: Path) -> None:
    db_file = tmp_path / "test_mlflow.db"
    art_dir = tmp_path / "artifacts"
    config = {
        "server": {
            "backend_store_uri": f"sqlite:///{db_file}",
            "default_artifact_root": str(art_dir),
        },
        "tracking": {
            "experiment_prefix": "TestW2V",
            "autolog_figures": True,
            "autolog_model": False,
        },
    }

    tracker = MLflowTracker(model_name="SkipGram", enabled=True, config=config)
    assert tracker.experiment_name == "TestW2V-SkipGram"

    # Start run
    params = {
        "embedding_dim": 16,
        "window_size": 2,
        "lr": 0.005,
        "epochs": 2,
    }
    run_id = tracker.start_run(params=params, run_name="test_run_1")
    assert run_id is not None
    assert tracker.run_id == run_id

    # Log metrics
    tracker.log_epoch_loss(1, 5.2)
    tracker.log_epoch_loss(2, 3.8)
    tracker.log_summary_metrics([5.2, 3.8], training_time_sec=1.5)

    # Log dummy artifact file
    dummy_art = tmp_path / "test_emb.txt"
    dummy_art.write_text("apple 0.1 0.2", encoding="utf-8")
    tracker.log_artifact(dummy_art)

    tracker.end_run()
    assert tracker.run_id is None

    # Verify SQLite database is populated
    assert db_file.exists()
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check experiments table
    cursor.execute("SELECT name FROM experiments WHERE name = 'TestW2V-SkipGram'")
    exp_row = cursor.fetchone()
    assert exp_row is not None
    assert exp_row[0] == "TestW2V-SkipGram"

    # Check runs table
    cursor.execute("SELECT run_uuid FROM runs WHERE run_uuid = ?", (run_id,))
    run_row = cursor.fetchone()
    assert run_row is not None

    # Check params table
    cursor.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_id,))
    logged_params = dict(cursor.fetchall())
    assert logged_params["embedding_dim"] == "16"
    assert logged_params["epochs"] == "2"

    # Check metrics table
    cursor.execute("SELECT key, value, step FROM metrics WHERE run_uuid = ?", (run_id,))
    metrics = cursor.fetchall()
    loss_steps = [m[2] for m in metrics if m[0] == "loss"]
    assert 1 in loss_steps
    assert 2 in loss_steps

    conn.close()
