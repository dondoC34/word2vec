import json
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


def test_mlflow_tracker_log_corpora(tmp_path: Path) -> None:
    db_file = tmp_path / "test_corpora.db"
    art_dir = tmp_path / "artifacts"
    config = {
        "server": {
            "backend_store_uri": f"sqlite:///{db_file}",
            "default_artifact_root": str(art_dir),
        },
        "tracking": {
            "experiment_prefix": "TestCorpora",
            "autolog_figures": False,
            "autolog_model": False,
        },
    }

    # Case 1: Explicit log_content=True (below 50MB threshold) -> manifest + .txt files
    c1 = tmp_path / "book1.txt"
    c1.write_text("the quick brown fox jumps over the lazy dog", encoding="utf-8")
    c2 = tmp_path / "book2.txt"
    c2.write_text("alice was beginning to get very tired of sitting by her sister", encoding="utf-8")

    tracker = MLflowTracker(model_name="SkipGram", enabled=True, config=config)
    run_id = tracker.start_run(run_name="corpora_run")
    assert run_id is not None

    tracker.log_corpora(
        corpus_files=[c1, c2],
        documents=[["the", "quick", "fox"], ["alice", "sister"]],
        log_content=True,
        max_content_size_bytes=50 * 1024 * 1024,
    )
    tracker.end_run()

    # Verify artifacts on disk
    manifest_candidates = list(art_dir.glob("**/corpora/corpora_manifest.json"))
    assert len(manifest_candidates) == 1
    manifest_path = manifest_candidates[0]
    assert manifest_path.exists()

    with open(manifest_path, encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["total_files"] == 2
    assert manifest_data["content_logged"] is True
    assert len(manifest_data["files"]) == 2
    assert manifest_data["files"][0]["filename"] == "book1.txt"
    assert "sha256" in manifest_data["files"][0]

    # Verify actual .txt files copied
    corpora_dir = manifest_path.parent
    assert (corpora_dir / "book1.txt").exists()
    assert (corpora_dir / "book2.txt").exists()

    # Case 2: Default log_content=False -> manifest only, NO .txt copy
    tracker_default = MLflowTracker(model_name="SkipGram", enabled=True, config=config)
    run_id_default = tracker_default.start_run(run_name="corpora_default_no_content")
    assert run_id_default is not None

    tracker_default.log_corpora(
        corpus_files=[c1, c2],
        log_content=False,
    )
    tracker_default.end_run()

    manifest_candidates_def = list(art_dir.glob("**/corpora/corpora_manifest.json"))
    assert len(manifest_candidates_def) == 2
    def_manifest_path = next(m for m in manifest_candidates_def if str(m) != str(manifest_path))
    with open(def_manifest_path, encoding="utf-8") as f:
        def_manifest_data = json.load(f)
    assert def_manifest_data["content_logged"] is False
    assert not (def_manifest_path.parent / "book1.txt").exists()

    # Case 3: Corpora size exceeds threshold (even if log_content=True) -> only manifest
    tracker2 = MLflowTracker(model_name="SkipGram", enabled=True, config=config)
    run_id2 = tracker2.start_run(run_name="corpora_exceed_threshold")
    assert run_id2 is not None

    tracker2.log_corpora(
        corpus_files=[c1, c2],
        log_content=True,
        max_content_size_bytes=10,  # 10 bytes threshold is smaller than the files
    )
    tracker2.end_run()

    manifest_candidates2 = list(art_dir.glob("**/corpora/corpora_manifest.json"))
    assert len(manifest_candidates2) == 3
    new_manifest_path = next(m for m in manifest_candidates2 if m not in (manifest_path, def_manifest_path))
    with open(new_manifest_path, encoding="utf-8") as f:
        manifest_data2 = json.load(f)
    assert manifest_data2["content_logged"] is False
    # Verify .txt files were NOT copied into this run's corpora artifact dir
    assert not (new_manifest_path.parent / "book1.txt").exists()

    # Case 4: In-memory dataset (empty corpus files list)
    tracker3 = MLflowTracker(model_name="SkipGram", enabled=True, config=config)
    tracker3.start_run(run_name="corpora_in_memory")
    tracker3.log_corpora(corpus_files=[])
    tracker3.end_run()

    manifest_candidates3 = list(art_dir.glob("**/corpora/corpora_manifest.json"))
    assert len(manifest_candidates3) == 4
    in_mem_manifest = next(
        m for m in manifest_candidates3 if m not in (manifest_path, def_manifest_path, new_manifest_path)
    )
    with open(in_mem_manifest, encoding="utf-8") as f:
        in_mem_data = json.load(f)
    assert in_mem_data["total_files"] == 0
    assert in_mem_data["source"] == "in-memory"


def test_mlflow_advanced_features(tmp_path: Path) -> None:
    db_file = tmp_path / "test_advanced.db"
    art_dir = tmp_path / "artifacts_advanced"
    config = {
        "server": {
            "backend_store_uri": f"sqlite:///{db_file}",
            "default_artifact_root": str(art_dir),
        },
        "tracking": {
            "experiment_prefix": "TestAdvanced",
            "autolog_figures": False,
            "autolog_model": False,
            "system_metrics": True,
        },
    }

    tracker = MLflowTracker(model_name="SkipGram", enabled=True, config=config)
    run_id = tracker.start_run(run_name="adv_run")
    assert run_id is not None

    # 1. Epoch metrics with timing
    tracker.log_epoch_metrics(epoch=1, loss=4.5, epoch_time_sec=0.25)
    tracker.log_epoch_metrics(epoch=2, loss=3.2, epoch_time_sec=0.20)

    # 2. Summary metrics with throughput
    tracker.log_summary_metrics(
        history=[4.5, 3.2],
        training_time_sec=0.50,
        total_tokens=1000,
        training_samples=500,
        epochs=2,
    )

    # 3. Semantic table evaluation
    mock_model = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset.stoi = {"apple": 0, "banana": 1, "cherry": 2}
    mock_dataset.counts = {"apple": 10, "banana": 8, "cherry": 5}
    mock_model.dataset = mock_dataset
    mock_model.most_similar.side_effect = lambda w, top_k=5: [("fruit", 0.88), ("tree", 0.65)]

    tracker.log_semantic_table(mock_model, candidate_words=["apple", "banana"])

    # End run
    tracker.end_run()
    assert tracker.run_id is None
    assert tracker.last_run_id == run_id

    # 4. Post-run artifact logging (e.g. exported embeddings)
    exported_emb = tmp_path / "w2v_skipgram_embeddings_20261101_2200.txt"
    exported_emb.write_text("apple 0.1 0.2\nbanana 0.3 0.4\n", encoding="utf-8")
    tracker.log_artifact(exported_emb, artifact_path="embeddings")

    # Verify artifacts and metrics
    emb_candidates = list(art_dir.glob("**/embeddings/w2v_skipgram_embeddings_20261101_2200.txt"))
    assert len(emb_candidates) == 1
    assert emb_candidates[0].exists()

    table_candidates = list(art_dir.glob("**/evaluations/semantic_neighbors.json"))
    assert len(table_candidates) == 1
    assert table_candidates[0].exists()

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT key, value FROM metrics WHERE run_uuid = ?", (run_id,))
    metrics_dict = {row[0]: row[1] for row in cursor.fetchall()}
    assert "tokens_per_second" in metrics_dict
    assert "samples_per_second" in metrics_dict
    assert "epoch_time_seconds" in metrics_dict
    assert "mean_top1_similarity" in metrics_dict
    assert metrics_dict["mean_top1_similarity"] == 0.88

    # Verify git tags if present
    cursor.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (run_id,))
    tags_dict = {row[0]: row[1] for row in cursor.fetchall()}
    assert "git_branch" in tags_dict
    assert "git_commit" in tags_dict

    conn.close()
