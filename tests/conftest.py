from pathlib import Path

import pytest

import w2v.cli
import w2v.tracking


@pytest.fixture(autouse=True)
def isolate_mlflow_during_tests(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Autouse fixture that completely isolates MLflow tracking during all pytest runs.
    Ensures that tests never create or modify the production 'mlflow.db' or 'mlartifacts/' folder.
    All test runs write to an isolated temporary SQLite database that is automatically destroyed.
    """
    temp_dir = tmp_path_factory.mktemp("test_mlflow_isolation")
    temp_db = temp_dir / "isolated_test_mlflow.db"
    temp_artifacts = temp_dir / "isolated_artifacts"

    # Set environment variable fallback
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{temp_db}")

    isolated_config = {
        "server": {
            "host": "127.0.0.1",
            "port": 5000,
            "backend_store_uri": f"sqlite:///{temp_db}",
            "default_artifact_root": str(temp_artifacts),
        },
        "tracking": {
            "experiment_prefix": "TestWord2Vec",
            "autolog_figures": True,
            "autolog_model": False,  # Keep standard unit tests fast
        },
    }

    # Patch DEFAULT_CONFIG
    monkeypatch.setattr(w2v.tracking, "DEFAULT_CONFIG", isolated_config)

    # Patch load_mlflow_config so that calls without explicit path use the isolated config
    original_load = w2v.tracking.load_mlflow_config

    def patched_load(config_path: str | Path | None = None) -> dict:
        if config_path is None:
            return isolated_config.copy()
        return original_load(config_path)

    monkeypatch.setattr(w2v.tracking, "load_mlflow_config", patched_load)
    monkeypatch.setattr(w2v.cli, "load_mlflow_config", patched_load)
