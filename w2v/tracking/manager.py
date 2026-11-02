"""
MLflow experiment queries, diagnostics, and storage management module.

This module provides high-level utilities to manage the local MLflow environment:
- Resolving storage filesystem paths for the SQLite backend and artifact directories.
- Querying server health, storage size, and run diagnostics (`get_mlflow_status`).
- Searching, filtering, and tabulating recent runs (`list_mlflow_runs`).
- Discovering and extracting the best performing model checkpoint (`get_best_mlflow_run`).
- Safely wiping SQLite database records and artifact directories (`clear_mlflow_data`).
- Exporting artifacts and embeddings for a given run ID to an external folder (`export_run_artifacts`).
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from w2v.tracking.config import load_mlflow_config
from w2v.tracking.server import check_mlflow_server_online

logger = logging.getLogger("MLflowManager")


def resolve_storage_paths(config_path: str | Path | None = None) -> tuple[Path | None, Path]:
    """
    Resolves the local filesystem paths for the SQLite database and default artifact root.

    Args:
        config_path: Optional custom path to YAML config file.

    Returns:
        tuple[Path | None, Path]: A pair (db_path, artifact_dir).
    """
    cfg = load_mlflow_config(config_path)
    server_cfg = cfg.get("server", {})
    backend_uri = server_cfg.get("backend_store_uri", "sqlite:///mlflow.db")
    artifact_root = server_cfg.get("default_artifact_root", "./mlartifacts")

    db_path: Path | None = None
    if backend_uri.startswith("sqlite:///"):
        db_raw = backend_uri[len("sqlite:///") :]
        db_path = Path(db_raw).resolve()

    artifact_path = Path(artifact_root).resolve()
    return db_path, artifact_path


def clear_mlflow_data(
    config_path: str | Path | None = None,
    artifacts_only: bool = False,
) -> tuple[bool, str]:
    """
    Cleans local MLflow storage by deleting artifacts and SQLite database records.

    Args:
        config_path: Optional custom path to YAML config file.
        artifacts_only: When True, only clears the artifacts folder and leaves SQLite DB intact.

    Returns:
        tuple[bool, str]: Status flag and human-readable confirmation or error message.
    """
    db_path, artifact_path = resolve_storage_paths(config_path)
    deleted_items: list[str] = []

    # 1. Clean artifacts directory
    if artifact_path.exists():
        try:
            for item in artifact_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            deleted_items.append(f"artifact directory '{artifact_path}'")
        except Exception as e:
            logger.error("Failed cleaning artifact directory '%s': %s", artifact_path, e)
            return False, f"Failed cleaning artifact directory '{artifact_path}': {e}"
    else:
        artifact_path.mkdir(parents=True, exist_ok=True)

    # 2. Clean SQLite database file and auxiliary journal/WAL files
    if not artifacts_only and db_path is not None:
        if db_path.exists():
            try:
                db_path.unlink()
                deleted_items.append(f"database file '{db_path.name}'")
            except PermissionError:
                msg = (
                    f"Database file '{db_path}' is currently locked by another active process. "
                    "If the MLflow tracking server is currently running, please stop it first."
                )
                logger.error(msg)
                return False, msg
            except Exception as e:
                logger.error("Failed deleting database file '%s': %s", db_path, e)
                return False, f"Failed deleting database file '{db_path}': {e}"

        # Clean auxiliary SQLite write-ahead log files
        for suffix in ("-wal", "-shm"):
            aux_file = db_path.with_name(db_path.name + suffix)
            if aux_file.exists():
                try:
                    aux_file.unlink()
                except Exception:
                    pass

    if not deleted_items:
        return True, "MLflow storage is already empty."
    return True, f"Successfully cleared {', '.join(deleted_items)}."


def get_mlflow_status(
    config_path: str | Path | None = None,
    host: str | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    """
    Collects comprehensive diagnostic metrics on server availability and storage usage.

    Args:
        config_path: Optional path to YAML config file.
        host: Optional server host override to probe.
        port: Optional server port override to probe.

    Returns:
        dict[str, Any]: Structured dictionary with server health, storage stats, and run counts.
    """
    cfg = load_mlflow_config(config_path)
    server_cfg = cfg.get("server", {})
    resolved_host = host or server_cfg.get("host", "127.0.0.1")
    resolved_port = int(port or server_cfg.get("port", 5000))
    backend_uri = server_cfg.get("backend_store_uri", "sqlite:///mlflow.db")
    db_path, artifact_path = resolve_storage_paths(config_path)

    server_online = check_mlflow_server_online(resolved_host, resolved_port)

    db_exists = db_path.exists() if db_path else False
    db_size = db_path.stat().st_size if db_exists and db_path else 0

    artifact_exists = artifact_path.exists()
    artifact_files = list(artifact_path.rglob("*")) if artifact_exists else []
    artifact_file_count = sum(1 for f in artifact_files if f.is_file())
    artifact_total_size = sum(f.stat().st_size for f in artifact_files if f.is_file())

    experiment_count = 0
    total_runs = 0
    runs_by_status: dict[str, int] = {}
    best_loss_by_experiment: dict[str, tuple[float, str]] = {}

    if db_exists or not backend_uri.startswith("sqlite:///"):
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=backend_uri)
            experiments = client.search_experiments()
            experiment_count = len(experiments)

            exp_ids = [e.experiment_id for e in experiments]
            if exp_ids:
                all_runs = client.search_runs(experiment_ids=exp_ids, max_results=5000)
                total_runs = len(all_runs)
                for r in all_runs:
                    status = r.info.status
                    runs_by_status[status] = runs_by_status.get(status, 0) + 1
                    exp_name = next(
                        (e.name for e in experiments if e.experiment_id == r.info.experiment_id), r.info.experiment_id
                    )
                    loss = r.data.metrics.get("final_loss") or r.data.metrics.get("loss")
                    if loss is not None:
                        current_best = best_loss_by_experiment.get(exp_name)
                        if current_best is None or loss < current_best[0]:
                            best_loss_by_experiment[exp_name] = (loss, r.info.run_id[:8])
        except Exception as e:
            logger.debug("Failed querying MlflowClient for status diagnostics: %s", e)

    return {
        "host": resolved_host,
        "port": resolved_port,
        "server_online": server_online,
        "backend_store_uri": backend_uri,
        "db_path": str(db_path) if db_path else None,
        "db_exists": db_exists,
        "db_size_bytes": db_size,
        "artifact_dir": str(artifact_path),
        "artifact_exists": artifact_exists,
        "artifact_file_count": artifact_file_count,
        "artifact_size_bytes": artifact_total_size,
        "experiment_count": experiment_count,
        "total_runs": total_runs,
        "runs_by_status": runs_by_status,
        "best_loss_by_experiment": best_loss_by_experiment,
    }


def list_mlflow_runs(
    config_path: str | Path | None = None,
    experiment_name: str | None = None,
    model: str | None = None,
    limit: int = 10,
    sort_by: str = "date",
) -> list[dict[str, Any]]:
    """
    Retrieves and formats a structured list of recent MLflow runs.

    Args:
        config_path: Optional path to YAML config file.
        experiment_name: Optional filter for a specific experiment name.
        model: Optional filter for model architecture (e.g. 'skipgram', 'bow').
        limit: Maximum number of run records to retrieve.
        sort_by: Sort strategy: 'date' (most recent first) or 'loss' (lowest first).

    Returns:
        list[dict[str, Any]]: List of dictionary summaries per matching run.
    """
    cfg = load_mlflow_config(config_path)
    backend_uri = cfg.get("server", {}).get("backend_store_uri", "sqlite:///mlflow.db")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=backend_uri)
    try:
        experiments = client.search_experiments()
    except Exception as e:
        logger.debug("No experiments found or failed connecting: %s", e)
        return []

    if not experiments:
        return []

    exp_map = {e.experiment_id: e.name for e in experiments}
    target_exp_ids = [
        e.experiment_id
        for e in experiments
        if not experiment_name or e.name.lower() == experiment_name.lower() or experiment_name.lower() in e.name.lower()
    ]
    if not target_exp_ids:
        return []

    order_by = ["metrics.final_loss ASC"] if sort_by == "loss" else ["attributes.start_time DESC"]
    try:
        runs = client.search_runs(
            experiment_ids=target_exp_ids,
            order_by=order_by,
            max_results=limit * 3 if model else limit,
        )
    except Exception as e:
        logger.debug("Failed querying runs: %s", e)
        return []

    results: list[dict[str, Any]] = []
    for r in runs:
        run_model = r.data.params.get("model_architecture") or r.data.tags.get("model_architecture", "-")
        if model and model.lower() not in run_model.lower():
            continue

        loss_val = r.data.metrics.get("final_loss") or r.data.metrics.get("loss")
        start_ts = r.info.start_time
        date_str = datetime.fromtimestamp(start_ts / 1000.0).strftime("%Y-%m-%d %H:%M:%S") if start_ts else "-"

        results.append(
            {
                "run_id": r.info.run_id,
                "experiment_name": exp_map.get(r.info.experiment_id, r.info.experiment_id),
                "model": run_model,
                "embedding_dim": r.data.params.get("embedding_dim", "-"),
                "epochs": r.data.params.get("epochs", "-"),
                "final_loss": f"{loss_val:.4f}" if loss_val is not None else "-",
                "status": r.info.status,
                "created_at": date_str,
                "artifact_uri": r.info.artifact_uri,
            }
        )
        if len(results) >= limit:
            break

    return results


def get_best_mlflow_run(
    config_path: str | Path | None = None,
    experiment_name: str | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """
    Finds the run with the lowest recorded loss for the target experiment or model architecture.

    Args:
        config_path: Optional path to YAML config file.
        experiment_name: Optional filter for a specific experiment name.
        model: Optional filter for model architecture (e.g. 'skipgram', 'bow').

    Returns:
        dict[str, Any] | None: Detailed dictionary containing parameters, metrics, and artifact paths.
    """
    runs = list_mlflow_runs(
        config_path=config_path,
        experiment_name=experiment_name,
        model=model,
        limit=1,
        sort_by="loss",
    )
    if not runs or runs[0].get("final_loss") == "-":
        return None

    run_entry = runs[0]
    cfg = load_mlflow_config(config_path)
    backend_uri = cfg.get("server", {}).get("backend_store_uri", "sqlite:///mlflow.db")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=backend_uri)
    full_run = client.get_run(run_entry["run_id"])

    run_entry["params"] = full_run.data.params
    run_entry["metrics"] = full_run.data.metrics
    run_entry["tags"] = {k: v for k, v in full_run.data.tags.items() if not k.startswith("mlflow.")}

    # Discover artifacts
    _, artifact_path = resolve_storage_paths(config_path)
    run_artifacts_dir = artifact_path / run_entry["run_id"] / "artifacts"
    run_entry["artifacts_dir"] = str(run_artifacts_dir) if run_artifacts_dir.exists() else None

    embeddings_files = (
        list((run_artifacts_dir / "embeddings").rglob("*.txt"))
        if (run_artifacts_dir / "embeddings").exists()
        else [f for f in run_artifacts_dir.rglob("*.txt") if "embeddings" in f.name.lower() or "w2v_" in f.name.lower()]
    )
    run_entry["embeddings_file"] = str(embeddings_files[0]) if embeddings_files else None

    return run_entry


def export_run_artifacts(
    run_id: str,
    dest_dir: str | Path,
    config_path: str | Path | None = None,
) -> list[Path]:
    """
    Copies artifacts from a specified run ID to a local destination directory.

    Args:
        run_id: The MLflow Run ID string.
        dest_dir: Target local directory where files will be copied.
        config_path: Optional custom path to YAML config file.

    Returns:
        list[Path]: List of copied file paths.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    _, artifact_path = resolve_storage_paths(config_path)
    run_artifacts_dir = artifact_path / run_id / "artifacts"
    if not run_artifacts_dir.exists():
        raise FileNotFoundError(f"Artifact directory for run '{run_id}' not found at '{run_artifacts_dir}'.")

    copied: list[Path] = []
    for item in run_artifacts_dir.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied.append(target)
    return copied
