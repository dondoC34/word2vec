"""
MLflow tracking and experiment management module for Word2Vec.
Handles SQLite backend tracking, dynamic macro-experiments (SkipGram and BoW),
automatic metric logging, model checkpointing, and artifact figure generation.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

# Disable MLflow automatic agent hint on import
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA

matplotlib.use("Agg")  # Non-interactive backend safe for CLI and background runs

logger = logging.getLogger("MLflowTracker")

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
    Loads MLflow configuration from a YAML file or falls back to defaults.
    Searches config_path first, then 'mlflow-config.yml' in the current working directory
    or repository root.
    """
    candidate_paths: list[Path] = []
    if config_path:
        candidate_paths.append(Path(config_path))
    else:
        candidate_paths.append(Path("mlflow-config.yml"))
        candidate_paths.append(Path(__file__).resolve().parent.parent / "mlflow-config.yml")

    for p in candidate_paths:
        if p.exists() and p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    # Deep merge with DEFAULT_CONFIG
                    cfg = {
                        "server": {**DEFAULT_CONFIG["server"], **data.get("server", {})},
                        "tracking": {**DEFAULT_CONFIG["tracking"], **data.get("tracking", {})},
                    }
                    logger.debug("Loaded MLflow config from '%s'.", p)
                    return cfg
            except Exception as e:
                logger.warning("Error reading config file '%s': %s. Using default config.", p, e)

    return DEFAULT_CONFIG.copy()


def create_loss_figure(history: list[float], model_name: str = "Word2Vec") -> plt.Figure:
    """Generates a Matplotlib figure plotting training loss convergence."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    epochs = list(range(1, len(history) + 1))
    ax.plot(epochs, history, marker="o", color="#1f77b4", linewidth=2.0, markersize=5)
    ax.set_title(f"{model_name} Loss Convergence Across {len(history)} Epochs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Negative Log-Likelihood Loss", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    if len(history) <= 20:
        ax.set_xticks(epochs)
    fig.tight_layout()
    return fig


def create_heatmap_figure(
    model: Any,
    dataset: Any,
    candidate_words: list[str] | None = None,
    max_words: int = 16,
) -> plt.Figure | None:
    """
    Generates a pairwise cosine similarity heatmap figure for representative words.
    Falls back gracefully if vocabulary is small.
    """
    stoi = getattr(dataset, "stoi", {})
    if not stoi or not hasattr(model, "get_embedding"):
        return None

    if candidate_words is None:
        candidate_words = [
            "dorian",
            "gray",
            "lord",
            "henry",
            "alice",
            "rabbit",
            "queen",
            "hatter",
            "sea",
            "ship",
            "captain",
            "ahab",
            "hamlet",
            "king",
            "macbeth",
            "ghost",
            "time",
            "world",
            "man",
            "life",
        ]

    # Filter candidates present in vocabulary
    selected = [w for w in candidate_words if w in stoi]
    if len(selected) < 4:
        # Fallback to most frequent words in vocabulary
        sorted_vocab = sorted(stoi.keys(), key=lambda w: getattr(dataset, "counts", {}).get(w, 0), reverse=True)
        selected = [w for w in sorted_vocab[:max_words] if w in stoi]

    selected = selected[:max_words]
    if len(selected) < 2:
        return None

    embeddings = []
    valid_words = []
    for w in selected:
        emb = model.get_embedding(w)
        if emb is not None:
            embeddings.append(emb)
            valid_words.append(w)

    if len(valid_words) < 2:
        return None

    emb_arr = np.array(embeddings)
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    norm_embs = emb_arr / np.where(norms == 0, 1e-8, norms)
    sim_matrix = np.dot(norm_embs, norm_embs.T)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(sim_matrix, cmap="coolwarm", vmin=-0.5, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", fontsize=10)

    ax.set_xticks(range(len(valid_words)))
    ax.set_xticklabels(valid_words, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(valid_words)))
    ax.set_yticklabels(valid_words, fontsize=9)
    ax.set_title("Pairwise Cosine Similarity Heatmap", fontsize=12, fontweight="bold", pad=10)

    for r in range(len(valid_words)):
        for c in range(len(valid_words)):
            val = sim_matrix[r, c]
            txt_color = "white" if abs(val) > 0.45 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=7)

    fig.tight_layout()
    return fig


def create_pca_figure(
    model: Any,
    dataset: Any,
    candidate_words: list[str] | None = None,
    max_words: int = 25,
) -> plt.Figure | None:
    """Generates a 2D PCA projection map figure for representative vocabulary embeddings."""
    stoi = getattr(dataset, "stoi", {})
    if not stoi or not hasattr(model, "get_embedding"):
        return None

    if candidate_words is None:
        candidate_words = [
            "dorian",
            "gray",
            "lord",
            "henry",
            "alice",
            "rabbit",
            "queen",
            "hatter",
            "sea",
            "ship",
            "captain",
            "ahab",
            "hamlet",
            "king",
            "macbeth",
            "ghost",
            "man",
            "woman",
            "father",
            "mother",
            "time",
            "day",
            "night",
            "life",
            "death",
        ]

    selected = [w for w in candidate_words if w in stoi]
    if len(selected) < 4:
        sorted_vocab = sorted(stoi.keys(), key=lambda w: getattr(dataset, "counts", {}).get(w, 0), reverse=True)
        selected = [w for w in sorted_vocab[:max_words] if w in stoi]

    selected = selected[:max_words]
    if len(selected) < 3:
        return None

    embeddings = []
    valid_words = []
    for w in selected:
        emb = model.get_embedding(w)
        if emb is not None:
            embeddings.append(emb)
            valid_words.append(w)

    if len(valid_words) < 3:
        return None

    emb_arr = np.array(embeddings)
    if emb_arr.shape[1] < 2:
        return None

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(emb_arr)
    var_ratio = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(coords[:, 0], coords[:, 1], color="#2980b9", s=100, alpha=0.85, edgecolors="white", linewidth=1.2)

    for i, w in enumerate(valid_words):
        ax.annotate(
            w,
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.set_title(
        f"2D PCA Projection Map of Learned Embeddings\n(PC 1: {var_ratio[0]:.1%}, PC 2: {var_ratio[1]:.1%})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(f"Principal Component 1 ({var_ratio[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"Principal Component 2 ({var_ratio[1]:.1%})", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


class MLflowTracker:
    """
    Manages the lifecycle of an MLflow tracking session for Word2Vec training.
    Supports SQLite backend persistence, macro-experiments per model architecture,
    hyperparameter logging, metrics per epoch, and visual artifact plots.
    """

    def __init__(
        self,
        model_name: str = "SkipGram",
        enabled: bool = True,
        config: dict[str, Any] | str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.run_id: str | None = None

        if isinstance(config, str | Path) or config is None:
            self.config = load_mlflow_config(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = DEFAULT_CONFIG.copy()

        self.backend_store_uri: str = self.config.get("server", {}).get(
            "backend_store_uri", DEFAULT_CONFIG["server"]["backend_store_uri"]
        )
        self.default_artifact_root: str = self.config.get("server", {}).get(
            "default_artifact_root", DEFAULT_CONFIG["server"]["default_artifact_root"]
        )
        self.experiment_prefix: str = self.config.get("tracking", {}).get(
            "experiment_prefix", DEFAULT_CONFIG["tracking"]["experiment_prefix"]
        )
        self.autolog_figures: bool = self.config.get("tracking", {}).get("autolog_figures", True)
        self.autolog_model: bool = self.config.get("tracking", {}).get("autolog_model", True)
        self.system_metrics: bool = self.config.get("tracking", {}).get("system_metrics", True)
        self.last_run_id: str | None = None

        # Macro-experiment name: e.g. "Word2Vec-SkipGram" or "Word2Vec-BoW"
        self.experiment_name = f"{self.experiment_prefix}-{self.model_name}"

    def start_run(
        self,
        params: dict[str, Any] | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """
        Initializes the MLflow tracking environment, creates/sets the macro-experiment,
        and starts an active run logging all initial hyperparameters and metadata tags.
        """
        if not self.enabled:
            return None

        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow

            # Suppress third-party and internal MLflow console warnings
            logging.getLogger("mlflow").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", module="mlflow")

            # Enable system metrics logging if configured
            if self.system_metrics:
                try:
                    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
                    interval = str(self.config.get("tracking", {}).get("system_metrics_sampling_interval", 10))
                    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = interval
                    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING"] = "1"
                    mlflow.enable_system_metrics_logging()
                except Exception as e:
                    logger.debug("Failed enabling system metrics logging: %s", e)

            # Configure SQLite backend storage URI
            mlflow.set_tracking_uri(self.backend_store_uri)

            # Ensure local artifact directory exists if using local path
            if not self.default_artifact_root.startswith(("s3://", "gs://", "hdfs://")):
                Path(self.default_artifact_root).mkdir(parents=True, exist_ok=True)

            # Resolve or create the macro-experiment
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                exp_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.default_artifact_root,
                )
            else:
                exp_id = exp.experiment_id

            mlflow.set_experiment(experiment_id=exp_id)

            # Default descriptive run name if not provided
            if run_name is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dim_str = f"d{params.get('embedding_dim', '')}" if params else ""
                run_name = f"{self.model_name.lower()}_{dim_str}_{ts}".rstrip("_")

            # Start the active run
            active_run = mlflow.start_run(
                run_name=run_name,
                log_system_metrics=self.system_metrics,
            )
            self.run_id = active_run.info.run_id

            # Set descriptive tags
            run_tags = {
                "model_architecture": self.model_name,
                "framework": "PyTorch",
            }
            try:
                import subprocess

                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
                ).strip()
                git_branch = subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
                ).strip()
                run_tags["git_commit"] = git_commit
                run_tags["git_branch"] = git_branch
            except Exception:
                pass

            if tags:
                run_tags.update(tags)
            mlflow.set_tags(run_tags)

            # Log hyperparameters
            if params:
                # Sanitize params so only scalar primitives are logged
                clean_params = {}
                for k, v in params.items():
                    if isinstance(v, int | float | str | bool):
                        clean_params[k] = v
                    elif v is None:
                        clean_params[k] = "None"
                    else:
                        clean_params[k] = str(v)
                mlflow.log_params(clean_params)

            logger.info(
                "MLflow Run started: '%s' (Run ID: %s) in experiment '%s'.",
                run_name,
                self.run_id,
                self.experiment_name,
            )
            return self.run_id

        except Exception as e:
            logger.warning("Failed to start MLflow run: %s. Continuing without MLflow tracking.", e)
            self.enabled = False
            return None

    def log_epoch_metrics(self, epoch: int, loss: float, epoch_time_sec: float | None = None) -> None:
        """Logs the average training loss and epoch duration for a specific epoch."""
        if not self.enabled or not self.run_id:
            return
        try:
            import mlflow

            mlflow.log_metric("loss", loss, step=epoch)
            if epoch_time_sec is not None:
                mlflow.log_metric("epoch_time_seconds", round(epoch_time_sec, 4), step=epoch)
        except Exception as e:
            logger.debug("Failed logging epoch metrics to MLflow: %s", e)

    def log_epoch_loss(self, epoch: int, loss: float) -> None:
        """Logs the average training loss for a specific epoch (backwards compatibility)."""
        self.log_epoch_metrics(epoch=epoch, loss=loss)

    def log_summary_metrics(
        self,
        history: list[float],
        training_time_sec: float,
        total_tokens: int = 0,
        training_samples: int = 0,
        epochs: int = 1,
    ) -> None:
        """Logs end-of-training summary metrics (final loss, best loss, total elapsed time, throughput)."""
        if not self.enabled or not self.run_id:
            return
        try:
            import mlflow

            if history:
                mlflow.log_metric("final_loss", history[-1])
                mlflow.log_metric("best_loss", min(history))
            mlflow.log_metric("training_time_seconds", training_time_sec)
            mlflow.log_metric("avg_epoch_time_seconds", training_time_sec / max(epochs, 1))

            if total_tokens > 0:
                total_tokens_processed = total_tokens * epochs
                tokens_per_sec = total_tokens_processed / max(training_time_sec, 0.001)
                mlflow.log_metric("tokens_per_second", round(tokens_per_sec, 2))

            if training_samples > 0:
                total_samples_processed = training_samples * epochs
                samples_per_sec = total_samples_processed / max(training_time_sec, 0.001)
                mlflow.log_metric("samples_per_second", round(samples_per_sec, 2))
        except Exception as e:
            logger.debug("Failed logging summary metrics to MLflow: %s", e)

    def log_figures(self, history: list[float], model_instance: Any, dataset_instance: Any) -> None:
        """Generates and logs convergence and embedding visualization figures as MLflow artifacts."""
        if not self.enabled or not self.run_id or not self.autolog_figures:
            return
        try:
            import mlflow

            # 1. Loss convergence curve
            if history:
                fig_loss = create_loss_figure(history, model_name=self.model_name)
                mlflow.log_figure(fig_loss, "loss_convergence.png")
                plt.close(fig_loss)

            # 2. Cosine similarity heatmap
            fig_hm = create_heatmap_figure(model_instance, dataset_instance)
            if fig_hm is not None:
                mlflow.log_figure(fig_hm, "similarity_heatmap.png")
                plt.close(fig_hm)

            # 3. 2D PCA projection map
            fig_pca = create_pca_figure(model_instance, dataset_instance)
            if fig_pca is not None:
                mlflow.log_figure(fig_pca, "pca_embeddings_2d.png")
                plt.close(fig_pca)

            logger.info("Successfully logged visualization figures to MLflow artifacts.")

        except Exception as e:
            logger.warning("Failed to generate or log figures to MLflow: %s", e)

    def log_model_checkpoint(self, torch_model: Any, artifact_path: str = "model") -> None:
        """Logs the PyTorch model to MLflow as a first-class Logged Model with READY status."""
        if not self.enabled or not self.run_id or not self.autolog_model:
            return
        try:
            import mlflow.pytorch

            # Log model with pickle serialization and use `name` (MLflow 3.x) to avoid deprecation warning
            mlflow.pytorch.log_model(
                torch_model,
                name=artifact_path,
                serialization_format="pickle",
            )
            logger.info("Logged PyTorch model to MLflow artifact path '%s'.", artifact_path)
        except Exception as e:
            logger.warning("Failed logging PyTorch model to MLflow: %s", e)

    def log_corpora(
        self,
        corpus_files: list[str | Path],
        documents: list[list[str]] | None = None,
        log_content: bool = False,
        max_content_size_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        """
        Logs training corpora metadata (manifest) and optionally text files as MLflow artifacts.
        Metadata manifest JSON is always logged. Text files are uploaded only if `log_content` is True
        and total size does not exceed `max_content_size_bytes` (50MB by default).
        """
        if not self.enabled or not self.run_id:
            return

        try:
            import hashlib

            import mlflow

            valid_paths: list[Path] = []
            for f in corpus_files:
                p = Path(f)
                if p.exists() and p.is_file():
                    valid_paths.append(p.resolve())

            if not valid_paths:
                manifest = {
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "content_logged": False,
                    "source": "in-memory",
                    "files": [],
                }
                mlflow.log_dict(manifest, "corpora/corpora_manifest.json")
                mlflow.set_tags(
                    {
                        "corpora_count": "0",
                        "corpora_files": "in-memory",
                        "corpora_content_logged": "False",
                    }
                )
                return

            total_size_bytes = sum(p.stat().st_size for p in valid_paths)
            actual_log_content = log_content and (total_size_bytes <= max_content_size_bytes)

            doc_token_counts: list[int] = []
            if documents:
                doc_token_counts = [len(doc) for doc in documents]

            file_entries: list[dict[str, Any]] = []
            for idx, p in enumerate(valid_paths):
                sha256 = hashlib.sha256()
                with open(p, "rb") as fp:
                    for chunk in iter(lambda: fp.read(65536), b""):
                        sha256.update(chunk)
                file_hash = sha256.hexdigest()

                entry: dict[str, Any] = {
                    "filename": p.name,
                    "original_path": str(p),
                    "size_bytes": p.stat().st_size,
                    "sha256": file_hash,
                }
                if idx < len(doc_token_counts):
                    entry["token_count"] = doc_token_counts[idx]
                file_entries.append(entry)

            manifest_data: dict[str, Any] = {
                "total_files": len(valid_paths),
                "total_size_bytes": total_size_bytes,
                "content_logged": actual_log_content,
                "max_content_size_threshold_bytes": max_content_size_bytes,
                "files": file_entries,
            }

            mlflow.log_dict(manifest_data, "corpora/corpora_manifest.json")

            if actual_log_content:
                for p in valid_paths:
                    mlflow.log_artifact(str(p), artifact_path="corpora")
                logger.info(
                    "Logged %d corpora file(s) and manifest (total: %.2f MB) to MLflow artifacts ('corpora/').",
                    len(valid_paths),
                    total_size_bytes / (1024 * 1024),
                )
            elif log_content and total_size_bytes > max_content_size_bytes:
                logger.info(
                    "Total corpora size (%.2f MB) exceeds threshold (%.2f MB). "
                    "Logged corpora manifest JSON only to MLflow artifacts ('corpora/corpora_manifest.json').",
                    total_size_bytes / (1024 * 1024),
                    max_content_size_bytes / (1024 * 1024),
                )
            else:
                logger.info(
                    "Logged corpora metadata manifest (total %d files, %.2f MB) to MLflow artifacts ('corpora/corpora_manifest.json'). "
                    "Full text upload disabled (enable with --log-corpora-content).",
                    len(valid_paths),
                    total_size_bytes / (1024 * 1024),
                )

            file_names = [p.name for p in valid_paths]
            summary_names = ", ".join(file_names[:5])
            if len(file_names) > 5:
                summary_names += f" (+{len(file_names) - 5} more)"

            mlflow.set_tags(
                {
                    "corpora_count": str(len(valid_paths)),
                    "corpora_files": summary_names,
                    "corpora_content_logged": str(actual_log_content),
                }
            )

        except Exception as e:
            logger.warning("Failed logging corpora to MLflow: %s", e)

    def log_semantic_table(
        self,
        model_instance: Any,
        candidate_words: list[str] | None = None,
        top_k: int = 5,
        artifact_file: str = "evaluations/semantic_neighbors.json",
    ) -> None:
        """
        Evaluates nearest neighbors for representative vocabulary words and logs the
        results as an interactive table artifact (mlflow.log_table). Also logs summary
        metrics (mean_top1_similarity).
        """
        if not self.enabled or not self.run_id:
            return

        try:
            import mlflow
            import pandas as pd

            dataset = getattr(model_instance, "dataset", None)
            stoi = getattr(dataset, "stoi", {}) if dataset else {}
            if not stoi or not hasattr(model_instance, "most_similar"):
                return

            if candidate_words is None:
                candidate_words = [
                    "dorian",
                    "gray",
                    "lord",
                    "henry",
                    "alice",
                    "rabbit",
                    "queen",
                    "hatter",
                    "sea",
                    "ship",
                    "captain",
                    "ahab",
                    "hamlet",
                    "king",
                    "macbeth",
                    "ghost",
                    "time",
                    "world",
                    "man",
                    "life",
                    "art",
                ]

            target_words = [w for w in candidate_words if w in stoi]
            if len(target_words) < 3:
                sorted_vocab = sorted(stoi.keys(), key=lambda w: getattr(dataset, "counts", {}).get(w, 0), reverse=True)
                target_words = sorted_vocab[:10]

            rows: list[dict[str, Any]] = []
            top1_sims: list[float] = []

            for word in target_words:
                sims = model_instance.most_similar(word, top_k=top_k)
                for rank, (neighbor, score) in enumerate(sims, start=1):
                    rows.append(
                        {
                            "target_word": word,
                            "rank": rank,
                            "neighbor": neighbor,
                            "cosine_similarity": round(float(score), 4),
                        }
                    )
                    if rank == 1:
                        top1_sims.append(float(score))

            if rows:
                df = pd.DataFrame(rows)
                mlflow.log_table(data=df, artifact_file=artifact_file)
                if top1_sims:
                    mlflow.log_metric("mean_top1_similarity", float(np.mean(top1_sims)))
                logger.info(
                    "Logged semantic evaluation table (%d rows across %d words) to MLflow artifact '%s'.",
                    len(rows),
                    len(target_words),
                    artifact_file,
                )
        except Exception as e:
            logger.warning("Failed logging semantic table to MLflow: %s", e)

    def log_artifact(
        self,
        file_path: str | Path,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Logs an arbitrary file (e.g. exported embeddings .txt) as an MLflow artifact."""
        if not self.enabled:
            return
        target_run_id = run_id or self.run_id or self.last_run_id
        if not target_run_id:
            return

        try:
            import mlflow

            p = Path(file_path)
            if not p.exists():
                logger.warning("Artifact file not found: %s", file_path)
                return

            if self.run_id and target_run_id == self.run_id:
                mlflow.log_artifact(str(p), artifact_path=artifact_path)
            else:
                client = mlflow.MlflowClient(self.backend_store_uri)
                client.log_artifact(target_run_id, str(p), artifact_path=artifact_path)

            logger.info("Logged artifact '%s' to MLflow (run %s).", p.name, target_run_id)
        except Exception as e:
            logger.warning("Failed logging artifact '%s' to MLflow: %s", file_path, e)

    def end_run(self) -> None:
        """Terminates the active MLflow run cleanly."""
        if not self.enabled or not self.run_id:
            return
        try:
            import mlflow

            mlflow.end_run()
            logger.info("MLflow Run %s finalized successfully.", self.run_id)
        except Exception as e:
            logger.debug("Failed ending MLflow run: %s", e)
        finally:
            self.last_run_id = self.run_id
            self.run_id = None
