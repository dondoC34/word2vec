"""
Core MLflow session tracker and experiment logger for Word2Vec models.

This module contains:
- `MLflowTracker`: High-level manager orchestrating experiment lifecycle, active run status,
  and artifact logging across Skip-Gram and Continuous Bag-of-Words architectures.
- Dynamic macro-experiments grouping runs cleanly (e.g. 'Word2Vec-SkipGram' and 'Word2Vec-BoW').
- Automatic logging of training loss convergence curves, 2D PCA projection maps, and cosine similarity heatmaps.
- Git version control metadata extraction (branch name and commit hash tags).
- Corpora manifest creation with SHA-256 integrity hashes, byte sizes, and token counts.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from w2v.tracking.config import DEFAULT_CONFIG, load_mlflow_config
from w2v.tracking.plotting import create_heatmap_figure, create_loss_figure, create_pca_figure

logger = logging.getLogger("MLflowTracker")


def _get_git_metadata() -> dict[str, str]:
    """
    Extracts current Git branch name and commit hash for experiment reproducibility tags.

    Returns:
        dict[str, str]: Dictionary containing 'git_commit' and 'git_branch' if available.
    """
    metadata: dict[str, str] = {}
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        metadata["git_commit"] = git_commit
        metadata["git_branch"] = git_branch
    except Exception:
        pass
    return metadata


class MLflowTracker:
    """
    Manages the lifecycle of an MLflow tracking session for Word2Vec training.

    Supports SQLite backend persistence, macro-experiments per model architecture,
    hyperparameter logging, metrics per epoch, visual artifact plots, and corpora manifests.
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

        Args:
            params: Dictionary of model/training hyperparameters to log.
            run_name: Optional explicit display name for the run.
            tags: Optional dictionary of metadata tags to register.

        Returns:
            str | None: Active Run ID string if tracking is enabled, None otherwise.
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
            run_tags.update(_get_git_metadata())

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
            logger.warning("Failed initializing MLflow run: %s. Continuing without tracking.", e)
            self.enabled = False
            return None

    def log_epoch_metrics(
        self,
        epoch: int,
        loss: float,
        epoch_time_sec: float | None = None,
        lr: float | None = None,
    ) -> None:
        """
        Logs fine-grained training loss and duration for a completed epoch.

        Args:
            epoch: 1-indexed epoch count.
            loss: Mean training loss computed across batches in the epoch.
            epoch_time_sec: Wall-clock duration of the epoch in seconds.
            lr: Current optimizer learning rate.
        """
        if not self.enabled or not self.run_id:
            return

        try:
            import mlflow

            mlflow.log_metric("epoch_loss", loss, step=epoch)
            mlflow.log_metric("loss", loss, step=epoch)
            if epoch_time_sec is not None:
                mlflow.log_metric("epoch_time_seconds", epoch_time_sec, step=epoch)
            if lr is not None:
                mlflow.log_metric("learning_rate", lr, step=epoch)
        except Exception as e:
            logger.debug("Failed logging epoch metrics to MLflow: %s", e)

    def log_epoch_loss(
        self,
        epoch: int,
        loss: float,
        epoch_time_sec: float | None = None,
        lr: float | None = None,
    ) -> None:
        """Backward-compatibility alias for log_epoch_metrics."""
        self.log_epoch_metrics(epoch=epoch, loss=loss, epoch_time_sec=epoch_time_sec, lr=lr)

    def log_summary_metrics(
        self,
        history: list[float],
        training_time_sec: float,
        total_tokens: int | None = None,
        training_samples: int | None = None,
        epochs: int = 1,
    ) -> None:
        """
        Logs aggregate summary performance and throughput metrics upon training completion.

        Args:
            history: List of epoch loss values.
            training_time_sec: Total duration of the training phase in seconds.
            total_tokens: Total token count in the loaded corpus documents.
            training_samples: Total number of training pairs or window contexts.
            epochs: Total number of completed epochs.
        """
        if not self.enabled or not self.run_id:
            return

        try:
            import mlflow

            final_loss = history[-1] if history else 0.0
            best_loss = min(history) if history else 0.0

            summary: dict[str, float] = {
                "final_loss": float(final_loss),
                "best_loss": float(best_loss),
                "training_time_seconds": float(training_time_sec),
                "avg_epoch_time_seconds": float(training_time_sec / max(epochs, 1)),
            }

            if training_time_sec > 0:
                if total_tokens is not None and total_tokens > 0:
                    summary["tokens_per_second"] = round((total_tokens * epochs) / training_time_sec, 2)
                if training_samples is not None and training_samples > 0:
                    summary["samples_per_second"] = round((training_samples * epochs) / training_time_sec, 2)

            mlflow.log_metrics(summary)
        except Exception as e:
            logger.debug("Failed logging summary metrics to MLflow: %s", e)

    def log_figures(
        self,
        history: list[float],
        model: Any,
        dataset: Any | None = None,
    ) -> None:
        """
        Generates and logs convergence, PCA projection, and similarity heatmap figures as artifacts.

        Args:
            history: Epoch loss values for the convergence curve.
            model: Trained Word2Vec model instance.
            dataset: Training dataset instance.
        """
        if not self.enabled or not self.run_id or not self.autolog_figures:
            return

        try:
            import matplotlib.pyplot as plt
            import mlflow

            # 1. Loss convergence curve
            if history:
                fig_loss = create_loss_figure(history, model_name=self.model_name)
                mlflow.log_figure(fig_loss, "loss_convergence.png")
                plt.close(fig_loss)

            # 2. Pairwise cosine similarity matrix heatmap
            fig_heatmap = create_heatmap_figure(model)
            if fig_heatmap is not None:
                mlflow.log_figure(fig_heatmap, "similarity_heatmap.png")
                plt.close(fig_heatmap)

            # 3. Two-dimensional PCA projection map
            ds = dataset or getattr(model, "dataset", None)
            if ds is not None:
                fig_pca = create_pca_figure(model, ds)
                if fig_pca is not None:
                    mlflow.log_figure(fig_pca, "pca_embeddings_2d.png")
                    plt.close(fig_pca)

            logger.info("Successfully logged visualization figures to MLflow artifacts.")
        except Exception as e:
            logger.warning("Failed generating or logging MLflow figures: %s", e)

    def log_semantic_table(
        self,
        model: Any,
        candidate_words: list[str] | None = None,
        top_k: int = 5,
    ) -> None:
        """
        Evaluates nearest semantic neighbors for anchor words and logs a structured JSON evaluation artifact.

        Args:
            model: Model instance implementing `most_similar(word, top_k)`.
            candidate_words: Benchmark tokens to probe.
            top_k: Number of nearest neighbors to retrieve per probe token.
        """
        if not self.enabled or not self.run_id:
            return

        if not hasattr(model, "most_similar") or not hasattr(model, "dataset"):
            return

        dataset = getattr(model, "dataset", None)
        stoi = getattr(dataset, "stoi", {}) if dataset else {}
        if not stoi:
            return

        if candidate_words is None:
            candidate_words = ["time", "day", "man", "king", "alice", "captain", "sea", "death"]

        valid_probes = [w for w in candidate_words if w in stoi]
        if not valid_probes:
            # Fall back to most frequent words in vocabulary
            counts = getattr(dataset, "counts", {})
            valid_probes = sorted(stoi.keys(), key=lambda w: counts.get(w, 0), reverse=True)[:5]

        valid_probes = valid_probes[:6]
        table_rows: list[dict[str, Any]] = []
        top1_sims: list[float] = []

        for probe in valid_probes:
            try:
                neighbors = model.most_similar(probe, top_k=top_k)
                for rank, (nbr_word, score) in enumerate(neighbors, start=1):
                    table_rows.append(
                        {
                            "target_word": probe,
                            "rank": rank,
                            "neighbor": nbr_word,
                            "cosine_similarity": round(float(score), 4),
                        }
                    )
                if neighbors:
                    top1_sims.append(float(neighbors[0][1]))
            except Exception as e:
                logger.debug("Failed computing neighbors for '%s': %s", probe, e)

        if not table_rows:
            return

        try:
            import mlflow

            evaluation_data = {
                "evaluation_date": datetime.now().isoformat(),
                "model_architecture": self.model_name,
                "top_k": top_k,
                "evaluated_words": len(valid_probes),
                "neighbors": table_rows,
            }
            mlflow.log_dict(evaluation_data, "evaluations/semantic_neighbors.json")

            if top1_sims:
                mean_top1 = sum(top1_sims) / len(top1_sims)
                mlflow.log_metric("mean_top1_similarity", round(mean_top1, 4))

            logger.info(
                "Logged semantic evaluation table (%d rows across %d words) to MLflow artifact 'evaluations/semantic_neighbors.json'.",
                len(table_rows),
                len(valid_probes),
            )
        except Exception as e:
            logger.debug("Failed logging semantic evaluation table: %s", e)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
    ) -> None:
        """
        Logs the PyTorch model checkpoint to the MLflow artifact repository.

        Args:
            model: PyTorch model instance (or pipeline exposing underlying `.model`).
            artifact_path: Target artifact subpath.
        """
        if not self.enabled or not self.run_id or not self.autolog_model:
            return

        try:
            import mlflow
            import torch.nn as nn

            target_nn = getattr(model, "model", model)
            if isinstance(target_nn, nn.Module):
                mlflow.pytorch.log_model(
                    pytorch_model=target_nn,
                    artifact_path=artifact_path,
                )
                logger.info("Logged PyTorch model to MLflow artifact path '%s'.", artifact_path)
        except Exception as e:
            logger.warning("Failed saving PyTorch model to MLflow: %s", e)

    def log_model_checkpoint(
        self,
        model: Any,
        artifact_path: str = "model",
    ) -> None:
        """Backward-compatibility alias for log_model."""
        self.log_model(model=model, artifact_path=artifact_path)

    def log_corpora(
        self,
        corpus_files: list[str | Path],
        documents: list[list[str]] | None = None,
        log_content: bool = False,
        max_content_size_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        """
        Logs training corpora metadata (manifest) and optionally raw text files as MLflow artifacts.

        Args:
            corpus_files: List of file paths pointing to plain text corpus files.
            documents: Optional list of tokenized documents corresponding to the corpus files.
            log_content: When True, uploads raw .txt files if total size <= max_content_size_bytes.
            max_content_size_bytes: Safety threshold (default: 50 MB) to prevent storage saturation.
        """
        if not self.enabled or not self.run_id:
            return

        try:
            import mlflow

            valid_paths: list[Path] = []
            for item in corpus_files:
                p = Path(item)
                if p.is_file():
                    valid_paths.append(p)

            # If corpus was ingested from memory (no files on disk)
            if not valid_paths:
                in_memory_manifest = {
                    "total_files": 0,
                    "source": "in-memory",
                    "timestamp": datetime.now().isoformat(),
                    "files": [],
                }
                mlflow.log_dict(in_memory_manifest, "corpora/corpora_manifest.json")
                return

            total_size_bytes = sum(p.stat().st_size for p in valid_paths)
            actual_log_content = log_content and (total_size_bytes <= max_content_size_bytes)

            doc_token_counts: list[int] = []
            if documents:
                doc_token_counts = [len(d) for d in documents]

            file_entries: list[dict[str, Any]] = []
            for idx, p in enumerate(valid_paths):
                hasher = hashlib.sha256()
                with open(p, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()

                tokens_count = doc_token_counts[idx] if idx < len(doc_token_counts) else None

                file_entries.append(
                    {
                        "filename": p.name,
                        "original_path": str(p.resolve()),
                        "size_bytes": p.stat().st_size,
                        "sha256": file_hash,
                        "token_count": tokens_count,
                    }
                )

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
                    "Logged %d corpora text file(s) (total %.2f MB) to MLflow artifacts ('corpora/').",
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
            logger.warning("Failed logging corpora manifest to MLflow: %s", e)

    def log_artifact(
        self,
        file_path: str | Path,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """
        Logs an arbitrary file (e.g. exported embeddings .txt) as an MLflow artifact.

        Args:
            file_path: Local path to the file to upload.
            artifact_path: Target directory inside the MLflow artifact repository.
            run_id: Optional Run ID override.
        """
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
        """Terminates the active MLflow run cleanly and records last_run_id."""
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
