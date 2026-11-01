import logging
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Sized
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from w2v.data.downloader import ensure_nltk_gutenberg_corpora
from w2v.preprocessing.tokenizer import BaseTokenizer, BasicTokenizer
from w2v.tracking import MLflowTracker


def setup_logging(
    log_dir: str | Path = "./logs",
    log_filename: str | None = None,
    level: int = logging.INFO,
) -> Path:
    """
    Configures centralized logging for the Word2Vec application.
    Logs are formatted with timestamp, log level, and component source name:
      YYYY-MM-DD HH:MM:SS [LEVEL] [source_of_log]: message
    Outputs simultaneously to the console and to an execution-specific .log file.

    Args:
        log_dir: Directory where log files are stored.
        log_filename: Optional explicit file name. Defaults to 'word2vec_YYYYMMDD_HHMMSS.log'.
        level: Logging severity threshold.

    Returns:
        Path to the generated log file.
    """
    dest = Path(log_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"word2vec_{timestamp}.log"

    log_path = (dest / log_filename).resolve()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging upon re-initialization
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # File Handler: writes to timestamped .log file
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console Handler: writes to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger("BaseWord2Vec").info("Logging initialized. Output file: %s", log_path)
    return log_path


class BaseWord2Vec(ABC):
    """
    Abstract base pipeline class for Word2Vec architectures (Skip-Gram, Continuous Bag-of-Words).

    Encapsulates shared functionality:
      - Centralized logging initialization
      - Vocabulary inspection & property access
      - Corpus directory loading & document boundary parsing
      - Custom PyTorch dataset attachment and validation
      - Vector extraction (get_embedding) with tied/untied weight handling
      - Semantic similarity queries (most_similar)
      - Multi-format embedding serialization (save_embeddings)
      - Epoch-level training orchestration with DataLoader & Adam optimizer
    """

    _logger_name: str = "BaseWord2Vec"
    _default_export_filename: str = "word2vec_embeddings.txt"

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        dataset: Dataset | None = None,
        vocab_size: int | None = None,
        embedding_dim: int = 128,
        window_size: int = 2,
        n_negative: int = 5,
        min_count: int = 2,
        tied_weights: bool = True,
        max_tokens: int | None = None,
        sync_corpora_hook: bool = False,
        device: str | torch.device | None = None,
        tokenizer: BaseTokenizer | None = None,
        log_dir: str | Path = "./logs",
        enable_mlflow: bool = True,
        mlflow_config: str | Path | dict | None = None,
    ):
        if not logging.getLogger().handlers:
            setup_logging(log_dir=log_dir)

        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.n_negative = n_negative
        self.min_count = min_count
        self.tied_weights = tied_weights
        self.max_tokens = max_tokens
        self.sync_corpora_hook = sync_corpora_hook
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.enable_mlflow = enable_mlflow
        self.mlflow_config = mlflow_config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.logger = logging.getLogger(self._logger_name)
        self.logger.info(
            "Initializing %s on device '%s' (embedding_dim=%d, window_size=%d, n_negative=%d, tied_weights=%s)",
            self._logger_name,
            self.device,
            embedding_dim,
            window_size,
            n_negative,
            tied_weights,
        )

        self.tokenizer: BaseTokenizer = tokenizer or BasicTokenizer()
        self.corpus_files: list[str] = []
        self.documents: list[list[str]] = []
        self.tokens: list[str] = []
        self.dataset: Dataset | None = None
        self.model: nn.Module | None = None
        self.history: list[float] = []

        if dataset is not None:
            self.set_dataset(dataset, vocab_size=vocab_size)
        elif corpus_dir is not None:
            self.dataset = self._create_dataset()
            self.documents = getattr(self.dataset, "documents", [])
            self.tokens = getattr(self.dataset, "tokens", [])
            self.corpus_files = getattr(self.dataset, "corpus_files", [])
            self.init_model()

    @abstractmethod
    def _create_dataset(self) -> Dataset:
        """Instantiates the architecture-specific dataset from corpus_dir."""
        pass

    @abstractmethod
    def _create_dataset_from_docs(self, docs: list[list[str]]) -> Dataset:
        """Instantiates the architecture-specific dataset from a list of documents."""
        pass

    @abstractmethod
    def _create_model(self, vocab_size: int) -> nn.Module:
        """Instantiates the architecture-specific PyTorch model."""
        pass

    @abstractmethod
    def _forward_batch(self, batch: Any) -> torch.Tensor:
        """Unpacks batch data, transfers to device, and computes batch loss."""
        pass

    def set_dataset(self, dataset: Dataset, vocab_size: int | None = None) -> None:
        """
        Sets a custom PyTorch Dataset and initializes the underlying model.
        """
        self.dataset = dataset

        resolved_vocab_size = vocab_size
        if resolved_vocab_size is None and hasattr(dataset, "vocab_size"):
            resolved_vocab_size = dataset.vocab_size
        elif resolved_vocab_size is None and hasattr(dataset, "stoi"):
            resolved_vocab_size = len(dataset.stoi)

        if resolved_vocab_size is None:
            self.logger.error("Could not infer `vocab_size` from custom dataset.")
            raise ValueError(
                "Could not infer `vocab_size` from the custom dataset. "
                f"Please pass `vocab_size` explicitly to {self.__class__.__name__}(dataset=..., vocab_size=...)."
            )

        dataset_len = len(dataset) if isinstance(dataset, Sized) else 0
        self.logger.info("Attached custom dataset: vocab_size=%d, length=%d.", resolved_vocab_size, dataset_len)
        self.model = self._create_model(vocab_size=resolved_vocab_size).to(self.device)

    def load_corpus_from_dir(self, corpus_dir: str | Path, max_tokens: int | None = None) -> list[str]:
        """
        Reads and tokenizes each text file (*.txt) inside corpus_dir independently.
        Stores individual document token lists in `self.documents` so that
        the sliding context window strictly respects document boundaries.
        """
        corpus_path = Path(corpus_dir)
        if not corpus_path.exists() or not corpus_path.is_dir():
            self.logger.error("Corpus directory not found: '%s'", corpus_dir)
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

        txt_files = sorted(list(corpus_path.glob("*.txt")))
        if not txt_files:
            self.logger.error(
                "Corpus directory '%s' does not contain any text files (*.txt) to train embeddings.",
                corpus_dir,
            )
            self.logger.info(
                "Automatic download of corpora is disabled by default. "
                "You can automatically download sample text files using the '--sync-corpora' parameter "
                "(or pass 'sync_corpora_hook=True' / 'download=True' in Python)."
            )
            self.logger.info('Example command:\n  w2v --corpus-dir "%s" --sync-corpora', corpus_dir)
            raise ValueError(
                f"Corpus directory '{corpus_dir}' does not contain any text files (*.txt) to train embeddings. "
                f"Enable automatic download using '--sync-corpora' (or pass sync_corpora_hook=True). "
                f'Example command: w2v --corpus-dir "{corpus_dir}" --sync-corpora'
            )

        self.corpus_files = [str(f) for f in txt_files]
        self.documents = []
        all_tokens = []

        self.logger.info("Loading corpus from directory '%s': %d document file(s) found.", corpus_dir, len(txt_files))

        for fpath in txt_files:
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                self.logger.warning("Failed reading '%s': %s. Skipping document.", fpath.name, e)
                continue

            file_tokens = self.tokenizer.tokenize(content)
            self.logger.info(
                "Parsed document '%s': %d tokens (%d characters).", fpath.name, len(file_tokens), len(content)
            )
            self.documents.append(file_tokens)
            all_tokens.extend(file_tokens)

        # Apply token budget cap if requested, slicing documents cleanly
        if max_tokens is not None and len(all_tokens) > max_tokens:
            self.logger.info("Applying max_tokens=%d cap (from %d total tokens).", max_tokens, len(all_tokens))
            truncated_docs = []
            accumulated = 0
            for d in self.documents:
                if accumulated + len(d) <= max_tokens:
                    truncated_docs.append(d)
                    accumulated += len(d)
                else:
                    remaining = max_tokens - accumulated
                    if remaining > 0:
                        truncated_docs.append(d[:remaining])
                    break
            self.documents = truncated_docs
            all_tokens = [tok for d in self.documents for tok in d]

        self.logger.info(
            "Corpus ingestion complete: %d document(s) with %d total tokens.", len(self.documents), len(all_tokens)
        )
        self.tokens = all_tokens
        return all_tokens

    def build_dataset(self) -> Dataset:
        """
        Builds the dataset using the list of distinct documents (`self.documents`).
        """
        if self.dataset is not None:
            return self.dataset

        if not self.documents and not self.tokens:
            self.logger.error("Cannot build dataset: no documents or tokens loaded.")
            raise ValueError("No tokens available. Please load corpus files first.")

        docs = self.documents if self.documents else [self.tokens]
        self.dataset = self._create_dataset_from_docs(docs)
        self.logger.info(
            "Dataset built: vocabulary size = %d, total training pairs/samples = %d.",
            getattr(self.dataset, "vocab_size", len(getattr(self.dataset, "stoi", {}))),
            len(self.dataset) if isinstance(self.dataset, Sized) else 0,
        )
        return self.dataset

    def init_model(self) -> nn.Module:
        """
        Initializes the underlying PyTorch model based on the dataset's vocabulary size.
        """
        if self.dataset is None:
            self.logger.error("Dataset not built yet; cannot initialize model.")
            raise ValueError("Dataset not built yet.")

        vocab_size = getattr(self.dataset, "vocab_size", None)
        if vocab_size is None and hasattr(self.dataset, "stoi"):
            vocab_size = len(self.dataset.stoi)
        if vocab_size is None:
            self.logger.error("Unable to determine vocab_size from dataset.")
            raise ValueError("Unable to determine vocab_size from dataset.")

        model = self._create_model(vocab_size=vocab_size).to(self.device)
        self.model = model
        self.logger.info(
            "%s instantiated on '%s' with %d vocabulary entries.",
            self.model.__class__.__name__,
            self.device,
            vocab_size,
        )
        return model

    def train(
        self,
        epochs: int = 5,
        batch_size: int = 512,
        lr: float = 0.005,
        num_workers: int = 0,
        enable_mlflow: bool | None = None,
        mlflow_config: str | Path | dict | None = None,
        run_name: str | None = None,
    ) -> list[float]:
        """
        Trains the model over the dataset across the specified number of epochs.
        Tracks hyperparameters, epoch losses, convergence figures, and model checkpoints in MLflow.
        """
        if self.sync_corpora_hook and self.corpus_dir is not None:
            ensure_nltk_gutenberg_corpora(self.corpus_dir)

        if self.model is None or self.dataset is None:
            self.logger.error("Model or dataset is not initialized before calling train().")
            raise ValueError("Model and dataset must be initialized before training.")

        # Resolve MLflow tracking configuration
        use_mlflow = self.enable_mlflow if enable_mlflow is None else enable_mlflow
        resolved_config = mlflow_config if mlflow_config is not None else self.mlflow_config

        tracker = MLflowTracker(
            model_name=self._logger_name,
            enabled=use_mlflow,
            config=resolved_config,
        )

        self.model.train()
        collate_fn = getattr(self.dataset, "collate_fn", None)
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        history: list[float] = []
        self.logger.info(
            "Beginning %s training: %d epochs, batch_size=%d, lr=%.5f on device '%s'.",
            self._logger_name,
            epochs,
            batch_size,
            lr,
            self.device,
        )

        if tracker.enabled:
            # Inform user how to view tracking UI
            self.logger.info("=" * 70)
            self.logger.info("MLflow Tracking active (Storage: %s)", tracker.backend_store_uri)
            self.logger.info("Macro-Experiment: '%s'", tracker.experiment_name)
            self.logger.info("To monitor metrics and visual plots in real-time, run: 'w2v mlflow start'")
            self.logger.info("=" * 70)

            # Start active run and log hyperparameters
            params = {
                "model_architecture": self._logger_name,
                "embedding_dim": self.embedding_dim,
                "window_size": self.window_size,
                "n_negative": self.n_negative,
                "min_count": self.min_count,
                "tied_weights": self.tied_weights,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "optimizer": "Adam",
                "device": str(self.device),
                "vocab_size": self.vocab_size,
                "num_documents": len(self.documents),
                "total_tokens": len(self.tokens),
                "training_samples": len(self.dataset) if isinstance(self.dataset, Sized) else 0,
            }
            tracker.start_run(params=params, run_name=run_name)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                optimizer.zero_grad()
                loss = self._forward_batch(batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            history.append(avg_loss)
            self.logger.info(
                "Epoch [%d/%d] - Average Loss: %.4f across %d batches.", epoch, epochs, avg_loss, num_batches
            )

            # Log epoch loss to MLflow
            if tracker.enabled:
                tracker.log_epoch_loss(epoch, avg_loss)

        elapsed = time.time() - start_time
        self.logger.info(
            "Training finished successfully in %.2f seconds (%.2f s/epoch).", elapsed, elapsed / max(epochs, 1)
        )
        self.history = history

        # Finalize MLflow tracking and log artifacts
        if tracker.enabled:
            tracker.log_summary_metrics(history, elapsed)
            tracker.log_figures(history, self, self.dataset)
            tracker.log_model_checkpoint(self.model)
            run_id = tracker.run_id
            tracker.end_run()

            self.logger.info("=" * 70)
            self.logger.info("MLflow Run completed (Run ID: %s)", run_id)
            self.logger.info("To inspect run metrics, loss curves, and embedding figures, launch the UI:")
            self.logger.info("    w2v mlflow start")
            self.logger.info("and open your browser at: http://127.0.0.1:5000")
            self.logger.info("=" * 70)

        return history

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size inferred from internal dataset or model."""
        if self.dataset is not None and hasattr(self.dataset, "vocab_size"):
            return self.dataset.vocab_size
        if self.model is not None and hasattr(self.model, "vocab_size"):
            return self.model.vocab_size
        return 0

    def _get_embedding_weights(self) -> torch.Tensor:
        """Returns the effective embedding weight matrix (handling tied/untied weights)."""
        if self.model is None:
            raise ValueError("Model is not initialized.")
        with torch.no_grad():
            in_weights = self.model.in_embedding.weight.detach()
            out_emb = getattr(self.model, "out_embedding", None)
            if not self.tied_weights and out_emb is not None and out_emb is not self.model.in_embedding:
                out_weights = out_emb.weight.detach()
                return (in_weights + out_weights) / 2.0
            return in_weights

    def get_embedding(self, word: str) -> np.ndarray | None:
        """
        Returns the learned embedding vector for a given word as a NumPy array.
        If tied_weights is False, returns the arithmetic mean of in_embedding and out_embedding.
        """
        if self.dataset is None or self.model is None:
            return None
        stoi = getattr(self.dataset, "stoi", None)
        if stoi is None:
            return None
        word = word.lower()
        if word not in stoi:
            return None
        idx = stoi[word]
        weights = self._get_embedding_weights()
        return weights[idx].cpu().numpy()

    def most_similar(self, word: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Finds the top_k most similar words using cosine similarity.
        If tied_weights is False, uses the mean of in_embedding and out_embedding matrices.
        """
        if self.dataset is None or self.model is None:
            return []
        stoi = getattr(self.dataset, "stoi", None)
        itos = getattr(self.dataset, "itos", None)
        if stoi is None or itos is None:
            self.logger.warning("Dataset does not provide word-to-index (stoi) or index-to-word (itos) mapping.")
            return []

        word = word.lower()
        if word not in stoi:
            self.logger.debug("Word '%s' not found in vocabulary for similarity query.", word)
            return []

        target_idx = stoi[word]
        with torch.no_grad():
            weights = self._get_embedding_weights()
            norm = weights.norm(dim=1, keepdim=True).clamp(min=1e-8)
            norm_weights = weights / norm
            target_vec = norm_weights[target_idx]
            sims = torch.mv(norm_weights, target_vec)
            sims[target_idx] = -float("inf")
            top_vals, top_indices = torch.topk(sims, k=min(top_k, len(itos) - 1))

            results = []
            for val, idx in zip(top_vals.cpu().tolist(), top_indices.cpu().tolist()):
                results.append((itos[idx], float(val)))
            return results

    def _resolve_export_filename(
        self,
        filename: str | None,
        default_base: str,
        ext: str,
        dest: Path,
    ) -> str:
        """
        Resolves the export filename for embeddings.
        If an explicit filename is provided, it is returned unmodified.
        Otherwise, a timestamp in format YYYYddmm_HHMM is appended before the extension
        (e.g. 'word2vec_embeddings_20260111_2135.txt') to prevent overwriting previous runs.
        If a file with the same timestamp already exists, an incremental counter is appended.
        """
        if filename is not None:
            return filename

        timestamp = datetime.now().strftime("%Y%d%m_%H%M")
        base_name = f"{default_base}_{timestamp}"
        candidate = f"{base_name}{ext}"
        if (dest / candidate).exists():
            counter = 1
            while (dest / f"{base_name}_{counter}{ext}").exists():
                counter += 1
            return f"{base_name}_{counter}{ext}"
        return candidate

    def save_embeddings(
        self,
        output_dir: str | Path = "./embeddings",
        filename: str | None = None,
        format: str = "word2vec",
    ) -> Path:
        """
        Saves the learned word embeddings to the specified directory.
        If the destination directory does not exist, it is created automatically.
        If filename is None, a timestamp (YYYYddmm_HHMM) is embedded into the filename
        to prevent new executions from overwriting previous embeddings.
        """
        if self.model is None or self.dataset is None:
            self.logger.error("Cannot save embeddings: model or dataset is not initialized.")
            raise ValueError("Model or dataset is not initialized. Please train a model first.")

        itos = getattr(self.dataset, "itos", None)
        if not itos:
            self.logger.error("Cannot save embeddings: vocabulary (itos) is not available.")
            raise ValueError("Vocabulary mapping (itos) is missing from the dataset.")

        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)

        embeddings_matrix = self._get_embedding_weights().cpu().numpy()
        vocab_size, embedding_dim = embeddings_matrix.shape
        fmt = format.lower()
        default_base = Path(self._default_export_filename).stem

        if fmt in ("word2vec", "txt", "vec"):
            out_filename = self._resolve_export_filename(filename, default_base, ".txt", dest)
            filepath = dest / out_filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"{vocab_size} {embedding_dim}\n")
                for word, vec in zip(itos, embeddings_matrix):
                    vec_str = " ".join(f"{val:.6f}" for val in vec)
                    f.write(f"{word} {vec_str}\n")
        elif fmt == "glove":
            out_filename = self._resolve_export_filename(filename, "glove_embeddings", ".txt", dest)
            filepath = dest / out_filename
            with open(filepath, "w", encoding="utf-8") as f:
                for word, vec in zip(itos, embeddings_matrix):
                    vec_str = " ".join(f"{val:.6f}" for val in vec)
                    f.write(f"{word} {vec_str}\n")
        elif fmt in ("numpy", "npz"):
            out_filename = self._resolve_export_filename(filename, default_base, ".npz", dest)
            filepath = dest / out_filename
            np.savez_compressed(
                filepath,
                words=np.array(itos, dtype=object),
                vectors=embeddings_matrix.astype(np.float32),
            )
        elif fmt == "pt":
            out_filename = self._resolve_export_filename(filename, default_base, ".pt", dest)
            filepath = dest / out_filename
            torch.save(
                {
                    "words": itos,
                    "vectors": torch.tensor(embeddings_matrix, dtype=torch.float32),
                },
                filepath,
            )
        else:
            raise ValueError(
                f"Unsupported embedding format: '{format}'. Choose from 'word2vec', 'glove', 'numpy', 'pt'."
            )

        self.logger.info(
            "Word embeddings successfully saved to '%s' (%d words, dim=%d, format=%s).",
            filepath,
            vocab_size,
            embedding_dim,
            fmt,
        )
        return filepath
