import logging
import time
from collections.abc import Sized
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from w2v.data.dataset import BoWDataset
from w2v.data.downloader import ensure_nltk_gutenberg_corpora
from w2v.preprocessing.tokenizer import BaseTokenizer, BasicTokenizer
from w2v.skipgram import setup_logging

logger = logging.getLogger("BoW")
model_logger = logging.getLogger("BoWModel")


class BoWModel(nn.Module):
    """
    PyTorch implementation of the Continuous Bag-of-Words (CBOW / BoW) architecture
    with Negative Sampling (Mikolov et al., 2013).

    Predicts the central target word given the mean embedding vector of its context words:
      context_words (bag) -> center_word (target)
    Optimizes Negative Log-Likelihood with log-sigmoid negative sampling.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        tied_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tied_weights = tied_weights

        # Input context embeddings: averaged via native PyTorch EmbeddingBag (fast & memory-efficient)
        self.in_embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            mode="mean",
        )

        if tied_weights:
            self.out_embedding = None
        else:
            self.out_embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
            )

        model_logger.debug(
            "Initialized BoWModel: vocab_size=%d, embedding_dim=%d, tied_weights=%s",
            vocab_size,
            embedding_dim,
            tied_weights,
        )

    @property
    def embedding(self) -> nn.EmbeddingBag:
        """Alias for backward compatibility."""
        return self.in_embedding

    def forward(
        self,
        flat_contexts: torch.LongTensor,
        offsets: torch.LongTensor,
        targets: torch.LongTensor,
        neg_samples: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Forward pass for Continuous Bag-of-Words with Negative Sampling.

        Args:
            flat_contexts: 1D LongTensor of all context tokens across the batch.
            offsets: 1D LongTensor of start positions in flat_contexts for each sample.
            targets: 1D LongTensor of true center target tokens (batch_size,).
            neg_samples: 2D LongTensor of drawn negative token IDs (batch_size, n_negative).

        Returns:
            Mean negative log-likelihood loss for the batch.
        """
        # 1. Mean context embedding for each sample in batch: (batch_size, embedding_dim)
        ctx_emb = self.in_embedding(flat_contexts, offsets)

        # 2. Positive target word embeddings
        if self.tied_weights or self.out_embedding is None:
            target_emb = F.embedding(targets, self.in_embedding.weight)
            neg_embs = F.embedding(neg_samples, self.in_embedding.weight)
        else:
            target_emb = self.out_embedding(targets)
            neg_embs = self.out_embedding(neg_samples)

        # 3. Dot product between mean context embedding and positive target: (batch_size,)
        pos_score = torch.sum(ctx_emb * target_emb, dim=1)

        # 4. Dot product between mean context embedding and negative samples: (batch_size, n_negative)
        neg_score = torch.einsum("bd,bnd->bn", ctx_emb, neg_embs)

        # 5. SGNS / CBOW Negative Sampling objective (Mikolov et al., 2013)
        loss = -F.logsigmoid(pos_score) - torch.sum(F.logsigmoid(-neg_score), dim=1)
        return loss.mean()


CBOWModel = BoWModel


class BoW:
    """
    High-level Continuous Bag-of-Words (CBOW / BoW) model and training pipeline.
    Maintains the exact same parameter fingerprint as Word2Vec (Skip-Gram).
    Accepts either:
      1. A corpus directory (`corpus_dir`), from which individual documents are read,
         tokenized with a `BaseTokenizer`, and packaged into a boundary-respecting `BoWDataset`.
      2. A custom PyTorch `Dataset` (`dataset`), allowing pre-built datasets.
    """

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
        sync_corpora_hook: bool = True,
        device: str | torch.device | None = None,
        tokenizer: BaseTokenizer | None = None,
        log_dir: str | Path = "./logs",
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

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            "Initializing BoW on device '%s' (embedding_dim=%d, window_size=%d, n_negative=%d, tied_weights=%s)",
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
        self.model: BoWModel | None = None

        if dataset is not None:
            self.set_dataset(dataset, vocab_size=vocab_size)
        elif corpus_dir is not None:
            self.dataset = BoWDataset(
                corpus_dir=self.corpus_dir,
                download=self.sync_corpora_hook,
                tokenizer=self.tokenizer,
                max_tokens=self.max_tokens,
                window_size=self.window_size,
                n_negative=self.n_negative,
                min_count=self.min_count,
            )
            self.documents = getattr(self.dataset, "documents", [])
            self.tokens = getattr(self.dataset, "tokens", [])
            self.corpus_files = getattr(self.dataset, "corpus_files", [])
            self.init_model()

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size inferred from internal dataset or model."""
        if self.dataset is not None and hasattr(self.dataset, "vocab_size"):
            return self.dataset.vocab_size
        if self.model is not None:
            return self.model.vocab_size
        return 0

    def set_dataset(self, dataset: Dataset, vocab_size: int | None = None) -> None:
        """
        Sets a custom PyTorch Dataset and initializes the underlying BoWModel.
        """
        self.dataset = dataset

        resolved_vocab_size = vocab_size
        if resolved_vocab_size is None and hasattr(dataset, "vocab_size"):
            resolved_vocab_size = dataset.vocab_size
        elif resolved_vocab_size is None and hasattr(dataset, "stoi"):
            resolved_vocab_size = len(dataset.stoi)

        if resolved_vocab_size is None:
            logger.error("Could not infer `vocab_size` from custom dataset.")
            raise ValueError(
                "Could not infer `vocab_size` from the custom dataset. "
                "Please pass `vocab_size` explicitly to BoW(dataset=..., vocab_size=...)."
            )

        dataset_len = len(dataset) if isinstance(dataset, Sized) else 0
        logger.info("Attached custom dataset to BoW: vocab_size=%d, length=%d.", resolved_vocab_size, dataset_len)
        self.model = BoWModel(
            vocab_size=resolved_vocab_size,
            embedding_dim=self.embedding_dim,
            tied_weights=self.tied_weights,
        ).to(self.device)

    def load_corpus_from_dir(self, corpus_dir: str | Path, max_tokens: int | None = None) -> list[str]:
        """
        Reads and tokenizes each text file (*.txt) inside corpus_dir independently.
        Stores individual document token lists in `self.documents` so that
        the sliding context window strictly respects document boundaries.
        """
        corpus_path = Path(corpus_dir)
        corpus_path.mkdir(parents=True, exist_ok=True)
        if self.sync_corpora_hook:
            ensure_nltk_gutenberg_corpora(corpus_path)

        txt_files = sorted(list(corpus_path.glob("*.txt")))
        if not txt_files:
            txt_files = [f for f in sorted(list(corpus_path.iterdir())) if f.is_file()]

        if not txt_files:
            logger.error("No text files found in corpus directory: '%s'", corpus_dir)
            raise ValueError(f"No text files found in corpus directory: {corpus_dir}")

        self.corpus_files = [str(f) for f in txt_files]
        self.documents = []
        all_tokens = []

        logger.info(
            "Loading corpus for BoW from directory '%s': %d document file(s) found.", corpus_dir, len(txt_files)
        )

        for fpath in txt_files:
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                logger.warning("Failed reading '%s': %s. Skipping document.", fpath.name, e)
                continue

            file_tokens = self.tokenizer.tokenize(content)
            logger.info("Parsed document '%s': %d tokens (%d characters).", fpath.name, len(file_tokens), len(content))
            self.documents.append(file_tokens)
            all_tokens.extend(file_tokens)

        if max_tokens is not None and len(all_tokens) > max_tokens:
            logger.info("Applying max_tokens=%d cap (from %d total tokens).", max_tokens, len(all_tokens))
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

        logger.info(
            "Corpus ingestion complete: %d document(s) with %d total tokens.", len(self.documents), len(all_tokens)
        )
        self.tokens = all_tokens
        return all_tokens

    def build_dataset(self) -> BoWDataset:
        """
        Builds the BoWDataset using the list of distinct documents (`self.documents`).
        Guarantees that the sliding context window never crosses document boundaries.
        """
        if self.dataset is not None and isinstance(self.dataset, BoWDataset):
            return self.dataset

        if not self.documents and not self.tokens:
            logger.error("Cannot build dataset: no documents or tokens loaded.")
            raise ValueError("No tokens available. Please load corpus files first.")

        logger.info(
            "Building BoW dataset: window_size=%d, n_negative=%d, min_count=%d across %d document(s).",
            self.window_size,
            self.n_negative,
            self.min_count,
            len(self.documents),
        )

        self.dataset = BoWDataset(
            documents=self.documents if self.documents else [self.tokens],
            window_size=self.window_size,
            n_negative=self.n_negative,
            min_count=self.min_count,
        )
        logger.info(
            "BoW Dataset built: vocabulary size = %d, total training samples = %d.",
            self.dataset.vocab_size,
            len(self.dataset),
        )
        return self.dataset

    def init_model(self) -> BoWModel:
        """
        Initializes the underlying BoWModel based on the dataset's vocabulary size.
        """
        if self.dataset is None:
            logger.error("Dataset not built yet; cannot initialize BoW model.")
            raise ValueError("Dataset not built yet.")

        vocab_size = getattr(self.dataset, "vocab_size", None)
        if vocab_size is None and hasattr(self.dataset, "stoi"):
            vocab_size = len(self.dataset.stoi)
        if vocab_size is None:
            logger.error("Unable to determine vocab_size from dataset.")
            raise ValueError("Unable to determine vocab_size from dataset.")

        model = BoWModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            tied_weights=self.tied_weights,
        ).to(self.device)
        self.model = model
        logger.info("BoWModel instantiated on '%s' with %d vocabulary entries.", self.device, vocab_size)
        return model

    def train(
        self,
        epochs: int = 5,
        batch_size: int = 512,
        lr: float = 0.005,
        num_workers: int = 0,
    ) -> list[float]:
        """
        Trains the BoW model over the dataset across the specified number of epochs.
        """
        if self.sync_corpora_hook and self.corpus_dir is not None:
            ensure_nltk_gutenberg_corpora(self.corpus_dir)

        if self.model is None or self.dataset is None:
            logger.error("Model or dataset is not initialized before calling train().")
            raise ValueError("Model and dataset must be initialized before training.")

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
        history = []
        logger.info(
            "Beginning BoW training: %d epochs, batch_size=%d, lr=%.5f on device '%s'.",
            epochs,
            batch_size,
            lr,
            self.device,
        )
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            num_batches = 0
            for flat_contexts, offsets, targets, neg_samples in dataloader:
                flat_contexts = flat_contexts.to(self.device)
                offsets = offsets.to(self.device)
                targets = targets.to(self.device)
                neg_samples = neg_samples.to(self.device)

                optimizer.zero_grad()
                loss = self.model(flat_contexts, offsets, targets, neg_samples)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            history.append(avg_loss)
            logger.info("Epoch [%d/%d] - Average Loss: %.4f across %d batches.", epoch, epochs, avg_loss, num_batches)

        elapsed = time.time() - start_time
        logger.info(
            "BoW Training finished successfully in %.2f seconds (%.2f s/epoch).", elapsed, elapsed / max(epochs, 1)
        )
        return history

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
        with torch.no_grad():
            in_emb = self.model.in_embedding.weight[idx]
            if not self.tied_weights and hasattr(self.model, "out_embedding") and self.model.out_embedding is not None:
                out_emb = self.model.out_embedding.weight[idx]
                emb = ((in_emb + out_emb) / 2.0).cpu().numpy()
            else:
                emb = in_emb.cpu().numpy()
        return emb

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
            logger.warning("Dataset does not provide word-to-index (stoi) or index-to-word (itos) mapping.")
            return []

        word = word.lower()
        if word not in stoi:
            logger.debug("Word '%s' not found in vocabulary for similarity query.", word)
            return []

        target_idx = stoi[word]
        with torch.no_grad():
            if not self.tied_weights and hasattr(self.model, "out_embedding") and self.model.out_embedding is not None:
                weights = (self.model.in_embedding.weight.detach() + self.model.out_embedding.weight.detach()) / 2.0
            else:
                weights = self.model.in_embedding.weight.detach()

            norm = weights.norm(dim=1, keepdim=True).clamp(min=1e-8)
            norm_weights = weights / norm
            target_vec = norm_weights[target_idx]  # (D,)
            sims = torch.mv(norm_weights, target_vec)  # (V,)
            sims[target_idx] = -float("inf")  # exclude word itself
            top_vals, top_indices = torch.topk(sims, k=min(top_k, len(itos) - 1))

            results = []
            for val, idx in zip(top_vals.cpu().tolist(), top_indices.cpu().tolist()):
                results.append((itos[idx], float(val)))
            return results

    def save_embeddings(
        self,
        output_dir: str | Path = "./embeddings",
        filename: str | None = None,
        format: str = "word2vec",
    ) -> Path:
        """
        Saves the learned word embeddings to the specified directory.
        If the destination directory does not exist, it is created automatically.

        If tied_weights is False, saves the arithmetic mean of in_embedding
        and out_embedding vectors for each token ((W_in + W_out) / 2.0).
        """
        if self.model is None or self.dataset is None:
            logger.error("Cannot save embeddings: model or dataset is not initialized.")
            raise ValueError("Model or dataset is not initialized. Please train a model first.")

        itos = getattr(self.dataset, "itos", None)
        if not itos:
            logger.error("Cannot save embeddings: vocabulary (itos) is not available.")
            raise ValueError("Vocabulary mapping (itos) is missing from the dataset.")

        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            in_weights = self.model.in_embedding.weight.detach().cpu().numpy()
            if not self.tied_weights and hasattr(self.model, "out_embedding") and self.model.out_embedding is not None:
                out_weights = self.model.out_embedding.weight.detach().cpu().numpy()
                embeddings_matrix = (in_weights + out_weights) / 2.0
                logger.info(
                    "Computed arithmetic mean of in_embedding and out_embedding across %d tokens (tied_weights=False).",
                    len(itos),
                )
            else:
                embeddings_matrix = in_weights
                logger.info("Using in_embedding weights across %d tokens (tied_weights=True).", len(itos))

        vocab_size, embedding_dim = embeddings_matrix.shape
        fmt = format.lower()

        if fmt in ("word2vec", "txt", "vec"):
            out_filename = filename or "bow_embeddings.txt"
            filepath = dest / out_filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"{vocab_size} {embedding_dim}\n")
                for word, vec in zip(itos, embeddings_matrix):
                    vec_str = " ".join(f"{val:.6f}" for val in vec)
                    f.write(f"{word} {vec_str}\n")
        elif fmt == "glove":
            out_filename = filename or "glove_embeddings.txt"
            filepath = dest / out_filename
            with open(filepath, "w", encoding="utf-8") as f:
                for word, vec in zip(itos, embeddings_matrix):
                    vec_str = " ".join(f"{val:.6f}" for val in vec)
                    f.write(f"{word} {vec_str}\n")
        elif fmt in ("numpy", "npz"):
            out_filename = filename or "bow_embeddings.npz"
            filepath = dest / out_filename
            np.savez_compressed(
                filepath,
                words=np.array(itos, dtype=object),
                vectors=embeddings_matrix.astype(np.float32),
            )
        elif fmt == "pt":
            out_filename = filename or "bow_embeddings.pt"
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

        logger.info(
            "BoW Word embeddings successfully saved to '%s' (%d words, dim=%d, format=%s).",
            filepath,
            vocab_size,
            embedding_dim,
            fmt,
        )
        return filepath


CBOW = BoW
