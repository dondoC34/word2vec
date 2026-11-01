import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from w2v.base import BaseWord2Vec
from w2v.base import setup_logging as setup_logging
from w2v.data.dataset import BoWDataset

__all__ = [
    "CBOW",
    "BoW",
    "BoWModel",
    "CBOWModel",
    "setup_logging",
]

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


class BoW(BaseWord2Vec):
    """
    High-level Continuous Bag-of-Words (CBOW / BoW) model and training pipeline.
    Maintains the exact same parameter fingerprint as Word2Vec (Skip-Gram).
    Accepts either:
      1. A corpus directory (`corpus_dir`), from which individual documents are read,
         tokenized with a `BaseTokenizer`, and packaged into a boundary-respecting `BoWDataset`.
      2. A custom PyTorch `Dataset` (`dataset`), allowing pre-built datasets.
    """

    _logger_name: str = "BoW"
    _model_name: str = "bow"
    _default_export_filename: str = "w2v_bow_embeddings.txt"

    def _create_dataset(self) -> BoWDataset:
        return BoWDataset(
            corpus_dir=self.corpus_dir,
            download=self.sync_corpora_hook,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            window_size=self.window_size,
            n_negative=self.n_negative,
            min_count=self.min_count,
        )

    def _create_dataset_from_docs(self, docs: list[list[str]]) -> BoWDataset:
        return BoWDataset(
            documents=docs,
            window_size=self.window_size,
            n_negative=self.n_negative,
            min_count=self.min_count,
        )

    def _create_model(self, vocab_size: int) -> BoWModel:
        return BoWModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            tied_weights=self.tied_weights,
        )

    def _forward_batch(self, batch: Any) -> torch.Tensor:
        flat_contexts, offsets, targets, neg_samples = batch
        flat_contexts = flat_contexts.to(self.device)
        offsets = offsets.to(self.device)
        targets = targets.to(self.device)
        neg_samples = neg_samples.to(self.device)
        assert self.model is not None
        return self.model(flat_contexts, offsets, targets, neg_samples)


CBOW = BoW
