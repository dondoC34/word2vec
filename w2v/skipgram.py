import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from w2v.base import BaseWord2Vec
from w2v.base import setup_logging as setup_logging
from w2v.data.dataset import SkipGramDataset

__all__ = [
    "PairScalarModel",
    "SkipGram",
    "SkipGramModel",
    "Word2Vec",
    "get_model",
    "setup_logging",
]

logger = logging.getLogger("SkipGram")
model_logger = logging.getLogger("SkipGramModel")


class SkipGramModel(nn.Module):
    """
    PyTorch implementation of the Skip-Gram with Negative Sampling (SGNS) architecture.
    Computes dot-product similarity between central word and context / negative words,
    optimizing Negative Log-Likelihood via log-sigmoid loss.
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

        self.in_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        if tied_weights:
            self.out_embedding = self.in_embedding
        else:
            self.out_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        model_logger.debug(
            "Initialized SkipGramModel: vocab_size=%d, embedding_dim=%d, tied_weights=%s",
            vocab_size,
            embedding_dim,
            tied_weights,
        )

    @property
    def embedding(self) -> nn.Embedding:
        """Alias for backward compatibility with external code referencing model.embedding."""
        return self.in_embedding

    def forward(
        self,
        id_central_word: torch.LongTensor,
        id_context_word: torch.LongTensor,
        ids_out_of_context_words: torch.LongTensor,
    ) -> torch.Tensor:
        # central_emb: (batch_size, embedding_dim)
        central_emb = self.in_embedding(id_central_word)
        # context_emb: (batch_size, embedding_dim)
        context_emb = self.out_embedding(id_context_word)
        # out_of_context_embs: (batch_size, n_negative, embedding_dim)
        out_of_context_embs = self.out_embedding(ids_out_of_context_words)

        # Dot product between central and positive context embeddings -> (batch_size,)
        pos_score = torch.sum(central_emb * context_emb, dim=1)

        # Dot product between central and each negative sample -> (batch_size, n_negative)
        neg_score = torch.einsum("bi,bni->bn", central_emb, out_of_context_embs)

        # Standard SGNS (Mikolov et al., 2013) Negative Log-Likelihood:
        # L = - log(sigmoid(pos_score)) - sum(log(sigmoid(-neg_score)))
        loss = -F.logsigmoid(pos_score) - torch.sum(F.logsigmoid(-neg_score), dim=1)
        return loss.mean()


# Backward compatibility alias
PairScalarModel = SkipGramModel


class SkipGram(BaseWord2Vec):
    """
    High-level Skip-Gram with Negative Sampling (SGNS) model and training pipeline.
    Accepts either:
      1. A corpus directory (`corpus_dir`), from which individual documents are read,
         tokenized with a `BaseTokenizer`, and packaged into a boundary-respecting `SkipGramDataset`.
      2. A custom PyTorch `Dataset` (`dataset`), allowing users to provide their own
         pre-built dataset implementation.
    """

    _logger_name: str = "SkipGram"
    _default_export_filename: str = "word2vec_embeddings.txt"

    def _create_dataset(self) -> SkipGramDataset:
        return SkipGramDataset(
            corpus_dir=self.corpus_dir,
            download=self.sync_corpora_hook,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            window_size=self.window_size,
            n_negative=self.n_negative,
            min_count=self.min_count,
        )

    def _create_dataset_from_docs(self, docs: list[list[str]]) -> SkipGramDataset:
        return SkipGramDataset(
            documents=docs,
            window_size=self.window_size,
            n_negative=self.n_negative,
            min_count=self.min_count,
        )

    def _create_model(self, vocab_size: int) -> SkipGramModel:
        return SkipGramModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            tied_weights=self.tied_weights,
        )

    def _forward_batch(self, batch: Any) -> torch.Tensor:
        centers, contexts, neg_samples = batch
        centers = centers.to(self.device)
        contexts = contexts.to(self.device)
        neg_samples = neg_samples.to(self.device)
        assert self.model is not None
        return self.model(centers, contexts, neg_samples)


# Backward compatibility alias
Word2Vec = SkipGram


def get_model(vocab_size: int, embedding_dim: int = 128) -> SkipGramModel:
    """
    Returns an instance of the SkipGramModel (SGNS).
    """
    return SkipGramModel(vocab_size, embedding_dim)
