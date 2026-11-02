import logging
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from w2v.data.downloader import ensure_nltk_gutenberg_corpora
from w2v.preprocessing.tokenizer import BaseTokenizer, BasicTokenizer

logger = logging.getLogger("Dataset")


class BaseWord2VecDataset(Dataset, ABC):
    """
    Abstract base class for Word2Vec datasets.
    Extends torch.utils.data.Dataset with unified corpus ingestion/download,
    document boundary isolation, vocabulary construction, and unigram negative sampling distribution.
    """

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        download: bool = False,
        tokenizer: BaseTokenizer | None = None,
        max_tokens: int | None = None,
        corpus: list[str] | list[list[str]] | None = None,
        tokens: list[str] | None = None,
        documents: list[list[str]] | None = None,
        window_size: int = 2,
        n_negative: int = 5,
        min_count: int = 1,
    ):
        """
        Initialize the common Word2Vec dataset components.

        Args:
            corpus_dir: Optional directory containing text files (*.txt) to ingest.
            download: If True and corpus_dir is provided, ensures standard NLTK corpora are downloaded.
            tokenizer: Tokenizer instance (defaults to BasicTokenizer()).
            max_tokens: Optional limit on total tokens across all documents.
            corpus: A list of tokens (single document) or a list of token lists (multiple documents).
            tokens: Alias for a single list of tokens (kept for backward compatibility).
            documents: Explicit list of document token lists.
            window_size: Half-window size for context words (e.g. 2 means 2 words left and 2 words right).
            n_negative: Number of negative samples drawn per positive sample.
            min_count: Words appearing fewer than this threshold are discarded.
        """
        self.window_size = window_size
        self.n_negative = n_negative
        self.min_count = min_count
        self.corpus_files: list[str] = []

        docs: list[list[str]] = []

        # Case 1: Ingest from corpus_dir directly (PyTorch dataset pattern)
        if corpus_dir is not None:
            corpus_path = Path(corpus_dir)
            existing_txt = (
                sorted(list(corpus_path.glob("*.txt"))) if corpus_path.exists() and corpus_path.is_dir() else []
            )

            if not existing_txt:
                logger.info(
                    "Corpus directory '%s' does not exist or does not contain any text files (*.txt). "
                    "Automatically triggering download of sample NLTK Gutenberg corpora into '%s'...",
                    corpus_dir,
                    corpus_dir,
                )
                corpus_path.mkdir(parents=True, exist_ok=True)
                ensure_nltk_gutenberg_corpora(corpus_path)
                txt_files = sorted(list(corpus_path.glob("*.txt")))
            else:
                if download:
                    ensure_nltk_gutenberg_corpora(corpus_path)
                txt_files = sorted(list(corpus_path.glob("*.txt")))

            if not txt_files:
                logger.error(
                    "Corpus directory '%s' does not contain any text files (*.txt) and automatic sample download could not provide any.",
                    corpus_dir,
                )
                raise ValueError(
                    f"Corpus directory '{corpus_dir}' does not contain any text files (*.txt) to train embeddings."
                )

            self.corpus_files = [str(f) for f in txt_files]
            tok = tokenizer or BasicTokenizer()

            logger.info("Loading corpus from directory '%s': %d document file(s) found.", corpus_dir, len(txt_files))
            for fpath in txt_files:
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception as e:
                    logger.warning("Failed reading '%s': %s. Skipping document.", fpath.name, e)
                    continue

                file_tokens = tok.tokenize(content)
                logger.info(
                    "Parsed document '%s': %d tokens (%d characters).", fpath.name, len(file_tokens), len(content)
                )
                docs.append(file_tokens)

            # Apply token budget cap if requested, slicing documents cleanly
            if max_tokens is not None:
                total_tok = sum(len(d) for d in docs)
                if total_tok > max_tokens:
                    logger.info("Applying max_tokens=%d cap (from %d total tokens).", max_tokens, total_tok)
                    truncated_docs = []
                    accumulated = 0
                    for d in docs:
                        if accumulated + len(d) <= max_tokens:
                            truncated_docs.append(d)
                            accumulated += len(d)
                        else:
                            remaining = max_tokens - accumulated
                            if remaining > 0:
                                truncated_docs.append(d[:remaining])
                            break
                    docs = truncated_docs

        # Case 2: Ingest from explicit token lists / documents
        elif documents is not None:
            docs = documents
        elif corpus is not None:
            if len(corpus) > 0 and isinstance(corpus[0], list):
                docs = corpus  # type: ignore[assignment]
            else:
                docs = [corpus]  # type: ignore[list-item]
        elif tokens is not None:
            docs = [tokens]
        else:
            raise ValueError(
                f"Must provide either 'corpus_dir', 'corpus', 'documents', or 'tokens' to {self.__class__.__name__}."
            )

        self.documents: list[list[str]] = docs
        self.tokens: list[str] = [tok for d in docs for tok in d]
        self.num_documents = len(docs)
        total_raw_tokens = len(self.tokens)
        logger.info(
            "Initializing %s: %d document(s) with %d total raw tokens.",
            self.__class__.__name__,
            self.num_documents,
            total_raw_tokens,
        )

        # Step 1: Compute global vocabulary frequencies across all documents
        raw_counts: Counter = Counter()
        for doc in docs:
            raw_counts.update(doc)

        # Filter words by minimum frequency
        filtered_words = [w for w, count in raw_counts.items() if count >= min_count]
        filtered_counts = {w: raw_counts[w] for w in filtered_words}

        self.itos: list[str] = list(filtered_counts.keys())
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}
        self._vocab_size: int = len(self.itos)

        if self._vocab_size == 0:
            logger.error("Vocabulary is empty after applying min_count=%d threshold.", min_count)
            raise ValueError(f"Vocabulary is empty after applying min_count={min_count} filter.")

        logger.info(
            "Vocabulary built: %d unique words retained (min_count=%d, discarded %d rare words).",
            self._vocab_size,
            min_count,
            len(raw_counts) - self._vocab_size,
        )

        # Step 2: Unigram distribution raised to the 0.75 power for negative sampling (Mikolov et al.)
        freqs = np.array([filtered_counts[w] for w in self.itos], dtype=np.float64)
        unigram_dist = freqs**0.75
        self.neg_sampling_dist = unigram_dist / unigram_dist.sum()
        self.neg_dist_tensor = torch.tensor(self.neg_sampling_dist, dtype=torch.float32)

    @property
    def vocab_size(self) -> int:
        """Total number of unique tokens in the vocabulary."""
        return self._vocab_size

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass


class SkipGramDataset(BaseWord2VecDataset):
    """
    Skip-Gram dataset generating (center, context) pairs within a sliding context window.
    Supports multi-document corpora (List[List[str]]), guaranteeing that the sliding context
    window NEVER crosses document boundaries. The end of one text and the beginning of another
    will never generate spurious cross-document training pairs.

    Standard PyTorch dataset paradigm:
    Optionally accepts `corpus_dir` and `download=True` to automatically fetch/synchronize
    standard benchmark literary corpora on disk before ingestion.
    """

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        download: bool = False,
        tokenizer: BaseTokenizer | None = None,
        max_tokens: int | None = None,
        corpus: list[str] | list[list[str]] | None = None,
        tokens: list[str] | None = None,
        documents: list[list[str]] | None = None,
        window_size: int = 2,
        n_negative: int = 5,
        min_count: int = 1,
    ):
        """
        Initialize the SkipGramDataset.

        Args:
            corpus_dir: Optional directory containing text files (*.txt) to ingest.
            download: If True and corpus_dir is provided, ensures standard NLTK corpora are downloaded.
            tokenizer: Tokenizer instance (defaults to BasicTokenizer()).
            max_tokens: Optional limit on total tokens across all documents.
            corpus: A list of tokens (single document) or a list of token lists (multiple documents).
            tokens: Alias for a single list of tokens (kept for backward compatibility).
            documents: Explicit list of document token lists.
            window_size: Half-window size for context words (e.g. 2 means 2 words left and 2 words right).
            n_negative: Number of negative samples drawn per positive pair.
            min_count: Words appearing fewer than this threshold are discarded.
        """
        super().__init__(
            corpus_dir=corpus_dir,
            download=download,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            corpus=corpus,
            tokens=tokens,
            documents=documents,
            window_size=window_size,
            n_negative=n_negative,
            min_count=min_count,
        )

        # Generate center-context pairs per document
        # CRITICAL: We iterate document by document so that the sliding window
        # is strictly bounded within each document. No pairs cross from doc N to doc N+1.
        self.pairs: list[tuple[int, int]] = []
        for doc in self.documents:
            # Map tokens in the current document to their integer vocabulary IDs
            doc_ids = [self.stoi[t] for t in doc if t in self.stoi]
            doc_len = len(doc_ids)

            for center_i in range(doc_len):
                center_word = doc_ids[center_i]
                start = max(0, center_i - self.window_size)
                end = min(doc_len, center_i + self.window_size + 1)
                for ctx_i in range(start, end):
                    if ctx_i != center_i:
                        self.pairs.append((center_word, doc_ids[ctx_i]))

        logger.info(
            "Skip-Gram pair generation complete: %d pairs generated across %d document(s) "
            "with zero boundary cross-contamination.",
            len(self.pairs),
            self.num_documents,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        """
        Fast O(1) lookup returning the (center_id, context_id) pair.
        Negative samples are generated in bulk at the batch level via collate_fn.
        """
        return self.pairs[idx]

    def collate_fn(self, batch: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized batch collation for high-performance training with DataLoader.
        Draws negative samples in bulk via a single C++/CUDA torch.multinomial kernel call.
        """
        centers = torch.tensor([item[0] for item in batch], dtype=torch.long)
        contexts = torch.tensor([item[1] for item in batch], dtype=torch.long)
        batch_size = len(batch)
        neg_samples = torch.multinomial(
            self.neg_dist_tensor,
            batch_size * self.n_negative,
            replacement=True,
        ).view(batch_size, self.n_negative)
        return centers, contexts, neg_samples


class BoWDataset(BaseWord2VecDataset):
    """
    Continuous Bag-of-Words (CBOW / BoW) dataset.
    Generates (context_tokens_list, target_center_token) pairs within a sliding context window.
    Guarantees document boundary protection: the sliding context window never crosses
    document boundaries. The beginning or end of one text never contaminates another text.
    """

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        download: bool = False,
        tokenizer: BaseTokenizer | None = None,
        max_tokens: int | None = None,
        corpus: list[str] | list[list[str]] | None = None,
        tokens: list[str] | None = None,
        documents: list[list[str]] | None = None,
        window_size: int = 2,
        n_negative: int = 5,
        min_count: int = 1,
    ):
        """
        Initialize the BoWDataset.

        Args:
            corpus_dir: Optional directory containing text files (*.txt) to ingest.
            download: If True and corpus_dir is provided, ensures standard NLTK corpora are downloaded.
            tokenizer: Tokenizer instance (defaults to BasicTokenizer()).
            max_tokens: Optional limit on total tokens across all documents.
            corpus: A list of tokens (single document) or a list of token lists (multiple documents).
            tokens: Alias for a single list of tokens (kept for backward compatibility).
            documents: Explicit list of document token lists.
            window_size: Half-window size for context words (e.g. 2 means 2 words left and 2 words right).
            n_negative: Number of negative samples drawn per positive target word.
            min_count: Words appearing fewer than this threshold are discarded.
        """
        super().__init__(
            corpus_dir=corpus_dir,
            download=download,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            corpus=corpus,
            tokens=tokens,
            documents=documents,
            window_size=window_size,
            n_negative=n_negative,
            min_count=min_count,
        )

        # Generate (context_tokens, center_target) samples per document
        # Strict boundary protection: context never crosses document boundaries
        self.samples: list[tuple[list[int], int]] = []
        for doc in self.documents:
            doc_ids = [self.stoi[t] for t in doc if t in self.stoi]
            doc_len = len(doc_ids)

            for center_i in range(doc_len):
                center_word = doc_ids[center_i]
                start = max(0, center_i - self.window_size)
                end = min(doc_len, center_i + self.window_size + 1)
                context = [doc_ids[ctx_i] for ctx_i in range(start, end) if ctx_i != center_i]
                if context:
                    self.samples.append((context, center_word))

        logger.info(
            "BoW sample generation complete: %d training samples generated across %d document(s) "
            "with zero boundary cross-contamination.",
            len(self.samples),
            self.num_documents,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.samples[idx]

    def collate_fn(
        self, batch: list[tuple[list[int], int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized collation for BoW/CBOW.
        Returns:
          - flat_contexts: 1D LongTensor of all context token IDs across the batch
          - offsets: 1D LongTensor marking the start index of each sample's context in flat_contexts
          - targets: 1D LongTensor of true center words (batch_size,)
          - neg_samples: 2D LongTensor of drawn negative sample IDs (batch_size, n_negative)
        """
        contexts_list = [item[0] for item in batch]
        targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
        batch_size = len(batch)

        offsets = [0]
        for ctx in contexts_list[:-1]:
            offsets.append(offsets[-1] + len(ctx))
        offsets_tensor = torch.tensor(offsets, dtype=torch.long)
        flat_contexts = torch.tensor([tok for ctx in contexts_list for tok in ctx], dtype=torch.long)

        neg_samples = torch.multinomial(
            self.neg_dist_tensor,
            batch_size * self.n_negative,
            replacement=True,
        ).view(batch_size, self.n_negative)

        return flat_contexts, offsets_tensor, targets, neg_samples


# Alias for explicit Continuous Bag-of-Words naming
CBOWDataset = BoWDataset
