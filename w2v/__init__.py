"""
w2v: A modular, high-performance PyTorch implementation of Word2Vec (Skip-Gram with Negative Sampling).
Provides end-to-end multi-document tokenization, boundary-protected pair generation,
PyTorch SGNS architecture, model training, and embedding export.
"""

from w2v.bow import BoW, BoWModel, CBOW, CBOWModel
from w2v.cli import create_parser, main as cli_main
from w2v.data.dataset import BaseWord2VecDataset, BoWDataset, CBOWDataset, SkipGramDataset
from w2v.data.downloader import ensure_nltk_gutenberg_corpora
from w2v.preprocessing.tokenizer import BaseTokenizer, BasicTokenizer
from w2v.skipgram import PairScalarModel, SkipGram, SkipGramModel, Word2Vec, get_model, setup_logging

__version__ = "0.1.0"

__all__ = [
    "SkipGram",
    "SkipGramModel",
    "Word2Vec",
    "PairScalarModel",
    "BoW",
    "CBOW",
    "BoWModel",
    "CBOWModel",
    "get_model",
    "setup_logging",
    "BaseWord2VecDataset",
    "SkipGramDataset",
    "BoWDataset",
    "CBOWDataset",
    "ensure_nltk_gutenberg_corpora",
    "BaseTokenizer",
    "BasicTokenizer",
    "cli_main",
    "create_parser",
]
