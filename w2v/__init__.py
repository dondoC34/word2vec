"""
w2v: A modular, high-performance PyTorch implementation of Word2Vec (Skip-Gram with Negative Sampling).
Provides end-to-end multi-document tokenization, boundary-protected pair generation,
PyTorch SGNS architecture, model training, and embedding export.
"""

from w2v.base import BaseWord2Vec
from w2v.bow import CBOW, BoW, BoWModel, CBOWModel
from w2v.cli import create_parser
from w2v.cli import main as cli_main
from w2v.data.dataset import BaseWord2VecDataset, BoWDataset, CBOWDataset, SkipGramDataset
from w2v.data.downloader import ensure_nltk_gutenberg_corpora
from w2v.preprocessing.tokenizer import BaseTokenizer, BasicTokenizer
from w2v.skipgram import PairScalarModel, SkipGram, SkipGramModel, Word2Vec, get_model, setup_logging

__version__ = "0.1.0"

__all__ = [
    "CBOW",
    "BaseTokenizer",
    "BaseWord2Vec",
    "BaseWord2VecDataset",
    "BasicTokenizer",
    "BoW",
    "BoWDataset",
    "BoWModel",
    "CBOWDataset",
    "CBOWModel",
    "PairScalarModel",
    "SkipGram",
    "SkipGramDataset",
    "SkipGramModel",
    "Word2Vec",
    "cli_main",
    "create_parser",
    "ensure_nltk_gutenberg_corpora",
    "get_model",
    "setup_logging",
]
