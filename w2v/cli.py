"""
Command Line Interface for w2v (Word2Vec embedding training and export).
Built using Python's standard library `argparse`.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from w2v.word2vec import Word2Vec, setup_logging

logger = logging.getLogger("CLI")


class W2VArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser that formats the CLI help message as a clean,
    structured table of parameters, default values, and descriptions.
    """

    def format_help(self) -> str:
        table_rows = [
            ("--model", "skipgram", "Model architecture to use (skipgram)."),
            ("--embedding-dim", "128", "Dimensionality of dense embedding vectors."),
            ("--window-size", "2", "Context half-window size (tokens left and right)."),
            ("--n-negative", "5", "Number of negative samples per positive pair."),
            ("--min-count", "3", "Minimum token frequency threshold (discards rare words)."),
            ("--tied-weights / --no-tied-weights", "True", "If False, saves arithmetic mean of (W_in + W_out) / 2."),
            ("--corpus-dir", "./corpuses", "Directory containing text documents (*.txt)."),
            ("--sync-corpora / --no-sync-corpora", "True", "Automatic download and sync of delta NLTK corpora."),
            ("--max-tokens", "None", "Optional maximum token budget across all documents."),
            ("--epochs", "5", "Number of training epochs."),
            ("--batch-size", "512", "Mini-batch size for DataLoader."),
            ("--lr", "0.005", "Learning rate for Adam optimizer."),
            ("--output-dir", "./embeddings", "Output directory (created automatically if missing)."),
            ("--output-filename", "None", "Custom output filename (e.g. custom.txt)."),
            ("--format", "word2vec", "Embedding export format (word2vec, glove, numpy, pt)."),
            ("--device", "None", "Compute device ('cuda', 'cpu', or auto-detected if None)."),
            ("--log-dir", "./logs", "Directory where execution log files are stored."),
            ("--log-level", "INFO", "Logging severity threshold (DEBUG, INFO, WARNING, ERROR)."),
            ("-h, --help", "", "Show this help message with parameters table and exit."),
        ]

        col1_w = max(len("Parameter"), max(len(r[0]) for r in table_rows)) + 2
        col2_w = max(len("Default"), max(len(r[1]) for r in table_rows)) + 2
        col3_w = max(len("Description"), max(len(r[2]) for r in table_rows)) + 2

        header = f"| {'Parameter'.ljust(col1_w)} | {'Default'.ljust(col2_w)} | {'Description'.ljust(col3_w)} |"
        separator = f"|{'-' * (col1_w + 2)}|{'-' * (col2_w + 2)}|{'-' * (col3_w + 2)}|"

        lines = [
            "\nUsage: w2v [OPTIONS]\n",
            "w2v: Train Skip-Gram with Negative Sampling (SGNS) embeddings and export vectors.\n",
            header,
            separator,
        ]

        for param, default, desc in table_rows:
            lines.append(f"| {param.ljust(col1_w)} | {default.ljust(col2_w)} | {desc.ljust(col3_w)} |")

        lines.append("\nExample:")
        lines.append("  w2v --model skipgram --embedding-dim 128 --output-dir ./abc/def/\n")

        return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    parser = W2VArgumentParser(
        prog="w2v",
        description="w2v: Train Skip-Gram with Negative Sampling (SGNS) embeddings and export vectors.",
    )

    # Core Model Architecture
    parser.add_argument(
        "--model",
        type=str,
        default="skipgram",
        choices=["skipgram"],
        help="Model architecture to use (skipgram).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Dimensionality of dense embedding vectors.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help="Context half-window size (tokens left and right).",
    )
    parser.add_argument(
        "--n-negative",
        type=int,
        default=5,
        help="Number of negative samples per positive pair.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Minimum token frequency threshold (discards rare words).",
    )
    parser.add_argument(
        "--tied-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If False, saves arithmetic mean of (W_in + W_out) / 2.",
    )

    # Data Ingestion & Sync
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="./corpuses",
        help="Directory containing text documents (*.txt).",
    )
    parser.add_argument(
        "--sync-corpora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatic download and sync of delta NLTK corpora.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional maximum token budget across all documents.",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size for DataLoader.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or auto-detected if None).",
    )

    # Embedding Export Options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./embeddings",
        help="Output directory (created automatically if missing).",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Custom output filename (e.g. custom.txt).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="word2vec",
        choices=["word2vec", "glove", "numpy", "pt"],
        help="Embedding export format (word2vec, glove, numpy, pt).",
    )

    # Logging Options
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory where execution log files are stored.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging severity threshold (DEBUG, INFO, WARNING, ERROR).",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Initialize unified logging inside the main CLI controller
    log_file = setup_logging(
        log_dir=parsed_args.log_dir,
        level=getattr(logging, parsed_args.log_level.upper()),
    )

    logger.info("=" * 70)
    logger.info("w2v CLI: Starting Word2Vec training and embedding generation")
    logger.info("Log file active: %s", log_file)
    logger.info("Configuration: model=%s, dim=%d, window=%d, n_neg=%d, min_count=%d, tied_weights=%s",
                parsed_args.model, parsed_args.embedding_dim, parsed_args.window_size,
                parsed_args.n_negative, parsed_args.min_count, parsed_args.tied_weights)
    logger.info("Corpus directory: %s (auto-sync: %s, max_tokens: %s)",
                parsed_args.corpus_dir, parsed_args.sync_corpora, parsed_args.max_tokens)
    logger.info("Output directory: %s (format: %s)", parsed_args.output_dir, parsed_args.format)
    logger.info("=" * 70)

    try:
        # Initialize Word2Vec pipeline with parsed configuration
        w2v = Word2Vec(
            corpus_dir=parsed_args.corpus_dir,
            embedding_dim=parsed_args.embedding_dim,
            window_size=parsed_args.window_size,
            n_negative=parsed_args.n_negative,
            min_count=parsed_args.min_count,
            tied_weights=parsed_args.tied_weights,
            max_tokens=parsed_args.max_tokens,
            sync_corpora_hook=parsed_args.sync_corpora,
            device=parsed_args.device,
            log_dir=parsed_args.log_dir,
        )

        # Train model
        logger.info("Beginning training (%d epochs, batch_size=%d, lr=%.5f)...",
                    parsed_args.epochs, parsed_args.batch_size, parsed_args.lr)
        history = w2v.train(
            epochs=parsed_args.epochs,
            batch_size=parsed_args.batch_size,
            lr=parsed_args.lr,
        )
        logger.info("Training complete. Final loss: %.4f", history[-1] if history else 0.0)

        # Save embeddings to destination directory (auto-created if missing)
        saved_file = w2v.save_embeddings(
            output_dir=parsed_args.output_dir,
            filename=parsed_args.output_filename,
            format=parsed_args.format,
        )
        logger.info("Embeddings successfully saved to: %s", saved_file)

        # Run sample semantic queries to verify quality
        sample_words = ["dorian", "alice", "time", "captain", "sea", "art"]
        logger.info("Verifying semantic neighbor queries for sample tokens...")
        for word in sample_words:
            sims = w2v.most_similar(word, top_k=3)
            if sims:
                formatted = ", ".join([f"{w} ({s:+.4f})" for w, s in sims])
                logger.info("Top neighbors for '%s': %s", word, formatted)

        logger.info("w2v process finished successfully. Output ready in '%s'.", saved_file)
        return 0

    except Exception as e:
        logger.error("Error during w2v execution: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
