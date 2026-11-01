"""
Command Line Interface for w2v (Word2Vec embedding training and export).
Built using Python's standard library `argparse`.
"""

import argparse
import logging
import subprocess
import sys

from w2v.bow import BoW
from w2v.skipgram import SkipGram, setup_logging
from w2v.tracking import load_mlflow_config

logger = logging.getLogger("CLI")


class W2VArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser that formats the CLI help message as a clean,
    structured table of parameters, default values, and descriptions.
    """

    def format_help(self) -> str:
        table_rows = [
            ("--model", "skipgram", "Model architecture to use ('skipgram', 'bow', 'cbow')."),
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
            ("--mlflow / --no-mlflow", "True", "Automatic MLflow experiment tracking (SQLite backend)."),
            ("--mlflow-config", "None", "Optional custom path to mlflow-config.yml."),
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
        choices=["skipgram", "bow", "cbow"],
        help="Model architecture to use ('skipgram', 'bow', 'cbow').",
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

    # MLflow Tracking Options
    parser.add_argument(
        "--mlflow",
        dest="mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatic MLflow experiment tracking (default: True).",
    )
    parser.add_argument(
        "--mlflow-config",
        type=str,
        default=None,
        help="Optional custom path to mlflow-config.yml.",
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


def start_mlflow_server(
    host: str = "127.0.0.1",
    port: int = 5000,
    backend_store_uri: str = "sqlite:///mlflow.db",
    default_artifact_root: str = "./mlartifacts",
) -> int:
    """
    Launches the MLflow tracking server and web UI as a subprocess.
    """
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        str(host),
        "--port",
        str(port),
        "--backend-store-uri",
        str(backend_store_uri),
        "--default-artifact-root",
        str(default_artifact_root),
    ]
    logger.info("=" * 70)
    logger.info("Starting MLflow Server...")
    logger.info("Backend SQLite URI: %s", backend_store_uri)
    logger.info("Default Artifact Root: %s", default_artifact_root)
    logger.info("MLflow Web Dashboard: http://%s:%s", host, port)
    logger.info("=" * 70)
    print(f"\n[w2v] MLflow UI running at: http://{host}:{port}")
    print("[w2v] Press Ctrl+C to terminate the server.\n")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        logger.info("MLflow server stopped by user.")
        return 0


def handle_mlflow_cli(args: list[str]) -> int:
    """
    Subcommand controller for 'w2v mlflow' commands.
    """
    mlflow_parser = argparse.ArgumentParser(
        prog="w2v mlflow",
        description="MLflow server management for Word2Vec experiments.",
    )
    subparsers = mlflow_parser.add_subparsers(dest="subcommand", required=True)

    start_parser = subparsers.add_parser(
        "start",
        help="Start the MLflow tracking server and web dashboard.",
    )
    start_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to mlflow-config.yml).",
    )
    start_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host IP address to bind to (overrides config).",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number to bind to (overrides config).",
    )
    start_parser.add_argument(
        "--backend-store-uri",
        type=str,
        default=None,
        help="Storage database URI (e.g. sqlite:///mlflow.db, overrides config).",
    )
    start_parser.add_argument(
        "--default-artifact-root",
        type=str,
        default=None,
        help="Directory where artifacts are stored (overrides config).",
    )

    parsed = mlflow_parser.parse_args(args)
    if parsed.subcommand == "start":
        cfg = load_mlflow_config(parsed.config)
        server_cfg = cfg.get("server", {})
        host = parsed.host or server_cfg.get("host", "127.0.0.1")
        port = parsed.port or server_cfg.get("port", 5000)
        backend_store_uri = parsed.backend_store_uri or server_cfg.get("backend_store_uri", "sqlite:///mlflow.db")
        default_artifact_root = parsed.default_artifact_root or server_cfg.get("default_artifact_root", "./mlartifacts")

        return start_mlflow_server(
            host=host,
            port=port,
            backend_store_uri=backend_store_uri,
            default_artifact_root=default_artifact_root,
        )

    return 0


def main(args: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if args is None else args)
    if raw_args and raw_args[0] == "mlflow":
        return handle_mlflow_cli(raw_args[1:])

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
    logger.info(
        "Configuration: model=%s, dim=%d, window=%d, n_neg=%d, min_count=%d, tied_weights=%s",
        parsed_args.model,
        parsed_args.embedding_dim,
        parsed_args.window_size,
        parsed_args.n_negative,
        parsed_args.min_count,
        parsed_args.tied_weights,
    )
    logger.info(
        "Corpus directory: %s (auto-sync: %s, max_tokens: %s)",
        parsed_args.corpus_dir,
        parsed_args.sync_corpora,
        parsed_args.max_tokens,
    )
    logger.info("Output directory: %s (format: %s)", parsed_args.output_dir, parsed_args.format)
    logger.info("=" * 70)

    try:
        # Instantiate requested architecture (Word2Vec / Skip-Gram or BoW / CBOW)
        is_bow = parsed_args.model.lower() in ("bow", "cbow")
        model_cls = BoW if is_bow else SkipGram
        logger.info(
            "Initializing %s engine...",
            "Continuous Bag-of-Words (BoW / CBOW)" if is_bow else "Skip-Gram with Negative Sampling (SGNS)",
        )
        model_pipeline = model_cls(
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
            enable_mlflow=parsed_args.mlflow,
            mlflow_config=parsed_args.mlflow_config,
        )

        # Train model
        logger.info(
            "Beginning training (%d epochs, batch_size=%d, lr=%.5f)...",
            parsed_args.epochs,
            parsed_args.batch_size,
            parsed_args.lr,
        )
        history = model_pipeline.train(
            epochs=parsed_args.epochs,
            batch_size=parsed_args.batch_size,
            lr=parsed_args.lr,
            enable_mlflow=parsed_args.mlflow,
            mlflow_config=parsed_args.mlflow_config,
        )
        logger.info("Training complete. Final loss: %.4f", history[-1] if history else 0.0)

        # Save embeddings to destination directory (auto-created if missing)
        saved_file = model_pipeline.save_embeddings(
            output_dir=parsed_args.output_dir,
            filename=parsed_args.output_filename,
            format=parsed_args.format,
        )
        logger.info("Embeddings successfully saved to: %s", saved_file)

        # Run sample semantic queries to verify quality
        sample_words = ["dorian", "alice", "time", "captain", "sea", "art"]
        logger.info("Verifying semantic neighbor queries for sample tokens...")
        for word in sample_words:
            sims = model_pipeline.most_similar(word, top_k=3)
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
