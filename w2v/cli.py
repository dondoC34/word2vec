"""
Command Line Interface for w2v (Word2Vec embedding training and export).
Built using Python's standard library `argparse`.
"""

import argparse
import logging
import sys
import webbrowser

from w2v.bow import BoW
from w2v.skipgram import SkipGram, setup_logging
from w2v.tracking import (
    check_mlflow_server_online,
    clear_mlflow_data,
    export_run_artifacts,
    get_best_mlflow_run,
    get_mlflow_status,
    list_mlflow_runs,
    load_mlflow_config,
    resolve_storage_paths,
    start_mlflow_server,
)

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
            ("--sync-corpora / --no-sync-corpora", "False", "Automatic download and sync of delta NLTK corpora."),
            ("--max-tokens", "None", "Optional maximum token budget across all documents."),
            ("--epochs", "5", "Number of training epochs."),
            ("--batch-size", "512", "Mini-batch size for DataLoader."),
            ("--lr", "0.005", "Learning rate for Adam optimizer."),
            ("--output-dir", "./embeddings", "Output directory (created automatically if missing)."),
            (
                "--output-filename",
                "None",
                "Custom output filename (defaults to w2v_{model}_embeddings_YYYYmmdd_hh_MM.txt).",
            ),
            ("--format", "word2vec", "Embedding export format (word2vec, glove, numpy, pt)."),
            ("--device", "None", "Compute device ('cuda', 'cpu', or auto-detected if None)."),
            ("--mlflow / --no-mlflow", "True", "Automatic MLflow experiment tracking (SQLite backend)."),
            ("--mlflow-config", "None", "Optional custom path to mlflow-config.yml."),
            (
                "--log-corpora-content / --no-log-corpora-content",
                "False",
                "Upload full text of training corpora to MLflow artifacts (default: False, manifest only).",
            ),
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
        default=False,
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
        help="Custom output filename (defaults to w2v_{model}_embeddings_YYYYmmdd_hh_MM.txt).",
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
    parser.add_argument(
        "--log-corpora-content",
        dest="log_corpora_content",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload full text of training corpora files to MLflow artifacts (default: False, manifest only).",
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


def format_cli_table(headers: list[str], rows: list[list[str]]) -> str:
    """Formats headers and string rows into a clean ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(col_widths)))
    row_lines = [" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(r)) for r in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def handle_mlflow_cli(args: list[str]) -> int:
    """
    Subcommand controller for 'w2v mlflow' commands:
    - start / run: Launch the tracking server daemon/subprocess
    - clear / clean: Clean SQLite database and/or artifact directories
    - status: Probe server health and storage usage metrics
    - list / runs: Display formatted table of runs and metrics
    - best: Display best run by loss and optionally export assets
    - ui / open: Open dashboard in default web browser
    """
    mlflow_parser = argparse.ArgumentParser(
        prog="w2v mlflow",
        description="MLflow experiment and tracking management for Word2Vec.",
    )

    # Common parent parser so all subcommands accept -c / --config
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to mlflow-config.yml).",
    )

    subparsers = mlflow_parser.add_subparsers(dest="subcommand", required=True)

    # 1. start / run
    start_parser = subparsers.add_parser(
        "start",
        aliases=["run"],
        parents=[common_parser],
        help="Start the MLflow tracking server and web dashboard.",
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

    # 2. clear / clean
    clear_parser = subparsers.add_parser(
        "clear",
        aliases=["clean"],
        parents=[common_parser],
        help="Clear MLflow tracking database records and artifact files.",
    )
    clear_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Confirm deletion automatically without interactive prompt.",
    )
    clear_parser.add_argument(
        "--artifacts-only",
        action="store_true",
        default=False,
        help="Only delete artifact files, leaving the SQLite database intact.",
    )

    # 3. status
    status_parser = subparsers.add_parser(
        "status",
        parents=[common_parser],
        help="Display MLflow server status, database size, and storage diagnostics.",
    )
    status_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host IP address to probe (overrides config).",
    )
    status_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number to probe (overrides config).",
    )

    # 4. list / runs
    list_parser = subparsers.add_parser(
        "list",
        aliases=["runs"],
        parents=[common_parser],
        help="List recent MLflow runs with metrics, architecture, and status.",
    )
    list_parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Filter runs by experiment name (e.g. 'Word2Vec-SkipGram').",
    )
    list_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter runs by model architecture ('skipgram' or 'bow').",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of runs to display (default: 10).",
    )
    list_parser.add_argument(
        "--sort-by",
        type=str,
        choices=["date", "loss"],
        default="date",
        help="Sort runs by 'date' (most recent first) or 'loss' (lowest first).",
    )

    # 5. best
    best_parser = subparsers.add_parser(
        "best",
        parents=[common_parser],
        help="Display details of the best performing run (lowest loss) and export assets.",
    )
    best_parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Filter by experiment name.",
    )
    best_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model architecture ('skipgram' or 'bow').",
    )
    best_parser.add_argument(
        "--export-to",
        type=str,
        default=None,
        help="Optional directory to copy the best run's embeddings and artifacts.",
    )

    # 6. ui / open
    ui_parser = subparsers.add_parser(
        "ui",
        aliases=["open"],
        parents=[common_parser],
        help="Open the MLflow web UI dashboard in the default browser.",
    )
    ui_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host IP address (overrides config).",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number (overrides config).",
    )

    parsed = mlflow_parser.parse_args(args)

    # ---------------- Subcommand Dispatchers ----------------

    # Command: start / run
    if parsed.subcommand in ("start", "run"):
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

    # Command: clear / clean
    if parsed.subcommand in ("clear", "clean"):
        db_path, artifact_path = resolve_storage_paths(parsed.config)
        db_size = db_path.stat().st_size if db_path and db_path.exists() else 0
        art_size = (
            sum(f.stat().st_size for f in artifact_path.rglob("*") if f.is_file()) if artifact_path.exists() else 0
        )

        print("\n" + "=" * 65)
        print("w2v MLflow: Storage Cleanup")
        print("=" * 65)
        if not parsed.artifacts_only and db_path:
            print(f"Target Database : {db_path} ({db_size / 1024:.1f} KB)")
        print(f"Target Artifacts: {artifact_path} ({art_size / (1024 * 1024):.2f} MB)")
        print("=" * 65)

        if not parsed.yes:
            target_desc = "all artifacts" if parsed.artifacts_only else "all MLflow runs, metrics, and artifacts"
            try:
                confirm = input(f"Are you sure you want to permanently delete {target_desc}? [y/N]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user.")
                return 1
            if confirm not in ("y", "yes"):
                print("Cleanup aborted by user.\n")
                return 0

        success, msg = clear_mlflow_data(parsed.config, artifacts_only=parsed.artifacts_only)
        if success:
            logger.info("MLflow storage cleanup: %s", msg)
            print(f"[w2v] {msg}\n")
            return 0
        else:
            logger.error("MLflow storage cleanup failed: %s", msg)
            print(f"[w2v ERROR] {msg}\n")
            return 1

    # Command: status
    if parsed.subcommand == "status":
        stats = get_mlflow_status(parsed.config, host=parsed.host, port=parsed.port)
        print("\n" + "=" * 65)
        print("w2v MLflow: System & Storage Status")
        print("=" * 65)
        server_badge = "[ONLINE]" if stats["server_online"] else "[OFFLINE]"
        print(f"Dashboard URL       : http://{stats['host']}:{stats['port']} {server_badge}")
        print(f"Backend Store URI   : {stats['backend_store_uri']}")
        if stats["db_path"]:
            db_status = f"Exists ({stats['db_size_bytes'] / 1024:.1f} KB)" if stats["db_exists"] else "Not created yet"
            print(f"Database File       : {stats['db_path']} -> {db_status}")
        art_status = (
            f"Exists ({stats['artifact_file_count']} files, {stats['artifact_size_bytes'] / (1024 * 1024):.2f} MB)"
            if stats["artifact_exists"]
            else "Not created yet"
        )
        print(f"Artifact Directory  : {stats['artifact_dir']} -> {art_status}")
        print("-" * 65)
        print(f"Total Experiments   : {stats['experiment_count']}")
        print(f"Total Runs Recorded : {stats['total_runs']}")
        if stats["runs_by_status"]:
            status_summary = ", ".join(f"{k}: {v}" for k, v in stats["runs_by_status"].items())
            print(f"Runs by Status      : {status_summary}")
        if stats["best_loss_by_experiment"]:
            print("Best Loss Recorded  :")
            for exp, (bloss, run_id) in stats["best_loss_by_experiment"].items():
                print(f"  * {exp.ljust(22)}: {bloss:.4f} (Run: {run_id})")
        print("=" * 65 + "\n")
        return 0

    # Command: list / runs
    if parsed.subcommand in ("list", "runs"):
        runs = list_mlflow_runs(
            config_path=parsed.config,
            experiment_name=parsed.experiment,
            model=parsed.model,
            limit=parsed.limit,
            sort_by=parsed.sort_by,
        )
        if not runs:
            print("\n[w2v] No MLflow runs found matching specified criteria.\n")
            return 0

        print("\n" + "=" * 80)
        print(f"w2v MLflow Runs (Showing top {len(runs)}, sorted by {parsed.sort_by}):")
        print("=" * 80)
        headers = ["Run ID", "Experiment", "Model", "Dim", "Epochs", "Loss", "Status", "Created At"]
        table_rows = [
            [
                r["run_id"][:8],
                r["experiment_name"],
                r["model"],
                str(r["embedding_dim"]),
                str(r["epochs"]),
                str(r["final_loss"]),
                r["status"],
                r["created_at"],
            ]
            for r in runs
        ]
        print(format_cli_table(headers, table_rows))
        print("=" * 80 + "\n")
        return 0

    # Command: best
    if parsed.subcommand == "best":
        best_run = get_best_mlflow_run(
            config_path=parsed.config,
            experiment_name=parsed.experiment,
            model=parsed.model,
        )
        if not best_run:
            print("\n[w2v] No runs with recorded loss found.\n")
            return 0

        print("\n" + "=" * 65)
        print("w2v MLflow: Best Performing Run")
        print("=" * 65)
        print(f"Run ID        : {best_run['run_id']}")
        print(f"Experiment    : {best_run['experiment_name']}")
        print(f"Architecture  : {best_run['model']}")
        print(f"Final Loss    : {best_run['final_loss']}")
        print(f"Created At    : {best_run['created_at']}")
        print("-" * 65)
        print("Key Hyperparameters:")
        params = best_run.get("params", {})
        for k in (
            "embedding_dim",
            "window_size",
            "n_negative",
            "min_count",
            "lr",
            "batch_size",
            "epochs",
            "tied_weights",
        ):
            if k in params:
                print(f"  * {k.ljust(15)}: {params[k]}")
        if best_run.get("embeddings_file"):
            print(f"Embeddings File: {best_run['embeddings_file']}")

        if parsed.export_to:
            try:
                copied = export_run_artifacts(best_run["run_id"], parsed.export_to, parsed.config)
                print("-" * 65)
                print(f"Exported {len(copied)} artifact file(s) to '{parsed.export_to}':")
                for f in copied:
                    print(f"  -> {f.name}")
            except Exception as e:
                print(f"[w2v ERROR] Failed exporting artifacts: {e}")
                return 1

        print("=" * 65 + "\n")
        return 0

    # Command: ui / open
    if parsed.subcommand in ("ui", "open"):
        cfg = load_mlflow_config(parsed.config)
        server_cfg = cfg.get("server", {})
        host = parsed.host or server_cfg.get("host", "127.0.0.1")
        port = parsed.port or server_cfg.get("port", 5000)
        url = f"http://{host}:{port}"

        online = check_mlflow_server_online(host, port)
        if not online:
            print(f"\n[w2v] Notice: MLflow tracking server appears offline at {url}.")
            print("[w2v] Launch it with: w2v mlflow start\n")
        else:
            print(f"\n[w2v] Opening MLflow Web UI at: {url}")

        webbrowser.open(url)
        return 0

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
        "Corpus directory: %s (auto-sync: %s, max_tokens: %s, upload_content: %s)",
        parsed_args.corpus_dir,
        parsed_args.sync_corpora,
        parsed_args.max_tokens,
        parsed_args.log_corpora_content,
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
            log_corpora_content=parsed_args.log_corpora_content,
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
            log_corpora_content=parsed_args.log_corpora_content,
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
