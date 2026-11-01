from pathlib import Path
from unittest.mock import patch

from w2v.cli import W2VArgumentParser, create_parser, main


def test_w2v_argument_parser_format_help() -> None:
    parser = W2VArgumentParser(prog="w2v")
    help_text = parser.format_help()
    assert "Usage: w2v" in help_text
    assert "--model" in help_text
    assert "--embedding-dim" in help_text
    assert "Example:" in help_text


def test_create_parser() -> None:
    parser = create_parser()
    args = parser.parse_args(["--model", "cbow", "--embedding-dim", "64", "--epochs", "2"])
    assert args.model == "cbow"
    assert args.embedding_dim == 64
    assert args.epochs == 2


def test_cli_main_skipgram(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "text.txt").write_text("worda wordb wordc worda wordb wordc", encoding="utf-8")

    out_dir = tmp_path / "embeddings"
    log_dir = tmp_path / "logs"

    ret = main(
        [
            "--model",
            "skipgram",
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(out_dir),
            "--log-dir",
            str(log_dir),
            "--no-sync-corpora",
            "--embedding-dim",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--min-count",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert ret == 0
    assert (out_dir / "word2vec_embeddings.txt").exists()


def test_cli_main_bow(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "text.txt").write_text("item1 item2 item3 item1 item2 item3", encoding="utf-8")

    out_dir = tmp_path / "embeddings"
    log_dir = tmp_path / "logs"

    ret = main(
        [
            "--model",
            "bow",
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(out_dir),
            "--log-dir",
            str(log_dir),
            "--no-sync-corpora",
            "--embedding-dim",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--min-count",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert ret == 0
    assert (out_dir / "bow_embeddings.txt").exists()


def test_cli_main_error_handling() -> None:
    # Invalid directory raises FileNotFoundError in pipeline, caught by main returning 1
    ret = main(["--corpus-dir", "/path/that/definitely/does/not/exist/12345", "--no-sync-corpora"])
    assert ret == 1


def test_cli_main_no_mlflow(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "text.txt").write_text("word1 word2 word3 word1 word2 word3", encoding="utf-8")

    out_dir = tmp_path / "embeddings"
    log_dir = tmp_path / "logs"

    ret = main(
        [
            "--model",
            "skipgram",
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(out_dir),
            "--log-dir",
            str(log_dir),
            "--no-sync-corpora",
            "--no-mlflow",
            "--embedding-dim",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--min-count",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert ret == 0
    assert (out_dir / "word2vec_embeddings.txt").exists()


@patch("w2v.cli.subprocess.call", return_value=0)
def test_cli_mlflow_start_mocked(mock_call: patch) -> None:
    ret = main(["mlflow", "start"])
    assert ret == 0
    assert mock_call.called
    cmd = mock_call.call_args[0][0]
    assert "mlflow" in cmd
    assert "server" in cmd
    assert "--host" in cmd
    assert "127.0.0.1" in cmd
    assert "--port" in cmd
    assert "5000" in cmd


@patch("w2v.cli.subprocess.call", return_value=0)
def test_cli_mlflow_start_custom_config_mocked(mock_call: patch, tmp_path: Path) -> None:
    custom_cfg = tmp_path / "test_cfg.yml"
    custom_cfg.write_text(
        """
server:
  host: "192.168.1.100"
  port: 6000
  backend_store_uri: "sqlite:///custom_test.db"
  default_artifact_root: "./custom_artifacts"
""",
        encoding="utf-8",
    )
    ret = main(["mlflow", "start", "-c", str(custom_cfg)])
    assert ret == 0
    assert mock_call.called
    cmd = mock_call.call_args[0][0]
    assert "--host" in cmd
    assert "192.168.1.100" in cmd
    assert "--port" in cmd
    assert "6000" in cmd
    assert "--backend-store-uri" in cmd
    assert "sqlite:///custom_test.db" in cmd


@patch("w2v.cli.subprocess.call", return_value=0)
def test_cli_mlflow_start_cli_overrides_mocked(mock_call: patch) -> None:
    ret = main(["mlflow", "start", "--host", "0.0.0.0", "--port", "9999"])
    assert ret == 0
    assert mock_call.called
    cmd = mock_call.call_args[0][0]
    assert "0.0.0.0" in cmd
    assert "9999" in cmd
