from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
    args_default = parser.parse_args([])
    assert args_default.sync_corpora is False
    assert args_default.log_corpora_content is False

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
    saved = list(out_dir.glob("w2v_skipgram_embeddings_*.txt"))
    assert len(saved) == 1
    assert saved[0].exists()


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
    saved = list(out_dir.glob("w2v_bow_embeddings_*.txt"))
    assert len(saved) == 1
    assert saved[0].exists()


def test_cli_main_error_handling(tmp_path: Path) -> None:
    # When directory cannot be used and download fails, main returns 1
    non_existent = tmp_path / "nonexistent_dir_12345"
    with patch("w2v.data.dataset.ensure_nltk_gutenberg_corpora", side_effect=RuntimeError("Download failed")):
        ret = main(["--corpus-dir", str(non_existent), "--no-sync-corpora", "--no-mlflow"])
        assert ret == 1


def test_cli_main_empty_corpus_dir(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_corpus"
    empty_dir.mkdir()
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "embeddings"

    with patch("w2v.data.dataset.ensure_nltk_gutenberg_corpora") as mock_dl:

        def create_sample(dest: Path | str) -> list[Path]:
            p = Path(dest) / "sample.txt"
            p.write_text("the quick brown fox jumps over the lazy dog " * 5, encoding="utf-8")
            return [p]

        mock_dl.side_effect = create_sample
        ret = main(
            [
                "--corpus-dir",
                str(empty_dir),
                "--output-dir",
                str(out_dir),
                "--log-dir",
                str(log_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--embedding-dim",
                "8",
                "--min-count",
                "1",
                "--no-mlflow",
            ]
        )
        assert ret == 0
        assert mock_dl.called


def test_cli_main_dir_without_txt_files(tmp_path: Path) -> None:
    dir_without_txt = tmp_path / "no_txt_corpus"
    dir_without_txt.mkdir()
    (dir_without_txt / "image.png").write_bytes(b"\x89PNG")
    (dir_without_txt / "data.csv").write_text("a,b,c", encoding="utf-8")
    log_dir = tmp_path / "logs"

    # Simulate download failing to produce .txt files
    with patch("w2v.data.dataset.ensure_nltk_gutenberg_corpora"):
        ret = main(["--corpus-dir", str(dir_without_txt), "--log-dir", str(log_dir), "--no-mlflow"])
        assert ret == 1


def test_cli_log_corpora_content_flag() -> None:
    parser = create_parser()
    args_default = parser.parse_args(["--corpus-dir", "./dummy"])
    assert args_default.log_corpora_content is False

    args_explicit = parser.parse_args(["--corpus-dir", "./dummy", "--log-corpora-content"])
    assert args_explicit.log_corpora_content is True

    args_disabled = parser.parse_args(["--corpus-dir", "./dummy", "--no-log-corpora-content"])
    assert args_disabled.log_corpora_content is False


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
    saved = list(out_dir.glob("w2v_skipgram_embeddings_*.txt"))
    assert len(saved) == 1
    assert saved[0].exists()


def test_cli_main_custom_filename_and_no_overwrite(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "text.txt").write_text("alpha beta gamma alpha beta gamma", encoding="utf-8")

    out_dir = tmp_path / "embeddings"
    log_dir = tmp_path / "logs"

    # Test custom output filename
    ret = main(
        [
            "--model",
            "skipgram",
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(out_dir),
            "--output-filename",
            "custom_embeddings.txt",
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
    assert (out_dir / "custom_embeddings.txt").exists()

    # Test two successive runs with default filename do not overwrite each other
    ret1 = main(
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
    assert ret1 == 0

    ret2 = main(
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
    assert ret2 == 0

    default_files = list(out_dir.glob("w2v_skipgram_embeddings_*.txt"))
    assert len(default_files) == 2


@patch("w2v.cli.subprocess.call", return_value=0)
def test_cli_mlflow_start_mocked(mock_call: MagicMock) -> None:
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
def test_cli_mlflow_start_custom_config_mocked(mock_call: MagicMock, tmp_path: Path) -> None:
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
def test_cli_mlflow_start_cli_overrides_mocked(mock_call: MagicMock) -> None:
    ret = main(["mlflow", "start", "--host", "0.0.0.0", "--port", "9999"])
    assert ret == 0
    assert mock_call.called
    cmd = mock_call.call_args[0][0]
    assert "0.0.0.0" in cmd
    assert "9999" in cmd


@patch("w2v.cli.subprocess.call", return_value=0)
def test_cli_mlflow_run_alias_mocked(mock_call: MagicMock) -> None:
    ret = main(["mlflow", "run"])
    assert ret == 0
    assert mock_call.called


@patch("w2v.cli.get_mlflow_status")
def test_cli_mlflow_status_mocked(mock_status: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    mock_status.return_value = {
        "host": "127.0.0.1",
        "port": 5000,
        "server_online": True,
        "backend_store_uri": "sqlite:///mlflow.db",
        "db_path": "/fake/mlflow.db",
        "db_exists": True,
        "db_size_bytes": 10240,
        "artifact_dir": "/fake/mlartifacts",
        "artifact_exists": True,
        "artifact_file_count": 5,
        "artifact_size_bytes": 20480,
        "experiment_count": 2,
        "total_runs": 4,
        "runs_by_status": {"FINISHED": 4},
        "best_loss_by_experiment": {"Word2Vec-SkipGram": (3.14, "abcd1234")},
    }
    ret = main(["mlflow", "status"])
    assert ret == 0
    captured = capsys.readouterr().out
    assert "System & Storage Status" in captured
    assert "[ONLINE]" in captured
    assert "Word2Vec-SkipGram" in captured


@patch("w2v.cli.list_mlflow_runs")
def test_cli_mlflow_list_and_runs_alias(mock_list: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    mock_list.return_value = [
        {
            "run_id": "12345678abcdef",
            "experiment_name": "Word2Vec-SkipGram",
            "model": "SkipGram",
            "embedding_dim": "128",
            "epochs": "5",
            "final_loss": "3.1415",
            "status": "FINISHED",
            "created_at": "2026-11-02 10:00:00",
        }
    ]
    ret1 = main(["mlflow", "list", "--limit", "1"])
    assert ret1 == 0
    out1 = capsys.readouterr().out
    assert "12345678" in out1
    assert "3.1415" in out1

    ret2 = main(["mlflow", "runs"])
    assert ret2 == 0


@patch("w2v.cli.get_best_mlflow_run")
def test_cli_mlflow_best_mocked(mock_best: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    mock_best.return_value = {
        "run_id": "87654321fedcba",
        "experiment_name": "Word2Vec-SkipGram",
        "model": "SkipGram",
        "final_loss": "2.7182",
        "created_at": "2026-11-02 10:00:00",
        "params": {"embedding_dim": "128", "lr": "0.005"},
        "embeddings_file": "/fake/embeddings.txt",
    }
    ret = main(["mlflow", "best"])
    assert ret == 0
    captured = capsys.readouterr().out
    assert "Best Performing Run" in captured
    assert "2.7182" in captured
    assert "embedding_dim" in captured


@patch("w2v.cli.clear_mlflow_data", return_value=(True, "Successfully cleared database."))
def test_cli_mlflow_clear_and_clean_alias(mock_clear: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    # Test with -y flag bypassing input prompt
    ret1 = main(["mlflow", "clear", "-y"])
    assert ret1 == 0
    assert mock_clear.called

    # Test clean alias
    ret2 = main(["mlflow", "clean", "--yes", "--artifacts-only"])
    assert ret2 == 0


@patch("w2v.cli.webbrowser.open")
@patch("w2v.cli.check_mlflow_server_online", return_value=True)
def test_cli_mlflow_ui_and_open_alias(
    mock_online: MagicMock,
    mock_browser: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ret1 = main(["mlflow", "ui"])
    assert ret1 == 0
    assert mock_browser.called
    assert "http://127.0.0.1:5000" in capsys.readouterr().out

    ret2 = main(["mlflow", "open", "--port", "5050"])
    assert ret2 == 0
