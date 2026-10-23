from pathlib import Path

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
