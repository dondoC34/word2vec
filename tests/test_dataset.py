from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from w2v.data.dataset import BaseWord2VecDataset, BoWDataset, CBOWDataset, SkipGramDataset
from w2v.data.downloader import ensure_nltk_gutenberg_corpora


@pytest.fixture
def sample_tokens() -> list[str]:
    return ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]


@pytest.fixture
def sample_docs() -> list[list[str]]:
    return [
        ["word_a", "word_b", "word_c"],
        ["word_d", "word_e", "word_f"],
    ]


def test_inheritance(sample_tokens: list[str]) -> None:
    sg = SkipGramDataset(tokens=sample_tokens)
    bow = BoWDataset(tokens=sample_tokens)
    assert isinstance(sg, BaseWord2VecDataset)
    assert isinstance(bow, BaseWord2VecDataset)
    assert CBOWDataset is BoWDataset


def test_skipgram_tokens_and_vocab(sample_tokens: list[str]) -> None:
    sg = SkipGramDataset(tokens=sample_tokens, window_size=2, min_count=1)
    assert sg.vocab_size > 0
    assert len(sg.itos) == sg.vocab_size
    assert len(sg.stoi) == sg.vocab_size
    assert len(sg.neg_dist_tensor) == sg.vocab_size
    assert abs(sg.neg_dist_tensor.sum().item() - 1.0) < 1e-4
    assert len(sg) > 0

    # __getitem__
    center, context = sg[0]
    assert isinstance(center, int)
    assert isinstance(context, int)

    # collate_fn
    batch = [sg[0], sg[1]]
    centers, contexts, neg = sg.collate_fn(batch)
    assert centers.shape == (2,)
    assert contexts.shape == (2,)
    assert neg.shape == (2, sg.n_negative)


def test_bow_tokens_and_vocab(sample_tokens: list[str]) -> None:
    bow = BoWDataset(tokens=sample_tokens, window_size=2, min_count=1)
    assert bow.vocab_size > 0
    assert len(bow.itos) == bow.vocab_size
    assert len(bow.stoi) == bow.vocab_size
    assert len(bow.neg_dist_tensor) == bow.vocab_size
    assert abs(bow.neg_dist_tensor.sum().item() - 1.0) < 1e-4
    assert len(bow) > 0

    # __getitem__
    context, center = bow[0]
    assert isinstance(context, list)
    assert isinstance(center, int)

    # collate_fn
    batch = [bow[0], bow[1]]
    flat_contexts, offsets, targets, neg = bow.collate_fn(batch)
    assert isinstance(flat_contexts, torch.Tensor)
    assert offsets.shape == (2,)
    assert targets.shape == (2,)
    assert neg.shape == (2, bow.n_negative)


def test_document_boundary_isolation(sample_docs: list[list[str]]) -> None:
    sg = SkipGramDataset(documents=sample_docs, window_size=2, min_count=1)
    doc1_ids = {sg.stoi["word_a"], sg.stoi["word_b"], sg.stoi["word_c"]}
    doc2_ids = {sg.stoi["word_d"], sg.stoi["word_e"], sg.stoi["word_f"]}

    for center, context in sg.pairs:
        if center in doc1_ids:
            assert context in doc1_ids
        if center in doc2_ids:
            assert context in doc2_ids


def test_corpus_dir_loading(tmp_path: Path) -> None:
    f1 = tmp_path / "doc1.txt"
    f1.write_text("apple banana orange apple", encoding="utf-8")
    f2 = tmp_path / "doc2.txt"
    f2.write_text("grape melon kiwi", encoding="utf-8")

    sg = SkipGramDataset(corpus_dir=tmp_path, max_tokens=10)
    assert sg.num_documents == 2
    assert "apple" in sg.stoi
    assert "banana" in sg.stoi

    bow = BoWDataset(corpus_dir=tmp_path, max_tokens=10)
    assert bow.num_documents == 2
    assert "melon" in bow.stoi


def test_download_flag_calls_ensure_corpora(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("hello world sample text", encoding="utf-8")

    with patch("w2v.data.dataset.ensure_nltk_gutenberg_corpora") as mock_ensure:
        SkipGramDataset(corpus_dir=tmp_path, download=True)
        mock_ensure.assert_called_once_with(tmp_path)

    with patch("w2v.data.dataset.ensure_nltk_gutenberg_corpora") as mock_ensure:
        BoWDataset(corpus_dir=tmp_path, download=True)
        mock_ensure.assert_called_once_with(tmp_path)


def test_invalid_inputs_raise_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SkipGramDataset()

    with pytest.raises(ValueError):
        BoWDataset()

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with patch("w2v.data.dataset.ensure_nltk_gutenberg_corpora") as mock_dl:

        def create_file(target):
            (Path(target) / "sample.txt").write_text("word1 word2 word3 word1 word2 word3", encoding="utf-8")
            return [Path(target) / "sample.txt"]

        mock_dl.side_effect = create_file
        ds = SkipGramDataset(corpus_dir=empty_dir)
        assert mock_dl.called
        assert len(ds.corpus_files) == 1

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    with patch("w2v.data.dataset.ensure_nltk_gutenberg_corpora"):
        with pytest.raises(ValueError, match="does not contain any text files"):
            BoWDataset(corpus_dir=failed_dir)


def test_downloader_target_corpora(tmp_path: Path) -> None:
    mock_gutenberg = MagicMock()
    mock_gutenberg.raw.return_value = "dummy text"
    with patch("nltk.download"), patch("nltk.corpus.gutenberg", mock_gutenberg):
        downloaded = ensure_nltk_gutenberg_corpora(tmp_path)
        assert len(downloaded) == 18

    # When all files already exist, should return [] without calling nltk
    downloaded_again = ensure_nltk_gutenberg_corpora(tmp_path)
    assert downloaded_again == []


def test_dataset_corpus_arg_and_budget_cap(tmp_path: Path) -> None:
    # Single list corpus
    ds1 = SkipGramDataset(corpus=["one", "two", "three"], min_count=1)
    assert ds1.vocab_size == 3

    # Nested list corpus
    ds2 = SkipGramDataset(corpus=[["a", "b"], ["c", "d"]], min_count=1)
    assert ds2.vocab_size == 4

    # Min count discards all words
    with pytest.raises(ValueError, match="Vocabulary is empty"):
        SkipGramDataset(tokens=["rare1", "rare2"], min_count=10)

    # Max tokens slicing across documents
    (tmp_path / "d1.txt").write_text("w1 w2 w3", encoding="utf-8")
    (tmp_path / "d2.txt").write_text("w4 w5 w6", encoding="utf-8")
    ds3 = SkipGramDataset(corpus_dir=tmp_path, max_tokens=4, min_count=1)
    assert sum(len(d) for d in ds3.documents) == 4
