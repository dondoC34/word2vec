from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from w2v.bow import CBOW, BoW, BoWModel
from w2v.data.dataset import BoWDataset


def test_bow_model_forward() -> None:
    # Tied weights
    model_tied = BoWModel(vocab_size=20, embedding_dim=16, tied_weights=True)
    assert model_tied.embedding is model_tied.in_embedding
    assert model_tied.out_embedding is None

    flat_contexts = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    targets = torch.tensor([6, 7], dtype=torch.long)
    neg_samples = torch.tensor([[8, 9, 10], [11, 12, 13]], dtype=torch.long)

    loss_tied = model_tied(flat_contexts, offsets, targets, neg_samples)
    assert isinstance(loss_tied, torch.Tensor)
    assert loss_tied.ndim == 0
    assert loss_tied.item() > 0

    # Untied weights
    model_untied = BoWModel(vocab_size=20, embedding_dim=16, tied_weights=False)
    assert model_untied.out_embedding is not None
    loss_untied = model_untied(flat_contexts, offsets, targets, neg_samples)
    assert isinstance(loss_untied, torch.Tensor)
    assert loss_untied.ndim == 0


def test_bow_training_and_inference(tmp_path: Path) -> None:
    tokens = ["sun", "moon", "star", "planet", "comet", "sun", "moon", "star"]
    dataset = BoWDataset(tokens=tokens, window_size=2, min_count=1, n_negative=2)

    bow = BoW(dataset=dataset, embedding_dim=8, tied_weights=True, device="cpu")
    assert bow.vocab_size == dataset.vocab_size

    history = bow.train(epochs=1, batch_size=4)
    assert len(history) == 1
    assert history[0] > 0

    emb = bow.get_embedding("sun")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (8,)
    assert bow.get_embedding("unknown_token") is None

    sims = bow.most_similar("sun", top_k=2)
    assert len(sims) <= 2
    assert bow.most_similar("unknown_token") == []

    p_w2v = bow.save_embeddings(output_dir=tmp_path, format="word2vec", filename="cbow.txt")
    assert p_w2v.exists()
    p_glove = bow.save_embeddings(output_dir=tmp_path, format="glove", filename="cbow_glove.txt")
    assert p_glove.exists()
    p_npz = bow.save_embeddings(output_dir=tmp_path, format="numpy", filename="cbow.npz")
    assert p_npz.exists()
    p_pt = bow.save_embeddings(output_dir=tmp_path, format="pt", filename="cbow.pt")
    assert p_pt.exists()

    # Default timestamped filename and no-overwrite behavior
    p_default1 = bow.save_embeddings(output_dir=tmp_path)
    assert p_default1.exists()
    assert "w2v_bow_embeddings_" in p_default1.name
    p_default2 = bow.save_embeddings(output_dir=tmp_path)
    assert p_default2.exists()
    assert p_default1 != p_default2

    with pytest.raises(ValueError):
        bow.save_embeddings(output_dir=tmp_path, format="invalid")


def test_bow_untied_weights(tmp_path: Path) -> None:
    tokens = ["red", "green", "blue", "yellow", "red", "green"]
    dataset = BoWDataset(tokens=tokens, window_size=1, min_count=1, n_negative=2)

    bow = CBOW(dataset=dataset, embedding_dim=8, tied_weights=False, device="cpu")
    bow.train(epochs=1, batch_size=2)

    emb = bow.get_embedding("red")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (8,)

    sims = bow.most_similar("red", top_k=2)
    assert len(sims) > 0

    p_w2v = bow.save_embeddings(output_dir=tmp_path, format="word2vec", filename="untied_cbow.txt")
    assert p_w2v.exists()


def test_bow_load_corpus_from_dir(tmp_path: Path) -> None:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "doc1.txt").write_text("deep learning neural networks", encoding="utf-8")
    (d / "doc2.txt").write_text("artificial intelligence word embeddings", encoding="utf-8")

    bow = BoW(corpus_dir=d, sync_corpora_hook=False, embedding_dim=8, max_tokens=6, min_count=1)
    assert len(bow.tokens) == 6
    assert len(bow.documents) > 0
    assert bow.dataset is not None
    assert bow.model is not None


def test_bow_error_handling(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    bow = BoW(embedding_dim=8, sync_corpora_hook=False)
    with patch("w2v.base.ensure_nltk_gutenberg_corpora"):
        with pytest.raises(ValueError):
            bow.load_corpus_from_dir(empty_dir)

    with pytest.raises(ValueError):
        bow.build_dataset()

    with pytest.raises(ValueError):
        bow.init_model()

    with pytest.raises(ValueError):
        bow.train()

    with pytest.raises(ValueError):
        bow.save_embeddings()

    class NoVocabDataset(Dataset):
        def __getitem__(self, idx: int):
            return idx

        def __len__(self) -> int:
            return 1

    with pytest.raises(ValueError):
        bow.set_dataset(NoVocabDataset())
