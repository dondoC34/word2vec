from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from w2v.data.dataset import SkipGramDataset
from w2v.skipgram import SkipGram, SkipGramModel, get_model, setup_logging


class DummyDataset(Dataset):
    def __init__(self) -> None:
        self.vocab_size = 10
        self.stoi = {f"word{i}": i for i in range(10)}
        self.itos = [f"word{i}" for i in range(10)]

    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int):
        return idx, (idx + 1) % 10

    def collate_fn(self, batch):
        centers = torch.tensor([b[0] for b in batch], dtype=torch.long)
        contexts = torch.tensor([b[1] for b in batch], dtype=torch.long)
        neg_samples = torch.zeros((len(batch), 3), dtype=torch.long)
        return centers, contexts, neg_samples


def test_skipgram_model_forward() -> None:
    # Tied weights
    model_tied = SkipGramModel(vocab_size=20, embedding_dim=16, tied_weights=True)
    assert model_tied.embedding is model_tied.in_embedding
    assert model_tied.out_embedding is model_tied.in_embedding

    center = torch.tensor([1, 2], dtype=torch.long)
    context = torch.tensor([3, 4], dtype=torch.long)
    neg = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.long)

    loss_tied = model_tied(center, context, neg)
    assert isinstance(loss_tied, torch.Tensor)
    assert loss_tied.ndim == 0
    assert loss_tied.item() > 0

    # Untied weights
    model_untied = SkipGramModel(vocab_size=20, embedding_dim=16, tied_weights=False)
    assert model_untied.out_embedding is not model_untied.in_embedding
    loss_untied = model_untied(center, context, neg)
    assert isinstance(loss_untied, torch.Tensor)
    assert loss_untied.ndim == 0


def test_setup_logging(tmp_path: Path) -> None:
    log_file = setup_logging(log_dir=tmp_path, log_filename="test.log")
    assert log_file.exists()
    assert log_file.name == "test.log"


def test_skipgram_training_and_inference(tmp_path: Path) -> None:
    tokens = ["alpha", "beta", "gamma", "delta", "epsilon", "alpha", "beta", "gamma"]
    dataset = SkipGramDataset(tokens=tokens, window_size=2, min_count=1, n_negative=2)

    sg = SkipGram(dataset=dataset, embedding_dim=8, tied_weights=True, device="cpu")
    assert sg.vocab_size == dataset.vocab_size

    # Train 1 epoch
    history = sg.train(epochs=1, batch_size=4)
    assert len(history) == 1
    assert history[0] > 0

    # Embeddings
    emb = sg.get_embedding("alpha")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (8,)
    assert sg.get_embedding("nonexistent_word") is None

    # Similar words
    sims = sg.most_similar("alpha", top_k=2)
    assert len(sims) <= 2
    assert all(isinstance(w, str) and isinstance(score, float) for w, score in sims)
    assert sg.most_similar("nonexistent_word") == []

    # Save formats
    p_w2v = sg.save_embeddings(output_dir=tmp_path, format="word2vec", filename="w2v.txt")
    assert p_w2v.exists()
    p_glove = sg.save_embeddings(output_dir=tmp_path, format="glove", filename="glove.txt")
    assert p_glove.exists()
    p_npz = sg.save_embeddings(output_dir=tmp_path, format="numpy", filename="emb.npz")
    assert p_npz.exists()
    p_pt = sg.save_embeddings(output_dir=tmp_path, format="pt", filename="emb.pt")
    assert p_pt.exists()

    # Default timestamped filename and no-overwrite behavior
    p_default1 = sg.save_embeddings(output_dir=tmp_path)
    assert p_default1.exists()
    assert "word2vec_embeddings_" in p_default1.name
    p_default2 = sg.save_embeddings(output_dir=tmp_path)
    assert p_default2.exists()
    assert p_default1 != p_default2

    with pytest.raises(ValueError):
        sg.save_embeddings(output_dir=tmp_path, format="unsupported_format")


def test_skipgram_untied_weights(tmp_path: Path) -> None:
    tokens = ["cat", "dog", "fish", "bird", "cat", "dog"]
    dataset = SkipGramDataset(tokens=tokens, window_size=1, min_count=1, n_negative=2)

    sg = SkipGram(dataset=dataset, embedding_dim=8, tied_weights=False, device="cpu")
    sg.train(epochs=1, batch_size=2)

    emb = sg.get_embedding("cat")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (8,)

    sims = sg.most_similar("cat", top_k=2)
    assert len(sims) > 0

    p_w2v = sg.save_embeddings(output_dir=tmp_path, format="word2vec", filename="untied.txt")
    assert p_w2v.exists()


def test_skipgram_load_corpus_from_dir(tmp_path: Path) -> None:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "doc1.txt").write_text("the quick brown fox jumps", encoding="utf-8")
    (d / "doc2.txt").write_text("over the lazy dog", encoding="utf-8")

    sg = SkipGram(corpus_dir=d, sync_corpora_hook=False, embedding_dim=8, max_tokens=6, min_count=1)
    assert len(sg.tokens) == 6
    assert len(sg.documents) > 0
    assert sg.dataset is not None
    assert sg.model is not None


def test_skipgram_error_handling(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    sg = SkipGram(embedding_dim=8)
    with pytest.raises(FileNotFoundError):
        sg.load_corpus_from_dir(tmp_path / "nonexistent")

    with pytest.raises(ValueError):
        sg.load_corpus_from_dir(empty_dir)

    with pytest.raises(ValueError):
        sg.build_dataset()

    with pytest.raises(ValueError):
        sg.init_model()

    with pytest.raises(ValueError):
        sg.train()

    with pytest.raises(ValueError):
        sg.save_embeddings()

    # set_dataset without inferable vocab_size
    class NoVocabDataset(Dataset):
        def __getitem__(self, idx: int):
            return idx

        def __len__(self) -> int:
            return 1

    with pytest.raises(ValueError):
        sg.set_dataset(NoVocabDataset())


def test_get_model() -> None:
    m = get_model(vocab_size=50, embedding_dim=16)
    assert isinstance(m, SkipGramModel)
    assert m.vocab_size == 50
    assert m.embedding_dim == 16
