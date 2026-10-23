from unittest.mock import MagicMock, patch

from w2v.preprocessing.tokenizer import BasicTokenizer


def test_basic_tokenizer_default() -> None:
    tokenizer = BasicTokenizer()
    text = "Hello, World! 123 Word2Vec."
    tokens = tokenizer.tokenize(text)
    assert tokens == ["hello", "world", "123", "word2vec"]


def test_basic_tokenizer_no_lowercase() -> None:
    tokenizer = BasicTokenizer(lowercase=False)
    text = "Hello, World!"
    tokens = tokenizer.tokenize(text)
    assert tokens == ["Hello", "World"]


def test_basic_tokenizer_none_text() -> None:
    tokenizer = BasicTokenizer()
    assert tokenizer.tokenize(None) == []  # type: ignore[arg-type]


def test_basic_tokenizer_stopwords() -> None:
    mock_stopwords = MagicMock()
    mock_stopwords.words.return_value = ["the", "a"]
    with patch("nltk.download"), patch("nltk.corpus.stopwords", mock_stopwords):
        tokenizer = BasicTokenizer(remove_stopwords=True)
        tokens = tokenizer.tokenize("The quick brown fox and a dog")
        assert "the" not in tokens
        assert "a" not in tokens
        assert "quick" in tokens


def test_basic_tokenizer_stopwords_fallback_on_error() -> None:
    with patch("nltk.download", side_effect=Exception("network down")):
        tokenizer = BasicTokenizer(remove_stopwords=True)
        assert len(tokenizer.stopwords) == 0
        tokens = tokenizer.tokenize("the dog")
        assert tokens == ["the", "dog"]


def test_tokenize_corpus() -> None:
    tokenizer = BasicTokenizer()
    docs = ["First document.", "Second document with more words."]
    results = list(tokenizer.tokenize_corpus(docs))
    assert len(results) == 2
    assert results[0] == ["first", "document"]
    assert results[1] == ["second", "document", "with", "more", "words"]
