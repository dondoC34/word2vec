import logging
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

logger = logging.getLogger("Tokenizer")


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.
    Custom tokenizers should inherit from this class and implement `tokenize`.
    """

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize a single text string into a list of string tokens.
        """
        pass

    def tokenize_corpus(self, corpus: Iterable[str]) -> Iterator[list[str]]:
        """
        Tokenize an iterable of texts, yielding token lists for each input text.
        """
        for doc in corpus:
            yield self.tokenize(doc)


class BasicTokenizer(BaseTokenizer):
    """
    Standard tokenizer for a corpus:
    - Normalizes Unicode characters (NFKC)
    - Strips leading and trailing whitespace
    - Retains alphabetic characters, digits, and internal whitespace
    - Lowercases text by default
    - Optionally removes English stopwords using NLTK
    """

    def __init__(self, lowercase: bool = True, remove_stopwords: bool = False):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords

        if self.remove_stopwords:
            try:
                import nltk

                nltk.download("stopwords", quiet=True)
                from nltk.corpus import stopwords

                self.stopwords = set([word.replace("'", "") for word in stopwords.words("english")])
                logger.info("Initialized BasicTokenizer with %d English stopwords removed.", len(self.stopwords))
            except Exception as e:
                logger.warning("Failed to load NLTK stopwords: %s. Proceeding without stopword removal.", e)
                self.stopwords = set()
        else:
            self.stopwords = set()
            logger.debug("Initialized BasicTokenizer (lowercase=%s, remove_stopwords=False).", lowercase)

    def _clean_text(self, text: str) -> str:
        """
        Normalize Unicode, filter out punctuation/symbols, and convert case if enabled.
        """
        text = unicodedata.normalize("NFKC", text)
        kept_chars = []
        for ch in text:
            cat = unicodedata.category(ch)
            if ch.isspace() or cat.startswith("L") or cat.startswith("N"):
                kept_chars.append(ch)
        cleaned = "".join(kept_chars)
        if self.lowercase:
            cleaned = cleaned.lower()
        return cleaned.strip()

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize a single text string into non-empty tokens.
        """
        if text is None:
            logger.debug("Received None text to tokenize; returning empty list.")
            return []
        cleaned = self._clean_text(text)
        tokens = [tok for tok in cleaned.split() if tok and tok not in self.stopwords]
        return tokens
