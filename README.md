# Word2Vec (`w2v`): Modular PyTorch Embeddings Framework

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B%20(CUDA%2FCPU)-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A clean, modular, and production-ready PyTorch implementation of **Word2Vec (Skip-Gram with Negative Sampling - SGNS)**, architected from the ground up for extensibility toward additional architectures such as **Bag of Words (BoW / CBOW)**.

---

## Key Highlights

- **Document Boundary Protection**: Sliding context windows strictly stop at the end of each document and restart at the beginning of the next. Tokens from adjacent books never contaminate each other with spurious training pairs.
- **Automated Corpus Synchronization**: Built-in hook automatically detects missing classic benchmark literature from NLTK Gutenberg and downloads only the delta, while fully preserving pre-existing custom texts (such as *The Picture of Dorian Gray*).
- **Tied vs. Untied Weights**: Supports both shared embedding matrices (`tied_weights=True`) and unshared matrices (`tied_weights=False`). In untied mode, exported embeddings automatically compute the arithmetic mean $\frac{\mathbf{W}_{\text{in}} + \mathbf{W}_{\text{out}}}{2}$, capturing complementary target and context semantics.
- **Universal Embedding Formats**: Exports directly to the standard Google Word2Vec / Gensim text format, Stanford GloVe format, compressed NumPy (`.npz`), and PyTorch (`.pt`).
- **Dedicated CLI & Python API**: Run training with a single shell command (`w2v`) or integrate directly into Python scripts.
- **High-Performance PyTorch Backend**: Vectorized negative sampling powered by native C++/CUDA multinomial sampling, Adam optimizer, and multi-threaded DataLoader batching.

---

## Repository Structure

```text
word2vec/
├── corpuses/                      # Text corpus files (*.txt)
│   ├── portrait_of_dorian_gray.txt # Custom user book (always preserved)
│   ├── austen-emma.txt            # NLTK Gutenberg synchronized books
│   └── melville-moby_dick.txt     # ...
├── embeddings/                    # Exported word embeddings (auto-created)
│   └── word2vec_embeddings.txt    # Standard Word2Vec KeyedVectors format
├── logs/                          # Timestamped execution logs
│   └── word2vec_YYYYMMDD_HHMMSS.log
├── w2v/                           # Core Python Package
│   ├── __init__.py                # Public API exports
│   ├── cli.py                     # Command-line interface (argparse)
│   ├── skipgram.py                # Skip-Gram (SGNS) architecture & pipeline
│   ├── bow.py                     # Continuous Bag-of-Words (CBOW) architecture & pipeline
│   ├── data/
│   │   ├── __init__.py            # Dataset & downloader exports
│   │   ├── dataset.py             # BaseWord2VecDataset, SkipGramDataset, BoWDataset
│   │   └── downloader.py          # Delta Gutenberg ingestion hook
│   └── preprocessing/
│       ├── __init__.py            # Tokenizer exports
│       └── tokenizer.py           # BaseTokenizer & BasicTokenizer
├── pyproject.toml                 # Packaging, dependencies & CLI entrypoints
└── README.md
```

---

## Installation

### Prerequisites
- Python >= 3.12
- PyTorch >= 2.5 with CUDA (recommended for GPU acceleration) or CPU

### 1. Using `uv` (Recommended)
[`uv`](https://github.com/astral-sh/uv) provides rapid environment resolution and dependency installation:

```bash
git clone https://github.com/your-username/word2vec.git
cd word2vec

# Install all dependencies and register the w2v CLI command in editable mode
uv sync
uv pip install -e .
```

### 2. Using standard `pip`
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e .
```

---

## Command-Line Interface (`w2v`)

Once installed, the CLI tool `w2v` is available directly in your terminal.

### Quick Command Example
```bash
w2v --model skipgram --embedding-dim 128 --output-dir ./embeddings
```

Or using `uv run`:
```bash
uv run w2v --model skipgram --embedding-dim 128 --output-dir ./embeddings
```

### Full CLI Options Table

Run `w2v --help` at any time to display the parameter table:

| Parameter | Default | Description |
|---|---|---|
| `--model` | `skipgram` | Model architecture to use (`skipgram`, `bow`, `cbow`). |
| `--embedding-dim` | `128` | Dimensionality of dense embedding vectors. |
| `--window-size` | `2` | Context half-window size (tokens left and right). |
| `--n-negative` | `5` | Number of negative samples drawn per positive pair. |
| `--min-count` | `3` | Minimum token frequency threshold (discards rare words). |
| `--tied-weights` / `--no-tied-weights` | `True` | If False, saves arithmetic mean of $(\mathbf{W}_{\text{in}} + \mathbf{W}_{\text{out}}) / 2$. |
| `--corpus-dir` | `./corpuses` | Directory containing text documents (`*.txt`). |
| `--sync-corpora` / `--no-sync-corpora` | `False` | Automatic download and sync of missing NLTK Gutenberg corpora (disabled by default). |
| `--max-tokens` | `None` | Optional maximum token budget across all documents. |
| `--epochs` | `5` | Number of training epochs. |
| `--batch-size` | `512` | Mini-batch size for DataLoader. |
| `--lr` | `0.005` | Learning rate for Adam optimizer. |
| `--output-dir` | `./embeddings` | Output directory (created automatically if missing). |
| `--output-filename` | `None` | Custom output filename (defaults to timestamped `word2vec_embeddings_YYYYddmm_HHMM.txt`). |
| `--format` | `word2vec` | Embedding export format (`word2vec`, `glove`, `numpy`, `pt`). |
| `--device` | `None` | Compute device (`cuda` or `cpu`, auto-detected if None). |
| `--log-dir` | `./logs` | Directory where timestamped execution log files are stored. |
| `--log-level` | `INFO` | Logging severity threshold (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `-h, --help` | | Show help message with parameters table and exit. |

---

## Python API Usage

You can also import and customize the pipeline directly in your Python code:

```python
from w2v import SkipGram, BoW, SkipGramDataset, BoWDataset, BasicTokenizer

# 1. Initialize Skip-Gram with automatic corpus download & boundary protection
sg = SkipGram(
    corpus_dir="./corpuses",
    embedding_dim=128,
    window_size=2,
    n_negative=5,
    min_count=3,
    tied_weights=True,
    sync_corpora_hook=True,
)

# 2. Train the model across epochs
history = sg.train(epochs=5, batch_size=512, lr=0.005)

# 3. Export embeddings to disk
saved_file = sg.save_embeddings(output_dir="./embeddings", format="word2vec")
print(f"Embeddings saved to: {saved_file}")

# 4. Perform cosine similarity semantic queries
neighbors = sg.most_similar("dorian", top_k=5)
for word, score in neighbors:
    print(f"  {word}: {score:+.4f}")

# 5. Extract a single word vector as a NumPy array
vector = sg.get_embedding("alice")
print("Vector shape:", vector.shape)

# Alternatively, initialize and train Continuous Bag-of-Words (CBOW / BoW)
bow = BoW(
    corpus_dir="./corpuses",
    embedding_dim=128,
    window_size=2,
    n_negative=5,
    min_count=3,
    tied_weights=True,
)
bow_history = bow.train(epochs=5, batch_size=512, lr=0.005)
```

---

## Corpora Management & Ingestion

### 1. Corpora Download & Empty Directory Handling
By default, automatic download of corpus files is **disabled**.

If the specified `--corpus-dir` is empty or does not contain any plain text files (`*.txt`) to train embeddings, `w2v` will:
1. Log an error that no text files (`*.txt`) were found in the folder.
2. Suggest that benchmark text corpora can be downloaded automatically using `--sync-corpora`.
3. Provide the exact equivalent command line example to run.

#### Enabling Automatic Download
To automatically download standard benchmark literary texts from NLTK Gutenberg:
- **CLI**: Pass `--sync-corpora`
  ```bash
  w2v --corpus-dir ./corpuses --sync-corpora
  ```
- **Python**: Pass `sync_corpora_hook=True` (or `download=True` on Dataset classes)
  ```python
  sg = SkipGram(corpus_dir="./corpuses", sync_corpora_hook=True)
  ```

Existing local files (such as `portrait_of_dorian_gray.txt`) are **never overwritten or removed**.

### 2. Using Custom Text Corpora
To train on your own domain-specific documents:
1. Place any plain UTF-8 text files (`*.txt`) inside your folder (e.g. `./my_data/book1.txt`, `./my_data/book2.txt`).
2. Point `corpus_dir` to that folder (download is disabled by default):
   ```bash
   w2v --corpus-dir ./my_data --output-dir ./my_embeddings
   ```

### 3. Document Boundary Protection
In conventional naive implementations, all corpus texts are concatenated into one long token list. This causes the last words of Book A and the first words of Book B to be treated as adjacent context neighbors, creating spurious pairings (e.g. pairing characters from different novels).

`w2v` tokenizes and stores documents as isolated sequences (`List[List[str]]`). During Skip-Gram pair generation, the sliding context window is strictly constrained to the boundaries of each individual document.

---

## Output & Results Structure

### 1. Where are the Results Saved?
By default, embeddings are exported to:
```text
./embeddings/word2vec_embeddings.txt
```
If the directory does not exist, it is created automatically.

### 2. File Format Details
The primary export format is the **standard Word2Vec KeyedVectors text format** (compatible with Google C Word2Vec, Gensim, spaCy, and fastText):

- **Header (Line 1)**: `<vocab_size> <embedding_dimension>`
- **Body (Lines 2 to N+1)**: `<word> <v_1> <v_2> ... <v_D>`

#### Example File Content:
```text
14967 128
emma 0.335441 0.153285 -0.228424 -0.198774 0.053688 ...
by 0.062310 -0.032598 0.009740 0.025141 -0.006778 ...
jane 0.153899 0.098117 -0.230463 0.048816 0.006228 ...
dorian 0.118905 -0.057310 0.081238 -0.003662 0.157168 ...
```

### 3. Loading the Saved Embeddings in Gensim
Because the format adheres strictly to the official standard, you can load it in Gensim with one line:

```python
from gensim.models import KeyedVectors

# Load trained embeddings
wv = KeyedVectors.load_word2vec_format("embeddings/word2vec_embeddings.txt", binary=False)

# Query semantic similarities
print(wv.most_similar("dorian", topn=5))
print("Cosine similarity:", wv.similarity("captain", "ahab"))
```

### 4. Alternative Output Formats
Using the `--format` option:
- `--format word2vec` (default): Text file with `<vocab_size> <dim>` header.
- `--format glove`: Pure text format without header line (`<word> <v_1> ...`).
- `--format numpy` (or `npz`): Compressed NumPy archive containing `words` and `vectors` arrays (`np.load("embeddings.npz")`).
- `--format pt`: PyTorch dictionary containing `{"words": list, "vectors": torch.Tensor}`.

---

## Architecture & Extensibility (Roadmap)

`w2v` separates tokenization, dataset generation, neural architecture, and training into modular components:

```text
[Raw Documents / NLTK Hook]
           │
           ▼
[BaseTokenizer / BasicTokenizer]
           │
           ▼
[BaseWord2VecDataset] ◄── Abstract interface for Word2Vec family
      ├── SkipGramDataset (Skip-Gram context pairs)
      └── BoWDataset / CBOWDataset (Continuous Bag-of-Words)
           │
           ▼
[SkipGramModel (SGNS) / BoWModel (CBOW)]
           │
           ▼
[SkipGram / BoW Engines]
           │
           ▼
[Vector Export: save_embeddings()]
```

### Continuous Bag-of-Words (CBOW / BoW) Support
The abstract base class `BaseWord2VecDataset` in [`w2v.data.dataset`](file:///C:/Users/utente/repos/word2vec/w2v/data/dataset.py) defines the contract for vocabulary mapping and tensor generation. The CBOW model slides across context windows to predict the central target word from the mean of context tokens using `nn.EmbeddingBag`, sharing the same tokenizer, corpus synchronization, and export utilities.

---

## Logging & Diagnostics

`w2v` features centralized structured logging. Each execution automatically logs timestamp, severity level, and component source:

```text
2026-10-20 21:34:42 [INFO] [CLI]: w2v CLI: Starting Word2Vec training
2026-10-20 21:34:42 [INFO] [DataDownloader]: Scanning directory 'corpuses' for NLTK Gutenberg corpora...
2026-10-20 21:34:43 [INFO] [Dataset]: Skip-Gram pair generation complete: 3,629,054 pairs (0 cross-doc contamination).
2026-10-20 21:34:46 [INFO] [Word2Vec]: Epoch [1/5] - Average Loss: 7.1884 across 7088 batches.
2026-10-20 21:34:46 [INFO] [Word2Vec]: Word embeddings successfully saved to 'embeddings/word2vec_embeddings.txt'.
```

Logs are displayed in real time on the console and simultaneously persisted to timestamped files in `./logs/`.

---

## License

This project is licensed under the MIT License.
