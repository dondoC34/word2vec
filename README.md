# Word2Vec (`w2v`) — Modular PyTorch Embeddings Framework

[![CI](https://github.com/dondoC34/word2vec/actions/workflows/ci.yml/badge.svg)](https://github.com/dondoC34/word2vec/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B%20(CUDA%2FCPU)-ee4c2c.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A modular, production-ready PyTorch framework for training **Word2Vec** static word embeddings using **Skip-Gram with Negative Sampling (SGNS)** and **Continuous Bag-of-Words (CBOW / BoW)**.

Designed with clean software engineering practices, `w2v` provides strict document-boundary isolation, automated corpus ingestion, multi-format vector export, and a full-featured **MLflow** experiment tracking and management CLI.

---

## Table of Contents

- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Interface (`w2v`)](#command-line-interface-w2v)
- [MLflow Management Suite (`w2v mlflow`)](#mlflow-management-suite-w2v-mlflow)
- [Python API & Notebooks](#python-api--notebooks)
- [Data Pipeline & Corpora Management](#data-pipeline--corpora-management)
- [Embedding Formats & Downstream Usage](#embedding-formats--downstream-usage)

---

## Key Features

- **Dual Neural Architectures**: Train either **Skip-Gram** (`SkipGram`) or **Continuous Bag-of-Words** (`BoW` / `CBOW`) models built on a unified `BaseWord2Vec` training engine with vectorized negative sampling.
- **Document Boundary Isolation**: Sliding context windows strictly stop at the end of each document and reset at the start of the next, preventing spurious cross-document co-occurrence pairs.
- **Smart Corpus Ingestion**: Automatically downloads classic benchmark literature (NLTK Project Gutenberg) if the target corpus directory is missing or contains no `.txt` files, while preserving custom local texts.
- **Tied & Untied Weight Topologies**: Supports single shared embedding matrices (`--tied-weights`) as well as separate target/context matrices (`--no-tied-weights`), automatically exporting the arithmetic mean $\frac{\mathbf{W}_{\text{in}} + \mathbf{W}_{\text{out}}}{2}$ when untied.
- **Integrated MLflow Experiment Tracking**: Automatically records hyperparameters, per-epoch loss, throughput (words/sec), CPU/RAM/GPU telemetry, Git commit metadata, SHA-256 corpus manifests, and diagnostic visualizations (loss curves, 2D PCA projections, cosine similarity heatmaps).
- **Safe Timestamped Exports**: Prevents accidental file overwrites by defaulting to timestamped filenames (`w2v_{model}_embeddings_YYYYmmdd_hh_MM.<ext>`) across standard Word2Vec (`KeyedVectors`), GloVe, NumPy (`.npz`), and PyTorch (`.pt`) formats.

---

## Repository Structure

```text
word2vec/
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI pipeline (lint, type-check, pytest)
├── notebooks/
│   ├── skipgram_training.ipynb    # Interactive Skip-Gram walkthrough & analysis
│   └── cbow_training.ipynb        # Interactive CBOW walkthrough & analysis
├── tests/                         # Unit and integration test suite (pytest)
│   └── ...
├── w2v/                           # Core Python package
│   ├── __init__.py                # Public package exports
│   ├── base.py                    # BaseWord2Vec shared training, querying & export engine
│   ├── skipgram.py                # Skip-Gram (SGNS) model architecture & pipeline
│   ├── bow.py                     # Continuous Bag-of-Words (CBOW) architecture & pipeline
│   ├── cli.py                     # CLI entrypoints (`w2v` and `w2v mlflow`)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # BaseWord2VecDataset, SkipGramDataset, BoWDataset
│   │   └── downloader.py          # Non-destructive NLTK Gutenberg corpus downloader
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── tokenizer.py           # BaseTokenizer & regex-based BasicTokenizer
│   └── tracking/                  # Modular MLflow tracking package
│       ├── __init__.py            # Clean public tracking API
│       ├── config.py              # YAML configuration loader & defaults
│       ├── manager.py             # Storage inspection, run queries, export & cleanup
│       ├── plotting.py            # Diagnostic Matplotlib figures (loss, PCA, heatmaps)
│       ├── server.py              # Tracking server process lifecycle & port probes
│       └── tracker.py             # MLflowTracker session manager & artifact logger
├── mlflow-config.yml              # Default local MLflow server & tracking configuration
├── pyproject.toml                 # Project metadata, dependencies & tool configurations
└── README.md
```

---

## Installation

### Prerequisites

- **Python**: `>= 3.12`
- **PyTorch**: `>= 2.5` (CUDA supported and auto-detected; falls back cleanly to CPU)

### Option 1: Using `uv` (Recommended)

[`uv`](https://docs.astral.sh/uv/) provides fast, reproducible dependency resolution using `uv.lock`:

```bash
git clone https://github.com/dondoC34/word2vec.git
cd word2vec

# Create virtual environment, install dependencies, and register the `w2v` CLI
uv sync
```

### Option 2: Using `pip`

```bash
git clone https://github.com/dondoC34/word2vec.git
cd word2vec

python -m venv .venv
# Activate on Linux/macOS:
source .venv/bin/activate
# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -e .
```

---

## Quick Start

### 1. Train a Skip-Gram Model

Run training with default hyperparameters. If `./corpuses` does not exist or contains no `.txt` files, sample benchmark corpora are downloaded automatically:

```bash
uv run w2v --model skipgram --embedding-dim 128 --epochs 5
```

### 2. Train a Continuous Bag-of-Words (CBOW) Model

```bash
uv run w2v --model bow --embedding-dim 128 --window-size 3 --epochs 5 --batch-size 512
```

### 3. Launch the MLflow Dashboard

Inspect loss curves, 2D PCA embedding projections, similarity heatmaps, and system metrics in your browser:

```bash
# Start the local MLflow tracking server (http://127.0.0.1:5000)
uv run w2v mlflow start

# Or open the dashboard directly in your default web browser
uv run w2v mlflow ui
```

---

## Command-Line Interface (`w2v`)

Once installed, the `w2v` command is available directly in your environment (or via `uv run w2v`).

### Example Commands

```bash
# Train Skip-Gram with untied weights and export in PyTorch (.pt) format
w2v --model skipgram --no-tied-weights --format pt --output-dir ./embeddings

# Train CBOW with a token budget cap and custom output filename
w2v --model bow --max-tokens 500000 --output-filename custom_cbow.txt

# Train without MLflow tracking
w2v --model skipgram --epochs 3 --no-mlflow
```

### CLI Parameters Reference

Run `w2v --help` at any time to display the full options table:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--model` | `skipgram` | Model architecture to train (`skipgram`, `bow`, `cbow`). |
| `--embedding-dim` | `128` | Dimensionality of dense word embedding vectors. |
| `--window-size` | `2` | Context half-window size (number of tokens to the left and right). |
| `--n-negative` | `5` | Number of negative samples drawn per positive context pair. |
| `--min-count` | `3` | Minimum word frequency threshold (rarer tokens are discarded). |
| `--tied-weights` / `--no-tied-weights` | `True` | Share input/output weights. If `False`, exports $\frac{\mathbf{W}_{\text{in}} + \mathbf{W}_{\text{out}}}{2}$. |
| `--corpus-dir` | `./corpuses` | Directory containing plain-text training documents (`*.txt`). |
| `--sync-corpora` / `--no-sync-corpora` | `False` | Download missing NLTK Gutenberg benchmark books when `.txt` files already exist. |
| `--max-tokens` | `None` | Optional cap on the total number of tokens ingested across all documents. |
| `--epochs` | `5` | Number of full passes over the training dataset. |
| `--batch-size` | `512` | Mini-batch size for the PyTorch `DataLoader`. |
| `--lr` | `0.005` | Learning rate for the Adam optimizer. |
| `--output-dir` | `./embeddings` | Target directory for exported embeddings (created if missing). |
| `--output-filename` | `None` | Custom output filename (defaults to `w2v_{model}_embeddings_YYYYmmdd_hh_MM.<ext>`). |
| `--format` | `word2vec` | Export file format (`word2vec`, `glove`, `numpy`, `pt`). |
| `--device` | `None` | Compute device (`cuda` or `cpu`; auto-detected when `None`). |
| `--mlflow` / `--no-mlflow` | `True` | Enable or disable automatic MLflow experiment tracking. |
| `--mlflow-config` | `None` | Path to a custom YAML configuration file (defaults to `mlflow-config.yml`). |
| `--log-corpora-content` / `--no-log-corpora-content` | `False` | Upload raw `.txt` corpus files to MLflow artifacts (by default only the manifest is logged). |
| `--log-dir` | `./logs` | Directory where timestamped execution logs are saved. |
| `--log-level` | `INFO` | Console and file logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `-h, --help` | — | Display the formatted parameter table and exit. |

---

## MLflow Management Suite (`w2v mlflow`)

`w2v` ships with a dedicated subcommand suite to manage the local SQLite-backed MLflow environment (`mlflow.db` and `./mlartifacts`) without requiring manual database or filesystem operations.

> [!NOTE]
> Every `w2v mlflow` subcommand accepts an optional `-c, --config <path>` flag to specify a custom YAML configuration file. By default, settings are read from `mlflow-config.yml`.

### Subcommands Overview

| Subcommand | Alias | Description |
| :--- | :--- | :--- |
| `w2v mlflow start` | `run` | Launch the local MLflow tracking server and web UI. |
| `w2v mlflow status` | — | Check server health, database size, artifact disk usage, and run counts. |
| `w2v mlflow list` | `runs` | Display an ASCII table of recent runs, hyperparameters, and loss metrics. |
| `w2v mlflow best` | — | Identify the best-performing run (lowest loss) and optionally export its artifacts. |
| `w2v mlflow clear` | `clean` | Safely purge tracking database records and/or artifact storage. |
| `w2v mlflow ui` | `open` | Open the MLflow web dashboard in the system's default browser. |

### Usage Examples

```bash
# 1. Start the tracking server on a custom port
w2v mlflow start --host 127.0.0.1 --port 5000

# 2. Check server status and storage footprint
w2v mlflow status

# 3. List top 5 Skip-Gram runs sorted by lowest training loss
w2v mlflow list --model skipgram --sort-by loss --limit 5

# 4. Inspect the best run and export its artifacts to a local directory
w2v mlflow best --model skipgram --export-to ./best_checkpoint

# 5. Clean up only heavy artifact files while keeping run metrics in SQLite
w2v mlflow clear --artifacts-only --yes

# 6. Full reset of both SQLite database and artifacts (with interactive confirmation)
w2v mlflow clear
```

---

## Python API & Notebooks

### Programmatic Usage

Integrate `SkipGram` or `BoW` directly into custom Python workflows:

```python
from w2v import BoW, SkipGram

# 1. Initialize and train a Skip-Gram model
sg = SkipGram(
    corpus_dir="./corpuses",
    embedding_dim=128,
    window_size=2,
    n_negative=5,
    min_count=3,
    tied_weights=True,
)
history = sg.train(epochs=5, batch_size=512, lr=0.005)

# 2. Export trained embeddings
saved_path = sg.save_embeddings(output_dir="./embeddings", format="word2vec")
print(f"Saved embeddings to: {saved_path}")

# 3. Query semantic nearest neighbors via cosine similarity
for word, similarity in sg.most_similar("dorian", top_k=5):
    print(f"{word:<15} {similarity:+.4f}")

# 4. Retrieve a single word vector as a NumPy array
vector = sg.get_embedding("london")
print(f"Vector shape: {vector.shape}")

# 5. Train a Continuous Bag-of-Words (CBOW) model
bow = BoW(
    corpus_dir="./corpuses",
    embedding_dim=128,
    window_size=2,
    n_negative=5,
    min_count=3,
)
bow.train(epochs=5, batch_size=512, lr=0.005)
```

### Interactive Jupyter Notebooks

Explore end-to-end training, loss convergence, and embedding space visualizations in the included notebooks:

- `notebooks/skipgram_training.ipynb` — Skip-Gram (SGNS) pipeline walkthrough.
- `notebooks/cbow_training.ipynb` — Continuous Bag-of-Words (CBOW) pipeline walkthrough.

---

## Data Pipeline & Corpora Management

### Automatic Corpus Fallback & Synchronization

1. **Empty or Missing Directory**:
   If `--corpus-dir` does not exist or contains zero `.txt` files, `w2v` logs an informational notice and **automatically downloads sample benchmark corpora** from NLTK Project Gutenberg into that directory so training works out of the box.
2. **Pre-Populated Directory**:
   If `--corpus-dir` already contains one or more `.txt` files, `w2v` uses only your local files by default. Pass `--sync-corpora` if you want to fetch any missing Gutenberg benchmark books alongside your custom files. Existing files are never overwritten or deleted.

### Document Boundary Protection

Rather than concatenating all files into a single flat token stream—which contaminates context windows at the boundaries between unrelated documents—`w2v` processes corpora as isolated document sequences (`list[list[str]]`). Context windows strictly terminate at the end of each file.

---

## Embedding Formats & Downstream Usage

By default, embeddings are saved to `./embeddings/w2v_{model}_embeddings_YYYYmmdd_hh_MM.<ext>`. You can override the filename with `--output-filename`.

| Format Flag | Extension | Structure |
| :--- | :--- | :--- |
| `--format word2vec` | `.txt` | `<vocab_size> <dim>` header followed by `<word> <v_1> ... <v_D>` lines. |
| `--format glove` | `.txt` | Headerless plain text with `<word> <v_1> ... <v_D>` per line. |
| `--format numpy` | `.npz` | Compressed NumPy archive with `words` and `vectors` arrays. |
| `--format pt` | `.pt` | PyTorch dictionary containing `{"words": list[str], "vectors": torch.Tensor}`. |

### Loading in Gensim

```python
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format(
    "embeddings/w2v_skipgram_embeddings_20261102_11_00.txt",
    binary=False,
)
print(wv.most_similar("whale", topn=5))
```

### Loading in NumPy / PyTorch

```python
import numpy as np
import torch

# NumPy (.npz)
data = np.load("embeddings/w2v_skipgram_embeddings_20261102_11_00.npz")
words, vectors = data["words"], data["vectors"]

# PyTorch (.pt)
checkpoint = torch.load("embeddings/w2v_skipgram_embeddings_20261102_11_00.pt", weights_only=True)
words, tensor_vectors = checkpoint["words"], checkpoint["vectors"]
```
