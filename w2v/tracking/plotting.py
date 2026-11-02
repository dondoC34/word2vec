"""
MLflow visualization and diagnostic figure generation module.

This module provides utility functions for creating Matplotlib figures to log as MLflow artifacts:
- Loss convergence curve across training epochs.
- Pairwise cosine similarity matrix heatmap for key semantic anchor words.
- Two-dimensional PCA projection map of learned word embedding representations.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib

# Set non-interactive Agg backend to ensure thread-safety and prevent GUI popups
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger("MLflowPlotting")


def create_loss_figure(history: list[float], model_name: str = "Word2Vec") -> plt.Figure:
    """
    Generates a Matplotlib figure plotting training loss convergence across epochs.

    Args:
        history: Sequence of scalar loss values recorded per epoch.
        model_name: Name of the neural model architecture (for title labeling).

    Returns:
        plt.Figure: The Matplotlib figure ready to log as an MLflow artifact.
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))
    epochs = list(range(1, len(history) + 1))
    ax.plot(epochs, history, marker="o", color="#1f77b4", linewidth=2.0, markersize=5)
    ax.set_title(f"{model_name} Loss Convergence Across {len(history)} Epochs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Negative Log-Likelihood Loss", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    if len(history) <= 20:
        ax.set_xticks(epochs)
    fig.tight_layout()
    return fig


def create_heatmap_figure(
    model: Any,
    dataset: Any | None = None,
    candidate_words: list[str] | None = None,
) -> plt.Figure | None:
    """
    Generates a cosine similarity matrix heatmap for representative vocabulary tokens.

    Args:
        model: Trained model instance exposing `get_embedding()`.
        dataset: Optional dataset instance exposing `stoi`. Falls back to `getattr(model, "dataset", None)`.
        candidate_words: Optional list of specific probe tokens to include in the matrix.

    Returns:
        plt.Figure | None: The heatmap figure, or None if fewer than 2 tokens are present in the vocabulary.
    """
    ds = dataset or getattr(model, "dataset", None)
    stoi = getattr(ds, "stoi", {}) if ds else {}
    if not stoi or not hasattr(model, "get_embedding"):
        return None

    # Benchmark terms covering literary themes, entities, and nature
    if candidate_words is None:
        candidate_words = [
            "dorian",
            "gray",
            "lord",
            "henry",
            "alice",
            "rabbit",
            "queen",
            "sea",
            "ship",
            "captain",
            "ahab",
            "whale",
            "king",
            "ghost",
            "man",
            "woman",
            "father",
            "time",
            "life",
            "death",
        ]

    # Filter to words that actually exist in the trained vocabulary
    valid_words = [w for w in candidate_words if w in stoi]
    if len(valid_words) < 2:
        return None

    # Limit to top 15 words for visual clarity
    valid_words = valid_words[:15]
    embeddings = []
    retained_words = []
    for w in valid_words:
        emb = model.get_embedding(w)
        if emb is not None:
            norm = np.linalg.norm(emb)
            if norm > 1e-9:
                embeddings.append(emb / norm)
                retained_words.append(w)

    valid_words = retained_words
    if len(valid_words) < 2:
        return None

    emb_matrix = np.array(embeddings)
    sim_matrix = np.dot(emb_matrix, emb_matrix.T)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cax = ax.imshow(sim_matrix, cmap="Blues", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(valid_words)))
    ax.set_yticks(range(len(valid_words)))
    ax.set_xticklabels(valid_words, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(valid_words, fontsize=9)
    ax.set_title("Cosine Similarity Heatmap of Representative Embeddings", fontsize=11, fontweight="bold")

    # Annotate cells with scalar similarity values
    for r in range(len(valid_words)):
        for c in range(len(valid_words)):
            val = sim_matrix[r, c]
            txt_color = "white" if abs(val) > 0.45 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=7)

    fig.tight_layout()
    return fig


def create_pca_figure(
    model: Any,
    dataset: Any,
    candidate_words: list[str] | None = None,
    max_words: int = 25,
) -> plt.Figure | None:
    """
    Generates a 2D PCA projection map figure for representative vocabulary embeddings.

    Args:
        model: Trained model instance exposing `get_embedding()`.
        dataset: Dataset instance exposing `stoi` and token `counts`.
        candidate_words: Optional probe words to project onto 2D space.
        max_words: Maximum number of tokens displayed on the projection scatter plot.

    Returns:
        plt.Figure | None: The 2D PCA projection figure, or None if insufficient words are found.
    """
    stoi = getattr(dataset, "stoi", {})
    if not stoi or not hasattr(model, "get_embedding"):
        return None

    if candidate_words is None:
        candidate_words = [
            "dorian",
            "gray",
            "lord",
            "henry",
            "alice",
            "rabbit",
            "queen",
            "hatter",
            "sea",
            "ship",
            "captain",
            "ahab",
            "hamlet",
            "king",
            "macbeth",
            "ghost",
            "man",
            "woman",
            "father",
            "mother",
            "time",
            "day",
            "night",
            "life",
            "death",
        ]

    # Keep candidate words present in vocabulary; fall back to most frequent if too few
    selected = [w for w in candidate_words if w in stoi]
    if len(selected) < 4:
        sorted_vocab = sorted(stoi.keys(), key=lambda w: getattr(dataset, "counts", {}).get(w, 0), reverse=True)
        selected = [w for w in sorted_vocab[:max_words] if w in stoi]

    selected = selected[:max_words]
    if len(selected) < 3:
        return None

    embeddings = []
    valid_words = []
    for w in selected:
        emb = model.get_embedding(w)
        if emb is not None:
            embeddings.append(emb)
            valid_words.append(w)

    if len(valid_words) < 3:
        return None

    emb_arr = np.array(embeddings)
    if emb_arr.shape[1] < 2:
        return None

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(emb_arr)
    var_ratio = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(coords[:, 0], coords[:, 1], color="#2980b9", s=100, alpha=0.85, edgecolors="white", linewidth=1.2)

    for i, w in enumerate(valid_words):
        ax.annotate(
            w,
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.set_title(
        f"2D PCA Projection Map of Learned Embeddings\n(PC 1: {var_ratio[0]:.1%}, PC 2: {var_ratio[1]:.1%})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(f"Principal Component 1 ({var_ratio[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"Principal Component 2 ({var_ratio[1]:.1%})", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig
