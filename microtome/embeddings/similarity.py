"""Similarity computation utilities for embeddings."""

from typing import List, Tuple

import numpy as np


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def compute_adjacent_similarities(embeddings: np.ndarray) -> List[float]:
    """Compute cosine similarities between adjacent embeddings.

    Args:
        embeddings: Array of shape (n, embedding_dim)

    Returns:
        List of n-1 similarity scores between adjacent pairs
    """
    if len(embeddings) < 2:
        return []

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms

    # Compute dot products between adjacent embeddings
    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

    return similarities.tolist()


def find_boundary_candidates(
    similarities: List[float],
    threshold: float = 0.5,
) -> List[Tuple[int, float]]:
    """Find potential boundary positions where similarity drops below threshold.

    Args:
        similarities: List of similarity scores between adjacent items
        threshold: Similarity threshold below which to propose a boundary

    Returns:
        List of (position, similarity_score) tuples for boundary candidates
    """
    candidates = []

    for i, similarity in enumerate(similarities):
        if similarity < threshold:
            # Position is the index where the boundary should occur
            # (i.e., between item i and item i+1)
            candidates.append((i + 1, similarity))

    return candidates


def compute_similarity_statistics(similarities: List[float]) -> dict:
    """Compute statistics about similarity scores.

    Args:
        similarities: List of similarity scores

    Returns:
        Dictionary with statistics (mean, std, min, max, etc.)
    """
    if not similarities:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
        }

    arr = np.array(similarities)

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(similarities),
        "median": float(np.median(arr)),
    }


def adaptive_threshold(
    similarities: List[float],
    percentile: float = 25.0,
) -> float:
    """Compute an adaptive threshold based on similarity distribution.

    Args:
        similarities: List of similarity scores
        percentile: Percentile to use as threshold

    Returns:
        Adaptive threshold value
    """
    if not similarities:
        return 0.5

    return float(np.percentile(similarities, percentile))
