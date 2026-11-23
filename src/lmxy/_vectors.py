__all__ = ['merge_cos_embeddings']

from collections.abc import Sequence

import numpy as np
from scipy.cluster import hierarchy

from ._types import Embedding


def merge_cos_embeddings(
    vectors: np.ndarray | Sequence[Embedding], max_similarity: float = 1.0
) -> np.ndarray:
    """
    Takes set of L2 normalized vectors and averages vectors that are too close.
    Returns new averaged and L2 normalized vectors.
    Each member has cosine similarity of at least max_similarity to its center.

    Negative thresholds use a pairwise similarity floor of zero.
    No-op for similarity >= 1.
    """
    vectors = np.asarray(vectors, np.float32)
    if vectors.ndim != 2:
        raise ValueError('Input should be array of (N vectors, D features)')
    if vectors.shape[1] < 2:
        raise ValueError('Too short vectors')
    if not vectors.any(axis=1).all():
        raise ValueError('Invalid vectors: some are all 0s')

    if max_similarity >= 1 or vectors.shape[0] <= 1:
        return vectors  # (n d)

    # Do clustering
    z = hierarchy.linkage(vectors, method='complete', metric='cosine')

    # Nonnegative pairwise similarity also bounds similarity to the unit center
    max_distance = 1 - max(0.0, max_similarity)
    labels = hierarchy.fcluster(z, max_distance, criterion='distance') - 1

    # Average vector for each cluster (not L2 normalized)
    n_clusters = labels.max() + 1
    centers = np.zeros((n_clusters, vectors.shape[1]), dtype=np.float32)
    np.add.at(centers, labels, vectors)

    # L2 norm
    centers /= np.linalg.vector_norm(centers, axis=1, keepdims=True)
    return centers
