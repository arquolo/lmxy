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
    z = hierarchy.linkage(vectors, method='average', metric='cosine')
    distances = z[:, 2]  # (n) of 0..2
    n_clusters = np.count_nonzero(distances >= 1 - max_similarity) + 1  # C
    assert n_clusters <= vectors.shape[0]

    # Assign labels
    labels = hierarchy.fcluster(z, n_clusters, 'maxclust') - 1  # (n) of 0..C-1

    # Average vector for each cluster (not L2 normalized)
    n_clusters = labels.max() + 1
    sums = np.zeros((n_clusters, vectors.shape[1]), dtype=vectors.dtype)
    np.add.at(sums, labels, vectors)
    counts = np.bincount(labels, minlength=n_clusters)[:, None]
    vectors = sums / counts  # (C d)

    # L2 norm
    vectors /= np.linalg.vector_norm(vectors, axis=1, keepdims=True)

    return vectors
