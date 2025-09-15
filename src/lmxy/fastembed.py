__all__ = ['get_sparse_encoder']

from collections.abc import Iterable

from fastembed import SparseTextEmbedding
from glow import memoize

from ._env import env
from ._types import BatchSparseEncoding, SparseEncode


@memoize()
def get_sparse_encoder(
    model_name: str, batch_size: int = 256, **kwargs
) -> SparseEncode:
    if env.HF_HUB_OFFLINE or env.TRANSFORMERS_OFFLINE:
        kwargs['local_files_only'] = True

    # prioritize GPU over CPU
    try:
        model = SparseTextEmbedding(
            model_name, providers=['CUDAExecutionProvider'], **kwargs
        )
    # If provider is not available, fallback to CPU
    except Exception:  # noqa: BLE001
        model = SparseTextEmbedding(model_name, **kwargs)

    def encode(texts: Iterable[str]) -> BatchSparseEncoding:
        embeddings = model.embed(texts, batch_size=batch_size)
        indices, values = zip(
            *((e.indices.tolist(), e.values.tolist()) for e in embeddings)
        )
        return list(indices), list(values)

    return encode
