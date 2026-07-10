__all__ = ['get_sparse_encoder']

from collections.abc import Iterable

from fastembed import SparseTextEmbedding
from glow import memoize, streaming

from ._env import env
from ._types import SparseEncode, SparseEncoding


@memoize()
def get_sparse_encoder(
    model_name: str,
    batch_size: int = 256,
    timeout: float | None = None,
    **kwargs,
) -> SparseEncode:
    if env.HF_HUB_OFFLINE or env.TRANSFORMERS_OFFLINE:
        kwargs['local_files_only'] = True

    try:
        # Prioritize GPU over CPU
        kwargs['providers'] = ['CUDAExecutionProvider']
        model = SparseTextEmbedding(model_name, **kwargs)
    except Exception:  # noqa: BLE001
        # If not available, fallback to CPU
        kwargs['providers'] = None
        model = SparseTextEmbedding(model_name, **kwargs)

    def encode(texts: Iterable[str]) -> list[SparseEncoding]:
        embeddings = model.embed(texts, batch_size=batch_size)
        return [(e.indices.tolist(), e.values.tolist()) for e in embeddings]

    if timeout is None:
        return encode
    return streaming(encode, batch_size=batch_size, timeout=timeout)
