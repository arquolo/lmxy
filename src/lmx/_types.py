__all__ = [
    'BatchSparseEncoding',
    'SparseEncode',
]

from collections.abc import Callable, Iterable

type BatchSparseEncoding = tuple[list[list[int]], list[list[float]]]
type SparseEncode = Callable[[Iterable[str]], BatchSparseEncoding]
