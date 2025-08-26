__all__ = [
    'BatchSparseEncoding',
    'SparseEncode',
    'VectorStore',
]

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode
    from llama_index.core.vector_stores.types import (
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from qdrant_client.http.models import Filter

type BatchSparseEncoding = tuple[list[list[int]], list[list[float]]]
type SparseEncode = Callable[[Iterable[str]], BatchSparseEncoding]


class VectorStore(Protocol):
    # CRUD: Create & Update (overwrite)
    async def async_add(self, nodes: list['BaseNode']) -> list[str]: ...

    # CRUD: Read
    async def aquery(
        self,
        query: 'VectorStoreQuery',
        /,
        *,
        qdrant_filters: 'Filter | None' = ...,
        dense_threshold: float | None = ...,
    ) -> 'VectorStoreQueryResult': ...

    # CRUD: Delete
    async def adelete(self, ref_doc_id: str) -> None: ...
    async def adelete_nodes(self, node_ids: list[str]) -> None: ...
