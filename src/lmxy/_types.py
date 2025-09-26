__all__ = [
    'BatchSparseEncoding',
    'LlmFunction',
    'LlmResponse',
    'SparseEncode',
    'Tokenize',
    'VectorStore',
]

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)
from typing import TYPE_CHECKING, Any, Protocol, Union

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import (
        AsyncStreamingResponse,
        PydanticResponse,
        Response,
        StreamingResponse,
    )
    from llama_index.core.chat_engine.types import (
        AgentChatResponse,
        StreamingAgentChatResponse,
    )
    from llama_index.core.schema import BaseNode
    from llama_index.core.vector_stores.types import (
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from qdrant_client.http.models import Filter

type BatchSparseEncoding = tuple[list[list[int]], list[list[float]]]
type SparseEncode = Callable[[Iterable[str]], BatchSparseEncoding]

type LlmResponse = Union[  # noqa: UP007
    'Response',
    'PydanticResponse',
    'StreamingResponse',
    'AsyncStreamingResponse',
    'AgentChatResponse',
    'StreamingAgentChatResponse',
    str,
]
type LlmFunction[**P] = Callable[
    P, Awaitable[LlmResponse | AsyncIterator[str]] | AsyncIterator[str]
]
type Tokenize = Callable[[str], list[Any]]


class VectorStore(Protocol):
    # CRUD: Create & Update (overwrite)
    async def async_add(
        self, nodes: Sequence['BaseNode']
    ) -> Sequence[str]: ...

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
    async def adelete_nodes(self, node_ids: Sequence[str]) -> None: ...
