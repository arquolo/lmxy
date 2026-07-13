__all__ = [
    'Embedding',
    'LlmFunction',
    'LlmResponse',
    'SparseEncode',
    'Tokenize',
    'Tokens',
    'VectorStore',
    'get_full_response',
]

from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Sequence,
)
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Never, Protocol, Union

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
    from llama_index.core.schema import BaseNode, NodeWithScore
    from llama_index.core.vector_stores.types import (
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from qdrant_client.http.models import Filter

type Embedding = list[float]
type SparseEncoding = tuple[list[int], Embedding]
type BatchSparseEmbedding = list[SparseEncoding]
type SparseEncode = Callable[[Sequence[str]], Sequence[SparseEncoding]]

type LlmResponse = Union[  # noqa: UP007
    'Response',
    'PydanticResponse',
    'StreamingResponse',
    'AsyncStreamingResponse',
    'AgentChatResponse',
    'StreamingAgentChatResponse',
    str,
]
type NativeResponse = Union[  # noqa: UP007
    tuple['Tokens', Sequence['NodeWithScore']],
    'Tokens',
]
type LlmFunction[**P] = Callable[
    P,
    Awaitable[LlmResponse | NativeResponse | AsyncIterator[str]]
    | NativeResponse
    | AsyncIterator[str],
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


@dataclass(frozen=True, slots=True)
class Tokens:
    obj: AsyncIterator[str] | str | None = None

    def __await__(self) -> Generator[Any, Any, str]:
        if self.obj is None or isinstance(self.obj, str):
            return _await(self.obj or '')
        return get_full_response(self.obj).__await__()

    def __aiter__(self) -> AsyncIterator[str]:
        if self.obj is None:
            return _empty_aiter()
        if isinstance(self.obj, str):
            return _ayield(self.obj) if self.obj else _empty_aiter()
        return self.obj


async def get_full_response(tokens: AsyncIterable[str]) -> str:
    buf = StringIO()
    async for tk in tokens:
        buf.write(tk)
    return buf.getvalue()


def _await[T](x: T) -> Generator[Never, Any, T]:
    yield from ()
    return x


async def _ayield[T](x: T) -> AsyncIterator[T]:
    yield x


async def _empty_aiter() -> AsyncIterator[Never]:
    return
    yield
