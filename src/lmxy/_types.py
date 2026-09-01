__all__ = [
    'Embedding',
    'LlmFunction',
    'SparseEncode',
    'Tokenize',
    'Tokens',
    'get_full_response',
]

from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
)
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore
    from pydantic import BaseModel

from ._async import ayield, ayield_never, genreturn

type Embedding = list[float]
type SparseEncoding = tuple[list[int], Embedding]
type BatchSparseEmbedding = list[SparseEncoding]
type SparseEncode = Callable[[Iterable[str]], list[SparseEncoding]]


@dataclass(frozen=True, slots=True)
class Tokens:
    obj: AsyncIterable[str] | str | None = None

    def __await__(self) -> Generator[Any, Any, str]:
        if self.obj is None or isinstance(self.obj, str):
            return genreturn(self.obj or '')
        return get_full_response(self.obj).__await__()

    def __aiter__(self) -> AsyncIterator[str]:
        if self.obj is None:
            return ayield_never()
        if isinstance(self.obj, str):
            return ayield(self.obj) if self.obj else ayield_never()
        return aiter(self.obj)


async def get_full_response(tokens: AsyncIterable[str]) -> str:
    buf = StringIO()
    async for tk in tokens:
        buf.write(tk)
    return buf.getvalue()


@runtime_checkable
class HasResponse(Protocol):
    @property
    def response(self) -> 'BaseModel | str | None': ...
    @property
    def source_nodes(self) -> list['NodeWithScore']: ...


@runtime_checkable
class HasResponseGen(Protocol):
    @property
    def response_gen(self) -> Iterable[str] | AsyncIterable[str]: ...
    @property
    def source_nodes(self) -> list['NodeWithScore']: ...


@runtime_checkable
class HasAsyncResponseGen(Protocol):
    def async_response_gen(self) -> AsyncIterable[str]: ...
    @property
    def source_nodes(self) -> list['NodeWithScore']: ...


type LlmResponse = (
    HasResponse
    | HasResponseGen
    | HasAsyncResponseGen
    | tuple[Tokens, list['NodeWithScore']]
    | Tokens
    | AsyncIterable[str]
    | str
)
type LlmFunction[**P] = Callable[P, LlmResponse | Awaitable[LlmResponse]]
type Tokenize = Callable[[str], list[Any]]
