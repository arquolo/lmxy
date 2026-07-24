__all__ = [
    'Embedding',
    'LlmFunction',
    'LlmResponse',
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
    Sequence,
)
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Union

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
    from llama_index.core.schema import NodeWithScore

from ._async import ayield, ayield_never, genreturn

type Embedding = list[float]
type SparseEncoding = tuple[list[int], Embedding]
type BatchSparseEmbedding = list[SparseEncoding]
type SparseEncode = Callable[[Sequence[str]], list[SparseEncoding]]

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
    tuple['Tokens', list['NodeWithScore']],
    'Tokens',
]
type LlmFunction[**P] = Callable[
    P,
    Awaitable[LlmResponse | NativeResponse | AsyncIterable[str]]
    | NativeResponse
    | AsyncIterable[str],
]
type Tokenize = Callable[[str], list[Any]]


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
