__all__ = ['get_full_response', 'tokens_from_response', 'unpack_response']

from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from io import StringIO

from llama_index.core.base.response.schema import (
    AsyncStreamingResponse,
    PydanticResponse,
    Response,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel

from ._types import LlmFunction, LlmResponse


async def unpack_response[**P](
    f: LlmFunction[P], *args: P.args, **kwargs: P.kwargs
) -> tuple[
    AsyncIterator[str] | str,
    list[NodeWithScore],
]:
    aw_agen = f(*args, **kwargs)
    ret = await aw_agen if isinstance(aw_agen, Awaitable) else aw_agen
    if isinstance(ret, AsyncIterator | str):
        return ret, []
    return tokens_from_response(ret), ret.source_nodes


def tokens_from_response(lrsp: LlmResponse) -> AsyncIterator[str] | str:
    match lrsp:
        # Chat.(a)chat
        # Synthesizer.(a)synthesize
        # Synthesizer.(a)synthesize if output_cls is set
        case Response(response=None) | PydanticResponse(response=None):
            return ''

        # Chat.(a)chat
        # Synthesizer.(a)synthesize
        case AgentChatResponse(response=obj) | Response(response=str(obj)):
            return obj

        # Synthesizer.(a)synthesize if output_cls is set
        case PydanticResponse(response=BaseModel() as rsp):
            return rsp.model_dump_json()

        # Synthesizer(stream=True).asynthesize
        case AsyncStreamingResponse():
            return lrsp.response_gen

        # Chat.(a)astream_chat
        case StreamingAgentChatResponse():
            return lrsp.async_response_gen()

        case _:
            msg = f'Unsupported type: {type(lrsp)}'
            raise NotImplementedError(msg)


async def get_full_response(tokens: AsyncIterable[str]) -> str:
    buf = StringIO()
    async for tk in tokens:
        buf.write(tk)
    return buf.getvalue()
