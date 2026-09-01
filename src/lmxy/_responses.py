__all__ = ['tokens_from_response', 'unpack_response']

from collections.abc import AsyncIterable, Iterable

from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel

from ._async import gen_to_agen
from ._types import (
    HasAsyncResponseGen,
    HasResponse,
    HasResponseGen,
    LlmResponse,
    Tokens,
)


async def unpack_response(
    ret: LlmResponse,
) -> tuple[Tokens, list[NodeWithScore]]:
    if isinstance(ret, tuple):
        return ret
    if isinstance(ret, Tokens):
        return ret, []
    if isinstance(ret, AsyncIterable | str):
        return Tokens(ret), []
    return tokens_from_response(ret), ret.source_nodes


def tokens_from_response(
    lrsp: 'HasResponse | HasResponseGen | HasAsyncResponseGen',
) -> Tokens:
    match lrsp:
        # Chat.(a)chat
        # Synthesizer.(a)synthesize
        # Synthesizer.(a)synthesize if output_cls is set
        # -> llama_index.core.base.response.schema.{Response,PydanticResponse}
        case HasResponse(response=None):
            return Tokens()

        # Chat.(a)chat
        # Synthesizer.(a)synthesize
        # -> llama_index.core.base.response.schema.Response
        # -> llama_index.core.chat_engine.types.AgentChatResponse
        case HasResponse(response=str(obj)):
            return Tokens(obj)

        # Synthesizer.(a)synthesize if output_cls is set
        # -> llama_index.core.base.response.schema.PydanticResponse
        case HasResponse(response=BaseModel() as rsp):
            return Tokens(rsp.model_dump_json())

        # Synthesizer(stream=True).asynthesize
        # -> llama_index.core.base.response.schema.AsyncStreamingResponse
        case HasResponseGen(response_gen=AsyncIterable() as agen):
            return Tokens(agen)

        # Synthesizer(stream=True).asynthesize
        # -> llama_index.core.base.response.schema.StreamingResponse
        case HasResponseGen(response_gen=Iterable() as gen):
            return Tokens(gen_to_agen(gen))

        # Chat.(a)astream_chat
        # -> llama_index.core.chat_engine.types.StreamingAgentChatResponse
        case HasAsyncResponseGen():
            return Tokens(lrsp.async_response_gen())

        case _:
            msg = f'Unsupported type: {type(lrsp)}'
            raise NotImplementedError(msg)
