__all__ = ['tokens_from_response', 'unpack_response']

from collections.abc import AsyncIterator, Awaitable

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

from ._types import LlmFunction, LlmResponse, Tokens


async def unpack_response[**P](
    f: LlmFunction[P], *args: P.args, **kwargs: P.kwargs
) -> tuple[Tokens, list[NodeWithScore]]:
    aw_agen = f(*args, **kwargs)
    ret = await aw_agen if isinstance(aw_agen, Awaitable) else aw_agen
    if isinstance(ret, AsyncIterator):
        return Tokens(ret), []
    if isinstance(ret, str):
        return Tokens(ret), []
    return tokens_from_response(ret), ret.source_nodes


def tokens_from_response(lrsp: LlmResponse) -> Tokens:
    match lrsp:
        # Chat.(a)chat
        # Synthesizer.(a)synthesize
        # Synthesizer.(a)synthesize if output_cls is set
        case Response(response=None) | PydanticResponse(response=None):
            return Tokens()

        # Chat.(a)chat
        # Synthesizer.(a)synthesize
        case AgentChatResponse(response=obj) | Response(response=str(obj)):
            return Tokens(obj)

        # Synthesizer.(a)synthesize if output_cls is set
        case PydanticResponse(response=BaseModel() as rsp):
            return Tokens(rsp.model_dump_json())

        # Synthesizer(stream=True).asynthesize
        case AsyncStreamingResponse():
            return Tokens(lrsp.response_gen)

        # Chat.(a)astream_chat
        case StreamingAgentChatResponse():
            return Tokens(lrsp.async_response_gen())

        case _:
            msg = f'Unsupported type: {type(lrsp)}'
            raise NotImplementedError(msg)
