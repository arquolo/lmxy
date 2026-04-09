__all__ = ['unpack_response']

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

from ._types import LlmFunction


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

    obj: AsyncIterator[str] | str
    match ret:
        # Chat.(a)chat
        # Synthesizer.(a)synthesize
        # Synthesizer.(a)synthesize if output_cls is set
        case Response(response=None) | PydanticResponse(response=None):
            obj = ''

        # Chat.(a)chat
        # Synthesizer.(a)synthesize
        case AgentChatResponse(response=obj) | Response(response=str(obj)):
            pass

        # Synthesizer.(a)synthesize if output_cls is set
        case PydanticResponse(response=BaseModel() as resp):
            obj = resp.model_dump_json()

        # Synthesizer(stream=True).asynthesize
        case AsyncStreamingResponse():
            obj = ret.response_gen

        # Chat.(a)astream_chat
        case StreamingAgentChatResponse():
            obj = ret.async_response_gen()

        case _:
            msg = f'Unsupported type: {type(ret)}'
            raise NotImplementedError(msg)

    return obj, ret.source_nodes
