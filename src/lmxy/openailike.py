__all__ = ['OpenAiLike']

import functools
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from json.decoder import JSONDecodeError
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import httpx
import openai
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    TextBlock,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.utils import parse_partial_json
from llama_index_instrumentation import get_dispatcher
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import Field, PrivateAttr

from ._types import Tokenize
from .tokenizer import get_tokenizer
from .util import aretry

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool
    from openai.types import Completion
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionMessageParam,
    )
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall


dispatcher = get_dispatcher(__name__)
_errors = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def _llm_retry[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        max_retries = getattr(args[0], 'max_retries', 0)
        if max_retries <= 0:
            return f(*args, **kwargs)

        r = aretry(
            *_errors,
            max_attempts=max_retries,
            timeout=60,
            wait_max=20,
        )
        return r(f)(*args, **kwargs)

    return functools.update_wrapper(wrapper, f)


class OpenAiLike(FunctionCallingLLM):
    model_config: ClassVar = {'extra': 'forbid'}

    model: str = Field(description='Model to use.')
    temperature: float = Field(
        default=0.1,
        description='The temperature to use during generation.',
        ge=0.0,
        le=2.0,
    )
    context_window: int = Field(
        default=3900,
        description=LLMMetadata.model_fields['context_window'].description,
    )
    max_new_tokens: int | None = Field(
        description='The maximum number of tokens to generate.',
        default=None,
        gt=0,
    )
    reasoning_effort: Literal['low', 'medium', 'high', 'minimal'] | None = (
        Field(
            default=None,
            description='The effort to use for reasoning models.',
        )
    )
    additional_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description='Additional kwargs for the API',
    )

    is_function_calling_model: bool = Field(
        default=True,
        description='Whether the model is a function calling model.',
    )
    tokenize: Tokenize | str | None = Field(
        default=None,
        description=(
            'Tokenize function or the name of a tokenizer model '
            'from Hugging Face. '
            'If left as None, then this disables inference of max_tokens.'
        ),
    )

    base_url: str = Field(description='The base URL for API.')
    api_key: str = Field(default='', description='API key.')
    default_headers: dict[str, str] | None = Field(
        default=None, description='The default headers for API requests.'
    )
    max_retries: int = Field(
        default=3, description='The maximum number of API retries.', ge=0
    )
    timeout: float | None = Field(
        default=60.0,
        description='The timeout, in seconds, for API requests.',
        ge=0,
    )
    http_client: httpx.Client | None = None
    async_http_client: httpx.AsyncClient | None = None

    _client: OpenAI = PrivateAttr()
    _aclient: AsyncOpenAI = PrivateAttr()

    def model_post_init(self, context) -> None:
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            http_client=self.http_client,
        )
        self._aclient = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            http_client=self.async_http_client,
        )
        if isinstance(self.tokenize, str):
            self.tokenize = get_tokenizer(self.tokenize)

    @classmethod
    def class_name(cls) -> str:
        return 'OpenAILike'

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens or -1,
            is_chat_model=True,  # * Bold assumption
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    @llm_completion_callback()
    @_llm_retry
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponse:
        """Complete the prompt."""
        kws = self._get_kwds(prompt=prompt, formatted=formatted, **kwargs)
        resp = self._client.completions.create(stream=False, **kws)
        return _Decoder().completion(resp)

    @llm_completion_callback()
    @_llm_retry
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponse:
        """Complete the prompt."""
        kws = self._get_kwds(prompt=prompt, formatted=formatted, **kwargs)
        resp = await self._aclient.completions.create(stream=False, **kws)
        return _Decoder().completion(resp)

    @llm_completion_callback()
    @_llm_retry  # NOTE: Stream breaks are on caller's side
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> Generator[CompletionResponse]:
        """Stream complete the prompt."""
        kws = self._get_kwds(prompt=prompt, formatted=formatted, **kwargs)
        s = self._client.completions.create(stream=True, **kws)
        return _map_ctx(s, _Decoder().completion)

    @llm_completion_callback()
    @_llm_retry  # NOTE: Stream breaks are on caller's side
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> AsyncGenerator[CompletionResponse]:
        """Stream complete the prompt."""
        kws = self._get_kwds(prompt=prompt, formatted=formatted, **kwargs)
        s = await self._aclient.completions.create(stream=True, **kws)
        return _amap_ctx(s, _Decoder().completion)

    @llm_chat_callback()
    @_llm_retry
    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        """Chat with the model."""
        kws = self._get_kwds(messages=messages, **kwargs)
        resp = self._client.chat.completions.create(stream=False, **kws)
        return _Decoder().chat(resp)

    @llm_chat_callback()
    @_llm_retry
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs
    ) -> ChatResponse:
        """Chat with the model."""
        kws = self._get_kwds(messages=messages, **kwargs)
        resp = await self._aclient.chat.completions.create(stream=False, **kws)
        return _Decoder().chat(resp)

    @llm_chat_callback()
    @_llm_retry  # NOTE: Stream breaks are on caller's side
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs
    ) -> Generator[ChatResponse]:
        kws = self._get_kwds(messages=messages, **kwargs)
        s = self._client.chat.completions.create(stream=True, **kws)
        return _map_ctx(s, _Decoder().chat)

    @llm_chat_callback()
    @_llm_retry  # NOTE: Stream breaks are on caller's side
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs
    ) -> AsyncGenerator[ChatResponse]:
        kwds = self._get_kwds(messages=messages, **kwargs)
        s = await self._aclient.chat.completions.create(stream=True, **kwds)
        return _amap_ctx(s, _Decoder().chat)

    def _get_kwds(
        self,
        prompt: str | None = None,
        formatted: bool = False,
        messages: Sequence[ChatMessage] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        all_kwargs: dict[str, Any] = {}

        if prompt is not None and not formatted and self.completion_to_prompt:
            prompt = self.completion_to_prompt(prompt)
            all_kwargs['prompt'] = prompt

        if messages:
            all_kwargs['messages'] = (
                _to_openai_message_dict(m) for m in messages
            )

        num_ctx = self.context_window
        all_kwargs |= {
            'model': self.model,
            'temperature': self.temperature,
            'extra_body': {'num_ctx': num_ctx} | self.additional_kwargs,
        }

        # Infer max_tokens for the payload, if possible.
        # NOTE: non-chat completion endpoint requires max_tokens to be set
        if self.max_new_tokens is not None:
            all_kwargs['max_tokens'] = self.max_new_tokens
        elif (
            prompt
            and self.tokenize is not None
            and not isinstance(self.tokenize, str)
        ):
            num_tokens = len(self.tokenize(prompt))
            if num_tokens >= num_ctx:
                msg = (
                    f'The prompt has {num_tokens} tokens, which is too long'
                    ' for the model. Please use a prompt that fits within'
                    f' {num_ctx} tokens.'
                )
                raise ValueError(msg)
            all_kwargs['max_tokens'] = num_ctx - num_tokens

        return all_kwargs | kwargs

    def _prepare_chat_with_tools(
        self,
        tools: Sequence['BaseTool'],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Predict and call the tool."""
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True)
            for tool in tools
        ]

        if self.metadata.is_function_calling_model:
            for tool_spec in tool_specs:
                if tool_spec['type'] == 'function':
                    tool_spec['function']['strict'] = False
                    tool_spec['function']['parameters'][
                        'additionalProperties'
                    ] = False

        messages = chat_history or []

        if user_msg is not None:
            if isinstance(user_msg, str):
                user_msg = ChatMessage(content=user_msg)
            messages.append(user_msg)

        return {
            'messages': messages,
            'tools': tool_specs or None,
            'tool_choice': (
                ('required' if tool_required else 'auto')
                if tool_specs
                else None
            ),
            **kwargs,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: Sequence['BaseTool'],
        allow_parallel_tool_calls: bool = False,
        **kwargs,
    ) -> ChatResponse:
        """Validate the response from chat_with_tools."""
        if not allow_parallel_tool_calls:
            additional_kwargs = response.message.additional_kwargs
            tool_calls = additional_kwargs.get('tool_calls', [])
            if len(tool_calls) > 1:
                additional_kwargs['tool_calls'] = [tool_calls[0]]
        return response

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs,
    ) -> list[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get('tool_calls', [])

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                msg = (
                    'Expected at least one tool call, '
                    f'but got {len(tool_calls)} tool calls.'
                )
                raise ValueError(msg)
            return []

        tool_selections = []
        for tool_call in tool_calls:
            if tool_call.type != 'function':
                msg = 'Invalid tool type. Unsupported by OpenAI llm'
                raise ValueError(msg)

            func = tool_call.function

            # this should handle both complete and partial jsons
            try:
                argument_dict = parse_partial_json(func.arguments)
            except (ValueError, TypeError, JSONDecodeError):
                argument_dict = {}

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.id,
                    tool_name=func.name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections


# -------------------------------- low level ---------------------------------


def _map_ctx[T, R](s: Stream[T], fn: Callable[[T], R]) -> Generator[R]:
    with s:
        for resp in s:
            yield fn(resp)


async def _amap_ctx[T, R](
    s: AsyncStream[T], fn: Callable[[T], R]
) -> AsyncGenerator[R]:
    async with s:
        async for resp in s:
            yield fn(resp)


def _to_openai_message_dict(m: ChatMessage) -> 'ChatCompletionMessageParam':
    txt = ''
    for block in m.blocks:
        if not isinstance(block, TextBlock):
            msg = f'Unsupported content block type: {type(block).__name__}'
            raise TypeError(msg)
        txt += block.text

    # NOTE: Sending a null value (None) for Tool Message to OpenAI
    # will cause error
    # It's only Allowed to send None if it's an Assistant Message and either
    # a function call or tool calls were performed
    # Reference: https://platform.openai.com/docs/api-reference/chat/create
    content = (
        None
        if not txt
        and m.role.value == 'assistant'
        and (
            'function_call' in m.additional_kwargs
            or 'tool_calls' in m.additional_kwargs
        )
        else txt
    )

    # NOTE: Despite what the openai docs say, if the role is
    # ASSISTANT, SYSTEM or TOOL,
    # 'content' cannot be a list and must be string instead.
    # Furthermore, if all blocks are text blocks,
    # we can use the content_txt as the content.
    # This will avoid breaking openai-like APIs.
    ret = {'role': m.role.value, 'content': content}

    # NOTE: openai messages have additional arguments:
    # - function messages have `name`
    # - assistant messages have optional `function_call`
    ret |= m.additional_kwargs

    return ret  # type: ignore


class _Decoder:
    def __init__(self) -> None:
        self.content = ''
        self.tool_calls: list = []

    def completion(self, x: 'Completion') -> CompletionResponse:
        txt = x.choices[0].text if x.choices else ''
        self.content += txt

        return CompletionResponse(
            delta=txt, text=self.content, raw=x.model_dump()
        )

    def chat(self, x: 'ChatCompletionChunk | ChatCompletion') -> ChatResponse:
        role: str
        delta: str | None = None

        if isinstance(x, ChatCompletion):  # Full response
            full = x.choices[0].message

            role = full.role
            self.content = full.content or ''
            self.tool_calls = full.tool_calls or []

        else:  # Delta only
            part = x.choices[0].delta if x.choices else ChoiceDelta()

            role = part.role or 'assistant'
            delta = part.content or ''
            self.content += delta
            _update_tool_calls(self.tool_calls, part.tool_calls)

        msg = ChatMessage(role=role, content=self.content)
        if self.tool_calls:
            msg.additional_kwargs['tool_calls'] = self.tool_calls

        return ChatResponse(message=msg, delta=delta, raw=x.model_dump())


def _update_tool_calls(
    tool_calls: list['ChoiceDeltaToolCall'],
    tool_calls_delta: list['ChoiceDeltaToolCall'] | None,
) -> None:
    if not tool_calls_delta:
        return

    # openai provides chunks consisting of tool_call deltas one tool at a time
    tc_delta = tool_calls_delta[0]
    if not tool_calls or tool_calls[-1].index != tc_delta.index:
        tool_calls.append(tc_delta)
        return

    # not the start of a new tool call,
    # so update last item of tool_calls
    t = tool_calls[-1]

    # validations to get passed by mypy
    assert t.function is not None
    assert tc_delta.function is not None
    func = t.function

    # Initialize fields if they're None
    # OpenAI(or Compatible)'s streaming API can return
    # partial tool call
    # information across multiple chunks where
    # some fields may be None in
    # initial chunks and populated in subsequent ones
    if func.arguments is None:
        func.arguments = ''
    if func.name is None:
        func.name = ''
    if t.id is None:
        t.id = ''

    # Update with delta values
    func.arguments += tc_delta.function.arguments or ''
    func.name += tc_delta.function.name or ''
    t.id += tc_delta.id or ''
    return
