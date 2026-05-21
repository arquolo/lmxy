__all__ = ['Embedder']

from asyncio import Semaphore as AsyncSemaphore
from collections.abc import Awaitable, Callable, Sequence
from threading import Semaphore as SyncSemaphore
from typing import Literal

from glow import astreaming, streaming
from httpx import (
    URL,
    AsyncClient,
    Client,
    ConnectError,
    HTTPStatusError,
    ReadError,
    Request,
    Response,
    Timeout,
)
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.utils.huggingface import (
    get_query_instruct_for_model_name,
    get_text_instruct_for_model_name,
)
from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter

from .util import aclient, aretry, client, raise_for_status

_endpoints = ['/embed', '/api/embed', '/embeddings', '/v1/embeddings']
_text_keys = ['input', 'inputs']


def _too_many_requests(e: BaseException) -> bool:
    return isinstance(e, ReadError) or (
        isinstance(e, HTTPStatusError) and e.response.status_code == 429
    )


# -------------------------------- embedding ---------------------------------


class Embedder(BaseEmbedding):
    """Generic class for remote embeddings via HTTP API.

    Supports:
    - Text Embeddings Inference
      - POST /embed --json {model: ..., inputs: [...]}
    - Infinity, vLLM
      - POST /embeddings --json {model: ..., input: [...]}
    - Ollama
      - POST /api/embed --json {model: ..., input: [...]}
    """

    # Inputs
    query_instruction: str | None = Field(
        default=None,
        description='Instruction prefix for query text for bi-encoder.',
    )
    text_instruction: str | None = Field(
        default=None, description='Instruction prefix for text for bi-encoder.'
    )

    # Connection
    base_url: str = Field(
        description='URL or base URL for the embeddings service.',
    )
    auth_token: str | Callable[[str], str] | None = Field(
        default=None,
        description=(
            'Authentication token or authentication token '
            'generating function for authenticated requests'
        ),
    )
    timeout: float | None = Field(
        default=360.0, description='HTTP connection timeout'
    )
    retries: int | None = 10
    concurrency: int = 10
    latency: float = 0  # >0 to enable automatic batching

    client: Client = client
    aclient: AsyncClient = aclient

    _instructions: dict[str, str] = PrivateAttr()
    _endpoint: str = PrivateAttr()
    _text_key: str = PrivateAttr()
    _ssemlock: SyncSemaphore = PrivateAttr()
    _asemlock: AsyncSemaphore = PrivateAttr()
    _embed: Callable[[Sequence[str]], Sequence[Embedding]]
    _aembed: Callable[[Sequence[str]], Awaitable[Sequence[Embedding]]]

    def model_post_init(self, context) -> None:
        if self.retries is not None:
            self.retries = max(self.retries, 0)
        self.base_url = self.base_url.removesuffix('/')
        if self.text_instruction is None:
            self.text_instruction = get_text_instruct_for_model_name(
                self.model_name
            )
        if self.query_instruction is None:
            self.query_instruction = get_query_instruct_for_model_name(
                self.model_name
            )
        self._instructions = {
            'text': self.text_instruction,
            'query': self.query_instruction,
        }
        self._endpoint = self._text_key = ''
        self._ssemlock = SyncSemaphore(self.concurrency)
        self._asemlock = AsyncSemaphore(self.concurrency)

        if self.latency > 0:
            self._embed = streaming(
                self._embed_impl,
                batch_size=self.embed_batch_size,
                timeout=self.latency,
                pool_timeout=self.timeout or 360,
            )
            self._aembed = astreaming(
                self._aembed_impl,
                batch_size=self.embed_batch_size,
                timeout=self.latency,
            )
        else:
            self._embed = self._embed_impl
            self._aembed = self._aembed_impl

    async def handshake(self) -> None:
        # Try to find working combo
        errors: list[Exception] = []
        try:
            for self._endpoint in _endpoints:
                # Find whether `input` or `inputs` must be in scheme
                for self._text_key in _text_keys:
                    try:
                        # Raw call, to not retry
                        req = self._request(['test line'])
                        await self._asend(req)
                    except HTTPStatusError as exc:
                        if exc.response.status_code == 404:  # Missing url
                            break  # Next `_text_key` will fail too, skip it.
                    except Exception as exc:  # noqa: BLE001
                        errors.append(exc)
                    else:
                        return

        except ConnectError as exc:
            raise RuntimeError(f'Cannot connect to {self.base_url!r}') from exc
        else:
            raise ExceptionGroup(
                f'{self.base_url} is not embeddings API', errors
            )

    @classmethod
    def class_name(cls) -> str:
        return 'RemoteEmbedding'

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""
        texts = self._with_inst([query], mode='query')
        return self._embed(texts)[0]

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get text embedding."""
        texts = self._with_inst([text], mode='text')
        return self._embed(texts)[0]

    def _get_text_embeddings(self, texts: Sequence[str]) -> list[Embedding]:
        """Get text embeddings."""
        texts = self._with_inst(texts, mode='text')
        return list(self._embed(texts))

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding async."""
        texts = self._with_inst([query], mode='query')
        return (await self._aembed(texts))[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Get text embedding async."""
        texts = self._with_inst([text], mode='text')
        return (await self._aembed(texts))[0]

    async def _aget_text_embeddings(
        self, texts: Sequence[str]
    ) -> list[Embedding]:
        """Get text embeddings async."""
        texts = self._with_inst(texts, mode='text')
        return list(await self._aembed(texts))

    def _embed_impl(self, texts: Sequence[str]) -> list[Embedding]:
        req = self._request(texts)
        return self._retry(self._send, req)

    async def _aembed_impl(self, texts: Sequence[str]) -> list[Embedding]:
        req = self._request(texts)
        return await self._retry(self._asend, req)

    def _send(self, req: Request) -> list[Embedding]:
        with self._ssemlock:
            rsp = self.client.send(req)
            return _handle_response(rsp)

    async def _asend(self, req: Request) -> list[Embedding]:
        async with self._asemlock:
            rsp = await self.aclient.send(req)
            return _handle_response(rsp)

    def _retry[**P, R](
        self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        fn = aretry(
            predicate=_too_many_requests,
            max_attempts=None if self.retries is None else 1 + self.retries,
            timeout=self.timeout,
        )(fn)
        return fn(*args, **kwargs)

    def _with_inst(
        self, texts: Sequence[str], mode: Literal['query', 'text']
    ) -> Sequence[str]:
        inst = self._instructions.get(mode, '')
        return [f'{inst} {t}'.strip() for t in texts]

    def _request(self, texts: Sequence[str]) -> Request:
        headers = {'Content-Type': 'application/json'}
        if callable(self.auth_token):
            headers['Authorization'] = self.auth_token(self.base_url)
        elif self.auth_token is not None:
            headers['Authorization'] = self.auth_token

        return Request(
            'POST',
            URL(self.base_url).join(self._endpoint),
            headers=headers,
            json={'model': self.model_name, self._text_key: texts},
            extensions={'timeout': Timeout(self.timeout).as_dict()},
        )


def _handle_response(response: Response) -> list[Embedding]:
    data = raise_for_status(response).result().content
    match _parse_embeddings(data):
        case _OllamaResponse() as obj:
            return obj.embeddings
        case _OpenAiResponse() as obj:
            return [x.embedding for x in obj.data]
        case _ as obj:
            return obj


class _OllamaResponse(BaseModel):
    embeddings: list[Embedding]


class _OpenAiEmbedding(BaseModel):
    embedding: Embedding


class _OpenAiResponse(BaseModel):
    data: list[_OpenAiEmbedding]


type _AnyEmbeddings = list[Embedding] | _OllamaResponse | _OpenAiResponse
_parse_embeddings = TypeAdapter[_AnyEmbeddings](_AnyEmbeddings).validate_json
