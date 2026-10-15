__all__ = ['Embedder']

import asyncio
import threading
from collections.abc import Awaitable, Callable, Iterable
from typing import Literal

import httpx
from glow import streaming
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.utils.huggingface import (
    get_query_instruct_for_model_name,
    get_text_instruct_for_model_name,
)
from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter

from ._retry import aretry
from ._types import Embedding
from .http import aclient, client, raise_for_status

_endpoints = ['/embed', '/api/embed', '/embeddings', '/v1/embeddings']
_text_keys = ['input', 'inputs']


def _too_many_requests(e: BaseException) -> bool:
    return isinstance(e, httpx.ReadError) or (
        isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429
    )


def _get_token_trimmer(
    max_chars: int, max_batch_size: int
) -> Callable[[Iterable[str]], int]:
    def usable_size(texts: Iterable[str]) -> int:
        texts = list(texts)
        if max_batch_size:
            texts = texts[:max_batch_size]
        if max_chars:
            nchars = 0
            for n, t in enumerate(texts, 1):
                nchars += len(t)
                if nchars > max_chars:  # Too long text
                    return max(n - 1, 1)
        if len(texts) == max_batch_size:
            return max_batch_size
        return 0

    return usable_size


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
    max_chars: int = 100_000  # >0 to limit N chars in single request

    client: httpx.Client = client
    aclient: httpx.AsyncClient = aclient

    _instructions: dict[str, str] = PrivateAttr()
    _endpoint: str = PrivateAttr()
    _text_key: str = PrivateAttr()
    _ssemlock: threading.Semaphore = PrivateAttr()
    _asemlock: asyncio.Semaphore = PrivateAttr()
    _embed: Callable[[Iterable[str]], list[Embedding]] = PrivateAttr()
    _aembed: Callable[[Iterable[str]], Awaitable[list[Embedding]]] = (
        PrivateAttr()
    )

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
            'text': self.text_instruction or '',
            'query': self.query_instruction or '',
        }
        self._endpoint = self._text_key = ''
        self._ssemlock = threading.Semaphore(self.concurrency)
        self._asemlock = asyncio.Semaphore(self.concurrency)

        usable_size = _get_token_trimmer(
            max_chars=self.max_chars,
            max_batch_size=self.embed_batch_size,
        )
        if self.latency > 0:
            self._embed = streaming(
                self._embed_impl,
                batch_size=usable_size,
                timeout=self.latency,
                pool_timeout=self.timeout or 360,
                workers=self.concurrency,
            )
            self._aembed = streaming(
                self._aembed_impl,
                batch_size=usable_size,
                timeout=self.latency,
            )
        else:
            self._embed = self._embed_impl
            self._aembed = self._aembed_impl

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

    def _get_text_embeddings(self, texts: Iterable[str]) -> list[Embedding]:
        """Get text embeddings."""
        texts = self._with_inst(texts, mode='text')
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding async."""
        texts = self._with_inst([query], mode='query')
        return (await self._aembed(texts))[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Get text embedding async."""
        texts = self._with_inst([text], mode='text')
        return (await self._aembed(texts))[0]

    async def _aget_text_embeddings(
        self, texts: Iterable[str]
    ) -> list[Embedding]:
        """Get text embeddings async."""
        texts = self._with_inst(texts, mode='text')
        return await self._aembed(texts)

    def _embed_impl(self, texts: Iterable[str]) -> list[Embedding]:
        texts = list(texts)
        if not texts:
            return []
        error: httpx.HTTPStatusError | None = None
        missing: set[str] = set()
        initialized = bool(self._endpoint)
        for endpoint, text_key in self._api_candidates():
            if endpoint in missing:
                continue
            req = self._request(texts, endpoint, text_key)
            try:
                embeddings = self._retry(self._send, req)
            except httpx.HTTPStatusError as e:
                if initialized or e.response.status_code not in {
                    400,
                    404,
                    422,
                }:
                    raise
                if e.response.status_code == 404:
                    missing.add(endpoint)
                error = e
            else:
                self._endpoint, self._text_key = endpoint, text_key
                return embeddings
        assert error is not None
        raise RuntimeError(f'{self.base_url} is not embeddings API') from error

    async def _aembed_impl(self, texts: Iterable[str]) -> list[Embedding]:
        texts = list(texts)
        if not texts:
            return []
        error: httpx.HTTPStatusError | None = None
        missing: set[str] = set()
        initialized = bool(self._endpoint)
        for endpoint, text_key in self._api_candidates():
            if endpoint in missing:
                continue
            req = self._request(texts, endpoint, text_key)
            try:
                embeddings = await self._retry(self._asend, req)
            except httpx.HTTPStatusError as e:
                if initialized or e.response.status_code not in {
                    400,
                    404,
                    422,
                }:
                    raise
                if e.response.status_code == 404:
                    missing.add(endpoint)
                error = e
            else:
                self._endpoint, self._text_key = endpoint, text_key
                return embeddings
        assert error is not None
        raise RuntimeError(f'{self.base_url} is not embeddings API') from error

    def _api_candidates(self) -> list[tuple[str, str]]:
        if self._endpoint and self._text_key:
            return [(self._endpoint, self._text_key)]
        return [(ep, key) for ep in _endpoints for key in _text_keys]

    def _send(self, req: httpx.Request) -> list[Embedding]:
        with self._ssemlock:
            rsp = self.client.send(req)
            return _handle_response(rsp)

    async def _asend(self, req: httpx.Request) -> list[Embedding]:
        async with self._asemlock:
            rsp = await self.aclient.send(req)
            return _handle_response(rsp)

    def _retry[**P, R](
        self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        fn = aretry(
            override_defaults=True,
            predicate=_too_many_requests,
            max_attempts=None if self.retries is None else 1 + self.retries,
            timeout=self.timeout,
        )(fn)
        return fn(*args, **kwargs)

    def _with_inst(
        self, texts: Iterable[str], mode: Literal['query', 'text']
    ) -> list[str]:
        inst = self._instructions.get(mode, '')
        return [f'{inst} {t}'.strip() for t in texts]

    def _request(
        self, texts: list[str], endpoint: str, text_key: str
    ) -> httpx.Request:
        headers = {'Content-Type': 'application/json'}
        if callable(self.auth_token):
            headers['Authorization'] = self.auth_token(self.base_url)
        elif self.auth_token is not None:
            headers['Authorization'] = self.auth_token

        return httpx.Request(
            'POST',
            httpx.URL(self.base_url).join(endpoint),
            headers=headers,
            json={'model': self.model_name, text_key: texts},
            extensions={'timeout': httpx.Timeout(self.timeout).as_dict()},
        )


def _handle_response(response: httpx.Response) -> list[Embedding]:
    raise_for_status(response, eager=True)
    data = response.content
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
