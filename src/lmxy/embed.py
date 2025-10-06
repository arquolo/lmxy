__all__ = ['Embedder']

from collections.abc import Callable, Sequence
from typing import Literal

from httpx import (
    URL,
    ConnectError,
    HTTPStatusError,
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

from .util import get_clients, raise_for_status

_endpoints = ['/embed', '/api/embed', '/embeddings', '/v1/embeddings']
_text_keys = ['input', 'inputs']
_client, _aclient = get_clients()

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

    _instructions: dict[str, str] = PrivateAttr()
    _endpoint: str = PrivateAttr()
    _text_key: str = PrivateAttr()

    def model_post_init(self, context) -> None:
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

    async def handshake(self) -> None:
        # Try to find working combo
        errors: list[Exception] = []
        try:
            for self._endpoint in _endpoints:
                # Find whether `input` or `inputs` must be in scheme
                for self._text_key in _text_keys:
                    try:
                        await self._aembed(['test line'])
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
        self, texts: Sequence[str]
    ) -> list[Embedding]:
        """Get text embeddings async."""
        texts = self._with_inst(texts, mode='text')
        return await self._aembed(texts)

    def _embed(self, texts: Sequence[str]) -> list[Embedding]:
        req = self._request(texts)
        resp = _client.send(req)
        return _handle_response(resp)

    async def _aembed(self, texts: Sequence[str]) -> list[Embedding]:
        req = self._request(texts)
        resp = await _aclient.send(req)
        return _handle_response(resp)

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
    obj = _parse_embeddings(data)
    match obj:
        case _OllamaResponse():
            return obj.embeddings
        case _OpenAiResponse():
            return [x.embedding for x in obj.data]
        case _:
            return obj


class _OllamaResponse(BaseModel):
    embeddings: list[Embedding]


class _OpenAiEmbedding(BaseModel):
    embedding: Embedding


class _OpenAiResponse(BaseModel):
    data: list[_OpenAiEmbedding]


type _AnyEmbeddings = list[Embedding] | _OllamaResponse | _OpenAiResponse
_parse_embeddings = TypeAdapter[_AnyEmbeddings](_AnyEmbeddings).validate_json
