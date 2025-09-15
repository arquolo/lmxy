__all__ = ['Embedder']

from collections.abc import Callable
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
from pydantic import Field, PrivateAttr

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

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding."""
        return self._embed([query], mode='query')[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get text embedding."""
        return self._embed([text], mode='text')[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings."""
        return self._embed(texts, mode='text')

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Get query embedding async."""
        return (await self._aembed([query], mode='query'))[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Get text embedding async."""
        return (await self._aembed([text], mode='text'))[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return await self._aembed(texts, mode='text')

    def _embed(
        self, texts: list[str], mode: Literal['query', 'text'] | None = None
    ) -> list[list[float]]:
        req = self._create_request(texts, mode=mode)
        resp = _client.send(req)
        return self._handle_response(resp)

    async def _aembed(
        self, texts: list[str], mode: Literal['query', 'text'] | None = None
    ) -> list[list[float]]:
        req = self._create_request(texts, mode=mode)
        resp = await _aclient.send(req)
        return self._handle_response(resp)

    def _create_request(
        self, texts: list[str], mode: Literal['query', 'text'] | None = None
    ) -> Request:
        if mode and (inst := self._instructions.get(mode)) is not None:
            texts = [f'{inst} {t}'.strip() for t in texts]

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

    @staticmethod
    def _handle_response(response: Response) -> list[list[float]]:
        j = raise_for_status(response).result().json()

        # NOTE: not match-case because PyArmor works only with attrs & literals
        # Text Embeddings Inference
        if isinstance(j, list):
            return j

        # Ollama
        if xs := j.get('embeddings', []):
            return list(xs)

        # OpenAI
        if xs := j.get('data', []):
            return [x['embedding'] for x in xs]

        raise NotImplementedError(f'Unknown embeddings schema: {j}')
