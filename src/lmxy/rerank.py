__all__ = ['CrossEvaluator', 'Reranker']

from asyncio import Future
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from httpx import URL, AsyncClient, Client, Request, Response, Timeout
from pydantic import BaseModel, Field

from .http import aclient, client, raise_for_status


class Reranker:
    def __init__(
        self,
        *,
        top_n: int,
        base_url: str,
        auth_token: str | Callable[[str], str] | None = None,
        extra_kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = 360.0,
        client: Client = client,
        aclient: AsyncClient = aclient,
    ) -> None:
        self._evaluator = CrossEvaluator(
            extra_kwargs=dict(extra_kwargs or {}),
            base_url=base_url,
            auth_token=auth_token,
            timeout=timeout,
            client=client,
            aclient=aclient,
        )
        self._top_n = top_n

    def rerank(
        self, texts: Sequence[str], query: str
    ) -> list[tuple[str, float]]:
        scores = self._evaluator.run(query, *texts)
        return self._rerank(texts, scores)

    async def arerank(
        self, texts: Sequence[str], query: str
    ) -> list[tuple[str, float]]:
        scores = await self._evaluator.arun(query, *texts)
        return self._rerank(texts, scores)

    def _rerank(
        self, texts: Sequence[str], scores: Sequence[float]
    ) -> list[tuple[str, float]]:
        scored = sorted(zip(texts, scores), key=lambda ts: ts[1], reverse=True)
        return scored[: self._top_n or None]


class CrossEvaluator(BaseModel):
    model_config = {'arbitrary_types_allowed': True, 'extra': 'forbid'}

    # Inputs and behavior
    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description='Extra options to append to request',
    )

    # Connection
    base_url: str = Field(
        description='Base URL for the text embeddings service.',
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

    client: Client = client
    aclient: AsyncClient = aclient

    def model_post_init(self, context) -> None:
        if keys := self.extra_kwargs.keys() & {'query', 'documents', 'top_n'}:
            raise ValueError(f'`extra_kwargs` contains forbidden keys: {keys}')

    def run(self, query: str, *texts: str) -> list[float]:
        if not texts:
            return []
        with self._request(query, *texts) as (req, rsp, scores):
            rsp.set_result(self.client.send(req))
        return scores

    async def arun(self, query: str, *texts: str) -> list[float]:
        if not texts:
            return []
        with self._request(query, *texts) as (req, rsp, scores):
            rsp.set_result(await self.aclient.send(req))
        return scores

    @contextmanager
    def _request(
        self, query: str, *texts: str
    ) -> Generator[tuple[Request, Future[Response], list[float]]]:
        headers = {'Content-Type': 'application/json'}
        if callable(self.auth_token):
            headers['Authorization'] = self.auth_token(self.base_url)
        elif self.auth_token is not None:
            headers['Authorization'] = self.auth_token

        req = Request(
            'POST',
            URL(self.base_url).join('/rerank'),
            headers=headers,
            json=(
                {'query': query, 'documents': texts, 'top_n': len(texts)}
                | self.extra_kwargs
            ),
            extensions={'timeout': Timeout(self.timeout).as_dict()},
        )
        f = Future[Response]()
        scores = [1.0] * len(texts)

        yield (req, f, scores)  # Pass to request handler to populate f.result

        # Populate scores
        rsp = f.result()
        raise_for_status(rsp, eager=True)
        for x in _RerankResponse.model_validate_json(rsp.content).results:
            scores[x.index] = x.score


class _RerankResult(BaseModel):
    index: int
    score: float


class _RerankResponse(BaseModel):
    results: list[_RerankResult]
