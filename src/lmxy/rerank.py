__all__ = ['CrossEvaluator', 'Reranker']

from asyncio import Future
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from httpx import (
    URL,
    AsyncClient,
    Client,
    HTTPStatusError,
    Request,
    Response,
    Timeout,
)
from itertools import batched
from pydantic import BaseModel, Field, TypeAdapter

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
        batch_size: int = 32,
    ) -> None:
        self._evaluator = CrossEvaluator(
            extra_kwargs=dict(extra_kwargs or {}),
            batch_size=batch_size,
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
    docs_field: str | None = None
    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description='Extra options to append to request',
    )
    batch_size: int = Field(default=32, ge=1)

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
        forbidden = {'query', 'texts', 'documents', 'top_n'}
        if self.docs_field:
            forbidden.add(self.docs_field)
        if keys := self.extra_kwargs.keys() & forbidden:
            raise ValueError(f'`extra_kwargs` contains forbidden keys: {keys}')

    def run(self, query: str, *texts: str) -> list[float]:
        if not texts:
            return []
        if len(texts) > self.batch_size:
            return [
                x
                for batch in batched(texts, self.batch_size, strict=False)
                for x in self.run(query, *batch)
            ]
        error: BaseException | None = None
        for f in self._docs_fields():
            try:
                with self._request(query, f, *texts) as (req, rsp, scores):
                    rsp.set_result(self.client.send(req))
            except HTTPStatusError as e:
                if self.docs_field or e.response.status_code not in {400, 422}:
                    raise
                error = e
            else:
                self.docs_field = f
                return scores
        assert error is not None
        raise RuntimeError(f'{self.base_url} is not rerank API') from error

    async def arun(self, query: str, *texts: str) -> list[float]:
        if not texts:
            return []
        if len(texts) > self.batch_size:
            return [
                x
                for batch in batched(texts, self.batch_size, strict=False)
                for x in await self.arun(query, *batch)
            ]
        error: BaseException | None = None
        for f in self._docs_fields():
            try:
                with self._request(query, f, *texts) as (req, rsp, scores):
                    rsp.set_result(await self.aclient.send(req))
            except HTTPStatusError as e:
                if self.docs_field or e.response.status_code not in {400, 422}:
                    raise
                error = e
            else:
                self.docs_field = f
                return scores
        assert error is not None
        raise RuntimeError(f'{self.base_url} is not rerank API') from error

    def _docs_fields(self) -> tuple[str, ...]:
        return (
            (self.docs_field,) if self.docs_field else ('texts', 'documents')
        )

    @contextmanager
    def _request(
        self, query: str, docs_field: str, *texts: str
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
                {'query': query, docs_field: texts, 'top_n': len(texts)}
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
        for idx, score in _handle_response(rsp.content):
            scores[idx] = score


class _RerankResult(BaseModel):
    index: int
    score: float


class _RerankResponse(BaseModel):
    results: list[_RerankResult]


def _handle_response(data: bytes) -> list[tuple[int, float]]:
    match _parse_scores(data):
        case _RerankResponse(results=scores):
            return [(x.index, x.score) for x in scores]
        case list() as obj:
            return [(x.index, x.score) for x in obj]
        case _ as obj:
            return obj


type _AnyScores = list[_RerankResult] | _RerankResponse
_parse_scores = TypeAdapter[_AnyScores](_AnyScores).validate_json
