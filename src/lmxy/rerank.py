__all__ = ['Reranker']

from asyncio import Future
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import httpx
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
)
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
)
from pydantic import BaseModel, Field, PrivateAttr

from .util import aclient, client, raise_for_status


class Reranker:
    def __init__(
        self,
        model_name: str,
        with_meta: bool,
        top_n: int,
        base_url: str,
        auth_token: str | Callable[[str], str] | None = None,
        timeout: float | None = 360.0,
        client: httpx.Client = client,
        aclient: httpx.AsyncClient = aclient,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        self._evaluator = QueryNodeSimilarityEvaluator(
            model_name=model_name,
            with_meta=with_meta,
            base_url=base_url,
            auth_token=auth_token,
            timeout=timeout,
            client=client,
            aclient=aclient,
            callback_manager=callback_manager or CallbackManager(),
        )
        self._top_n = top_n

    def postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
        query_str: str | None = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        if query_str is None:
            if query_bundle is None:
                raise ValueError('Missing query bundle in extra info.')
            query_str = query_bundle.query_str

        nodes_ = [n.node for n in nodes]
        scores = self._evaluator.run(query_str, *nodes_)
        return self._rerank(nodes, scores)

    async def apostprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
        query_str: str | None = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        if query_str is None:
            if query_bundle is None:
                raise ValueError('Missing query bundle in extra info.')
            query_str = query_bundle.query_str

        nodes_ = [n.node for n in nodes]
        scores = await self._evaluator.arun(query_str, *nodes_)
        return self._rerank(nodes, scores)

    def _rerank(
        self,
        nodes: Sequence[NodeWithScore],
        scores: Sequence[float],
    ) -> list[NodeWithScore]:
        for n, s in zip(nodes, scores):
            n.node.metadata['retrieval_score'] = n.score
            n.score = s

        nodes = sorted(
            nodes,
            key=lambda n: n.score or 1.0,
            reverse=True,
        )
        return nodes[: self._top_n or None]


class QueryNodeSimilarityEvaluator(BaseModel):
    # Inputs and behavior
    model_name: str = Field(
        description='The name of the reranker model.',
    )

    # Behavior
    with_meta: bool = Field(
        default=False, description='Use node metadata in reranking'
    )
    _metadata_mode: MetadataMode = PrivateAttr()

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

    client: httpx.Client = client
    aclient: httpx.AsyncClient = aclient
    callback_manager: CallbackManager = Field(default_factory=CallbackManager)

    def model_post_init(self, context) -> None:
        self._metadata_mode = (
            MetadataMode.EMBED if self.with_meta else MetadataMode.NONE
        )

    def run(self, query: str, *nodes: BaseNode) -> list[float]:
        if not nodes:
            return []
        with self._request(query, *nodes) as (req, rsp, scores):
            rsp.set_result(self.client.send(req))
        return scores

    async def arun(self, query: str, *nodes: BaseNode) -> list[float]:
        if not nodes:
            return []
        with self._request(query, *nodes) as (req, rsp, scores):
            rsp.set_result(await self.aclient.send(req))
        return scores

    @contextmanager
    def _request(
        self,
        query: str,
        *nodes: BaseNode,
    ) -> Iterator[tuple[httpx.Request, Future[httpx.Response], list[float]]]:
        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: list(nodes),
                EventPayload.QUERY_STR: query,
                EventPayload.MODEL_NAME: self.model_name,
            },
        ) as event:
            texts = [node.get_content(self._metadata_mode) for node in nodes]

            headers = {'Content-Type': 'application/json'}
            if callable(self.auth_token):
                headers['Authorization'] = self.auth_token(self.base_url)
            elif self.auth_token is not None:
                headers['Authorization'] = self.auth_token

            req = httpx.Request(
                'POST',
                httpx.URL(self.base_url).join('/rerank'),
                headers=headers,
                json={'query': query, 'documents': texts, 'top_n': len(texts)},
                extensions={'timeout': httpx.Timeout(self.timeout).as_dict()},
            )
            f = Future[httpx.Response]()
            scores = [1.0] * len(nodes)
            yield (req, f, scores)

            rsp = f.result()
            raise_for_status(rsp).result()

            for x in _RerankResponse.model_validate_json(rsp.content).results:
                scores[x.index] = x.score

            scored_nodes = [
                NodeWithScore(node=n, score=s) for n, s in zip(nodes, scores)
            ]
            event.on_end({EventPayload.NODES: scored_nodes})


class _RerankResult(BaseModel):
    index: int
    score: float


class _RerankResponse(BaseModel):
    results: list[_RerankResult]
