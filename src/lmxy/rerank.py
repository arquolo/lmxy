__all__ = ['Reranker']

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from httpx import URL, Request, Response, Timeout
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from pydantic import BaseModel, Field, PrivateAttr

from .util import get_clients, raise_for_status

_client, _aclient = get_clients()


class Reranker(BaseNodePostprocessor):
    # Inputs and behavior
    model_name: str = Field(
        description='The name of the reranker model.',
    )

    # Behavior
    with_meta: bool = Field(
        default=False, description='Use node metadata in reranking'
    )
    _metadata_mode: MetadataMode = PrivateAttr()

    # Outputs
    top_n: int = Field(
        description='Number of nodes to return sorted by score.'
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description='Whether to keep the retrieval score in metadata.',
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
    # TODO: support caching (by node ID)

    def model_post_init(self, context) -> None:
        self._metadata_mode = (
            MetadataMode.EMBED if self.with_meta else MetadataMode.NONE
        )

    @classmethod
    def class_name(cls) -> str:
        return 'RemoteReranker'

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError('Missing query bundle in extra info.')
        if not nodes:
            return nodes
        with self._event(nodes, query_bundle) as new_nodes:
            req = self._new_request(nodes, query_bundle)
            resp = _client.send(req)
            new_nodes.extend(self._reorder_nodes(nodes, resp)[: self.top_n])
            return new_nodes

    async def _apostprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError('Missing query bundle in extra info.')
        if not nodes:
            return nodes
        with self._event(nodes, query_bundle) as new_nodes:
            req = self._new_request(nodes, query_bundle)
            resp = await _aclient.send(req)
            new_nodes.extend(self._reorder_nodes(nodes, resp)[: self.top_n])
            return new_nodes

    @contextmanager
    def _event(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle
    ) -> Iterator[list[NodeWithScore]]:
        query = query_bundle.query_str

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.QUERY_STR: query,
                EventPayload.TOP_K: self.top_n,
                EventPayload.MODEL_NAME: self.model_name,
            },
        ) as event:
            new_nodes: list[NodeWithScore] = []
            yield new_nodes
            event.on_end(payload={EventPayload.NODES: nodes})

    def _new_request(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle
    ) -> Request:
        query = query_bundle.query_str
        texts = [node.get_content(self._metadata_mode) for node in nodes]

        headers = {'Content-Type': 'application/json'}
        if callable(self.auth_token):
            headers['Authorization'] = self.auth_token(self.base_url)
        elif self.auth_token is not None:
            headers['Authorization'] = self.auth_token

        return Request(
            'POST',
            URL(self.base_url).join('/rerank'),
            headers=headers,
            json={'query': query, 'documents': texts, 'top_n': len(texts)},
            extensions={'timeout': Timeout(self.timeout).as_dict()},
        )

    def _reorder_nodes(
        self, nodes: list[NodeWithScore], resp: Response
    ) -> list[NodeWithScore]:
        raise_for_status(resp).result()
        return [
            _update_node(nodes[x.index], x.score, self.keep_retrieval_score)
            for x in _RerankResponse.model_validate_json(resp.content).results
        ]


def _update_node(
    x: NodeWithScore, score: float, keep_retrieval_score: bool = False
) -> NodeWithScore:
    if keep_retrieval_score:
        x.node.metadata['retrieval_score'] = x.score
    x.score = score
    return x


class _RerankResult(BaseModel):
    index: int
    score: float


class _RerankResponse(BaseModel):
    results: list[_RerankResult]
