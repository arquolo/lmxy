__all__ = ['LlamaReranker']

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from httpx import AsyncClient, Client
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
)
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

from .rerank import CrossEvaluator
from .http import aclient, client


class LlamaReranker:
    def __init__(
        self,
        *,
        with_meta: bool,
        top_n: int,
        base_url: str,
        auth_token: str | Callable[[str], str] | None = None,
        extra_kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = 360.0,
        client: Client = client,
        aclient: AsyncClient = aclient,
        callback_manager: CallbackManager | None = None,
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
        self._metadata_mode = (
            MetadataMode.EMBED if with_meta else MetadataMode.NONE
        )
        self._callback_manager = callback_manager or CallbackManager()

    def postprocess_nodes(
        self,
        nodes: Sequence[NodeWithScore],
        query_bundle: QueryBundle | None = None,
        query_str: str | None = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        if query_str is None:
            if query_bundle is None:
                raise ValueError('Missing query bundle in extra info.')
            query_str = query_bundle.query_str

        with self._callback_manager.event(
            CBEventType.RERANKING,
            {
                EventPayload.NODES: list(nodes),
                EventPayload.QUERY_STR: query_str,
            },
        ) as event:
            texts = [node.get_content(self._metadata_mode) for node in nodes]
            scores = self._evaluator.run(query_str, *texts)
            nodes = self._rerank(nodes, scores)
            event.on_end({EventPayload.NODES: nodes})

        return nodes

    async def apostprocess_nodes(
        self,
        nodes: Sequence[NodeWithScore],
        query_bundle: QueryBundle | None = None,
        query_str: str | None = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        if query_str is None:
            if query_bundle is None:
                raise ValueError('Missing query bundle in extra info.')
            query_str = query_bundle.query_str

        with self._callback_manager.event(
            CBEventType.RERANKING,
            {
                EventPayload.NODES: list(nodes),
                EventPayload.QUERY_STR: query_str,
            },
        ) as event:
            texts = [node.get_content(self._metadata_mode) for node in nodes]
            scores = await self._evaluator.arun(query_str, *texts)
            nodes = self._rerank(nodes, scores)
            event.on_end({EventPayload.NODES: nodes})

        return nodes

    def _rerank(
        self,
        nodes: Sequence[NodeWithScore],
        scores: Sequence[float],
    ) -> list[NodeWithScore]:
        for n, s in zip(nodes, scores):
            n.node.metadata['retrieval_score'] = n.score
            n.score = s

        ret = sorted(
            nodes,
            key=lambda n: 1.0 if n.score is None else n.score,
            reverse=True,
        )
        return ret[: self._top_n or None]
