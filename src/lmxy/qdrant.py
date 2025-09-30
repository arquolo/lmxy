"""Qdrant vector store, built on top of an existing Qdrant collection."""

__all__ = [
    'HybridFuse',
    'QdrantVectorStore',
]

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Iterable, Sequence
from typing import Any, Protocol, cast, runtime_checkable

import orjson
from glow import astreaming
from grpc import RpcError
from llama_index.core.schema import (
    BaseNode,
    ImageNode,
    IndexNode,
    MetadataMode,
    Node,
    TextNode,
)
from llama_index.core.vector_stores import (
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from pydantic import BaseModel, PrivateAttr
from qdrant_client import AsyncQdrantClient
from qdrant_client.conversions.common_types import QuantizationConfig
from qdrant_client.fastembed_common import IDF_EMBEDDING_MODELS
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from ._types import SparseEncode
from .fastembed import get_sparse_encoder
from .util import aretry

_DENSE_NAME = 'text-dense'
_SPARSE_NAME = 'text-sparse'
_SPARSE_MODIFIERS = dict.fromkeys(IDF_EMBEDDING_MODELS, rest.Modifier.IDF)
_LOCK = asyncio.Lock()
logger = logging.getLogger(__name__)


@runtime_checkable
class HybridFuse(Protocol):
    def __call__(
        self,
        dense: VectorStoreQueryResult,
        sparse: VectorStoreQueryResult,
        /,
        *,
        alpha: float = ...,
        top_k: int = ...,
    ) -> VectorStoreQueryResult: ...


def _relative_score_fusion(
    dense: VectorStoreQueryResult,
    sparse: VectorStoreQueryResult,
    # NOTE: only for hybrid search (0 for sparse search, 1 for dense search)
    alpha: float = 0.5,
    top_k: int = 2,
) -> VectorStoreQueryResult:
    """Fuse dense and sparse results using relative score fusion."""
    if not dense.nodes and not sparse.nodes:
        return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    if not sparse.nodes or alpha >= 1:
        return dense
    if not dense.nodes or alpha <= 0:
        return sparse

    assert dense.similarities is not None
    assert sparse.similarities is not None

    # deconstruct results
    dense_scores = {
        n.id_: x for n, x in zip(dense.nodes, _min_max(dense.similarities))
    }
    sparse_scores = {
        n.id_: x for n, x in zip(sparse.nodes, _min_max(sparse.similarities))
    }

    # fuse scores
    all_nodes = {n.id_: n for n in [*dense.nodes, *sparse.nodes]}
    fused = [
        (
            (1 - alpha) * sparse_scores.get(id_, 0)
            + alpha * dense_scores.get(id_, 0),
            node,
        )
        for id_, node in all_nodes.items()
    ]
    fused = sorted(fused, key=lambda x: x[0], reverse=True)[:top_k]

    # create final response object
    return VectorStoreQueryResult(
        nodes=[x[1] for x in fused],
        similarities=[x[0] for x in fused],
        ids=[x[1].id_ for x in fused],
    )


class QdrantVectorStore(BaseModel):
    """Fork of LlamaIndex's Qdrant Vector Store.

    Differences:
    - async only
    - no legacy formats
    - no legacy sparse embeddings
    - Qdrant Query API

    In this vector store, embeddings and docs are stored within a
    Qdrant collection.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes.
    """

    model_config = {'arbitrary_types_allowed': True}

    collection_name: str
    aclient: AsyncQdrantClient
    upsert_timeout: float | None = None  # enable to batch upserts
    upsert_batch_size: int | None = 64
    query_timeout: float | None = None  # enable to batch upserts
    query_batch_size: int | None = 64
    max_retries: int = 3

    # Collection construction parameters
    dense_config: rest.VectorParams | None = None
    sparse_config: rest.SparseVectorParams | None = None
    shard_number: int | None = None
    hnsw_config: rest.HnswConfigDiff | None = None
    optimizers_config: rest.OptimizersConfigDiff | None = None
    quantization_config: QuantizationConfig | None = None
    tenant_field_name: str | None = None  # For multitenancy

    # Sparse search parameters
    sparse_doc_fn: SparseEncode | None = None
    sparse_query_fn: SparseEncode | None = None
    sparse_model: str | None = None
    sparse_model_kwargs: dict[str, Any] = {}

    # Hybrid search fusion
    hybrid_fusion_fn: HybridFuse = _relative_score_fusion

    _upsert: Callable[
        [Sequence[BaseNode]],
        Awaitable[Sequence[str]],
    ] = PrivateAttr()
    _query: Callable[
        [Sequence[rest.QueryRequest]],
        Awaitable[Sequence[VectorStoreQueryResult]],
    ] = PrivateAttr()

    _is_initialized: bool = PrivateAttr()
    _is_legacy: bool = PrivateAttr()

    def model_post_init(self, context) -> None:
        retry_ = aretry(
            RpcError,
            UnexpectedResponse,
            max_attempts=self.max_retries,
        )
        upsert = self._ll_upsert
        if self.upsert_timeout is not None:
            upsert = astreaming(
                upsert,
                batch_size=self.upsert_batch_size,
                timeout=self.upsert_timeout,
            )
        self._upsert = retry_(upsert)

        query = self._ll_query
        if self.query_timeout is not None:
            query = astreaming(
                query,
                batch_size=self.query_batch_size,
                timeout=self.query_timeout,
            )
        self._query = retry_(query)

        self._is_initialized = False
        self._is_legacy = False

    async def initialize(self, vector_size: int) -> None:
        async with _LOCK:
            await self._initialize_unsafe(vector_size)

    async def is_initialized(self) -> bool:
        async with _LOCK:
            return await self._is_initialized_unsafe()

    async def _initialize_unsafe(self, vector_size: int) -> None:
        if self._is_initialized:
            return

        dense_config = self.dense_config or rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        )
        vectors_config = {_DENSE_NAME: dense_config}

        await self._load_models()
        if self.sparse_query_fn and self.sparse_doc_fn:
            sparse_config = self.sparse_config or rest.SparseVectorParams(
                modifier=_SPARSE_MODIFIERS.get(self.sparse_model or '')
            )
            sparse_vectors_config = {_SPARSE_NAME: sparse_config}
        else:
            sparse_vectors_config = None

        try:
            await self.aclient.create_collection(
                self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                shard_number=self.shard_number,
                hnsw_config=self.hnsw_config,
                optimizers_config=self.optimizers_config,
                quantization_config=self.quantization_config,
            )

            # To improve search performance Qdrant recommends setting up
            # a payload index for fields used in filters.
            # https://qdrant.tech/documentation/concepts/indexing
            await self.aclient.create_payload_index(
                self.collection_name,
                field_name='doc_id',
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
            self._is_initialized = True
        except (RpcError, ValueError, UnexpectedResponse) as exc:
            if 'already exists' not in str(exc):
                raise exc  # noqa: TRY201
            logger.warning(
                'Collection %s already exists, skipping collection creation.',
                self.collection_name,
            )
            assert await self._is_initialized_unsafe()

        if self.tenant_field_name is not None:
            await self.aclient.create_payload_index(
                self.collection_name,
                field_name=self.tenant_field_name,
                field_schema=rest.KeywordIndexParams(
                    type=rest.KeywordIndexType.KEYWORD, is_tenant=True
                ),
            )

    async def _is_initialized_unsafe(self) -> bool:
        if self._is_initialized:
            return True

        if not await self.aclient.collection_exists(self.collection_name):
            return False
        await self._load_models()
        info = await self.aclient.get_collection(self.collection_name)

        dense = info.config.params.vectors
        if isinstance(dense, rest.VectorParams):
            logger.warning(
                'Collection %s is using legacy anonymous vectors. '
                'Recreate it to allow sparse/hybrid search',
                self.collection_name,
            )
            self._is_legacy = True
        elif isinstance(dense, dict) and _DENSE_NAME in dense:
            self._is_legacy = False
        else:
            msg = f'Bad vector config {dense}'
            raise TypeError(msg)

        sparse = info.config.params.sparse_vectors
        if isinstance(sparse, dict) and _SPARSE_NAME in sparse:
            if not self.sparse_query_fn:
                logger.warning(
                    'Collection %s support '
                    'sparse search, but neither '
                    'sparse_query_fn nor sparse_model was provided',
                    self.collection_name,
                )
            if not self.sparse_doc_fn:
                logger.warning(
                    'Collection %s support '
                    'sparse search, but neither '
                    'sparse_doc_fn nor sparse_model was provided',
                    self.collection_name,
                )
        else:
            self.sparse_query_fn = self.sparse_doc_fn = None

        self._is_initialized = True
        return True

    async def _load_models(self) -> None:
        if self.sparse_doc_fn is None and self.sparse_model is not None:
            self.sparse_doc_fn = await asyncio.to_thread(
                get_sparse_encoder,
                self.sparse_model,
                **self.sparse_model_kwargs,
            )
        if self.sparse_query_fn is None and self.sparse_model is not None:
            self.sparse_query_fn = await asyncio.to_thread(
                get_sparse_encoder,
                self.sparse_model,
                **self.sparse_model_kwargs,
            )

    # CRUD: create or update
    async def async_add(self, nodes: Sequence[BaseNode]) -> Sequence[str]:
        """Add nodes with embeddings to Qdrant index.

        Returns node IDs that were added to the index.
        """
        if not nodes:
            return []
        await self.initialize(vector_size=len(nodes[0].get_embedding()))
        return await self._upsert(nodes)

    async def _build_points(
        self, nodes: Sequence[BaseNode]
    ) -> list[rest.PointStruct]:
        if not nodes:
            return []
        sparse_embeddings: Sequence[rest.SparseVector | None]
        if self.sparse_doc_fn:
            sparse_embeddings = await _aembed_sparse(
                *(n.get_content(MetadataMode.EMBED) for n in nodes),
                fn=self.sparse_doc_fn,
            )
        else:
            sparse_embeddings = [None for _ in nodes]

        points: list[rest.PointStruct] = []
        for node, semb in zip(nodes, sparse_embeddings, strict=True):
            demb = node.get_embedding()
            vector: rest.VectorStruct = (
                demb  # type: ignore[assignment]
                if self._is_legacy
                else (
                    {_DENSE_NAME: demb}
                    | ({} if semb is None else {_SPARSE_NAME: semb})
                )
            )
            payload = node_to_metadata_dict(node)

            pt = rest.PointStruct(id=node.id_, vector=vector, payload=payload)
            points.append(pt)
        return points

    # CRUD: read
    async def aquery(
        self,
        query: VectorStoreQuery,
        *,
        qdrant_filters: rest.Filter | None = None,
        with_payload: list[str] | bool = True,
        dense_threshold: float | None = None,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        if not await self.is_initialized():
            return VectorStoreQueryResult([], [], [])

        #  NOTE: users can pass in qdrant_filters
        # (nested/complicated filters) to override the default MetadataFilters
        filter_ = (
            _build_filter(query.doc_ids, query.node_ids, query.filters)
            if qdrant_filters is None
            else qdrant_filters
        )
        if query.query_embedding is None and query.query_str is None:
            assert query.node_ids
            assert query.doc_ids is None
            assert query.filters is None
            assert qdrant_filters is None
            records = await self.aclient.retrieve(
                self.collection_name, query.node_ids
            )
            return _parse_to_query_result(records)

        dense_k, sparse_k, hybrid_k, alpha = self._parse_query(query)

        # TODO: possible optimization.
        # Use prefetch={filter=filter_, lookup_from=<other collection>}
        #   and no filter in QueryRequest itself,
        # or call scroll first.
        # But for this we need to ensure that limit is infinite,
        #  otherwise we should use another storage for filters.

        # TODO: handle MMR in qdrant

        reqs: list[rest.QueryRequest] = []
        if dense_k:
            # Dense scores are absolute, i.e. depend only on (query, node),
            # thus we can apply some globally fixed score threshold.
            assert query.query_embedding
            reqs.append(
                rest.QueryRequest(
                    query=query.query_embedding,
                    using=None if self._is_legacy else _DENSE_NAME,
                    filter=filter_,
                    score_threshold=dense_threshold,
                    limit=dense_k,
                    with_payload=with_payload,
                )
            )

        if sparse_k:
            assert query.query_str is not None
            assert self.sparse_query_fn
            [sparse_embedding] = await _aembed_sparse(
                query.query_str, fn=self.sparse_query_fn
            )
            # Sparse scores are computed relative to the whole candidate list,
            # so we cannot threshold them,
            # and only able to directly limit their count.
            reqs.append(
                rest.QueryRequest(
                    query=sparse_embedding,
                    using=_SPARSE_NAME,
                    filter=filter_,
                    limit=sparse_k,
                    with_payload=with_payload,
                )
            )

        if not reqs:
            return VectorStoreQueryResult([], [], [])

        results = await self._query(reqs)
        if len(results) != 2:  # (dense) or (sparse)
            return results[0]

        # (dense, sparse)
        assert self.hybrid_fusion_fn is not None
        return self.hybrid_fusion_fn(*results, alpha=alpha, top_k=hybrid_k)

    def _parse_query(self, q: VectorStoreQuery) -> tuple[int, int, int, float]:
        match q.mode:
            case VectorStoreQueryMode.DEFAULT:
                alpha = 1.0
            case VectorStoreQueryMode.HYBRID:
                alpha = 0.5 if q.alpha is None else q.alpha
            case VectorStoreQueryMode.SPARSE:
                alpha = 0.0
            case _ as unsupported:
                msg = f'Unsupported query mode: {unsupported}'
                raise NotImplementedError(msg)

        dense_k = q.similarity_top_k
        sparse_k = dense_k if q.sparse_top_k is None else q.sparse_top_k
        hybrid_k = dense_k if q.hybrid_top_k is None else q.hybrid_top_k

        dense_k, sparse_k = min(dense_k, hybrid_k), min(sparse_k, hybrid_k)

        # Full sparse
        if alpha <= 0:
            if not self.sparse_query_fn:
                msg = (
                    f'Collection {self.collection_name} does not '
                    'have sparse vectors to do sparse search. '
                    'Please reinitialize it with sparse model '
                    'to allow sparse/hybrid search'
                )
                raise ValueError(msg)
            return 0, sparse_k, hybrid_k, 0.0  # Skip dense search

        # Full dense or no data for sparse
        if alpha >= 1 or not self.sparse_query_fn:
            return dense_k, 0, hybrid_k, 1.0  # Skip sparse search

        return dense_k, sparse_k, hybrid_k, alpha

    # CRUD: delete
    async def adelete(self, ref_doc_id: str) -> None:
        if not await self.is_initialized():
            return
        cond = rest.FieldCondition(
            key='doc_id', match=rest.MatchValue(value=ref_doc_id)
        )
        await self.aclient.delete(
            self.collection_name, rest.Filter(must=[cond])
        )

    # CRUD: delete
    async def adelete_nodes(self, node_ids: Sequence[str]) -> None:
        if not await self.is_initialized():
            return
        cond = rest.HasIdCondition(has_id=node_ids)  # type: ignore[arg-type]
        await self.aclient.delete(
            self.collection_name, rest.Filter(must=[cond])
        )

    async def aclear(self) -> None:
        async with _LOCK:
            await self.aclient.delete_collection(self.collection_name)
            self._is_initialized = False

    # low levels

    async def _ll_upsert(self, nodes: Sequence[BaseNode], /) -> Sequence[str]:
        if not nodes:
            return []
        points = await self._build_points(nodes)
        await self.aclient.upsert(self.collection_name, points)
        return [node.id_ for node in nodes]

    async def _ll_query(
        self, reqs: Sequence[rest.QueryRequest], /
    ) -> Sequence[VectorStoreQueryResult]:
        if not reqs:
            return []
        qrs = await self.aclient.query_batch_points(self.collection_name, reqs)
        return [_parse_to_query_result(r.points) for r in qrs]


async def _aembed_sparse(
    *queries: str, fn: SparseEncode
) -> list[rest.SparseVector]:
    ichunk, vchunk = await asyncio.to_thread(fn, queries)
    return [
        rest.SparseVector(indices=ids, values=vs)
        for ids, vs in zip(ichunk, vchunk, strict=True)
    ]


def _min_max(xs: Collection[float]) -> Collection[float]:
    if ptp := max(xs) - (lo := min(xs)):
        return [(x - lo) / ptp for x in xs]
    return xs


# --------------- from llama index metadata to qdrant filters ----------------


def _build_filter(
    doc_ids: list[str] | None = None,
    node_ids: list[str] | None = None,
    filters: MetadataFilters | None = None,
) -> rest.Filter | None:
    conditions: list[rest.Condition] = []

    if doc_ids:
        conditions.append(
            rest.FieldCondition(key='doc_id', match=rest.MatchAny(any=doc_ids))
        )

    # Point id is a "service" id, it is not stored in payload.
    # There is 'HasId' condition to filter by point id
    # https://qdrant.tech/documentation/concepts/filtering/#has-id
    if node_ids:
        conditions.append(
            rest.HasIdCondition(has_id=node_ids),  # type: ignore
        )

    if c := _build_subfilter(filters):
        conditions.append(c)

    return rest.Filter(must=conditions) if conditions else None


def _build_subfilter(mfs: MetadataFilters | None) -> rest.Filter | None:
    if not mfs or not mfs.filters:
        return None
    nullable_conditions = [
        (
            _build_subfilter(mf)
            if isinstance(mf, MetadataFilters)
            else _meta_to_condition(mf)
        )
        for mf in mfs.filters
    ]
    conditions = [c for c in nullable_conditions if c]
    match mfs.condition:
        case FilterCondition.AND:
            return rest.Filter(must=conditions)
        case FilterCondition.OR:
            return rest.Filter(should=conditions)
        case FilterCondition.NOT:
            return rest.Filter(must_not=conditions)
        case _:
            return rest.Filter()


def _meta_to_condition(f: MetadataFilter) -> rest.Condition | None:
    op = f.operator
    if op.name in {'LT', 'GT', 'LTE', 'GTE'}:
        return rest.FieldCondition(
            key=f.key,
            range=rest.Range(**{op.name.lower(): f.value}),  # type: ignore
        )

    # Missing value, `None` or [].
    # https://qdrant.tech/documentation/concepts/filtering/#is-empty
    if op.value == 'is_empty':
        return rest.IsEmptyCondition(is_empty=rest.PayloadField(key=f.key))

    if f.value is None:
        msg = f'Invalid filter {f}'
        raise ValueError(msg)

    values = cast(
        'list[int] | list[str]',
        f.value if isinstance(f.value, list) else [f.value],
    )

    m: rest.Match | None = None
    match op.value:
        case 'text_match' | 'text_match_insensitive':
            assert isinstance(f.value, str)
            m = rest.MatchText(text=f.value)

        case '==':
            if isinstance(f.value, float):
                return rest.FieldCondition(
                    key=f.key, range=rest.Range(gte=f.value, lte=f.value)
                )
            m = rest.MatchValue(value=f.value)  # type: ignore

        # Any of
        # https://qdrant.tech/documentation/concepts/filtering/#match-any
        case 'in':
            m = rest.MatchAny(any=values)

        # None of
        # https://qdrant.tech/documentation/concepts/filtering/#match-except
        case '!=' | 'nin':
            m = rest.MatchExcept(**{'except': values})

    if m:
        return rest.FieldCondition(key=f.key, match=m)
    return None


# ------------------------ from qdrant to llama index ------------------------


def _parse_to_query_result(
    points: Iterable[rest.Record | rest.ScoredPoint],
) -> VectorStoreQueryResult:
    nodes: list[BaseNode] = []
    similarities: list[float] = []
    ids: list[str] = []

    for pt in points:
        assert pt.payload is not None
        node = metadata_dict_to_node(pt.payload)

        if node.embedding is None:
            vecs = pt.vector
            if isinstance(vecs, list):
                node.embedding = vecs  # type: ignore[assignment]
            elif isinstance(vecs, dict) and (
                isinstance(vec := vecs.get(_DENSE_NAME), list)
                or isinstance(vec := vecs.get(''), list)
            ):
                node.embedding = vec  # type: ignore[assignment]

        nodes.append(node)
        ids.append(str(pt.id))
        similarities.append(
            pt.score if isinstance(pt, rest.ScoredPoint) else 1.0
        )

    if any(similarities):
        logger.debug(
            'Retrieved %d nodes with score: %.3g - %.3g',
            len(similarities),
            min(similarities),
            max(similarities),
        )

    return VectorStoreQueryResult(
        nodes=nodes, similarities=similarities, ids=ids
    )


# ------------------------ metadata to payload and back ----------------------


def node_to_metadata_dict(node: BaseNode) -> dict:
    """Common logic for saving Node data into metadata dict."""
    # See: llama_index.core.vector_stores.utils:node_to_metadata_dict
    # NOTE: original greatly bloats qdrant because of JSON dump below

    # Using mode="json" here because BaseNode may have fields
    # of type bytes (e.g. images in ImageBlock),
    # which would cause serialization issues.
    node_dict = node.model_dump(mode='json')

    # Remove embedding from node_dict
    node_dict.pop('embedding', None)  # ! originally set None

    # Make metadata the top level
    metadata: dict = node_dict.pop('metadata', {})  # ! originally `get()`

    return metadata | {
        # dump remainder of node_dict to json string
        '_node_content': orjson.dumps(node_dict).decode(),
        '_node_type': node.class_name(),
        # store ref doc id at top level to allow metadata filtering
        'doc_id': src.node_id if (src := node.source_node) else 'None',
    }


def metadata_dict_to_node(metadata: dict) -> BaseNode:
    """Load generic Node from metadata dict."""
    # See: llama_index.core.vector_stores.utils:metadata_dict_to_node
    # ! This one is altered to be compatible with above.
    node_json = metadata.pop('_node_content', None)
    node_type = metadata.pop('_node_type', None)
    if node_json is None:
        msg = 'Node content not found in metadata dict.'
        raise ValueError(msg)

    metadata = {
        k: v
        for k, v in metadata.items()
        if k not in {'document_id', 'doc_id', 'ref_doc_id'}
    }
    data = orjson.loads(node_json)
    data.setdefault('metadata', metadata)
    data.pop('class_name', None)

    tps = {tp.class_name(): tp for tp in (Node, IndexNode, ImageNode)}
    tp = tps.get(node_type, TextNode)
    return tp(**data)
