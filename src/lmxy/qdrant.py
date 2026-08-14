"""Qdrant vector store, built on top of an existing Qdrant collection."""

__all__ = ['Qdrant']

import asyncio
from collections.abc import Awaitable, Callable, Generator, Iterable, Sequence
from typing import Any, Literal, NotRequired, TypedDict, cast
from uuid import UUID

from glow import astreaming
from grpc import RpcError, StatusCode
from grpc.aio import AioRpcError
from httpx import Timeout
from loguru import logger
from pydantic import BaseModel, PrivateAttr
from qdrant_client import AsyncQdrantClient
from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.conversions.common_types import QuantizationConfig
from qdrant_client.fastembed_common import IDF_EMBEDDING_MODELS
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from ._retry import aretry
from ._types import Embedding, SparseEncode
from .fastembed import get_sparse_encoder
from .util import min_max

_Id = int | str | UUID

# embedding/text, top K, score threshold
type DenseQuery = tuple[Embedding, int, float | None]
type SparseQuery = tuple[str, int]
type FusionMode = Literal['hsf', 'rrf', 'dbsf']

_SPARSE_MODIFIERS = dict.fromkeys(IDF_EMBEDDING_MODELS, rest.Modifier.IDF)
_LOCK = asyncio.Lock()
_log = logger.opt(depth=1)


class Record(TypedDict):
    id_: _Id
    data: dict[str, str]
    embeddings: NotRequired[list[Embedding]]


class EmbedRecord(Record):
    embed_text: NotRequired[str]


class ScoredRecord(Record):
    score: float


class Qdrant(BaseModel):
    """Fork of LlamaIndex's Qdrant Vector Store.

    Differences:
    - async only
    - no legacy formats
    - no legacy sparse embeddings
    - Qdrant Query API
    - no Llama Index dependency

    In this vector store, embeddings and docs are stored within a
    Qdrant collection.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes.
    """

    model_config = {'arbitrary_types_allowed': True}

    collection_name: str
    aclient: AsyncQdrantClient
    upsert_timeout: float | None = None  # enable to batch upserts
    upsert_batch_size: int = 64
    query_timeout: float | None = None  # enable to batch upserts
    query_batch_size: int = 64
    retries: int | None = 3  # None = retry forever

    # Collection construction parameters
    dense_config: rest.VectorParams = rest.VectorParams(
        size=0,
        distance=rest.Distance.COSINE,
        multivector_config=rest.MultiVectorConfig(
            comparator=rest.MultiVectorComparator.MAX_SIM
        ),
    )
    sparse_config: rest.SparseVectorParams = rest.SparseVectorParams()
    shard_number: int | None = None
    hnsw_config: rest.HnswConfigDiff | None = None
    optimizers_config: rest.OptimizersConfigDiff | None = None
    quantization_config: QuantizationConfig | None = None
    tenant_fields: list[str] = []  # For multitenancy

    # Sparse search parameters
    sparse_doc_fn: SparseEncode | None = None
    sparse_query_fn: SparseEncode | None = None
    sparse_model: str | None = None
    sparse_model_kwargs: dict[str, Any] = {}

    # Field names
    dense_field_name: str = 'text-dense'
    sparse_field_name: str = 'text-sparse'

    _update: Callable[
        [Iterable[EmbedRecord | _Id]],
        Awaitable[list[_Id]],
    ] = PrivateAttr()
    _qd_query: Callable[
        [Iterable[rest.QueryRequest]],
        Awaitable[list[Sequence[rest.ScoredPoint]]],
    ] = PrivateAttr()

    _is_initialized: bool = PrivateAttr()

    def model_post_init(self, context) -> None:
        if self.retries is not None:
            self.retries = max(self.retries, 0)
        retry_ = aretry(
            RpcError,
            UnexpectedResponse,
            max_attempts=None if self.retries is None else 1 + self.retries,
        )
        update = self._ll_update
        if self.upsert_timeout is not None:
            update = astreaming(
                update,
                batch_size=self.upsert_batch_size,
                timeout=self.upsert_timeout,
            )
        self._update = retry_(update)

        qd_query = self._ll_qd_query
        if self.query_timeout is not None:
            qd_query = astreaming(
                qd_query,
                batch_size=self.query_batch_size,
                timeout=self.query_timeout,
            )
        self._qd_query = retry_(qd_query)

        modifier = _SPARSE_MODIFIERS.get(self.sparse_model or '')
        self.sparse_config.modifier = modifier
        self._is_initialized = False

    @classmethod
    def create(
        cls,
        collection_name: str,
        vector_size: int,
        # connection
        host: str,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        retries: int | None = None,
        # sparse model
        sparse_model: str = 'Qdrant/bm25',
        sparse_model_kwargs: dict[str, Any] | None = None,
        # speed
        upsert_timeout: float = 0.01,
        query_timeout: float = 0.01,
        upsert_batch_size: int = 10,
        query_batch_size: int = 10,
        shard_number: int = 1,
        # collection options
        tenant_fields: Iterable[str] = (),
    ) -> 'Qdrant':
        aclient = AsyncQdrantClient(
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            host=host,
        )
        assert isinstance(aclient._client, AsyncQdrantRemote)
        aclient._client.http.client._async_client.timeout = Timeout(None)

        dense_config = rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
            on_disk=True,
            datatype=rest.Datatype.FLOAT32,
        )
        hnsw_config = rest.HnswConfigDiff(
            # Number of edges per node (16).
            # 12-16, higher = more accuracy / more memory.
            m=0,
            payload_m=32,
            # Size of the dynamic candidate list during construction (100).
            # 100-200, higher = better quality / slower construct.
            ef_construct=64,
            # Threshold for using HNSW vs exhaustive search (10000)
            # 5000-20000 depending on vector dimensions.
            full_scan_threshold=10_000,
            on_disk=False,
        )
        optimizers_config = rest.OptimizersConfigDiff(
            flush_interval_sec=5,
            # When to start building index (20000), 10k-20k
            indexing_threshold=20_000,
            # When to switch to disk-based storage (None), 2-5x indexing
            memmap_threshold=50_000,
            # Target number of segments (0).
            # 3-7, higher = faster updates / slower search.
            default_segment_number=4,
            # GC thresholds
            vacuum_min_vector_number=1_000,  # as count per segment
            deleted_threshold=0.2,  # fraction of segment
        )

        return cls(
            collection_name=collection_name,
            aclient=aclient,
            # speed
            upsert_timeout=upsert_timeout,
            query_timeout=query_timeout,
            upsert_batch_size=upsert_batch_size,
            query_batch_size=query_batch_size,
            retries=retries,
            # collection construction
            dense_config=dense_config,
            shard_number=shard_number,  # To allow parallel indexing
            hnsw_config=hnsw_config,
            tenant_fields=list(tenant_fields),
            optimizers_config=optimizers_config,
            # sparsity
            sparse_model=sparse_model,
            sparse_model_kwargs=sparse_model_kwargs or {},
        )

    async def initialize(self, vector_size: int) -> None:
        if self._is_initialized:
            return
        async with _LOCK:
            await self._initialize_unsafe(vector_size)

    async def is_initialized(self) -> bool:
        if self._is_initialized:
            return True
        async with _LOCK:
            return await self._is_initialized_unsafe()

    async def _initialize_unsafe(self, vector_size: int) -> None:
        await self._load_models()

        self.dense_config.size = self.dense_config.size or vector_size
        if vector_size != self.dense_config.size:
            raise ValueError(
                f'Invalid vector size {vector_size} '
                f'for dense config {self.dense_config}'
            )

        try:
            await self.aclient.create_collection(
                self.collection_name,
                vectors_config={self.dense_field_name: self.dense_config},
                sparse_vectors_config=(
                    {self.sparse_field_name: self.sparse_config}
                    if self.sparse_query_fn and self.sparse_doc_fn
                    else None
                ),
                shard_number=self.shard_number,
                hnsw_config=self.hnsw_config,
                optimizers_config=self.optimizers_config,
                quantization_config=self.quantization_config,
            )

            self._is_initialized = True
        except (RpcError, ValueError, UnexpectedResponse) as exc:
            if 'already exists' not in str(exc):
                raise exc  # noqa: TRY201
            _log.warning(f'Reusing existing collection {self.collection_name}')
            assert await self._is_initialized_unsafe()

        await self._setup_indices()

    async def _setup_indices(self) -> None:
        tenant_schema = rest.KeywordIndexParams(
            type=rest.KeywordIndexType.KEYWORD, is_tenant=True
        )
        name_n_schema = [('doc_id', rest.PayloadSchemaType.KEYWORD)] + [
            (field, tenant_schema) for field in self.tenant_fields
        ]

        # To improve search performance set up a payload index
        # for fields used in filters.
        # https://qdrant.tech/documentation/concepts/indexing
        aws = (
            self.aclient.create_payload_index(
                self.collection_name, field_name=name, field_schema=schema
            )
            for name, schema in name_n_schema
        )
        await asyncio.gather(*aws)

    async def _is_initialized_unsafe(self) -> bool:
        if self._is_initialized:
            return True
        if not await self.aclient.collection_exists(self.collection_name):
            return False
        await self._load_models()
        info = await self.aclient.get_collection(self.collection_name)

        dense = info.config.params.vectors
        if not isinstance(dense, dict):
            msg = (
                f'Collection {self.collection_name} is using '
                'legacy anonymous vectors. '
                'Recreate it to allow sparse/hybrid search'
            )
            raise TypeError(msg)

        sparse = info.config.params.sparse_vectors
        if isinstance(sparse, dict) and self.sparse_field_name in sparse:
            if not self.sparse_query_fn:
                _log.warning(
                    'Collection {} support '
                    'sparse search, but neither '
                    'sparse_query_fn nor sparse_model was provided',
                    self.collection_name,
                )
            if not self.sparse_doc_fn:
                _log.warning(
                    'Collection {} support '
                    'sparse search, but neither '
                    'sparse_doc_fn nor sparse_model was provided',
                    self.collection_name,
                )
        else:
            self.sparse_query_fn = self.sparse_doc_fn = None

        self._is_initialized = True
        return True

    async def _load_models(self) -> None:
        if self.sparse_model is None or (
            self.sparse_doc_fn is not None and self.sparse_query_fn is not None
        ):
            return

        encoder = await asyncio.to_thread(
            get_sparse_encoder, self.sparse_model, **self.sparse_model_kwargs
        )
        self.sparse_doc_fn = self.sparse_doc_fn or encoder
        self.sparse_query_fn = self.sparse_query_fn or encoder

    # CRUD: create
    async def add(self, records: Sequence[EmbedRecord]) -> list[_Id]:
        return await self._update(records)

    # CRUD: read
    async def retrieve(
        self,
        ids: Sequence[_Id],
        *,
        with_payload: Sequence[str] | bool = True,
    ) -> list[ScoredRecord]:
        points = await self.qd_retrieve(ids, with_payload=with_payload)
        return [_qd_to_record(pt, self.dense_field_name) for pt in points]

    async def query(
        self,
        dense: DenseQuery | None = None,
        sparse: SparseQuery | None = None,
        *,
        fuse: tuple[int, float] = (1, 0.5),
        filters: rest.Filter | None = None,
        with_payload: Sequence[str] | bool = True,
        mode: FusionMode = 'hsf',
    ) -> list[ScoredRecord]:
        dq = sq = None
        k, alpha = fuse
        assert 0 <= alpha <= 1

        k_max = 0
        if dense and (dk := dense[1]) and alpha > 0.0:
            if not dense[0]:
                msg = 'query embedding is required for dense queries'
                raise ValueError(msg)
            k_max += dk
            dq = self.query1(
                *dense, filters=filters, with_payload=with_payload
            )
        if sparse and (sk := sparse[1]) and alpha < 1.0:
            if not sparse[0]:
                msg = 'query str is required for sparse queries'
                raise ValueError(msg)
            k_max += sk
            sq = self.query1(
                *sparse, filters=filters, with_payload=with_payload
            )

        # With hybrid search we get:
        # - some nodes from dense search;
        # - some nodes from sparse search;
        # - and some nodes coming from both with merged scores.
        # The larger `dense_k`/`sparse_k` the higher chances to get these.

        # `k` is effective only up to `dense_k+sparse_k`
        k = min(k, k_max)
        q = sq.hybrid(dq, alpha, k, mode=mode) if dq and sq else (dq or sq)
        return (await q) if q else []

    def query1(
        self,
        q: str | Embedding,
        limit: int = 1,
        threshold: float | None = None,
        filters: rest.Filter | None = None,
        with_payload: Sequence[str] | bool = True,
    ) -> '_Request':
        return _Request(
            lambda: self._make_qd_query(q, limit, threshold, filters=filters),
            self,
            with_payload=with_payload,
        )

    async def qd_retrieve(
        self,
        ids: Sequence[_Id],
        with_payload: Sequence[str] | bool = True,
    ) -> list[rest.Record]:
        aw = self.aclient.retrieve(self.collection_name, ids, with_payload)
        try:
            return await aw
        except AioRpcError as e:
            if e.code() is StatusCode.NOT_FOUND:
                _log.warning(f'Not initialized: {self.collection_name}')
                return []
            raise
        except UnexpectedResponse as e:
            if e.status_code == 404:
                _log.warning(f'Not initialized: {self.collection_name}')
                return []
            raise

    async def qd_query(
        self,
        q: str | Embedding,
        limit: int = 1,
        threshold: float | None = None,
        filters: rest.Filter | None = None,
        with_payload: Sequence[str] | bool = True,
    ) -> list[rest.Record | rest.ScoredPoint]:
        if isinstance(with_payload, Sequence):
            with_payload = list(with_payload)
        ptss = await self._resolve(
            lambda: self._make_qd_query(q, limit, threshold, filters=filters),
            with_payload=with_payload,
        )
        return list(ptss[0]) if ptss else []

    async def _make_qd_query(
        self,
        q: str | Embedding,
        limit: int = 1,
        threshold: float | None = None,
        filters: rest.Filter | None = None,
    ) -> rest.Prefetch | None:
        if not limit:
            return None
        vec: Embedding | rest.SparseVector
        if isinstance(q, str):
            if not self.sparse_query_fn:
                msg = (
                    f'Collection {self.collection_name} does not '
                    'have sparse vectors to do sparse search. '
                    'Please reinitialize it with sparse model '
                    'to allow sparse/hybrid search'
                )
                raise ValueError(msg)
            [(ids, vals)] = await asyncio.to_thread(self.sparse_query_fn, [q])
            vec = rest.SparseVector(indices=ids, values=vals)
            using = self.sparse_field_name
        else:
            vec = q
            using = self.dense_field_name

        return rest.Prefetch(
            query=vec,
            using=using,
            filter=filters,
            score_threshold=threshold,
            limit=limit,
        )

    # CRUD: delete
    async def delete_by(self, value: str, key: str) -> None:
        cond = rest.FieldCondition(key=key, match=rest.MatchValue(value=value))
        selector = rest.Filter(must=[cond])
        try:
            await self.aclient.delete(self.collection_name, selector)
        except AioRpcError as e:
            if e.code() is StatusCode.NOT_FOUND:
                _log.warning(f'Not initialized: {self.collection_name}')
                return
            raise
        except UnexpectedResponse as e:
            if e.status_code == 404:
                _log.warning(f'Not initialized: {self.collection_name}')
                return
            raise

    # CRUD: delete
    async def delete(self, ids: Sequence[str], /) -> None:
        await self._update(ids)

    async def clear(self) -> None:
        async with _LOCK:
            await self.aclient.delete_collection(self.collection_name)
            self._is_initialized = False

    # low levels

    async def _resolve(
        self,
        *subqueries: Callable[[], Awaitable[rest.Prefetch | None]],
        with_payload: list[str] | bool = True,
        query: rest.FusionQuery | rest.RrfQuery | None = None,
        limit: int = 1,
    ) -> list[Sequence[rest.ScoredPoint]]:
        prefetches = [
            p for p in await asyncio.gather(*(cr() for cr in subqueries)) if p
        ]
        if query is not None and len(prefetches) > 1:
            req = rest.QueryRequest(
                prefetch=list(prefetches),
                query=query,
                limit=limit,
                with_payload=with_payload,
            )
            reqs = [req]
        else:
            reqs = [
                rest.QueryRequest(
                    query=p.query,
                    using=p.using,
                    filter=p.filter,
                    score_threshold=p.score_threshold,
                    limit=p.limit,
                    with_payload=with_payload,
                )
                for p in prefetches
            ]
        return [pts for pts in await self._qd_query(reqs) if pts]

    async def _ll_update(
        self, records: Iterable[EmbedRecord | _Id], /
    ) -> list[_Id]:
        # Merge and deduplicate updates & deletions
        ids: list[_Id] = []
        actions: dict[_Id, EmbedRecord | None] = {}
        for r in records:  # For same ID of add/rm last takes precedence
            if isinstance(r, _Id):
                ids.append(r)
                actions[r] = None
            else:
                ids.append(r['id_'])
                actions[r['id_']] = r

        try:
            async with asyncio.TaskGroup() as tg:
                if recs := [a for a in actions.values() if a]:
                    vecs = (v for r in recs for v in r.get('embeddings', []))
                    vec = next(vecs, None)
                    if vec is None:
                        raise ValueError('No dense vectors to store')
                    await self.initialize(len(vec))
                    tg.create_task(self._ll_upsert(recs))

                if rm_ids := [i for i, a in actions.items() if not a]:
                    tg.create_task(
                        self.aclient.delete(self.collection_name, list(rm_ids))
                    )
        except* AioRpcError as eg:
            errors = cast('list[AioRpcError]', eg.exceptions)
            if any(e.code() == StatusCode.NOT_FOUND for e in errors):
                _log.warning(f'Not initialized: {self.collection_name}')
            if rest := [e for e in errors if e.code() != StatusCode.NOT_FOUND]:
                raise eg.derive(rest) from eg
        except* UnexpectedResponse as eg:
            errors = cast('list[UnexpectedResponse]', eg.exceptions)
            if any(e.status_code == 404 for e in errors):
                _log.warning(f'Not initialized: {self.collection_name}')
            if rest := [e for e in errors if e.status_code != 404]:
                raise eg.derive(rest) from eg

        return ids

    async def _ll_upsert(self, recs: Sequence[EmbedRecord]) -> None:
        svs = await _aembed_sparse_records(self.sparse_doc_fn, recs)
        points = [
            _record_to_qd(
                r,
                dense_field=self.dense_field_name,
                sparse_field=self.sparse_field_name,
                sparse_vec=sv,
            )
            for r, sv in zip(recs, svs, strict=True)
        ]
        await self.aclient.upsert(self.collection_name, points)

    async def _ll_qd_query(
        self, reqs: Iterable[rest.QueryRequest], /
    ) -> list[Sequence[rest.ScoredPoint]]:
        reqs = list(reqs)
        if not reqs:
            return []
        aw = self.aclient.query_batch_points(self.collection_name, reqs)
        try:
            qrs = await aw
        except AioRpcError as e:
            if e.code() is StatusCode.NOT_FOUND:
                _log.warning(f'Not initialized: {self.collection_name}')
                return [[] for _ in reqs]
            raise
        except UnexpectedResponse as e:
            if e.status_code == 404:
                _log.warning(f'Not initialized: {self.collection_name}')
                return [[] for _ in reqs]
            raise
        return [r.points for r in qrs]


async def _aembed_sparse_records(
    fn: SparseEncode | None, records: Sequence[EmbedRecord]
) -> list[rest.SparseVector | None]:
    if not fn:
        return [None for _ in records]

    vectors: list[rest.SparseVector | None] = []
    embed_ids: list[int] = []
    embed_texts: list[str] = []
    for i, r in enumerate(records):
        if txt := r.get('embed_text'):
            embed_ids.append(i)
            embed_texts.append(txt)
        vectors.append(None)

    if embed_texts:
        svecs = await asyncio.to_thread(fn, embed_texts)
        for i, (ids, vs) in zip(embed_ids, svecs, strict=True):
            vectors[i] = rest.SparseVector(indices=ids, values=vs)

    return vectors


# -------------------------- from qdrant to native ---------------------------


def _record_to_qd(
    record: EmbedRecord,
    dense_field: str,
    sparse_field: str,
    sparse_vec: rest.SparseVector | None = None,
) -> rest.PointStruct:
    vector: dict[str, rest.Vector] = {}
    if dembs := record.get('embeddings'):
        vector[dense_field] = dembs
    if sparse_vec is not None:
        vector[sparse_field] = sparse_vec
    if not vector:
        raise ValueError(f'Embedding is not set: keys={record.keys()}')

    return rest.PointStruct(
        id=record['id_'], vector=vector, payload=record['data']
    )


def _qd_to_record(
    pt: rest.Record | rest.ScoredPoint,
    dense_field_name: str = 'text-dense',
) -> ScoredRecord:
    assert pt.payload is not None
    s = pt.score if isinstance(pt, rest.ScoredPoint) else 1.0
    rec = ScoredRecord(id_=pt.id, data=pt.payload, score=s)

    vecs = pt.vector
    if vecs is None:
        return rec
    if isinstance(vecs, list):
        raise TypeError('Anonimous dense vectors are not supported')
    vec_or_vecs = vecs.get(dense_field_name)
    if vec_or_vecs is not None:
        if isinstance(vec_or_vecs, rest.SparseVector):
            raise TypeError('sparse vector in dense field')
        rec['embeddings'] = (
            cast('list[Embedding]', vec_or_vecs)
            if all(isinstance(v, list) for v in vec_or_vecs)
            else [cast('Embedding', vec_or_vecs)]
        )

    return rec


class _Request:
    def __init__(
        self,
        query: Callable[[], Awaitable[rest.Prefetch | None]],
        qd: Qdrant,
        with_payload: Sequence[str] | bool = True,
    ) -> None:
        self.query = query
        self.qd = qd
        if isinstance(with_payload, Sequence):
            with_payload = list(with_payload)
        self.with_payload = with_payload

    def __await__(self) -> Generator[Any, Any, list[ScoredRecord]]:
        return self.acall().__await__()

    async def hybrid(
        self,
        other: '_Request',
        t: float = 0.5,
        limit: int = 1,
        mode: FusionMode = 'hsf',
        rrf_k: int | None = None,
    ) -> list[ScoredRecord]:
        match mode:
            case 'hsf':
                return await self.hsf(other, t, limit)
            case 'rrf':
                return await self._fuse(
                    other,
                    rest.RrfQuery(rrf=rest.Rrf(k=rrf_k, weights=[1 - t, t])),
                    limit=limit,
                )
            case 'dbsf':
                return await self._fuse(
                    other,
                    rest.FusionQuery(fusion=rest.Fusion.DBSF),
                    limit=limit,
                )
        raise NotImplementedError

    async def _fuse(
        self,
        other: '_Request',
        query: rest.FusionQuery | rest.RrfQuery,
        limit: int = 1,
    ) -> list[ScoredRecord]:
        if limit <= 0:
            return []
        ptss = await self.qd._resolve(
            self.query,
            other.query,
            with_payload=self.with_payload,
            query=query,
            limit=limit,
        )
        if not ptss:
            return []
        [pts] = ptss
        _log_scores(pts)
        return [_qd_to_record(pt) for pt in pts]

    async def hsf(
        self,
        other: '_Request',
        t: float = 0.5,
        limit: int = 1,
    ) -> list[ScoredRecord]:
        """
        `= (normalized(self) * (1-t) + normalized(other) * t)[:n]`
        """
        if limit <= 0:
            return []
        fns = [self.query] if t < 1 else []
        fns = [*fns, other.query] if t > 0 else fns
        if not fns:
            return []
        ptss = await self.qd._resolve(*fns, with_payload=self.with_payload)
        if not ptss:
            return []
        if len(ptss) == 1:
            [pts] = ptss
        else:
            [lhs, rhs] = ptss
            uniq = {r.id: r for r in [*lhs, *rhs]}
            zeros = dict.fromkeys(uniq, 0.0)
            pts1 = zeros | _min_max_scores(lhs)
            pts2 = zeros | _min_max_scores(rhs)
            pts = [
                p.model_copy(update={'score': (1 - t) * pts1[i] + t * pts2[i]})
                for i, p in uniq.items()
            ]

        pts = sorted(pts, key=lambda p: p.score, reverse=True)[:limit]
        _log_scores(pts, name='fused records')
        return [_qd_to_record(pt) for pt in pts]

    async def acall(self) -> list[ScoredRecord]:
        ptss = await self.qd._resolve(
            self.query, with_payload=self.with_payload
        )
        if not ptss:
            return []
        [points] = ptss
        _log_scores(points)
        return [_qd_to_record(pt) for pt in points]


def _min_max_scores(
    pts: Sequence[rest.ScoredPoint],
) -> dict[rest.ExtendedPointId, float]:
    return dict(
        zip([p.id for p in pts], min_max(p.score for p in pts), strict=True)
    )


def _log_scores(
    pts: Sequence[rest.ScoredPoint], name: str = 'records'
) -> None:
    scores = [p.score for p in pts]
    if not any(scores):
        return
    n, lo, hi = len(scores), min(scores), max(scores)
    _log.info(f'Retrieved {n} {name} with score: {lo:.3g} .. {hi:.3g}')
