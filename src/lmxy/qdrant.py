"""Qdrant vector store, built on top of an existing Qdrant collection."""

__all__ = ['Qdrant']

import asyncio
from asyncio import TaskGroup
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Sequence,
)
from typing import Any, Literal, NotRequired, TypedDict, cast
from uuid import UUID

import qdrant_client.http.models as qm
from glow import flatten_exc, streaming
from grpc import RpcError, StatusCode
from grpc.aio import AioRpcError
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, ValidationError
from pydantic_core import to_json
from qdrant_client import AsyncQdrantClient
from qdrant_client.conversions.common_types import QuantizationConfig
from qdrant_client.fastembed_common import IDF_EMBEDDING_MODELS
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

_SPARSE_MODIFIERS = dict.fromkeys(IDF_EMBEDDING_MODELS, qm.Modifier.IDF)
_LOCK = asyncio.Lock()
_log = logger.opt(depth=1)


class Record(TypedDict):
    id_: _Id
    data: dict[str, Any]
    embeddings: NotRequired[list[Embedding]]


class EmbedRecord(Record):
    embed_text: NotRequired[str]
    dense_texts: NotRequired[list[str]]


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
    upsert_batch_size: int = Field(default=64, gt=0)
    upsert_batch_bytes: int = Field(default=1024 * 1024, gt=0)
    query_timeout: float | None = None  # enable to batch upserts
    query_batch_size: int = 64
    retries: int | None = 3  # None = retry forever

    # Collection construction parameters
    dense_config: qm.VectorParams = qm.VectorParams(
        size=0,
        distance=qm.Distance.COSINE,
        multivector_config=qm.MultiVectorConfig(
            comparator=qm.MultiVectorComparator.MAX_SIM
        ),
    )
    sparse_config: qm.SparseVectorParams = qm.SparseVectorParams()
    shard_number: int | None = None
    hnsw_config: qm.HnswConfigDiff | None = None
    optimizers_config: qm.OptimizersConfigDiff | None = None
    quantization_config: QuantizationConfig | None = None
    kw_fields: list[str] = []  # For search
    int_fields: list[str] = []  # For search
    uuid_fields: list[str] = []  # For search
    tenant_fields: list[str] = []  # For multitenancy

    # Sparse search parameters
    sparse_doc_fn: SparseEncode | None = None
    sparse_query_fn: SparseEncode | None = None
    sparse_model: str | None = None
    sparse_model_kwargs: dict[str, Any] = {}

    # Field names
    dense_field_name: str = 'text-dense'
    sparse_field_name: str = 'text-sparse'

    _qd_query: Callable[
        [Sequence[qm.QueryRequest]],
        Awaitable[list[Sequence[qm.ScoredPoint]]],
    ] = PrivateAttr()
    _update: Callable[
        [Sequence[EmbedRecord | _Id]],
        Awaitable[list[_Id]],
    ] = PrivateAttr()
    _delete_by: Callable[
        [Sequence[tuple[str, str]]],
        Awaitable[list[None]],
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

        qd_query = self._ll_qd_query
        if self.query_timeout is not None:
            qd_query = streaming(
                qd_query,
                batch_size=self.query_batch_size,
                timeout=self.query_timeout,
            )
        self._qd_query = retry_(qd_query)

        update = self._ll_update
        delete_by = self._ll_delete_by
        if self.upsert_timeout is not None:
            deco = streaming(
                batch_size=self._update_batch_size,
                timeout=self.upsert_timeout,
            )
            update = deco(update)
            delete_by = deco(delete_by)
        self._update = retry_(update)
        self._delete_by = retry_(delete_by)

        modifier = _SPARSE_MODIFIERS.get(self.sparse_model or '')
        self.sparse_config.modifier = modifier
        self._is_initialized = False

    async def initialize(self, vector_size: int) -> None:
        if self._is_initialized:
            return
        async with _LOCK:
            await self._load_models_unsafe()
            await self._initialize_unsafe(vector_size)

    async def is_initialized(self) -> bool:
        if self._is_initialized:
            return True
        async with _LOCK:
            return await self._is_initialized_unsafe()

    async def _initialize_unsafe(self, vector_size: int) -> None:
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
        tenant_schema = qm.KeywordIndexParams(
            type=qm.KeywordIndexType.KEYWORD, is_tenant=True
        )
        name_n_schema = (
            [
                (field, qm.PayloadSchemaType.KEYWORD)
                for field in {'doc_id', *self.kw_fields}
            ]
            + [
                (field, qm.PayloadSchemaType.INTEGER)
                for field in self.int_fields
            ]
            + [
                (field, qm.PayloadSchemaType.UUID)
                for field in self.uuid_fields
            ]
            + [(field, tenant_schema) for field in self.tenant_fields]
        )

        # To improve search performance set up a payload index
        # for fields used in filters.
        # https://qdrant.tech/documentation/concepts/indexing
        async with TaskGroup() as tg:
            for field_name, field_schema in name_n_schema:
                aw = self.aclient.create_payload_index(
                    self.collection_name, field_name, field_schema
                )
                tg.create_task(aw)

    async def _is_initialized_unsafe(self) -> bool:
        if self._is_initialized:
            return True
        if not await self.aclient.collection_exists(self.collection_name):
            return False
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
            self.sparse_model = None
            self.sparse_query_fn = self.sparse_doc_fn = None

        self._is_initialized = True
        return True

    async def _load_models(self) -> None:
        if self.sparse_model is None or (
            self.sparse_doc_fn is not None and self.sparse_query_fn is not None
        ):
            return
        async with _LOCK:
            await self._load_models_unsafe()

    async def _load_models_unsafe(self) -> None:
        if self.sparse_model is None or (
            self.sparse_doc_fn is not None and self.sparse_query_fn is not None
        ):
            return
        encoder = await asyncio.to_thread(
            get_sparse_encoder,
            self.sparse_model,
            **self.sparse_model_kwargs,
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
        with_payload: qm.WithPayloadInterface | Sequence[str] = True,
    ) -> list[ScoredRecord]:
        points = await self.qd_retrieve(ids, with_payload=with_payload)
        return [_qd_to_record(pt, self.dense_field_name) for pt in points]

    async def query(
        self,
        dense: DenseQuery | None = None,
        sparse: SparseQuery | None = None,
        *,
        fuse: tuple[int, float, FusionMode] = (1, 0.5, 'hsf'),
        filters: qm.Filter | None = None,
        with_payload: qm.WithPayloadInterface | Sequence[str] = True,
    ) -> list[ScoredRecord]:
        dq = sq = None
        k, alpha, mode = fuse
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
        q: str | Embedding | None = None,
        limit: int = 1,
        threshold: float | None = None,
        filters: qm.Filter | None = None,
        with_payload: qm.WithPayloadInterface | Sequence[str] = True,
    ) -> '_Request':
        if isinstance(with_payload, Sequence):
            with_payload = list(with_payload)
        return _Request(
            lambda: self._make_qd_query(q, limit, threshold, filters=filters),
            self,
            with_payload=with_payload,
        )

    async def scroll(
        self,
        where: qm.Filter,
        with_payload: qm.WithPayloadInterface | Sequence[str] = True,
        batch_size: int = 100,
    ) -> AsyncGenerator[ScoredRecord]:
        if isinstance(with_payload, Sequence):
            with_payload = list(with_payload)
        offset = None
        while True:
            points, offset = await self.aclient.scroll(
                self.collection_name,
                where,
                limit=batch_size,
                offset=offset,
                with_payload=with_payload,
            )
            for pt in points:
                yield _qd_to_record(pt, self.dense_field_name)
            if len(points) < batch_size or offset is None:
                return

    async def qd_retrieve(
        self,
        ids: Sequence[_Id],
        with_payload: qm.WithPayloadInterface | Sequence[str] = True,
    ) -> list[qm.Record]:
        if isinstance(with_payload, Sequence):
            with_payload = list(with_payload)
        aw = self.aclient.retrieve(self.collection_name, ids, with_payload)
        try:
            return await aw
        except (AioRpcError, UnexpectedResponse) as exc1:
            if exc2 := self._remove_404_errors(exc1):
                raise exc2 from None
            return []

    async def qd_query(
        self,
        q: str | Embedding | None = None,
        limit: int = 1,
        threshold: float | None = None,
        filters: qm.Filter | None = None,
        with_payload: qm.WithPayloadInterface | Sequence[str] = True,
    ) -> list[qm.Record | qm.ScoredPoint]:
        if isinstance(with_payload, Sequence):
            with_payload = list(with_payload)
        ptss = await self._resolve(
            lambda: self._make_qd_query(q, limit, threshold, filters=filters),
            with_payload=with_payload,
        )
        return list(ptss[0]) if ptss else []

    async def _make_qd_query(
        self,
        q: str | Embedding | None = None,
        limit: int = 1,
        threshold: float | None = None,
        filters: qm.Filter | None = None,
    ) -> qm.Prefetch | None:
        if not limit:
            return None
        await self._load_models()

        if q is None:  # Scroll
            if threshold is not None:
                raise ValueError('score_threshold requires query')
            return qm.Prefetch(query=q, filter=filters, limit=limit)

        if not isinstance(q, str):  # Dense embedding
            return qm.Prefetch(
                query=q,
                using=self.dense_field_name,
                filter=filters,
                score_threshold=threshold,
                limit=limit,
            )

        # Raw string for sparse embedding
        if not self.sparse_query_fn:
            msg = (
                f'Collection {self.collection_name} does not '
                'have sparse vectors to do sparse search. '
                'Please reinitialize it with sparse model '
                'to allow sparse/hybrid search'
            )
            raise ValueError(msg)

        [(ids, vals)] = await asyncio.to_thread(self.sparse_query_fn, [q])
        return qm.Prefetch(
            query=qm.SparseVector(indices=ids, values=vals),
            using=self.sparse_field_name,
            filter=filters,
            score_threshold=threshold,
            limit=limit,
        )

    # CRUD: delete
    async def delete_by(self, *values: str, key: str) -> None:
        await self._delete_by([(key, v) for v in values])

    async def _ll_delete_by(
        self, kvs: Sequence[tuple[str, str]]
    ) -> list[None]:
        groups: dict[str, list[str]] = {}
        for k, v in kvs:
            groups.setdefault(k, []).append(v)
        if should := [
            qm.FieldCondition(key=k, match=qm.MatchAny(any=vs))
            for k, vs in groups.items()
        ]:
            await self.delete_where(should=should)
        return [None] * sum(len(vs) for vs in groups.values())

    # CRUD: delete
    async def delete_where(
        self,
        *,
        should: Sequence[qm.Condition] = (),
        min_should: qm.MinShould | None = None,
        must: Sequence[qm.Condition] = (),
        must_not: Sequence[qm.Condition] = (),
    ) -> None:
        where = qm.Filter(
            should=list(should),
            min_should=min_should,
            must=list(must),
            must_not=list(must_not),
        )
        try:
            await self.aclient.delete(self.collection_name, where)
        except (AioRpcError, UnexpectedResponse) as exc1:
            if exc2 := self._remove_404_errors(exc1):
                raise exc2 from None
            return

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
        *subqueries: Callable[[], Awaitable[qm.Prefetch | None]],
        with_payload: qm.WithPayloadInterface = True,
        query: qm.QueryInterface | None = None,
        limit: int = 1,
    ) -> list[Sequence[qm.ScoredPoint]]:
        pfs = await asyncio.gather(*(cr() for cr in subqueries))
        prefetches = [p for p in pfs if p]
        if query is not None and len(prefetches) > 1:
            req = qm.QueryRequest(
                prefetch=prefetches,
                query=query,
                limit=limit,
                with_payload=with_payload,
            )
            [pts] = await self._qd_query([req])
            return [pts] if pts else []

        reqs = [
            qm.QueryRequest(
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
        self, records: Sequence[EmbedRecord | _Id], /
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

        recs = [a for a in actions.values() if a]
        if recs:
            vecs = (v for r in recs for v in r.get('embeddings', []))
            vec = next(vecs, None)
            if vec is None:
                raise ValueError('No dense vectors to store')
            await self.initialize(len(vec))

        try:
            async with asyncio.TaskGroup() as tg:
                if recs:
                    tg.create_task(self._ll_upsert(recs))

                if rm_ids := [i for i, a in actions.items() if not a]:
                    tg.create_task(
                        self.aclient.delete(self.collection_name, list(rm_ids))
                    )
        except BaseException as exc1:  # noqa: BLE001
            if exc2 := self._remove_404_errors(exc1):
                raise exc2 from None
        return ids

    async def _ll_upsert(self, recs: Sequence[EmbedRecord]) -> None:
        await self._load_models()
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

    def _update_batch_size(self, items: list[_Id | EmbedRecord]) -> int:
        batch_bytes = 0
        for i, x in enumerate(items, 1):
            batch_bytes += len(to_json(x))
            if batch_bytes > self.upsert_batch_bytes:
                return max(i - 1, 1)
            if i >= self.upsert_batch_size:
                return i
        return 0  # Wait for more points or the streaming timeout.

    async def _ll_qd_query(
        self, reqs: Sequence[qm.QueryRequest], /
    ) -> list[Sequence[qm.ScoredPoint]]:
        if not reqs:
            return []
        aw = self.aclient.query_batch_points(self.collection_name, reqs)
        try:
            qrs = await aw
        except (AioRpcError, UnexpectedResponse) as exc1:
            if exc2 := self._remove_404_errors(exc1):
                raise exc2 from None
            return [[] for _ in reqs]
        return [r.points for r in qrs]

    def _remove_404_errors(self, ex: BaseException) -> BaseException | None:
        ex = flatten_exc(ex)
        exs: list[BaseException] = []
        not_found = False
        for e in ex.exceptions if isinstance(ex, BaseExceptionGroup) else [ex]:
            if (
                isinstance(e, UnexpectedResponse) and e.status_code == 404
            ) or (
                isinstance(e, AioRpcError) and e.code() == StatusCode.NOT_FOUND
            ):
                not_found = True
            else:
                exs.append(e)
        if not_found:
            _log.warning(f'Not initialized: {self.collection_name}')
        if not exs:
            return None
        return flatten_exc(BaseExceptionGroup('Got multiple errors', exs))


async def _aembed_sparse_records(
    fn: SparseEncode | None, records: Sequence[EmbedRecord]
) -> list[qm.SparseVector | None]:
    if not fn:
        return [None for _ in records]

    vectors: list[qm.SparseVector | None] = []
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
            vectors[i] = qm.SparseVector(indices=ids, values=vs)

    return vectors


# -------------------------- from qdrant to native ---------------------------


def _record_to_qd(
    record: EmbedRecord,
    dense_field: str,
    sparse_field: str,
    sparse_vec: qm.SparseVector | None = None,
) -> qm.PointStruct:
    vector: dict[str, qm.Vector] = {}
    if dembs := record.get('embeddings'):
        vector[dense_field] = dembs
    if sparse_vec is not None:
        vector[sparse_field] = sparse_vec
    if not vector:
        raise ValueError(f'Embedding is not set: keys={record.keys()}')

    try:
        return qm.PointStruct(
            id=record['id_'], vector=vector, payload=record['data']
        )
    except ValidationError:
        return qm.PointStruct(
            id=str(record['id_']), vector=vector, payload=record['data']
        )


def _qd_to_record(
    pt: qm.Record | qm.ScoredPoint,
    dense_field_name: str = 'text-dense',
) -> ScoredRecord:
    assert pt.payload is not None
    s = pt.score if isinstance(pt, qm.ScoredPoint) else 1.0
    rec = ScoredRecord(id_=pt.id, data=pt.payload, score=s)

    vecs = pt.vector
    if vecs is None:
        return rec
    if isinstance(vecs, list):
        raise TypeError('Anonimous dense vectors are not supported')
    vec_or_vecs = vecs.get(dense_field_name)
    if vec_or_vecs is not None:
        if isinstance(vec_or_vecs, qm.SparseVector):
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
        query: Callable[[], Awaitable[qm.Prefetch | None]],
        qd: Qdrant,
        with_payload: qm.WithPayloadInterface = True,
    ) -> None:
        self.query = query
        self.qd = qd
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
        q: qm.QueryInterface
        match mode:
            case 'hsf':
                return await self.hsf(other, t, limit)

            case 'rrf':
                weights = [1 - t, t] if t != 0.5 else None
                q = qm.RrfQuery(rrf=qm.Rrf(k=rrf_k, weights=weights))
                return await self._fuse(other, q, limit=limit)

            case 'dbsf':
                q = qm.FusionQuery(fusion=qm.Fusion.DBSF)
                return await self._fuse(other, q, limit=limit)

        raise NotImplementedError

    async def _fuse(
        self, other: '_Request', query: qm.QueryInterface, limit: int = 1
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
        return [_qd_to_record(pt, self.qd.dense_field_name) for pt in pts]

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
        return [_qd_to_record(pt, self.qd.dense_field_name) for pt in pts]

    async def acall(self) -> list[ScoredRecord]:
        ptss = await self.qd._resolve(
            self.query, with_payload=self.with_payload
        )
        if not ptss:
            return []
        [points] = ptss
        _log_scores(points)
        return [_qd_to_record(pt, self.qd.dense_field_name) for pt in points]


def _min_max_scores(
    pts: Sequence[qm.ScoredPoint],
) -> dict[qm.ExtendedPointId, float]:
    return dict(
        zip([p.id for p in pts], min_max(p.score for p in pts), strict=True)
    )


def _log_scores(pts: Sequence[qm.ScoredPoint], name: str = 'records') -> None:
    scores = [p.score for p in pts]
    if not any(scores):
        return
    n, lo, hi = len(scores), min(scores), max(scores)
    _log.info(f'Retrieved {n} {name} with score: {lo:.3g} .. {hi:.3g}')
