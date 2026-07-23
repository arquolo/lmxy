"""Qdrant vector store, built on top of an existing Qdrant collection."""

__all__ = ['Qdrant']

import asyncio
from collections.abc import Awaitable, Callable, Generator, Sequence
from functools import partial
from typing import Any, NotRequired, TypedDict, cast
from uuid import UUID

from glow import astreaming
from grpc import RpcError
from loguru import logger
from pydantic import BaseModel, PrivateAttr
from qdrant_client import AsyncQdrantClient
from qdrant_client.conversions.common_types import QuantizationConfig
from qdrant_client.fastembed_common import IDF_EMBEDDING_MODELS
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from ._types import Embedding, SparseEncode
from .fastembed import get_sparse_encoder
from .util import aretry, min_max

_Id = int | str | UUID

# embedding/text, top K, score threshold
type DenseQuery = tuple[Embedding, int, float | None]
type SparseQuery = tuple[str, int]

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
    upsert_batch_size: int | None = 64
    query_timeout: float | None = None  # enable to batch upserts
    query_batch_size: int | None = 64
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
        [Sequence[EmbedRecord | _Id]],
        Awaitable[Sequence[_Id]],
    ] = PrivateAttr()
    _qd_query: Callable[
        [Sequence[rest.QueryRequest]],
        Awaitable[Sequence[Sequence[rest.ScoredPoint]]],
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
            _log.warning(
                'Collection %s already exists, skipping collection creation.',
                self.collection_name,
            )
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
                    'Collection %s support '
                    'sparse search, but neither '
                    'sparse_query_fn nor sparse_model was provided',
                    self.collection_name,
                )
            if not self.sparse_doc_fn:
                _log.warning(
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
        ids = await self._update(records)
        return list(ids)

    # CRUD: read
    async def retrieve(
        self,
        ids: Sequence[_Id],
        *,
        with_payload: Sequence[str] | bool = True,
    ) -> Sequence[ScoredRecord]:
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
    ) -> Sequence[ScoredRecord]:
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
        q = sq.lerp(dq, alpha).limit(k) if dq and sq else (dq or sq)
        return (await q) if q else []

    def query1(
        self,
        q: str | Embedding,
        limit: int = 1,
        threshold: float | None = None,
        filters: rest.Filter | None = None,
        with_payload: Sequence[str] | bool = True,
    ) -> '_Request':
        async def call() -> list[ScoredRecord]:
            points = await self.qd_query(
                q, limit, threshold, filters=filters, with_payload=with_payload
            )
            rs = [_qd_to_record(pt, self.dense_field_name) for pt in points]
            _log_scores(rs)
            return rs

        return _Request(call)

    async def qd_retrieve(
        self,
        ids: Sequence[_Id],
        with_payload: Sequence[str] | bool = True,
    ) -> list[rest.Record]:
        if not await self.is_initialized():
            return []
        return await self.aclient.retrieve(
            self.collection_name, ids, with_payload=with_payload
        )

    async def qd_query(
        self,
        q: str | Embedding,
        limit: int = 1,
        threshold: float | None = None,
        filters: rest.Filter | None = None,
        with_payload: Sequence[str] | bool = True,
    ) -> Sequence[rest.Record | rest.ScoredPoint]:
        if not limit or not await self.is_initialized():
            return []

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

        # TODO: possible optimization.
        # Use prefetch={filter=filter_, lookup_from=<other collection>}
        #   and no filter in QueryRequest itself,
        # or call scroll first.
        # But for this we need to ensure that limit is infinite,
        #  otherwise we should use another storage for filters.

        # TODO: handle MMR in qdrant

        if isinstance(with_payload, Sequence):
            with_payload = list(with_payload)
        req = rest.QueryRequest(
            query=vec,
            using=using,
            filter=filters,
            score_threshold=threshold,
            limit=limit,
            with_payload=with_payload,
        )
        [points] = await self._qd_query([req])
        return points

    # CRUD: delete
    async def delete_by(self, value: str, key: str) -> None:
        if not await self.is_initialized():
            return
        cond = rest.FieldCondition(key=key, match=rest.MatchValue(value=value))
        await self.aclient.delete(
            self.collection_name, rest.Filter(must=[cond])
        )

    # CRUD: delete
    async def delete(self, ids: Sequence[str], /) -> None:
        await self._update(ids)

    async def clear(self) -> None:
        async with _LOCK:
            await self.aclient.delete_collection(self.collection_name)
            self._is_initialized = False

    # low levels

    async def _ll_update(
        self, records: Sequence[EmbedRecord | _Id], /
    ) -> Sequence[_Id]:
        # Merge and deduplicate updates & deletions
        ids: list[_Id] = []
        add_recs: list[EmbedRecord] = []
        rm_ids: list[_Id] = []
        for r in records:
            if isinstance(r, _Id):
                ids.append(r)
                rm_ids.append(r)
            else:
                ids.append(r['id_'])
                add_recs.append(r)
        add_recs = list({d['id_']: d for d in add_recs}.values())

        aws: list[Awaitable] = []

        if add_recs:
            rec0 = add_recs[0]
            if not (vecs := rec0.get('embeddings')):
                raise ValueError('No dense vectors to store')
            vec_size = len(vecs[0])
            await self.initialize(vec_size)

            svs = await _aembed_sparse_records(self.sparse_doc_fn, add_recs)
            points = [
                _record_to_qd(
                    r,
                    dense_field=self.dense_field_name,
                    sparse_field=self.sparse_field_name,
                    sparse_vec=sv,
                )
                for r, sv in zip(add_recs, svs, strict=True)
            ]
            aws.append(self.aclient.upsert(self.collection_name, points))

        if rm_ids and await self.is_initialized():
            cond = rest.HasIdCondition(has_id=rm_ids)
            aws.append(
                self.aclient.delete(
                    self.collection_name, rest.Filter(must=[cond])
                )
            )

        if aws:
            await asyncio.gather(*aws)

        return ids

    async def _ll_qd_query(
        self, reqs: Sequence[rest.QueryRequest], /
    ) -> Sequence[Sequence[rest.ScoredPoint]]:
        if not reqs:
            return []
        qrs = await self.aclient.query_batch_points(self.collection_name, reqs)
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


class _Request(partial[Awaitable[list[ScoredRecord]]]):
    def __await__(self) -> Generator[Any, Any, list[ScoredRecord]]:
        return self().__await__()

    def limit(self, n: int) -> '_Request':
        """`= self.scores[:n]`"""
        return _Request(self._limit, n)

    def lerp(self, rhs: '_Request', t: float) -> '_Request':
        """`= normalized(self.scores) * (1-t) + normalized(rhs.scores) * t`"""
        return _Request(self._lerp, rhs, t)

    async def _limit(self, n: int) -> list[ScoredRecord]:
        if n <= 0:
            return []
        rs = await self()
        return rs[:n]

    async def _lerp(self, rhs: '_Request', t: float) -> list[ScoredRecord]:
        if t <= 0:
            return await self()
        if t >= 1:
            return await rhs()
        rs1, rs2 = await asyncio.gather(self(), rhs())
        if not (rs1 and rs2):
            return rs1 or rs2
        uniq = {r['id_']: r for r in [*rs1, *rs2]}
        zeros = dict.fromkeys(uniq, 0.0)
        xs1 = zeros | _min_max_scores(rs1)
        xs2 = zeros | _min_max_scores(rs2)
        rs: list[ScoredRecord] = [
            {**r, 'score': (1 - t) * xs1[i] + t * xs2[i]}
            for i, r in uniq.items()
        ]
        _log_scores(rs, name='fused records')
        return sorted(rs, key=lambda r: r['score'], reverse=True)


def _min_max_scores(rs: Sequence[ScoredRecord]) -> dict[_Id, float]:
    ids = [r['id_'] for r in rs]
    xs = min_max(r['score'] for r in rs)
    return dict(zip(ids, xs, strict=True))


def _log_scores(rs: Sequence[ScoredRecord], name: str = 'records') -> None:
    scores = [r['score'] for r in rs]
    if not any(scores):
        return
    n, lo, hi = len(scores), min(scores), max(scores)
    _log.info(f'Retrieved {n} {name} with score: {lo:.3g} .. {hi:.3g}')
