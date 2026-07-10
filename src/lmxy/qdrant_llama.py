"""Qdrant vector store, built on top of an existing Qdrant collection."""

__all__ = ['QdrantVectorStore']

from collections.abc import Sequence
from typing import cast

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from pydantic import BaseModel
from qdrant_client.http import models as rest

from ._nodes import metadata_dict_to_node, node_to_metadata_dict
from ._types import Embedding
from .qdrant import EmbedRecord, Qdrant, Record

type _ScoredNode = tuple[BaseNode, float]


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
    qdrant: Qdrant

    # CRUD: create or update
    async def add(self, nodes: Sequence['BaseNode'], /) -> Sequence[str]:
        """Add nodes with embeddings to Qdrant index.

        Returns node IDs that were added to the index.
        """
        records = [_node_to_record(n) for n in nodes]
        ids = await self.qdrant.add(records)
        assert all(isinstance(i, str) for i in ids)
        return cast('list[str]', ids)

    # CRUD: read
    async def query(
        self,
        query: 'VectorStoreQuery',
        /,
        *,
        qdrant_filters: rest.Filter | None = None,
        with_payload: Sequence[str] | bool = True,
        dense_threshold: float | None = None,
    ) -> Sequence['_ScoredNode']:
        """Query index for top k most similar nodes."""
        #  NOTE: users can pass in qdrant_filters
        # (nested/complicated filters) to override the default MetadataFilters
        if qdrant_filters is None:
            qdrant_filters = _build_filter(
                query.doc_ids, query.node_ids, query.filters
            )

        if query.query_embedding is None and query.query_str is None:
            assert query.node_ids
            assert query.doc_ids is None
            assert query.filters is None
            records = await self.qdrant.retrieve(
                query.node_ids, with_payload=with_payload
            )
        else:
            q = query
            match q.mode.value:
                case 'default':
                    alpha = 1.0
                case 'hybrid':
                    alpha = 0.5 if q.alpha is None else max(0, min(q.alpha, 1))
                case 'sparse':
                    alpha = 0.0
                case _ as unsupported:
                    msg = f'Unsupported query mode: {unsupported}'
                    raise NotImplementedError(msg)

            top_k = q.similarity_top_k
            sparse_k = top_k if q.sparse_top_k is None else q.sparse_top_k
            fuse_k = top_k if q.hybrid_top_k is None else q.hybrid_top_k

            dq = sq = None
            if q.query_embedding and top_k and alpha > 0:
                dq = self.qdrant.query1(
                    q.query_embedding,
                    top_k,
                    dense_threshold,
                    filters=qdrant_filters,
                    with_payload=with_payload,
                )
            if q.query_str and sparse_k and alpha < 1:
                sq = self.qdrant.query1(
                    q.query_str,
                    sparse_k,
                    filters=qdrant_filters,
                    with_payload=with_payload,
                )
            rsp = (
                (dq.unit() * alpha + sq.unit() * (1 - alpha)).limit(fuse_k)
                if dq and sq
                else (dq or sq)
            )
            records = (await rsp) if rsp else []

        return [(_record_to_node(r), r['score']) for r in records]

    # CRUD: delete
    async def delete(self, doc_id: str) -> None:
        return await self.qdrant.delete_by(doc_id, key='doc_id')

    async def delete_nodes(self, node_ids: Sequence[str], /) -> None:
        await self.qdrant.delete(node_ids)

    async def clear(self) -> None:
        await self.qdrant.clear()


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
    if mfs.condition is None:
        return rest.Filter()

    match mfs.condition.value:
        case 'and':
            return rest.Filter(must=conditions)
        case 'or':
            return rest.Filter(should=conditions)
        case 'not':
            return rest.Filter(must_not=conditions)
        case _ as unknown:
            raise NotImplementedError(f'Unknown FilterCondition: {unknown}')


def _meta_to_condition(f: 'MetadataFilter') -> rest.Condition | None:
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


# ------------------------ from native to llama index ------------------------


def _record_to_node(record: Record) -> BaseNode:
    node = metadata_dict_to_node(record['data'], with_id=record['id_'])
    if (
        node.embedding is None
        and not node.metadata.get('embeddings')
        and (vecs := record.get('embeddings'))
    ):
        if len(vecs) == 1:
            node.embedding = vecs[0]
        else:
            node.metadata['embeddings'] = vecs
    return node


def _node_to_record(node: BaseNode) -> EmbedRecord:
    r = EmbedRecord(id_=node.id_, data=node_to_metadata_dict(node))

    if text := node.get_content(MetadataMode.EMBED).strip():
        r['embed_text'] = text

    if (vec := node.embedding) is not None:
        r['embeddings'] = [vec]
    elif vecs := node.metadata.get('embeddings'):
        r['embeddings'] = cast('list[Embedding]', vecs)

    return r
