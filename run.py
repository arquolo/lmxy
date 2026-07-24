import asyncio

from glow import init_loguru, timer
from httpx import Timeout
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from qdrant_client import AsyncQdrantClient
from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.http.models import (
    Datatype,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)

from lmxy import Embedder, Qdrant, QdrantVectorStore

FILES_COLLECTION_NAME = 'default-deepvk--USER-bge-m3_chunks'


async def aget_vector_store(vector_size: int) -> 'QdrantVectorStore':
    aclient = AsyncQdrantClient(
        port=6333,
        grpc_port=6334,
        prefer_grpc=False,
        https=None,
        prefix=None,
        timeout=None,
        host='bs03',
    )
    assert isinstance(aclient._client, AsyncQdrantRemote)
    aclient._client.http.client._async_client.timeout = Timeout(None)

    dense_config = VectorParams(
        size=vector_size,
        distance=Distance.COSINE,
        on_disk=True,
        datatype=Datatype.FLOAT32,
    )
    hnsw_config = HnswConfigDiff(
        # Number of edges per node (16).
        # 12-16, higher = more accuracy / more memory.
        # m=32,
        # ! For multitenancy - requires direct control over nodes
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
    optimizers_config = OptimizersConfigDiff(
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

    # stores Nodes as id(uuid), embedding, metadata, relationships & text
    q = Qdrant(
        collection_name=FILES_COLLECTION_NAME,
        aclient=aclient,
        # upsert_timeout=0.1,
        # query_timeout=0.1,
        upsert_batch_size=10,
        query_batch_size=10,
        retries=10,
        dense_config=dense_config,
        shard_number=4,  # To allow parallel indexing
        hnsw_config=hnsw_config,
        tenant_fields=['subset'],
        optimizers_config=optimizers_config,
        sparse_model='Qdrant/bm25',
        sparse_model_kwargs={'language': 'russian', 'disable_stemmer': True},
    )
    return QdrantVectorStore(qdrant=q)


async def main():
    with timer('connect to embedder'):
        embedder = Embedder(base_url='http://bs03:8010', latency=0.1)
        await embedder.handshake()

    with timer('embed test'):
        e = await embedder.aget_text_embedding('test')
        vector_size = len(e)

    with timer('vector init'):
        qvs = await aget_vector_store(vector_size)

    # with timer('vector add'):
    #     tn = TextNode(
    #         id=12345,
    #         text='hello peter',
    #         metadata={
    #             'id': 12345,
    #             'card': {'a': ['b', 'c']},
    #             'embeddings': embedder.get_text_embedding_batch(['aaa', 'bbb']),
    #         },
    #         excluded_embed_metadata_keys=['embeddings', 'id'],
    #         excluded_llm_metadata_keys=['embeddings', 'id'],
    #     )
    #     print(tn)
    #     await qvs.add([tn])

    # with timer('vector add'):
    #     await qvs.qdrant.add(
    #         [
    #             {
    #                 'id_': 56789,
    #                 'data': {'q': 'v'},
    #                 'embeddings': embedder.get_text_embedding_batch(
    #                     ['hello john']
    #                 ),
    #                 'embed_text': 'hello john',
    #             }
    #         ]
    #     )

    squery = 'hello'
    with timer('embed query'):
        dquery = 'aaa'
        e = await embedder.aget_query_embedding(dquery)
    print('emb', len(e))

    alpha = 0.5
    dk, sk, hk = 3, 3, 3

    # Native complex
    # with timer('query native complex'):
    #     recs_nc = await qvs.qdrant.query(
    #         dense=(e, dk, None),
    #         sparse=(squery, sk),
    #         fuse=(hk, alpha),
    #     )
    # print(len(recs_nc))
    # print(*recs_nc, sep='\n')

    # Native simple
    with timer('query native'):
        sq = qvs.qdrant.query1(squery, sk) if alpha < 1 else None
        dq = qvs.qdrant.query1(e, dk) if alpha > 0 else None
        if sq and dq:
            recs = await sq.lerp(dq, alpha).limit(hk)
        elif v := (sq or dq):
            recs = await v
        else:
            recs = []
    print(len(recs))
    print(*recs, sep='\n')

    # Llama
    with timer('query llama'):
        nodes = await qvs.query(
            VectorStoreQuery(
                query_embedding=e,
                similarity_top_k=dk,
                query_str=squery,
                mode=VectorStoreQueryMode.HYBRID,
                alpha=alpha,
                sparse_top_k=sk,
                hybrid_top_k=hk,
            )
        )
    print(len(nodes))
    print(*nodes, sep='\n')

    # assert recs_nc == recs
    assert [r['id_'] for r in recs] == [n.id_ for n, _ in nodes]
    scores_l = [r['score'] for r in recs]
    scores_n = [x for _, x in nodes]
    assert scores_l == scores_n, (scores_l, scores_n)


if __name__ == '__main__':
    init_loguru('INFO')
    asyncio.run(main())
