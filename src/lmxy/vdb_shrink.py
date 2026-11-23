import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import cast

import numpy as np
import qdrant_client.conversions.common_types as qcm
import qdrant_client.models as qm
from glow import cli
from qdrant_client import AsyncQdrantClient
from tqdm.auto import tqdm

from ._vectors import merge_cos_embeddings


def norm(vectors: np.ndarray) -> np.ndarray:
    vectors /= np.linalg.vector_norm(vectors, axis=1, keepdims=True)
    return vectors


async def scroll(
    client: AsyncQdrantClient,
    collection: str,
    tq: tqdm,
    *,
    batch: int = 1,
    offset: qcm.PointId | None = None,
    vectors: tuple[str, ...] = (),
) -> AsyncGenerator[qm.Record]:
    base = (
        int(str(offset).replace('-', '')[:12], 16)
        if isinstance(offset, str | uuid.UUID)
        else 0
    )

    async def sample(
        offset: qcm.PointId | None,
    ) -> tuple[list[qm.Record], qcm.PointId | None]:
        return await client.scroll(
            collection,
            limit=batch,
            with_payload=False,
            offset=offset,
            with_vectors=list(vectors) or True,
        )

    task = None
    while True:
        if task is None:
            rs, offset = await sample(offset)
        else:
            rs, offset = await task
        tq.update(len(rs)) if rs else None

        if offset is not None:
            task = asyncio.create_task(sample(offset))

        for r in rs:
            yield r

        if offset is None:
            tq.total = tq.n
            return
        tq.set_postfix(next_offset=offset)
        if isinstance(offset, str | uuid.UUID):  # Try to estimate end
            tq.total = round(
                tq.n
                * ((1 << 48) - base)
                / (int(str(offset).replace('-', '')[:12], 16) - base)
            )


async def squeeze_and_write(
    client: AsyncQdrantClient,
    collection: str,
    tqs: tuple[tqdm, tqdm],
    id_: qm.ExtendedPointId,
    vectors: list[list[float]],
    *,
    max_similarity: float,
    src_vector: str,
    dst_vector: str,
    move: bool = False,
    check: bool = False,
    dry: bool = False,
) -> None:
    assert vectors
    aw1 = asyncio.to_thread(
        merge_cos_embeddings,
        norm(np.asarray(vectors, np.float32)),
        max_similarity,
    )
    if not dry and check:
        aw2 = client.query_points(
            collection, id_, with_payload=False, with_vectors=[dst_vector]
        )
        vecs, qrsp = await asyncio.gather(aw1, aw2, return_exceptions=True)
    else:
        vecs = await aw1
        qrsp = ValueError()

    if isinstance(vecs, BaseException):
        raise vecs
    tqs[0].update(vecs.shape[0])
    if dry:
        tqs[1].update()
        return

    do_update = False
    if isinstance(qrsp, BaseException):
        do_update = True
    else:
        pt = qrsp.points[0]
        existing_vectors = cast('dict[str, list[list[float]]]', pt.vector).get(
            dst_vector, []
        )
        do_update = len(existing_vectors) != len(vecs) or not np.allclose(
            vecs,
            np.asarray(existing_vectors, np.float32),
            atol=1e-6,
        )

    if do_update:
        update_points = [
            qm.PointVectors(id=id_, vector={dst_vector: vecs.tolist()})
        ]
        await client.update_vectors(collection, update_points)

    if move and src_vector != dst_vector:
        await client.delete_vectors(
            collection, vectors=[src_vector], points=[id_]
        )

    tqs[1].update()


async def main(
    host: str,
    collection: str,
    *,
    port: int = 6333,
    src_vector: str = 'dense1',
    dst_vector: str = 'dense2',
    max_similarity: float = 0.95,
    timeout: int = 3600,
    offset: str | None = None,
    move: bool = False,
    dry: bool = False,
) -> None:
    cl = AsyncQdrantClient(host=host, port=port, timeout=timeout)
    with (
        tqdm(desc='points written') as tq2,
        tqdm(desc='dst vectors') as tqo,
        tqdm(desc='src vectors') as tqi,
        tqdm(desc='points read') as tq1,
    ):
        async with asyncio.TaskGroup() as tg:
            async for r in scroll(
                cl,
                collection,
                tq1,
                batch=100,
                offset=offset,
                vectors=(src_vector,),
            ):
                vstruct = cast('dict[str, list[list[float]]]', r.vector)
                src_vectors = vstruct.get(src_vector, [])
                if not src_vectors:  # Already moved
                    tq2.update()
                    continue

                tqi.update(len(src_vectors))
                aw = squeeze_and_write(
                    cl,
                    collection,
                    (tqo, tq2),
                    r.id,
                    src_vectors,
                    max_similarity=max_similarity,
                    src_vector=src_vector,
                    dst_vector=dst_vector,
                    move=move,
                    dry=dry,
                )
                tg.create_task(aw)

                while tg._tasks and tq2.n + 60 < tq1.n:  # limit prefetch
                    await asyncio.wait(
                        tg._tasks, return_when='FIRST_COMPLETED'
                    )


if __name__ == '__main__':
    asyncio.run(cli.run(main))
