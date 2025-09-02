import asyncio
from asyncio import CancelledError, QueueShutDown

import pytest

from lmxy import MulticastQueue


async def worker(mq: MulticastQueue[int], total: int) -> None:
    with mq:
        for x in range(total):
            await mq.put(x)


@pytest.mark.asyncio
async def test_seq():
    mq = MulticastQueue[int]()
    t = asyncio.create_task(worker(mq, 2))

    seq = [x async for x in mq]
    assert seq == [0, 1]

    seq = [x async for x in mq]
    assert seq == [0, 1]

    await t


@pytest.mark.asyncio
async def test_overlap():
    mq = MulticastQueue[int]()
    t = asyncio.create_task(worker(mq, 2))

    ag1 = aiter(mq)
    x1 = await anext(ag1)
    assert x1 == 0

    ag2 = aiter(mq)
    x1, x2 = await asyncio.gather(anext(ag1), anext(ag2))
    assert x1 == 1
    assert x2 == 0

    x1, x2 = await asyncio.gather(anext(ag1, None), anext(ag2))
    assert x1 is None
    assert x2 == 1

    with pytest.raises(StopAsyncIteration):
        await anext(ag2)

    await t


@pytest.mark.asyncio
async def test_broken_sub_anext():
    mq = MulticastQueue[int]()
    t = asyncio.create_task(worker(mq, 2))

    qi1 = mq.subscribe()
    x1 = await anext(qi1)
    assert x1 == 0

    await asyncio.sleep(0.001)  # Awake worker

    ag2 = aiter(mq)
    try:
        x2 = await anext(ag2)
        assert x2 == 0
        x2 = await anext(ag2)
        assert x2 == 1
        with pytest.raises(StopAsyncIteration):
            x2 = await anext(ag2)
    finally:
        await ag2.aclose()

    await t


@pytest.mark.asyncio
async def test_broken_sub_with_anext():
    mq = MulticastQueue[int]()
    t = asyncio.create_task(worker(mq, 2))

    with mq.subscribe() as qi1:
        x1 = await anext(qi1)
    assert x1 == 0

    await asyncio.sleep(0.001)  # Awake worker

    ag2 = aiter(mq)
    try:
        x2 = await anext(ag2)
        assert x2 == 0
        # x2 = await anext(ag2)
        # assert x2 == 1
        with pytest.raises(QueueShutDown):
            x2 = await anext(ag2)
    finally:
        await ag2.aclose()

    assert t.cancelled()
    with pytest.raises(CancelledError):
        await t


@pytest.mark.asyncio
async def test_broken_aiter_anext():
    mq = MulticastQueue[int]()
    t = asyncio.create_task(worker(mq, 2))

    ag1 = aiter(mq)
    x1 = await anext(ag1)
    assert x1 == 0

    del ag1
    await asyncio.sleep(0.001)  # Awake worker

    ag2 = aiter(mq)
    try:
        x2 = await anext(ag2)
        assert x2 == 0
        with pytest.raises(QueueShutDown):
            x2 = await anext(ag2)
    finally:
        await ag2.aclose()

    assert t.cancelled()
    with pytest.raises(CancelledError):
        await t


@pytest.mark.asyncio
async def test_broken_aiter_anext_aclose():
    mq = MulticastQueue[int]()
    t = asyncio.create_task(worker(mq, 2))

    ag1 = aiter(mq)
    x1 = await anext(ag1)
    assert x1 == 0

    await ag1.aclose()

    ag2 = aiter(mq)
    try:
        x2 = await anext(ag2)
        assert x2 == 0
        with pytest.raises(QueueShutDown):
            x2 = await anext(ag2)
    finally:
        await ag2.aclose()

    with pytest.raises(CancelledError):
        await t
