__all__ = ['MulticastQueue']

from asyncio import CancelledError, Future
from collections.abc import AsyncGenerator, Awaitable, Iterable
from typing import Literal, Self
from weakref import finalize


class QueueShutdownError(Exception):
    """Raised when putting on to or getting from a shut-down Queue."""


class MulticastQueue[T]:
    """Single producer, multiple consumer queue.

    Each consumer gets each message put by producer.
    Late joined consumer gets all messages from the beginning.

    Usage:

        mq = MulticastQueue()
        async def worker():
            with mq:
                for x in range(3):
                    mq.put(x)

        t = asyncio.create_task(worker())
        assert [x async for x in mq] == [0, 1, 2]
        assert [x async for x in mq] == [0, 1, 2]

    """

    def __init__(self) -> None:
        self._buf: list[T] = []
        self._pos = 0
        self._putters = set[Future[None]]()
        self._getters = set[Future[T]]()
        self._state: Literal['running', 'done', 'closed'] = 'running'
        self._nsubs = 0

    # ------------------------------- consumer -------------------------------

    async def __aiter__(self) -> AsyncGenerator[T]:
        # finally doesn't work unless `aclose` is manually called,
        # or anything is `await`ed (even via `asyncio.sleep(0.001)`)
        with _QueueIterator(self) as it:
            async for x in it:
                yield x

    def subscribe(self) -> '_QueueIterator[T]':
        return _QueueIterator(self)

    def aget(self, idx: int) -> Awaitable[T]:
        self._pos = max(self._pos, idx)

        if idx < len(self._buf):  # Read from buffer
            value = self._buf[idx]
            f = Future[T]()
            f.set_result(value)
            return f

        if self._state == 'closed':  # Closed before finish
            raise QueueShutdownError

        if self._state == 'done':  # Finished
            raise IndexError

        # Wait for the new data
        _release_all(self._putters, None)
        return _acquire(self._getters)

    # ------------------------------- producer -------------------------------

    def __enter__(self) -> None:
        pass

    def __exit__(self, *_) -> None:
        if self._state == 'running':
            self._state = 'done'
            assert not self._putters
            _cancel_all(self._getters, msg='Queue is finalized')

    def put(self, value: T) -> Awaitable[None]:
        if self._state != 'running':
            raise QueueShutdownError

        # Pass value to all getters
        self._buf.append(value)
        _release_all(self._getters, value)

        # Wait till all of them succeed
        return _acquire(self._putters)

    # ------------------------------- private --------------------------------

    def incref(self) -> None:
        self._nsubs += 1

    def decref(self) -> None:
        self._nsubs -= 1

        # No waiters, close
        if not self._nsubs and self._state == 'running':
            self._state = 'closed'
            assert not self._getters
            _cancel_all(self._putters, msg='No waiters to store values for')


class _QueueIterator[T]:
    def __init__(self, mq: MulticastQueue[T]) -> None:
        self._mq = mq
        self._pos = 0
        mq.incref()
        self.close = finalize(self, mq.decref)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        try:
            value = await self._mq.aget(self._pos)
        except (IndexError, CancelledError):
            raise StopAsyncIteration from None
        else:
            self._pos += 1
            return value


async def _acquire[T](waiters: set[Future[T]]) -> T:
    f = Future[T]()
    waiters.add(f)
    try:
        return await f
    finally:
        f.cancel()
        waiters.discard(f)


def _release_all[T](waiters: Iterable[Future[T]], value: T) -> None:
    for f in waiters:
        if not f.done():
            f.set_result(value)


def _cancel_all(waiters: Iterable[Future], msg: str | None = None) -> None:
    for f in waiters:
        f.cancel(msg)
        f.cancel(msg)
