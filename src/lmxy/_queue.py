__all__ = ['MulticastQueue']

from asyncio import CancelledError, Future
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Literal


class MulticastQueue[T]:
    """Single producer, multiple consumer queue.

    Each consumer gets each message put by producer.
    Late joined consumer gets all messages from the beginning.
    """

    def __init__(self) -> None:
        self._buf: list[T] = []
        self._state: Literal['pending', 'running', 'done'] = 'pending'
        self._waiters = set[Future[None]]()
        self._num_waiters = 0

    async def __aiter__(self) -> AsyncIterator[T]:
        pos = 0
        with self.subscribe():
            while True:
                if pos < len(self._buf):
                    yield self._buf[pos]
                    pos += 1
                elif self._state != 'done':
                    await self._wait_data()
                else:
                    return

    @contextmanager
    def subscribe(self) -> Iterator[None]:
        self._num_waiters += 1
        try:
            yield
        finally:
            self._num_waiters -= 1

    async def put(self, msg: T) -> None:
        if self._state == 'done':
            raise RuntimeError('Cannot put message to closed stream')

        if self._state == 'running' and not self._num_waiters:
            self._state = 'done'
            self._notify_all()
            raise CancelledError('All waiters exited')

        self._state = 'running'
        self._buf.append(msg)
        self._notify_all()

    def close(self) -> None:
        self._state = 'done'
        self._notify_all()

    async def _wait_data(self) -> None:
        f = Future[None]()
        self._waiters.add(f)
        try:
            await f
        finally:
            self._waiters.discard(f)

    def _notify_all(self) -> None:
        for f in self._waiters:
            if not f.done():
                f.set_result(None)
