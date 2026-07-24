from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Callable,
    Generator,
    Iterable,
)
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    ExitStack,
)
from typing import Any, Never, assert_never, overload


class Areturn[T]:
    def __init__(self, obj: T) -> None:
        self.obj = obj

    def __await__(self) -> Generator[Never, Any, T]:
        return genreturn(self.obj)


def genreturn[T](x: T) -> Generator[Never, Any, T]:
    yield from ()
    return x


async def ayield[T](x: T) -> AsyncGenerator[T]:
    yield x


async def ayield_never() -> AsyncGenerator[Never]:
    never: Never
    for never in ():
        assert_never(never)
        yield


@overload
def map_ctx[T, R](
    cm: (
        AbstractAsyncContextManager[AsyncIterable[T] | Iterable[T]]
        | AbstractContextManager[AsyncIterable[T]]
    ),
    fn: Callable[[T], R],
) -> AsyncGenerator[R]: ...
@overload
def map_ctx[T, R](
    cm: AbstractContextManager[Iterable[T]],
    fn: Callable[[T], R],
) -> Generator[R]: ...


def map_ctx[T, R](  # noqa: C901
    cm: (
        AbstractAsyncContextManager[AsyncIterable[T] | Iterable[T]]
        | AbstractContextManager[AsyncIterable[T] | Iterable[T]]
    ),
    fn: Callable[[T], R],
) -> Generator[R] | AsyncGenerator[R]:
    s = ExitStack()
    it = None
    if isinstance(cm, AbstractContextManager):
        it = s.enter_context(cm)
    if isinstance(it, Iterable):

        def call() -> Generator[R]:
            with s:
                yield from map(fn, it)

        return call()

    async def acall() -> AsyncGenerator[R]:
        if isinstance(cm, AbstractAsyncContextManager):
            async with cm as it_:
                if isinstance(it_, AsyncIterable):
                    async for x in it_:
                        yield fn(x)
                else:
                    for x in it_:
                        yield fn(x)
        elif it is not None:
            with s:
                async for x in it:
                    yield fn(x)

    return acall()
